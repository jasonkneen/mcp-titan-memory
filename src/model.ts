import * as tf from '@tensorflow/tfjs';
import { ITensor, IMemoryModel, TensorWrapper, wrapTensor, unwrapTensor } from './types.js';
import * as fs from 'fs/promises';

export interface TitanMemoryConfig {
  inputDim?: number;
  hiddenDim?: number;
  outputDim?: number; // We'll treat this as "memoryDim" internally
  memoryDim?: number; // Explicit memory dimension parameter
  learningRate?: number;
  useManifold?: boolean;
  momentumFactor?: number;
  forgetGateInit?: number;
  maxStepSize?: number;
  tangentEpsilon?: number;
  numHeads?: number; // Number of attention heads
  numLayers?: number; // Number of hierarchical memory layers
  useMemoryReplay?: boolean; // Enable memory replay for enhanced learning
  replayBufferSize?: number; // Size of memory replay buffer
  compressionRate?: number; // Rate for memory compression
  longTermMemorySize?: number; // Size of long-term memory storage
  dynamicAllocation?: boolean; // Enable dynamic memory allocation
  cacheTTL?: number; // Time-to-live for LLM cache entries in milliseconds
}

interface ForwardResult extends tf.TensorContainerObject {
  predicted: ITensor;
  newMemory: ITensor;
  surprise: ITensor;
  [key: string]: any;
}

export class TitanMemoryModel implements IMemoryModel {
  private inputDim: number;
  private hiddenDim: number;
  private memoryDim: number;
  private learningRate: number;
  public useManifold: boolean;
  private momentumFactor: number;
  private forgetGateInit: number;
  private maxStepSize: number;
  private tangentEpsilon: number;
  private numHeads: number;
  private numLayers: number;
  
  // Advanced memory features
  private useMemoryReplay: boolean;
  private replayBufferSize: number;
  private compressionRate: number;
  private longTermMemorySize: number;
  private dynamicAllocation: boolean;
  private cacheTTL: number;

  // New memory mechanisms
  private persistentDim: number;
  private useMomentum: boolean;
  private variant: 'mac' | 'mag' | 'mal';
  private persistentMemory: tf.Variable | null;
  private momentumMemory: tf.Variable | null;
  
  // Memory replay buffer
  private replayBuffer: Array<{input: number[], memory: number[], target: number[]}>;
  
  // Long-term memory storage
  private longTermMemory: Array<{key: string, value: number[], timestamp: number}>;
  
  // LLM Cache
  private llmCache: Map<string, {value: number[], timestamp: number}>;

  // For convenience, total output dimension = memoryDim + inputDim
  private fullOutputDim: number;

  // Trainable parameters
  private W1: tf.Variable;
  private b1: tf.Variable;
  private W2: tf.Variable;
  private b2: tf.Variable;
  private forgetGate: tf.Variable;
  private optimizer: tf.Optimizer;

  // Attention parameters
  private queryWeights: tf.Variable[];
  private keyWeights: tf.Variable[];
  private valueWeights: tf.Variable[];
  private attentionFinalOutputWeight: tf.Variable;

  // Hierarchical memory
  private hierarchicalMemory: tf.Variable[];
  
  // Memory compression parameters
  private compressionWeights: tf.Variable;
  private compressionBias: tf.Variable;

  constructor(config: TitanMemoryConfig = {}) {
    this.inputDim = config.inputDim || 64;
    this.hiddenDim = config.hiddenDim || 32;

    // Use memoryDim directly if available, fall back to outputDim for backward compatibility
    this.memoryDim = config.memoryDim || config.outputDim || 64;

    this.fullOutputDim = this.inputDim + this.memoryDim;

    this.learningRate = config.learningRate || 1e-3;
    this.useManifold = config.useManifold || false;
    this.momentumFactor = config.momentumFactor || 0.9;
    this.forgetGateInit = config.forgetGateInit || 0.01;
    this.maxStepSize = config.maxStepSize || 0.1;
    this.tangentEpsilon = config.tangentEpsilon || 1e-8;
    this.numHeads = config.numHeads || 4;
    this.numLayers = config.numLayers || 3;
    
    // Initialize advanced memory features
    this.useMemoryReplay = config.useMemoryReplay || false;
    this.replayBufferSize = config.replayBufferSize || 1000;
    this.compressionRate = config.compressionRate || 0.5;
    this.longTermMemorySize = config.longTermMemorySize || 10000;
    this.dynamicAllocation = config.dynamicAllocation || false;
    this.cacheTTL = config.cacheTTL || 3600000; // Default 1 hour

    // New memory mechanism settings
    this.persistentDim = config.persistentDim || 0;
    this.useMomentum = config.useMomentum || false;
    this.variant = config.variant || 'mac';
    
    // Initialize memory structures
    this.replayBuffer = [];
    this.longTermMemory = [];
    this.llmCache = new Map();

    // Initialize trainable parameters
    // First layer receives inputDim + memoryDim + persistentDim:
    const inputSize = this.inputDim + this.memoryDim + this.persistentDim;
    this.W1 = tf.variable(tf.randomNormal([this.hiddenDim, inputSize], 0, 0.1));
    this.b1 = tf.variable(tf.zeros([this.hiddenDim]));

    // Second layer outputs (memoryDim + inputDim):
    this.W2 = tf.variable(tf.randomNormal([this.fullOutputDim, this.hiddenDim], 0, 0.1));
    this.b2 = tf.variable(tf.zeros([this.fullOutputDim]));

    // Forget gate
    this.forgetGate = tf.variable(tf.scalar(this.forgetGateInit));

    // Initialize optimizer
    this.optimizer = tf.train.adam(this.learningRate);

    // Initialize attention parameters
    this.queryWeights = [];
    this.keyWeights = [];
    this.valueWeights = [];
    for (let i = 0; i < this.numHeads; i++) {
      this.queryWeights.push(tf.variable(tf.randomNormal([this.memoryDim, this.memoryDim], 0, 0.1)));
      this.keyWeights.push(tf.variable(tf.randomNormal([this.memoryDim, this.memoryDim], 0, 0.1)));
      this.valueWeights.push(tf.variable(tf.randomNormal([this.memoryDim, this.memoryDim], 0, 0.1)));
    }
    this.attentionFinalOutputWeight = tf.variable(tf.randomNormal([this.numHeads * this.memoryDim, this.memoryDim], 0, 0.1));

    // Initialize hierarchical memory
    this.hierarchicalMemory = [];
    for (let i = 0; i < this.numLayers; i++) {
      this.hierarchicalMemory.push(tf.variable(tf.zeros([this.memoryDim])));
    }
    
    // Initialize memory compression parameters
    this.compressionWeights = tf.variable(tf.randomNormal([Math.floor(this.memoryDim * this.compressionRate), this.memoryDim], 0, 0.1));
    this.compressionBias = tf.variable(tf.zeros([Math.floor(this.memoryDim * this.compressionRate)]));

    // Initialize persistent and momentum memories
    this.persistentMemory = this.persistentDim > 0 ?
      tf.variable(tf.randomNormal([this.persistentDim], 0, 0.1)) :
      null;
    this.momentumMemory = this.useMomentum ?
      tf.variable(tf.zeros([this.memoryDim])) :
      null;
  }

  private multiHeadAttention(query: tf.Tensor, key: tf.Tensor, value: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const attentionHeads = [];
      for (let i = 0; i < this.numHeads; i++) {
        const q = tf.matMul(query.reshape([1, -1]), this.queryWeights[i]);
        const k = tf.matMul(key.reshape([1, -1]), this.keyWeights[i]);
        const v = tf.matMul(value.reshape([1, -1]), this.valueWeights[i]);

        const attentionScores = tf.softmax(tf.matMul(q, k.transpose()).div(tf.scalar(Math.sqrt(this.memoryDim))));
        const attentionOutput = tf.matMul(attentionScores, v);

        attentionHeads.push(attentionOutput);
      }

      const concatenatedHeads = tf.concat(attentionHeads, -1);
      const output = tf.matMul(concatenatedHeads, this.attentionFinalOutputWeight);

      return output;
    });
  }

  public forward(xTensor: ITensor, memoryState: ITensor): ForwardResult {
    return tf.tidy(() => {
      const x = unwrapTensor(xTensor);       // shape [inputDim]
      const memory = unwrapTensor(memoryState); // shape [memoryDim]

      // Gate the memory state - wrap in tidy to ensure cleanup
      const gatedMemory = tf.tidy(() => {
        const forgetVal = this.forgetGate; // shape []
        const one = tf.scalar(1.0);
        return tf.mul(memory, tf.sub(one, forgetVal)); // shape [memoryDim]
      });

      // Combine input, gated memory, and persistent memory if available
      const parts: tf.Tensor[] = [x, gatedMemory];
      if (this.persistentMemory) {
        parts.push(this.persistentMemory);
      }
      const combined = tf.concat(parts, 0);
      const combinedReshaped = combined.reshape([1, this.inputDim + this.memoryDim + this.persistentDim]);

      // MLP forward pass
      const hidden1 = tf.add(
        tf.matMul(combinedReshaped, this.W1.transpose()),
        this.b1
      ).relu(); // shape [1, hiddenDim]

      let out = tf.add(
        tf.matMul(hidden1, this.W2.transpose()),
        this.b2
      ).squeeze(); 

      // If out was completely scalar (which it shouldn't be), fix shape:
      if (out.shape.length === 0) {
        out = out.reshape([1]);
      }

      // Split output into new memory [0: memoryDim], predicted [memoryDim: memoryDim+inputDim]
      const newMemory = out.slice([0], [this.memoryDim]);
      const predicted = out.slice([this.memoryDim], [this.inputDim]);

      // Calculate surprise (MSE between predicted and x)
      const diff = tf.sub(predicted, x);
      const surprise = tf.mean(tf.square(diff)); // scalar

      // Momentum memory update
      if (this.useMomentum && this.momentumMemory) {
        const momentumUpdate = tf.add(
          this.momentumMemory.mul(tf.scalar(this.momentumFactor)),
          newMemory.mul(tf.scalar(1 - this.momentumFactor))
        );
        this.momentumMemory.assign(momentumUpdate);
      }

      // Multi-head attention for memory update
      const attentionOutput = this.multiHeadAttention(newMemory, memory, memory);

      // Hierarchical memory update
      for (let i = 0; i < this.numLayers; i++) {
        const layerMemory = this.hierarchicalMemory[i];
        const updatedLayerMemory = tf.add(layerMemory, attentionOutput.reshape(layerMemory.shape));
        this.hierarchicalMemory[i].assign(updatedLayerMemory);
      }
      
      // Store in replay buffer if memory replay is enabled
      if (this.useMemoryReplay) {
        this.addToReplayBuffer(x.flatten().arraySync() as number[], memory.flatten().arraySync() as number[], predicted.flatten().arraySync() as number[]);
      }
      
      // Integrate momentum memory if enabled
      let integrationMemory = newMemory;
      if (this.useMomentum && this.momentumMemory) {
        if (this.variant === 'mag') {
          const gate = tf.sigmoid(this.forgetGate);
          integrationMemory = tf.add(
            tf.mul(gate, newMemory),
            tf.mul(tf.sub(tf.scalar(1.0), gate), this.momentumMemory)
          );
        }
      }

      // Dynamic memory allocation if enabled
      let dynamicMemory = integrationMemory;
      if (this.dynamicAllocation) {
        dynamicMemory = this.allocateMemoryDynamically(integrationMemory, surprise);
      }
      
      // Cache the result if surprise is low (indicating high confidence)
      const surpriseValue = surprise.dataSync()[0];
      if (surpriseValue < 0.01) {
        const cacheKey = this.generateCacheKey(x.flatten().arraySync() as number[]);
        this.cacheMemoryState(cacheKey, dynamicMemory.flatten().arraySync() as number[]);
      }

      return {
        predicted: wrapTensor(predicted),
        newMemory: wrapTensor(dynamicMemory),
        surprise: wrapTensor(surprise)
      };
    });
  }

  public manifoldStep(base: ITensor, velocity: ITensor): ITensor {
    // Riemannian "update" if useManifold is true
    if (!this.useManifold) {
      // Standard Euclidean update
      return wrapTensor(tf.add(unwrapTensor(base), unwrapTensor(velocity)));
    }

    const result = tf.tidy<ITensor>(() => {
      const baseTensor = unwrapTensor(base);
      const velocityTensor = unwrapTensor(velocity);

      const dot = baseTensor.mul(velocityTensor).sum(); // shape []
      const radial = baseTensor.mul(dot);               // shape [inputDim]
      const tangent = velocityTensor.sub(radial);       // shape [inputDim]
      const tnorm = tangent.norm();                     // shape []

      const tNormVal = tnorm.dataSync()[0];
      if (tNormVal < this.tangentEpsilon) {
        // Very small velocity => no movement
        return wrapTensor(baseTensor);
      }

      const stepSize = Math.min(tNormVal, this.maxStepSize);
      const direction = tangent.div(tf.scalar(tNormVal));
      const cosV = tf.cos(tf.scalar(stepSize));
      const sinV = tf.sin(tf.scalar(stepSize));
      const part1 = baseTensor.mul(cosV);
      const part2 = direction.mul(sinV);
      const newParam = part1.add(part2);
      const newParamNorm = newParam.norm();

      // Return a normalized param
      return wrapTensor(newParam.div(newParamNorm.add(tf.scalar(1e-12))));
    });

    return result;
  }

  public trainStep(x_t: ITensor, x_next: ITensor, memoryState: ITensor): ITensor {
    return tf.tidy(() => {
      const xt = unwrapTensor(x_t);
      const xn = unwrapTensor(x_next);
      const mem = unwrapTensor(memoryState);

      // Minimizing a combined loss
      const cost = this.optimizer.minimize(() => {
        const { predicted, newMemory, surprise } = this.forward(x_t, memoryState);

        // MSE wrt x_next plus a small "surprise" penalty
        const diff = tf.sub(unwrapTensor(predicted), xn);
        const mse = tf.mean(tf.square(diff)).asScalar();
        const spr = unwrapTensor(surprise);

        // Clean up intermediate tensors
        diff.dispose();
        predicted.dispose();
        newMemory.dispose();

        // Weighted combination
        const totalLoss = tf.add(mse, tf.mul(tf.scalar(0.01), spr)).asScalar();

        // Clean up more tensors
        mse.dispose();
        spr.dispose();

        return totalLoss;
      }, true);

      // If memory replay is enabled, perform replay training
      if (this.useMemoryReplay && this.replayBuffer.length > 0) {
        this.trainOnReplayBuffer();
      }

      // If cost is null for some reason, return scalar(0)
      const result = wrapTensor(cost || tf.scalar(0));

      return result;
    });
  }

  public async saveModel(path: string): Promise<void> {
    // Extract weights data without creating intermediate tensors
    const weights = {
      W1: await this.W1.array(),
      b1: await this.b1.array(),
      W2: await this.W2.array(),
      b2: await this.b2.array(),
      forgetGate: await this.forgetGate.array(),
      queryWeights: await Promise.all(this.queryWeights.map(w => w.array())),
      keyWeights: await Promise.all(this.keyWeights.map(w => w.array())),
      valueWeights: await Promise.all(this.valueWeights.map(w => w.array())),
      attentionFinalOutputWeight: await this.attentionFinalOutputWeight.array(),
      hierarchicalMemory: await Promise.all(this.hierarchicalMemory.map(m => m.array())),
      compressionWeights: await this.compressionWeights.array(),
      compressionBias: await this.compressionBias.array(),
      persistentMemory: this.persistentMemory ? await this.persistentMemory.array() : null,
      momentumMemory: this.momentumMemory ? await this.momentumMemory.array() : null,
      // Save advanced memory structures
      replayBuffer: this.replayBuffer,
      longTermMemory: this.longTermMemory,
      llmCache: Array.from(this.llmCache.entries())
    };

    // Use proper path normalization
    const normalizedPath = path.replace('file://', '');
    await fs.writeFile(normalizedPath, JSON.stringify(weights));
  }

  public async loadModel(path: string): Promise<void> {
    try {
      // Normalize path
      const normalizedPath = path.replace('file://', '');
      const weightsJson = await fs.readFile(normalizedPath, 'utf8');
      const weights = JSON.parse(weightsJson);

      // Use tf.tidy to handle intermediate tensors
      tf.tidy(() => {
        // Assign weights to variables
        this.W1.assign(tf.tensor2d(weights.W1));
        this.b1.assign(tf.tensor1d(weights.b1));
        this.W2.assign(tf.tensor2d(weights.W2));
        this.b2.assign(tf.tensor1d(weights.b2));
        this.forgetGate.assign(tf.scalar(weights.forgetGate));
        
        // Handle array-based weights
        this.queryWeights.forEach((w, i) => w.assign(tf.tensor2d(weights.queryWeights[i])));
        this.keyWeights.forEach((w, i) => w.assign(tf.tensor2d(weights.keyWeights[i])));
        this.valueWeights.forEach((w, i) => w.assign(tf.tensor2d(weights.valueWeights[i])));
        this.attentionFinalOutputWeight.assign(tf.tensor2d(weights.attentionFinalOutputWeight));
        this.hierarchicalMemory.forEach((m, i) => m.assign(tf.tensor1d(weights.hierarchicalMemory[i])));

        // Load compression weights if they exist
        if (weights.compressionWeights && weights.compressionBias) {
          this.compressionWeights.assign(tf.tensor2d(weights.compressionWeights));
          this.compressionBias.assign(tf.tensor1d(weights.compressionBias));
        }

        if (weights.persistentMemory && this.persistentMemory) {
          this.persistentMemory.assign(tf.tensor1d(weights.persistentMemory));
        }

        if (weights.momentumMemory && this.momentumMemory) {
          this.momentumMemory.assign(tf.tensor1d(weights.momentumMemory));
        }
      });
      
      // Load advanced memory structures if they exist
      if (weights.replayBuffer) {
        this.replayBuffer = weights.replayBuffer;
      }
      
      if (weights.longTermMemory) {
        this.longTermMemory = weights.longTermMemory;
      }
      
      if (weights.llmCache) {
        this.llmCache = new Map(weights.llmCache);
      }
    } catch (error) {
      throw new Error(`Failed to load model from ${path}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  public getConfig(): TitanMemoryConfig {
    return {
      inputDim: this.inputDim,
      hiddenDim: this.hiddenDim,
      memoryDim: this.memoryDim, // Use memoryDim consistently
      outputDim: this.memoryDim, // Keep outputDim for backward compatibility
      learningRate: this.learningRate,
      useManifold: this.useManifold,
      momentumFactor: this.momentumFactor,
      forgetGateInit: this.forgetGateInit,
      maxStepSize: this.maxStepSize,
      tangentEpsilon: this.tangentEpsilon,
      numHeads: this.numHeads,
      numLayers: this.numLayers,
      useMemoryReplay: this.useMemoryReplay,
      replayBufferSize: this.replayBufferSize,
      compressionRate: this.compressionRate,
      longTermMemorySize: this.longTermMemorySize,
      dynamicAllocation: this.dynamicAllocation,
      cacheTTL: this.cacheTTL,
      persistentDim: this.persistentDim,
      useMomentum: this.useMomentum,
      variant: this.variant
    };
  }

  public getWeights() {
    return {
      W1: this.W1.arraySync(),
      b1: this.b1.arraySync(),
      W2: this.W2.arraySync(),
      b2: this.b2.arraySync(),
      forgetGate: this.forgetGate.arraySync(),
      queryWeights: this.queryWeights.map(w => w.arraySync()),
      keyWeights: this.keyWeights.map(w => w.arraySync()),
      valueWeights: this.valueWeights.map(w => w.arraySync()),
      attentionFinalOutputWeight: this.attentionFinalOutputWeight.arraySync(),
      hierarchicalMemory: this.hierarchicalMemory.map(m => m.arraySync()),
      compressionWeights: this.compressionWeights.arraySync(),
      compressionBias: this.compressionBias.arraySync(),
      persistentMemory: this.persistentMemory ? this.persistentMemory.arraySync() : null,
      momentumMemory: this.momentumMemory ? this.momentumMemory.arraySync() : null
    };
  }

  public getLayerMemoryState(layerIndex: number): number[] {
    if (layerIndex < 0 || layerIndex >= this.numLayers) {
      throw new Error(`Layer index out of bounds: ${layerIndex}`);
    }
    // Ensure proper conversion to number[]
    return Array.from(this.hierarchicalMemory[layerIndex].dataSync());
  }


  public train_sequence(sequence: ITensor[], epochs: number = 1): number[] {
    const costs: number[] = [];

    try {
      for (let epoch = 0; epoch < epochs; epoch++) {
        let epochCost = 0;

        // Use tidy to clean up all intermediate tensors
        tf.tidy(() => {
          let memory = this.zeroVector(this.getConfig().outputDim);

          for (let i = 0; i < sequence.length - 1; i++) {
            const x_t = sequence[i];
            const x_next = sequence[i + 1];

            // Wrap in another tidy to clear intermediate tensors for each step
            tf.tidy(() => {
              const cost = this.trainStep(x_t, x_next, memory);
              const { newMemory } = this.forward(x_t, memory);

              // Update memory for next iteration (making sure to dispose old memory)
              const oldMemory = memory;
              memory = newMemory;
              epochCost += cost.dataSync()[0];
              
              // Dispose cost tensor
              cost.dispose();
            });
          }

          costs.push(epochCost / (sequence.length - 1));
        });
      }

      return costs;
    } catch (error) {
      console.error('Error in train_sequence:', error);
      throw error;
    } finally {
      // Only report memory stats in debug mode
      if (process.env.DEBUG) {
        console.log('TensorFlow memory stats:', tf.memory());
      }
    }
  }

  private zeroVector(dim: number | undefined): ITensor {
    const dimension = dim || this.memoryDim;  // Default to memoryDim if dim is undefined
    return wrapTensor(tf.zeros([dimension]));
  }
  
  /**
   * Adds a sample to the memory replay buffer
   */
  private addToReplayBuffer(input: number[], memory: number[], target: number[]): void {
    // Add to replay buffer
    this.replayBuffer.push({ input, memory, target });
    
    // Ensure buffer doesn't exceed maximum size
    if (this.replayBuffer.length > this.replayBufferSize) {
      this.replayBuffer.shift(); // Remove oldest sample
    }
  }
  
  /**
   * Trains the model on a batch from the replay buffer
   */
  private trainOnReplayBuffer(): void {
    if (this.replayBuffer.length === 0) return;
    
    // Sample a batch from the replay buffer
    const batchSize = Math.min(16, this.replayBuffer.length);
    const indices: number[] = [];
    
    // Random sampling without replacement
    for (let i = 0; i < batchSize; i++) {
      const idx = Math.floor(Math.random() * this.replayBuffer.length);
      indices.push(idx);
    }
    
    // Train on each sample in the batch
    tf.tidy(() => {
      for (const idx of indices) {
        const sample = this.replayBuffer[idx];
        const x_t = wrapTensor(tf.tensor1d(sample.input));
        const x_next = wrapTensor(tf.tensor1d(sample.target));
        const memory = wrapTensor(tf.tensor1d(sample.memory));
        
        // Train step on this sample
        const cost = this.optimizer.minimize(() => {
          const { predicted } = this.forward(x_t, memory);
          
          // MSE loss
          const diff = tf.sub(unwrapTensor(predicted), unwrapTensor(x_next));
          const mse = tf.mean(tf.square(diff)).asScalar();
          
          // Clean up
          predicted.dispose();
          diff.dispose();
          
          return mse;
        }, true);
        
        // Clean up
        x_t.dispose();
        x_next.dispose();
        memory.dispose();
        if (cost) cost.dispose();
      }
    });
  }
  
  /**
   * Compresses memory using the compression network
   */
  private compressMemory(memory: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      // Apply compression
      const compressed = tf.add(
        tf.matMul(memory.reshape([1, this.memoryDim]), this.compressionWeights.transpose()),
        this.compressionBias
      ).tanh().reshape([Math.floor(this.memoryDim * this.compressionRate)]);
      
      return compressed;
    });
  }
  
  /**
   * Decompresses memory from compressed representation
   */
  private decompressMemory(compressed: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      // Apply decompression (approximate inverse)
      const decompressed = tf.matMul(
        compressed.reshape([1, Math.floor(this.memoryDim * this.compressionRate)]),
        this.compressionWeights
      ).reshape([this.memoryDim]);
      
      return decompressed;
    });
  }
  
  /**
   * Stores memory in long-term storage with a key
   */
  public storeInLongTermMemory(key: string, memory: number[]): void {
    // Add to long-term memory
    this.longTermMemory.push({
      key,
      value: memory,
      timestamp: Date.now()
    });
    
    // Ensure long-term memory doesn't exceed maximum size
    if (this.longTermMemory.length > this.longTermMemorySize) {
      // Remove oldest entry
      this.longTermMemory.sort((a, b) => a.timestamp - b.timestamp);
      this.longTermMemory.shift();
    }
  }
  
  /**
   * Retrieves memory from long-term storage by key
   */
  public retrieveFromLongTermMemory(key: string): number[] | null {
    const entry = this.longTermMemory.find(item => item.key === key);
    if (!entry) return null;
    
    // Update timestamp to indicate recent use
    entry.timestamp = Date.now();
    return entry.value;
  }
  
  /**
   * Dynamically allocates memory based on surprise level
   */
  private allocateMemoryDynamically(memory: tf.Tensor, surprise: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const surpriseValue = surprise.dataSync()[0];
      
      // If surprise is high, allocate more memory capacity
      if (surpriseValue > 0.5) {
        // Enhance memory representation based on surprise level
        const enhancementFactor = tf.scalar(Math.min(1.0 + surpriseValue, 2.0));
        const enhancedMemory = tf.mul(memory, enhancementFactor);
        return enhancedMemory;
      }
      
      // If surprise is low, consider compressing the memory
      if (surpriseValue < 0.1) {
        // Compress and then decompress to save memory while preserving information
        const compressed = this.compressMemory(memory);
        const decompressed = this.decompressMemory(compressed);
        return decompressed;
      }
      
      // Otherwise, return the original memory
      return memory.clone();
    });
  }
  
  /**
   * Generates a cache key from input vector
   */
  private generateCacheKey(input: number[]): string {
    // Simple hashing function for array of numbers
    return input.map(x => Math.round(x * 100) / 100).join('|');
  }
  
  /**
   * Caches a memory state with the given key
   */
  private cacheMemoryState(key: string, memory: number[]): void {
    // Store in cache with current timestamp
    this.llmCache.set(key, {
      value: memory,
      timestamp: Date.now()
    });
    
    // Clean up expired cache entries
    this.cleanupCache();
  }
  
  /**
   * Retrieves a cached memory state by key
   */
  public retrieveCachedMemory(key: string): number[] | null {
    if (!this.llmCache.has(key)) return null;
    
    const entry = this.llmCache.get(key)!;
    
    // Check if entry has expired
    if (Date.now() - entry.timestamp > this.cacheTTL) {
      this.llmCache.delete(key);
      return null;
    }
    
    // Update timestamp to indicate recent use
    entry.timestamp = Date.now();
    return entry.value;
  }
  
  /**
   * Cleans up expired cache entries
   */
  private cleanupCache(): void {
    const now = Date.now();
    
    // Remove expired entries
    for (const [key, entry] of this.llmCache.entries()) {
      if (now - entry.timestamp > this.cacheTTL) {
        this.llmCache.delete(key);
      }
    }
    
    // If cache is still too large, remove oldest entries
    if (this.llmCache.size > 1000) {
      const entries = Array.from(this.llmCache.entries())
        .sort((a, b) => a[1].timestamp - b[1].timestamp);
      
      // Remove oldest 20% of entries
      const toRemove = Math.floor(entries.length * 0.2);
      for (let i = 0; i < toRemove; i++) {
        this.llmCache.delete(entries[i][0]);
      }
    }
  }
}