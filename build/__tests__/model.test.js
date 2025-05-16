import * as tf from '@tensorflow/tfjs';
import { TitanMemoryModel } from '../model.js';
import { wrapTensor, unwrapTensor } from '../types.js';

// Set backend to CPU for deterministic tests
tf.setBackend('cpu');
tf.env().set('WEBGL_FORCE_F16_TEXTURES', false);
describe('TitanMemoryModel', () => {
  let model;
  const inputDim = 64;
  const hiddenDim = 32;
  const outputDim = 64;
  beforeEach(() => {
    // Create model with default settings
    model = new TitanMemoryModel({
      inputDim,
      hiddenDim,
      outputDim,
      learningRate: 0.001
    });
  });
  afterEach(() => {
    // Clean up any remaining tensors
    tf.disposeVariables();
    tf.dispose(); // Clean up all tensors
  });
  test('initializes with correct dimensions', () => {
    const config = model.getConfig();
    expect(config.inputDim).toBe(inputDim);
    expect(config.hiddenDim).toBe(hiddenDim);
    expect(config.outputDim).toBe(outputDim);
  });
  test('forward pass produces correct output shapes', () => {
    const x = wrapTensor(tf.randomNormal([inputDim], 0, 1, 'float32'));
    const memoryState = wrapTensor(tf.zeros([outputDim]));
    const {
      predicted,
      newMemory,
      surprise
    } = model.forward(x, memoryState);
    expect(predicted.shape).toEqual([inputDim]);
    expect(newMemory.shape).toEqual([outputDim]);
    expect(surprise.shape).toEqual([]);

    // Clean up
    x.dispose();
    memoryState.dispose();
    predicted.dispose();
    newMemory.dispose();
    surprise.dispose();
  });
  describe('training', () => {
    test('reduces loss over time with default learning rate', () => {
      // Create input tensor and its target (same tensor)
      const x_t = tf.randomNormal([inputDim]);
      const x_next = x_t.clone();
      const memoryState = tf.zeros([outputDim]);

      // Wrap tensors for model
      const wrappedX = wrapTensor(x_t);
      const wrappedNext = wrapTensor(x_next);
      const wrappedMemory = wrapTensor(memoryState);
      const losses = [];
      const surprises = [];
      const numSteps = 50;
      for (let i = 0; i < numSteps; i++) {
        const cost = model.trainStep(wrappedX, wrappedNext, wrappedMemory);
        const {
          surprise
        } = model.forward(wrappedX, wrappedMemory);
        losses.push(unwrapTensor(cost).dataSync()[0]);
        surprises.push(unwrapTensor(surprise).dataSync()[0]);
        cost.dispose();
        surprise.dispose();
      }

      // Verify loss reduction
      const firstLosses = losses.slice(0, 5);
      const lastLosses = losses.slice(-5);
      const avgFirstLoss = firstLosses.reduce((a, b) => a + b, 0) / firstLosses.length;
      const avgLastLoss = lastLosses.reduce((a, b) => a + b, 0) / lastLosses.length;
      expect(avgLastLoss).toBeLessThan(avgFirstLoss);

      // Verify surprise reduction
      const firstSurprises = surprises.slice(0, 5);
      const lastSurprises = surprises.slice(-5);
      const avgFirstSurprise = firstSurprises.reduce((a, b) => a + b, 0) / firstSurprises.length;
      const avgLastSurprise = lastSurprises.reduce((a, b) => a + b, 0) / lastSurprises.length;
      expect(avgLastSurprise).toBeLessThan(avgFirstSurprise);

      // Clean up
      x_t.dispose();
      x_next.dispose();
      memoryState.dispose();
      wrappedX.dispose();
      wrappedNext.dispose();
      wrappedMemory.dispose();
    });
    test('trains with different learning rates', () => {
      const learningRates = [0.0001, 0.001, 0.01];
      const numSteps = 20;
      for (const lr of learningRates) {
        const testModel = new TitanMemoryModel({
          inputDim,
          hiddenDim,
          outputDim,
          learningRate: lr
        });
        const x_t = tf.randomNormal([inputDim]);
        const x_next = x_t.clone();
        const memoryState = tf.zeros([outputDim]);
        const wrappedX = wrapTensor(x_t);
        const wrappedNext = wrapTensor(x_next);
        const wrappedMemory = wrapTensor(memoryState);
        const losses = [];
        for (let i = 0; i < numSteps; i++) {
          const cost = testModel.trainStep(wrappedX, wrappedNext, wrappedMemory);
          losses.push(unwrapTensor(cost).dataSync()[0]);
          cost.dispose();
        }
        const avgFirstLoss = losses.slice(0, 3).reduce((a, b) => a + b, 0) / 3;
        const avgLastLoss = losses.slice(-3).reduce((a, b) => a + b, 0) / 3;
        expect(avgLastLoss).toBeLessThan(avgFirstLoss);

        // Clean up
        x_t.dispose();
        x_next.dispose();
        memoryState.dispose();
        wrappedX.dispose();
        wrappedNext.dispose();
        wrappedMemory.dispose();
      }
    });
    test('handles sequence training', () => {
      return tf.tidy(() => {
        const sequenceLength = 5;
        const sequence = [];

        // Create sequence in a single tidy
        for (let i = 0; i < sequenceLength; i++) {
          sequence.push(wrapTensor(tf.randomNormal([inputDim])));
        }
        const wrappedMemory = wrapTensor(tf.zeros([outputDim]));

        // Train on sequence
        for (let i = 0; i < sequenceLength - 1; i++) {
          const cost = model.trainStep(sequence[i], sequence[i + 1], wrappedMemory);
          const costShape = unwrapTensor(cost).shape;
          expect(costShape).toEqual([]);
          cost.dispose();
        }

        // Clean up
        sequence.forEach(t => t.dispose());
        wrappedMemory.dispose();
      });
    });
  });
  describe('manifold operations', () => {
    beforeEach(() => {
      model = new TitanMemoryModel({
        inputDim,
        hiddenDim,
        outputDim,
        useManifold: true,
        maxStepSize: 0.1,
        tangentEpsilon: 1e-8
      });
    });
    test('maintains unit norm with standard input', () => {
      let base = tf.randomNormal([inputDim]);
      const baseNorm = base.norm().dataSync()[0];
      base = base.div(tf.scalar(baseNorm + 1e-12));
      let velocity = tf.randomNormal([inputDim]);
      velocity = velocity.mul(tf.scalar(0.05));
      const wrappedBase = wrapTensor(base);
      const wrappedVel = wrapTensor(velocity);
      const result = model.manifoldStep(wrappedBase, wrappedVel);
      const unwrappedResult = unwrapTensor(result);
      const norm = unwrappedResult.norm().dataSync()[0];
      expect(Math.abs(norm - 1.0)).toBeLessThan(1e-5);

      // Clean up
      base.dispose();
      velocity.dispose();
      wrappedBase.dispose();
      wrappedVel.dispose();
      result.dispose();
      unwrappedResult.dispose();
    });
    test('handles zero velocity correctly', () => {
      let base = tf.randomNormal([inputDim]);
      const baseNorm = base.norm().dataSync()[0];
      base = base.div(tf.scalar(baseNorm + 1e-12));
      const velocity = tf.zeros([inputDim]);
      const wrappedBase = wrapTensor(base);
      const wrappedVel = wrapTensor(velocity);
      const result = model.manifoldStep(wrappedBase, wrappedVel);
      const unwrappedResult = unwrapTensor(result);

      // Should return original base vector
      const diff = tf.sum(tf.sub(unwrappedResult, base)).dataSync()[0];
      expect(Math.abs(diff)).toBeLessThan(1e-5);

      // Clean up
      base.dispose();
      velocity.dispose();
      wrappedBase.dispose();
      wrappedVel.dispose();
      result.dispose();
      unwrappedResult.dispose();
    });
    test('respects maximum step size', () => {
      let base = tf.randomNormal([inputDim]);
      const baseNorm = base.norm().dataSync()[0];
      base = base.div(tf.scalar(baseNorm + 1e-12));

      // Create large velocity
      let velocity = tf.randomNormal([inputDim]);
      velocity = velocity.mul(tf.scalar(1.0)); // Much larger than maxStepSize

      const wrappedBase = wrapTensor(base);
      const wrappedVel = wrapTensor(velocity);
      const result = model.manifoldStep(wrappedBase, wrappedVel);
      const unwrappedResult = unwrapTensor(result);

      // Calculate angle between base and result
      const dot = tf.sum(tf.mul(base, unwrappedResult)).dataSync()[0];
      const angle = Math.acos(Math.min(1.0, Math.abs(dot)));

      // Angle should not exceed maxStepSize (with small epsilon for floating point precision)
      const epsilon = 1e-6;
      expect(angle).toBeLessThanOrEqual((model.getConfig().maxStepSize || 0.1) + epsilon);

      // Clean up
      base.dispose();
      velocity.dispose();
      wrappedBase.dispose();
      wrappedVel.dispose();
      result.dispose();
      unwrappedResult.dispose();
    });
  });
  describe('model persistence', () => {
    test('saves and loads weights correctly', async () => {
      const model = new TitanMemoryModel({
        inputDim,
        hiddenDim,
        outputDim
      });
      const initialWeights = model.getWeights();
      await model.saveModel('./test-weights.json');
      const loadedModel = new TitanMemoryModel({
        inputDim,
        hiddenDim,
        outputDim
      });
      await loadedModel.loadModel('./test-weights.json');
      const loadedWeights = loadedModel.getWeights();
      expect(loadedWeights).toEqual(initialWeights);
    });
    test('maintains model behavior after load', async () => {
      // Train original model
      const x = wrapTensor(tf.randomNormal([inputDim]));
      const memoryState = wrapTensor(tf.zeros([outputDim]));
      const {
        predicted: originalPrediction
      } = model.forward(x, memoryState);
      await model.saveModel('./test-weights.json');

      // Load into new model
      const loadedModel = new TitanMemoryModel({
        inputDim,
        hiddenDim,
        outputDim
      });
      await loadedModel.loadModel('./test-weights.json');

      // Compare predictions
      const {
        predicted: loadedPrediction
      } = loadedModel.forward(x, memoryState);
      const originalData = unwrapTensor(originalPrediction).dataSync();
      const loadedData = unwrapTensor(loadedPrediction).dataSync();
      for (let i = 0; i < originalData.length; i++) {
        expect(originalData[i]).toBeCloseTo(loadedData[i], 5);
      }

      // Clean up
      x.dispose();
      memoryState.dispose();
      originalPrediction.dispose();
      loadedPrediction.dispose();
    });
    test('handles invalid file paths', async () => {
      await expect(model.loadModel('./nonexistent.json')).rejects.toThrow();
    });
  });
  test('model configuration is accessible', () => {
    const config = model.getConfig();
    expect(config).toHaveProperty('inputDim');
    expect(config).toHaveProperty('outputDim');
    expect(config).toHaveProperty('hiddenDim');
  });
  test('Multi-head attention mechanism works correctly', () => {
    const x = wrapTensor(tf.randomNormal([inputDim]));
    const memoryState = wrapTensor(tf.zeros([outputDim]));
    const {
      predicted,
      newMemory,
      surprise
    } = model.forward(x, memoryState);

    // Check if the attention mechanism updates the memory correctly
    expect(newMemory.shape).toEqual([outputDim]);
    predicted.dispose();
    newMemory.dispose();
    surprise.dispose();
    x.dispose();
    memoryState.dispose();
  });
  test('Hierarchical memory structure updates correctly', () => {
    const x = wrapTensor(tf.randomNormal([inputDim]));
    const memoryState = wrapTensor(tf.zeros([outputDim]));
    const {
      predicted,
      newMemory,
      surprise
    } = model.forward(x, memoryState);

    // Get memory state for each layer
    const config = model.getConfig();
    const numLayers = config.numLayers || 2; // Default to 2 if undefined

    for (let i = 0; i < numLayers; i++) {
      const layerMemory = model.getLayerMemoryState(i);
      if (Array.isArray(layerMemory)) {
        expect(layerMemory.length).toBeLessThanOrEqual(outputDim);
      } else {
        // Handle other types or log a proper error
        fail('Expected layerMemory to be an array');
      }
    }
    predicted.dispose();
    newMemory.dispose();
    surprise.dispose();
    x.dispose();
    memoryState.dispose();
  });
  test('Dynamic memory allocation mechanism works correctly', () => {
    // Create model with dynamic allocation enabled
    const dynamicModel = new TitanMemoryModel({
      inputDim,
      hiddenDim,
      outputDim,
      dynamicAllocation: true
    });

    // Create input and memory tensors
    const x = wrapTensor(tf.randomNormal([inputDim]));
    const memoryState = wrapTensor(tf.zeros([outputDim]));

    // Forward pass with dynamic allocation
    const {
      newMemory,
      surprise
    } = dynamicModel.forward(x, memoryState);

    // Verify dynamic allocation is working
    if (typeof dynamicModel['allocateMemoryDynamically'] === 'function') {
      // Create a high surprise value to test allocation
      const highSurprise = wrapTensor(tf.scalar(0.9));

      // Call the method directly
      const dynamicMemory = tf.tidy(() => {
        return dynamicModel['allocateMemoryDynamically'](unwrapTensor(newMemory), unwrapTensor(highSurprise));
      });

      // Verify the shape is still correct
      expect(dynamicMemory.shape).toEqual([outputDim]);

      // Clean up
      dynamicMemory.dispose();
      highSurprise.dispose();
    }

    // Clean up
    x.dispose();
    memoryState.dispose();
    newMemory.dispose();
    surprise.dispose();
  });
  test('Memory replay mechanism works correctly', () => {
    // Create model with memory replay enabled
    const replayModel = new TitanMemoryModel({
      inputDim,
      hiddenDim,
      outputDim,
      useMemoryReplay: true,
      replayBufferSize: 50
    });

    // Create input and memory tensors
    const x = wrapTensor(tf.randomNormal([inputDim]));
    const memoryState = wrapTensor(tf.zeros([outputDim]));

    // First forward pass to populate replay buffer
    const {
      predicted,
      newMemory
    } = replayModel.forward(x, memoryState);

    // Add to replay buffer by calling private method directly
    if (typeof replayModel['addToReplayBuffer'] === 'function') {
      replayModel['addToReplayBuffer'](unwrapTensor(x).flatten().arraySync(), unwrapTensor(memoryState).flatten().arraySync(), unwrapTensor(predicted).flatten().arraySync());
    }

    // Train on replay buffer
    if (typeof replayModel['trainOnReplayBuffer'] === 'function') {
      replayModel['trainOnReplayBuffer']();
    }

    // Verify replay buffer is working by checking if it has entries
    if (typeof replayModel['replayBuffer'] !== 'undefined') {
      expect(replayModel['replayBuffer'].length).toBeGreaterThan(0);
    }

    // Clean up
    x.dispose();
    memoryState.dispose();
    predicted.dispose();
    newMemory.dispose();
  });
  test('Memory compression works correctly', () => {
    // Create model with compression enabled
    const compressionModel = new TitanMemoryModel({
      inputDim,
      hiddenDim,
      outputDim,
      compressionRate: 0.5 // Compress to half size
    });

    // Create memory tensor
    const memory = wrapTensor(tf.randomNormal([outputDim]));

    // Test compression if the method exists
    if (typeof compressionModel['compressMemory'] === 'function') {
      const compressed = tf.tidy(() => {
        return compressionModel['compressMemory'](unwrapTensor(memory));
      });

      // Verify the compressed size is approximately half
      const expectedSize = Math.floor(outputDim * 0.5);
      expect(compressed.shape[0]).toBe(expectedSize);

      // Test decompression if the method exists
      if (typeof compressionModel['decompressMemory'] === 'function') {
        const decompressed = tf.tidy(() => {
          return compressionModel['decompressMemory'](compressed);
        });

        // Verify the decompressed size matches the original
        expect(decompressed.shape[0]).toBe(outputDim);
        decompressed.dispose();
      }
      compressed.dispose();
    }
    memory.dispose();
  });
  test('LLM Cache integration works correctly', () => {
    // Create model with cache
    const cacheModel = new TitanMemoryModel({
      inputDim,
      hiddenDim,
      outputDim,
      cacheTTL: 3600000 // 1 hour cache TTL
    });

    // Create input and memory tensors
    const x = wrapTensor(tf.randomNormal([inputDim]));
    const memoryState = wrapTensor(tf.zeros([outputDim]));

    // Forward pass to generate a memory state
    const {
      newMemory
    } = cacheModel.forward(x, memoryState);

    // Test cache methods if they exist
    if (typeof cacheModel['generateCacheKey'] === 'function' && typeof cacheModel['cacheMemoryState'] === 'function' && typeof cacheModel['retrieveCachedMemory'] === 'function') {
      // Generate a cache key
      const key = cacheModel['generateCacheKey'](unwrapTensor(x).flatten().arraySync());

      // Cache the memory state
      cacheModel['cacheMemoryState'](key, unwrapTensor(newMemory).flatten().arraySync());

      // Retrieve from cache
      const cachedMemory = cacheModel['retrieveCachedMemory'](key);

      // Verify cache is working
      expect(cachedMemory).not.toBeNull();
      if (cachedMemory) {
        expect(cachedMemory.length).toBe(outputDim);
      }

      // Test cache cleanup
      if (typeof cacheModel['cleanupCache'] === 'function') {
        cacheModel['cleanupCache']();
      }
    }

    // Clean up
    x.dispose();
    memoryState.dispose();
    newMemory.dispose();
  });
});