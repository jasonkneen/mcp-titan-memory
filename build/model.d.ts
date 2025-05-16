import * as tf from '@tensorflow/tfjs';
import { ITensor, IMemoryModel } from './types.js';
export interface TitanMemoryConfig {
    inputDim?: number;
    hiddenDim?: number;
    outputDim?: number;
    memoryDim?: number;
    learningRate?: number;
    useManifold?: boolean;
    momentumFactor?: number;
    forgetGateInit?: number;
    maxStepSize?: number;
    tangentEpsilon?: number;
    numHeads?: number;
    numLayers?: number;
    useMemoryReplay?: boolean;
    replayBufferSize?: number;
    compressionRate?: number;
    longTermMemorySize?: number;
    dynamicAllocation?: boolean;
    cacheTTL?: number;
}
interface ForwardResult extends tf.TensorContainerObject {
    predicted: ITensor;
    newMemory: ITensor;
    surprise: ITensor;
    [key: string]: any;
}
export declare class TitanMemoryModel implements IMemoryModel {
    private inputDim;
    private hiddenDim;
    private memoryDim;
    private learningRate;
    useManifold: boolean;
    private momentumFactor;
    private forgetGateInit;
    private maxStepSize;
    private tangentEpsilon;
    private numHeads;
    private numLayers;
    private useMemoryReplay;
    private replayBufferSize;
    private compressionRate;
    private longTermMemorySize;
    private dynamicAllocation;
    private cacheTTL;
    private replayBuffer;
    private longTermMemory;
    private llmCache;
    private fullOutputDim;
    private W1;
    private b1;
    private W2;
    private b2;
    private forgetGate;
    private optimizer;
    private queryWeights;
    private keyWeights;
    private valueWeights;
    private attentionFinalOutputWeight;
    private hierarchicalMemory;
    private compressionWeights;
    private compressionBias;
    constructor(config?: TitanMemoryConfig);
    private multiHeadAttention;
    forward(xTensor: ITensor, memoryState: ITensor): ForwardResult;
    manifoldStep(base: ITensor, velocity: ITensor): ITensor;
    trainStep(x_t: ITensor, x_next: ITensor, memoryState: ITensor): ITensor;
    saveModel(path: string): Promise<void>;
    loadModel(path: string): Promise<void>;
    getConfig(): TitanMemoryConfig;
    getWeights(): {
        W1: number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][];
        b1: number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][];
        W2: number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][];
        b2: number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][];
        forgetGate: number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][];
        queryWeights: (number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][])[];
        keyWeights: (number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][])[];
        valueWeights: (number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][])[];
        attentionFinalOutputWeight: number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][];
        hierarchicalMemory: (number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][])[];
        compressionWeights: number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][];
        compressionBias: number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][];
    };
    getLayerMemoryState(layerIndex: number): number[];
    train_sequence(sequence: ITensor[], epochs?: number): number[];
    private zeroVector;
    /**
     * Adds a sample to the memory replay buffer
     */
    private addToReplayBuffer;
    /**
     * Trains the model on a batch from the replay buffer
     */
    private trainOnReplayBuffer;
    /**
     * Compresses memory using the compression network
     */
    private compressMemory;
    /**
     * Decompresses memory from compressed representation
     */
    private decompressMemory;
    /**
     * Stores memory in long-term storage with a key
     */
    storeInLongTermMemory(key: string, memory: number[]): void;
    /**
     * Retrieves memory from long-term storage by key
     */
    retrieveFromLongTermMemory(key: string): number[] | null;
    /**
     * Dynamically allocates memory based on surprise level
     */
    private allocateMemoryDynamically;
    /**
     * Generates a cache key from input vector
     */
    private generateCacheKey;
    /**
     * Caches a memory state with the given key
     */
    private cacheMemoryState;
    /**
     * Retrieves a cached memory state by key
     */
    retrieveCachedMemory(key: string): number[] | null;
    /**
     * Cleans up expired cache entries
     */
    private cleanupCache;
}
export {};
