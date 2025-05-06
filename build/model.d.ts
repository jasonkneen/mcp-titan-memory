import * as tf from '@tensorflow/tfjs';
import { ITensor, IMemoryModel } from './types.js';
export interface TitanMemoryConfig {
    inputDim?: number;
    hiddenDim?: number;
    outputDim?: number;
    learningRate?: number;
    useManifold?: boolean;
    momentumFactor?: number;
    forgetGateInit?: number;
    maxStepSize?: number;
    tangentEpsilon?: number;
    numHeads?: number;
    numLayers?: number;
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
    private attentionOutputWeights;
    private hierarchicalMemory;
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
        attentionOutputWeights: (number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][])[];
        hierarchicalMemory: (number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][] | number[][][][][][])[];
    };
    train_sequence(sequence: ITensor[], epochs?: number): number[];
    private zeroVector;
}
export {};
