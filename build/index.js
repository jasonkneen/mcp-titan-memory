#!/usr/bin/env node
import '@tensorflow/tfjs-node';
import { createMCPServer, CallToolResultSchema } from '@modelcontextprotocol/sdk';
import { z } from 'zod';
import { TitanMemoryModel } from './model.js';
import * as tf from '@tensorflow/tfjs';
import { wrapTensor } from './types.js';
// Memory cache for storing model states
const LLM_CACHE = {};
// Initialize with null model
let model = null;
let memoryVec = null;
// Initialize the model with configuration
const initModelTool = {
    name: 'init_model',
    description: 'Initialize the Titan memory model with the given configuration',
    parameters: z.object({
        inputDim: z.number().optional(),
        hiddenDim: z.number().optional(),
        outputDim: z.number().optional(),
        learningRate: z.number().optional(),
        numLayers: z.number().optional(),
        useAttention: z.boolean().optional(),
        useManifold: z.boolean().optional(),
    }),
    execute: async (params) => {
        try {
            const config = params.arguments || {};
            // Initialize the model
            model = new TitanMemoryModel(config);
            // Initialize memory vector
            if (memoryVec) {
                memoryVec.dispose();
            }
            const memDim = model.getConfig().outputDim || 64; // Memory dimension
            memoryVec = tf.variable(tf.zeros([memDim]));
            return CallToolResultSchema.parse({
                content: [
                    {
                        type: 'text',
                        text: `Model initialized with config: ${JSON.stringify(model.getConfig())}`
                    }
                ]
            });
        }
        catch (error) {
            throw new Error(`Failed to initialize model: ${error instanceof Error ? error.message : String(error)}`);
        }
    }
};
// Forward pass through the model
const forwardTool = {
    name: 'forward',
    description: 'Perform a forward pass through the model with the given input vector',
    parameters: z.object({
        x: z.array(z.number())
    }),
    execute: async (params) => {
        if (!model || !memoryVec) {
            throw new Error('Model not initialized');
        }
        try {
            const { x } = params.arguments;
            return tf.tidy(() => {
                // Convert to tensors
                const xT = wrapTensor(tf.tensor1d(x));
                const memoryT = wrapTensor(memoryVec);
                // Run forward pass
                const { predicted, newMemory, surprise } = model.forward(xT, memoryT);
                // Extract values
                const predVal = predicted.dataSync();
                const memVal = newMemory.dataSync();
                const surVal = surprise.dataSync()[0];
                // Update memory
                memoryVec.assign(tf.tensor(memVal));
                return CallToolResultSchema.parse({
                    content: [
                        {
                            type: 'text',
                            text: JSON.stringify({
                                predicted: Array.from(predVal),
                                surprise: surVal
                            })
                        }
                    ]
                });
            });
        }
        catch (error) {
            throw new Error(`Forward pass failed: ${error instanceof Error ? error.message : String(error)}`);
        }
    }
};
// Train step
const trainStepTool = {
    name: 'train_step',
    description: 'Perform a training step with the given input and target vectors',
    parameters: z.object({
        x_t: z.array(z.number()),
        x_next: z.array(z.number())
    }),
    execute: async (params) => {
        if (!model || !memoryVec) {
            throw new Error('Model not initialized');
        }
        try {
            const { x_t, x_next } = params.arguments;
            return tf.tidy(() => {
                // Convert to tensors
                const x_tT = wrapTensor(tf.tensor1d(x_t));
                const x_nextT = wrapTensor(tf.tensor1d(x_next));
                const memoryT = wrapTensor(memoryVec);
                // Run training step
                const cost = model.trainStep(x_tT, x_nextT, memoryT);
                // Forward pass results
                const { predicted, newMemory, surprise } = model.forward(x_tT, memoryT);
                // Extract values
                const costVal = cost.dataSync()[0];
                const predVal = predicted.dataSync();
                const surVal = surprise.dataSync()[0];
                // Update memory
                memoryVec.assign(tf.tensor(newMemory.dataSync()));
                return CallToolResultSchema.parse({
                    content: [
                        {
                            type: 'text',
                            text: JSON.stringify({
                                cost: costVal,
                                predicted: Array.from(predVal),
                                surprise: surVal
                            })
                        }
                    ]
                });
            });
        }
        catch (error) {
            throw new Error(`Training step failed: ${error instanceof Error ? error.message : String(error)}`);
        }
    }
};
// Train sequence
const trainSequenceTool = {
    name: 'train_sequence',
    description: 'Train the model on a sequence of vectors',
    parameters: z.object({
        sequence: z.array(z.array(z.number())),
        epochs: z.number().optional(),
    }),
    execute: async (params) => {
        if (!model || !memoryVec) {
            throw new Error('Model not initialized');
        }
        try {
            const { sequence, epochs = 1 } = params.arguments;
            if (sequence.length < 2) {
                throw new Error('Sequence must contain at least 2 vectors');
            }
            // Train for specified number of epochs
            const costs = [];
            for (let epoch = 0; epoch < epochs; epoch++) {
                let epochCost = 0;
                // Reset memory at start of each epoch
                memoryVec.assign(tf.zeros([model.getConfig().outputDim || 64]));
                for (let i = 0; i < sequence.length - 1; i++) {
                    const result = await trainStepTool.execute({
                        name: 'train_step',
                        arguments: {
                            x_t: sequence[i],
                            x_next: sequence[i + 1]
                        }
                    });
                    const content = result.content[0].text;
                    const parsed = JSON.parse(content);
                    epochCost += parsed.cost;
                }
                costs.push(epochCost / (sequence.length - 1));
            }
            return CallToolResultSchema.parse({
                content: [
                    {
                        type: 'text',
                        text: JSON.stringify({
                            costs
                        })
                    }
                ]
            });
        }
        catch (error) {
            throw new Error(`Training sequence failed: ${error instanceof Error ? error.message : String(error)}`);
        }
    }
};
// Save model
const saveModelTool = {
    name: 'save_model',
    description: 'Save the model weights to the specified path',
    parameters: z.object({
        path: z.string()
    }),
    execute: async (params) => {
        if (!model) {
            throw new Error('Model not initialized');
        }
        try {
            const { path } = params.arguments;
            await model.saveModel(path);
            return CallToolResultSchema.parse({
                content: [
                    {
                        type: 'text',
                        text: `Model saved to ${path}`
                    }
                ]
            });
        }
        catch (error) {
            throw new Error(`Failed to save model: ${error instanceof Error ? error.message : String(error)}`);
        }
    }
};
// Load model
const loadModelTool = {
    name: 'load_model',
    description: 'Load the model weights from the specified path',
    parameters: z.object({
        path: z.string()
    }),
    execute: async (params) => {
        if (!model) {
            throw new Error('Model not initialized');
        }
        try {
            const { path } = params.arguments;
            await model.loadModel(path);
            return CallToolResultSchema.parse({
                content: [
                    {
                        type: 'text',
                        text: `Model loaded from ${path}`
                    }
                ]
            });
        }
        catch (error) {
            throw new Error(`Failed to load model: ${error instanceof Error ? error.message : String(error)}`);
        }
    }
};
// Get model status
const getStatusTool = {
    name: 'get_status',
    description: 'Get the current status of the model',
    parameters: z.object({}),
    execute: async () => {
        if (!model) {
            return CallToolResultSchema.parse({
                content: [
                    {
                        type: 'text',
                        text: JSON.stringify({ status: 'No model initialized' })
                    }
                ]
            });
        }
        return CallToolResultSchema.parse({
            content: [
                {
                    type: 'text',
                    text: JSON.stringify(model.getConfig())
                }
            ]
        });
    }
};
// Store memory state
const storeMemoryStateTool = {
    name: 'store_memory_state',
    description: 'Store the current memory state with the given key',
    parameters: z.object({
        key: z.string()
    }),
    execute: async (params) => {
        if (!memoryVec) {
            throw new Error('Model not initialized');
        }
        try {
            const { key } = params.arguments;
            // Store current memory state
            LLM_CACHE[key] = Array.from(memoryVec.dataSync());
            return CallToolResultSchema.parse({
                content: [
                    {
                        type: 'text',
                        text: `Memory state stored with key: ${key}`
                    }
                ]
            });
        }
        catch (error) {
            throw new Error(`Failed to store memory state: ${error instanceof Error ? error.message : String(error)}`);
        }
    }
};
// Retrieve memory state
const retrieveMemoryStateTool = {
    name: 'retrieve_memory_state',
    description: 'Retrieve a memory state with the given key',
    parameters: z.object({
        key: z.string()
    }),
    execute: async (params) => {
        if (!memoryVec) {
            throw new Error('Model not initialized');
        }
        try {
            const { key } = params.arguments;
            // Check if key exists
            if (!LLM_CACHE[key]) {
                throw new Error(`No memory state found with key: ${key}`);
            }
            // Restore memory state
            memoryVec.assign(tf.tensor1d(LLM_CACHE[key]));
            return CallToolResultSchema.parse({
                content: [
                    {
                        type: 'text',
                        text: `Memory state retrieved with key: ${key}`
                    }
                ]
            });
        }
        catch (error) {
            throw new Error(`Failed to retrieve memory state: ${error instanceof Error ? error.message : String(error)}`);
        }
    }
};
// Setup tools
const tools = [
    initModelTool,
    forwardTool,
    trainStepTool,
    trainSequenceTool,
    saveModelTool,
    loadModelTool,
    getStatusTool,
    storeMemoryStateTool,
    retrieveMemoryStateTool
];
// Create MCP server
const server = createMCPServer({ tools });
// Start server
server.listen(3000, '0.0.0.0', () => {
    console.log('Titan Memory MCP Server listening on port 3000');
});
// Handle process termination
process.on('SIGINT', () => {
    console.log('Shutting down server...');
    // Clean up TensorFlow resources
    if (memoryVec) {
        memoryVec.dispose();
    }
    server.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
});
//# sourceMappingURL=index.js.map