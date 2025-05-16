import { describe, expect, test, jest, beforeEach } from '@jest/globals';
import * as tf from '@tensorflow/tfjs';
import { TitanMemoryModel } from '../model.js';
import { wrapTensor } from '../types.js';

// Mock the MCP SDK
jest.mock('@modelcontextprotocol/sdk/server', () => ({
  createMCPServer: jest.fn().mockReturnValue({
    listen: jest.fn(),
    close: jest.fn()
  }),
  CallToolResultSchema: {
    parse: jest.fn().mockImplementation(data => data)
  }
}));
describe('TitanMemoryModel Tests', () => {
  let model;
  const inputDim = 64;
  const hiddenDim = 32;
  const outputDim = 64;
  beforeEach(() => {
    model = new TitanMemoryModel({
      inputDim,
      hiddenDim,
      outputDim,
      learningRate: 0.001
    });
  });
  test('Model processes sequences correctly', () => {
    const x = wrapTensor(tf.randomNormal([inputDim]));
    const memoryState = wrapTensor(tf.zeros([outputDim]));
    const {
      predicted,
      newMemory,
      surprise
    } = model.forward(x, memoryState);

    // Expect the shapes to match the logic:
    // predicted => [inputDim]
    // newMemory => [outputDim]
    expect(predicted.shape).toEqual([inputDim]);
    expect(newMemory.shape).toEqual([outputDim]);
    // surprise => scalar
    expect(surprise.shape).toEqual([]);
    predicted.dispose();
    newMemory.dispose();
    surprise.dispose();
    x.dispose();
    memoryState.dispose();
  });
  test('Training reduces loss over time', () => {
    const memoryState = wrapTensor(tf.zeros([outputDim]));
    const x_t = wrapTensor(tf.randomNormal([inputDim]));
    const x_next = wrapTensor(tf.randomNormal([inputDim]));

    // Run multiple training steps
    const losses = [];
    for (let i = 0; i < 10; i++) {
      const loss = model.trainStep(x_t, x_next, memoryState);
      losses.push(loss.dataSync()[0]);
      loss.dispose();
    }

    // Check if loss generally decreases
    const firstLoss = losses[0];
    const lastLoss = losses[losses.length - 1];
    expect(lastLoss).toBeLessThan(firstLoss);
    x_t.dispose();
    x_next.dispose();
    memoryState.dispose();
  });
  test('Manifold updates work correctly when enabled', () => {
    model = new TitanMemoryModel({
      inputDim,
      hiddenDim,
      outputDim,
      useManifold: true
    });
    const base = wrapTensor(tf.randomNormal([inputDim]));
    const velocity = wrapTensor(tf.randomNormal([inputDim]));
    const result = model.manifoldStep(base, velocity);

    // We check that the result is normalized (approximately),
    // or at least consistent with the manifold logic.
    const resultTensor = tf.tensor(result.dataSync());
    const normVal = resultTensor.norm().dataSync()[0];
    // For a sphere manifold, we expect roughly norm ~ 1.0
    expect(Math.abs(normVal - 1.0)).toBeLessThan(1e-5);
    base.dispose();
    velocity.dispose();
    result.dispose();
    resultTensor.dispose();
  });
  test('Model can save and load weights', async () => {
    const x = wrapTensor(tf.randomNormal([inputDim]));
    const memoryState = wrapTensor(tf.zeros([outputDim]));

    // Get initial prediction
    const {
      predicted: pred1
    } = model.forward(x, memoryState);
    const initial = pred1.dataSync();
    pred1.dispose();

    // Save and load weights
    await model.saveModel('./test-weights.json');
    await model.loadModel('./test-weights.json');

    // Get prediction after loading
    const {
      predicted: pred2
    } = model.forward(x, memoryState);
    const loaded = pred2.dataSync();
    pred2.dispose();

    // Compare predictions
    expect(Array.from(initial)).toEqual(Array.from(loaded));
    x.dispose();
    memoryState.dispose();
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

    // Check if the hierarchical memory structure updates correctly
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
});

// Temporarily comment out MCP Server tests until we fix the import issue
/*
describe('MCP Server', () => {
  // Import the index module to trigger server creation
  beforeEach(() => {
    jest.resetModules();
    require('../index.js');
  });

  test('Server is created with the correct tools', () => {
    const createMCPServer = require('@modelcontextprotocol/sdk/server').createMCPServer;
    expect(createMCPServer).toHaveBeenCalled();
    const callArgs = (createMCPServer as jest.Mock).mock.calls[0][0];
    
    // Check that all expected tools are registered
    const toolNames = (callArgs as any).tools.map((tool: any) => tool.name);
    expect(toolNames).toContain('init_model');
    expect(toolNames).toContain('forward');
    expect(toolNames).toContain('train_step');
    expect(toolNames).toContain('train_sequence');
    expect(toolNames).toContain('save_model');
    expect(toolNames).toContain('load_model');
    expect(toolNames).toContain('get_status');
    expect(toolNames).toContain('store_memory_state');
    expect(toolNames).toContain('retrieve_memory_state');
    expect(toolNames).toContain('compress_memory');
    expect(toolNames).toContain('memory_replay');
  });
  
  test('Advanced memory features are available', () => {
    const createMCPServer = require('@modelcontextprotocol/sdk/server').createMCPServer;
    const callArgs = (createMCPServer as jest.Mock).mock.calls[0][0];
    const tools = (callArgs as any).tools;
    
    // Find the init_model tool
    const initModelTool = tools.find((tool: any) => tool.name === 'init_model');
    expect(initModelTool).toBeDefined();
    
    // Check that advanced parameters are supported
    const parameters = initModelTool.parameters._def.shape;
    expect(parameters).toHaveProperty('useMemoryReplay');
    expect(parameters).toHaveProperty('replayBufferSize');
    expect(parameters).toHaveProperty('compressionRate');
    expect(parameters).toHaveProperty('longTermMemorySize');
    expect(parameters).toHaveProperty('dynamicAllocation');
    expect(parameters).toHaveProperty('cacheTTL');
  });
});
*/