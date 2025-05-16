#!/usr/bin/env node
import '@tensorflow/tfjs-node';
import express from 'express';
import bodyParser from 'body-parser';
import { z } from 'zod';
import { TitanMemoryModel } from './model.js';
import * as tf from '@tensorflow/tfjs';
import { wrapTensor, unwrapTensor } from './types.js';
import logger from './logger.js';
import * as fs from 'fs/promises';
import * as path from 'path';
import dotenv from 'dotenv';

// Create a simple Express server instead of MCP
const app = express();
app.use(bodyParser.json());

// Initialize with null model
let model: TitanMemoryModel | null = null;
let memoryVec: tf.Variable | null = null;

// Initialize model endpoint
app.post('/init-model', async (req, res) => {
  try {
    const config = req.body || {};
    
    // Initialize the model
    model = new TitanMemoryModel(config);
    
    // Initialize memory vector
    if (memoryVec) {
      memoryVec.dispose();
    }
    const memDim = model.getConfig().memoryDim || model.getConfig().outputDim || 64;
    memoryVec = tf.variable(tf.zeros([memDim]));
    
    res.json({
      status: 'success',
      config: model.getConfig()
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: `Failed to initialize model: ${error instanceof Error ? error.message : String(error)}`
    });
  }
});

// Forward pass endpoint
app.post('/forward', async (req, res) => {
  if (!model || !memoryVec) {
    return res.status(400).json({
      status: 'error',
      message: 'Model not initialized'
    });
  }
  
  try {
    const { x } = req.body;
    
    if (!Array.isArray(x)) {
      return res.status(400).json({
        status: 'error',
        message: 'Input must be an array of numbers'
      });
    }
    
    // Convert to tensors
    const xT = wrapTensor(tf.tensor1d(x));
    const memoryT = wrapTensor(memoryVec);
    
    // Run forward pass
    const { predicted, newMemory, surprise } = model.forward(xT, memoryT);
    
    // Extract values
    const predVal = Array.from(predicted.dataSync());
    const surVal = surprise.dataSync()[0];
    
    // Update memory
    memoryVec.assign(tf.tensor(newMemory.dataSync()));
    
    res.json({
      status: 'success',
      predicted: predVal,
      surprise: surVal
    });
    
    // Clean up
    xT.dispose();
    predicted.dispose();
    newMemory.dispose();
    surprise.dispose();
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: `Forward pass failed: ${error instanceof Error ? error.message : String(error)}`
    });
  }
});

// Get status endpoint
app.get('/status', (req, res) => {
  if (!model) {
    return res.json({
      status: 'No model initialized'
    });
  }
  
  res.json({
    status: 'Model initialized',
    config: model.getConfig()
  });
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  logger.info(`Titan Memory Server listening on port ${PORT}`);
});