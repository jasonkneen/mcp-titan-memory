
import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config();

// Helper function to parse boolean environment variables
const parseBool = (value: string | undefined): boolean => {
  return value?.toLowerCase() === 'true';
};

// Helper function to parse number environment variables
const parseNumber = (value: string | undefined, defaultValue: number): number => {
  if (!value) return defaultValue;
  const parsed = parseFloat(value);
  return isNaN(parsed) ? defaultValue : parsed;
};

// Configuration object
export const config = {
  server: {
    port: parseNumber(process.env.PORT, 3000),
    host: process.env.HOST || '0.0.0.0'
  },
  model: {
    inputDim: parseNumber(process.env.MODEL_INPUT_DIM, 64),
    hiddenDim: parseNumber(process.env.MODEL_HIDDEN_DIM, 32),
    memoryDim: parseNumber(process.env.MODEL_MEMORY_DIM, 64),
    learningRate: parseNumber(process.env.MODEL_LEARNING_RATE, 0.001),
    useManifold: parseBool(process.env.MODEL_USE_MANIFOLD),
    numHeads: parseNumber(process.env.MODEL_NUM_HEADS, 4),
    numLayers: parseNumber(process.env.MODEL_NUM_LAYERS, 3)
  },
  debug: parseBool(process.env.DEBUG)
};

export default config;
