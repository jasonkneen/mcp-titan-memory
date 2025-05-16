# Titan Memory Server

A Model Context Protocol (MCP) server implementation with an enhanced Titan Memory model.

## Overview

This project implements a memory model for large language models (LLMs) that is designed to enhance memory capabilities in generative AI systems. It's built using TensorFlow.js and implemented as an MCP server, making it easy to integrate with any MCP-compatible client.

## Features

Currently implemented:
- Multi-head attention mechanism
- Hierarchical memory structure
- Memory state persistence
- Integration with Model Context Protocol (MCP)
- Memory replay for enhanced learning
- LLM Cache integration
- Dynamic memory allocation
- Long-term memory storage
- Advanced memory compression
- Persistent task-specific memory
- Momentum-based memory updates
- Configurable memory integration variants (MAC/MAG)

## Usage

The server exposes several tools via the Model Context Protocol (MCP):

- `init_model`: Initialize the memory model with custom configurations
- `forward`: Perform a forward pass through the model
- `train_step`: Perform a single training step
- `train_sequence`: Train on a sequence of vectors
- `save_model`: Save the current model weights
- `load_model`: Load model weights from a saved file
- `get_status`: Get the current status of the model
- `store_memory_state`: Store the current memory state with a key
- `retrieve_memory_state`: Retrieve a stored memory state
- `compress_memory`: Compress the current memory state to save space
- `memory_replay`: Perform memory replay training to enhance learning

## Installation

```bash
npm install
```

## Running the Server

```bash
npm run build
npm start
```

This will start the MCP server on port 3000.

## Development

```bash
npm run watch
```

## Testing

```bash
npm test
```

## Advanced Features

### Memory Replay
The memory replay mechanism stores past input-output pairs and periodically retrains on them to reinforce learning. This helps prevent catastrophic forgetting and improves overall model performance.

### Dynamic Memory Allocation
The model can dynamically adjust memory allocation based on the complexity of the input and the surprise level (prediction error). This allows it to allocate more resources to complex patterns and compress simpler ones.

### Long-term Memory Storage
The system maintains a persistent long-term memory that survives across sessions. This memory is stored on disk and loaded when the server starts, allowing for continuity in learning.

### Memory Compression
Advanced compression techniques reduce the memory footprint while preserving important information. This is particularly useful for deployment in resource-constrained environments.

### LLM Cache Integration
The system maintains a cache of frequently accessed memory states, improving performance for repeated queries and reducing computational overhead.

## Citation

If you use this implementation in your research, please cite:

```
@misc{titanmemory2023,
  author = {Titan Memory Team},
  title = {Titan Memory: Enhanced Memory Framework for Language Models},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/titan-memory/titan-cognitive-memory-framework}
}
```

## License

MIT