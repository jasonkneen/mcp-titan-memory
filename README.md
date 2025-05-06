
# Titan Memory Server

A Model Context Protocol (MCP) server implementation with an enhanced Titan Memory model.

## Overview

This project implements a memory model for large language models (LLMs) that is designed to enhance memory capabilities in generative AI systems. It's built using TensorFlow.js and implemented as an MCP server, making it easy to integrate with any MCP-compatible client.

## Features

Current implementation includes:
- Multi-head attention mechanism for improved memory focus
- Basic hierarchical memory structure for organizing information
- Manifold operations for geometric memory representation
- Surprise-based memory updates to focus on unexpected information
- TensorFlow.js backend for efficient tensor operations

Planned features:
- Enhanced contextual memory updates
- Memory replay mechanism
- Dynamic memory allocation

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

## Citation

If you use this implementation in your research, please cite:

```
@misc{titanmemory2023,
  author = {Replit Team},
  title = {Titan Memory: Enhanced Memory Framework for Language Models},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/replit/titan-memory}
}
```

## License

MIT
