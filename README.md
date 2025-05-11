# Titan Memory Server

A Model Context Protocol (MCP) server implementation with an enhanced Titan Memory model inspired by the Google white paper.

## Overview

This project implements a memory model for large language models (LLMs) that is designed to enhance memory capabilities in generative AI systems. It's built using TensorFlow.js and implemented as an MCP server, making it easy to integrate with any MCP-compatible client.

## Features

Currently implemented:
- Multi-head attention mechanism
- Basic hierarchical memory structure
- Memory state persistence
- Integration with Model Context Protocol (MCP)

In development:
- Memory replay for enhanced learning
- LLM Cache integration
- Dynamic memory allocation

Planned features:
- Long-term memory storage
- Advanced memory compression

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

## Thanks to

- Google for the Titan Paper
- Me for the first version
- @ExpressionsBot for their contributions to the early versions


## License

MIT
