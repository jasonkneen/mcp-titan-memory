/**
 * This file is deprecated. All server functionality has been consolidated
 * into the MCP server implementation in index.ts.
 * 
 * This file now contains a stub class that redirects to the MCP server.
 * It remains for backward compatibility with existing code references.
 */

// @ts-ignore
import { createMCPServer } from '@modelcontextprotocol/sdk/server/index.js';

export class TitanExpressServer {
  private mcpServer: any;

  constructor() {
    console.warn('TitanExpressServer is deprecated. Use the MCP server in index.ts instead.');
    this.mcpServer = createMCPServer({ tools: [] });
  }

  listen(port: number, callback?: () => void) {
    console.warn('Using deprecated TitanExpressServer. Redirecting to MCP server in index.ts');
    return this.mcpServer.listen(port, '0.0.0.0', callback);
  }

  close(callback?: () => void) {
    return this.mcpServer.close(callback);
  }
}