/**
 * This file is deprecated. All server functionality has been consolidated
 * into the MCP server implementation in index.ts.
 *
 * This file now contains a stub class that redirects to the MCP server.
 * It remains for backward compatibility with existing code references.
 */
export declare class TitanExpressServer {
    private mcpServer;
    constructor();
    listen(port: number, callback?: () => void): any;
    close(callback?: () => void): any;
}
