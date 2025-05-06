/**
 * This test file is for the deprecated Express server implementation.
 * All server functionality has been consolidated into the MCP server in index.ts.
 */
import { test, expect } from '@jest/globals';
import { TitanExpressServer } from '../server';
test('Server implementation consolidated to MCP', () => {
    // This is just a placeholder test to indicate the architectural change
    expect(() => new TitanExpressServer()).not.toThrow();
});
//# sourceMappingURL=server.test.js.map