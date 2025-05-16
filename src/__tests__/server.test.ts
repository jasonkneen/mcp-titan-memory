
/**
 * This test file is for the deprecated Express server implementation.
 * All server functionality has been consolidated into the MCP server in index.ts.
 */

import { test, expect } from '@jest/globals';

// Skip the test since the server is deprecated
test.skip('Server implementation consolidated to MCP', () => {
  // This is just a placeholder test to indicate the architectural change
  expect(true).toBe(true);
});
