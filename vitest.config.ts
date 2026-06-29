import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // Disable the CTC Python sidecar in tests — it spawns a real process
    // that takes >4s and causes timeouts. The sidecar is tested separately
    // via the evolva-mragent Python test suite.
    env: {
      CTC_DISABLED: '1',
    },
    testTimeout: 10_000,
    hookTimeout: 15_000,
  },
});
