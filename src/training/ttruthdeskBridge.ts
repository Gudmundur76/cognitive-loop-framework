/**
 * ttruthdeskBridge.ts — DB Connection Bridge
 *
 * Provides a single function to fetch verified claims from the ttruthdesk
 * MySQL database. Keeps the DB dependency isolated so that if the ttruthdesk
 * schema changes, only this file changes.
 *
 * Connection: reads DATABASE_URL from the environment (same env var used by
 * ttruthdesk-platform/server/db.ts). Both services share the same MySQL
 * instance in the docker-compose network.
 *
 * Type mapping: ttruthdesk claims table → ClaimRecord (internal type)
 *
 * Design constraints: max 200 lines, max 20 lines/function, max 3 params
 */

import mysql from 'mysql2/promise';

// ─── Types ─────────────────────────────────────────────────────────────────────

export interface ClaimRecord {
  id: number;
  claimText: string;
  verdict: string;
  verdictRationale: string | null;
  confidenceScore: number | null;
  compositeTruthLabel: string | null;
  verticalDomain: string | null;
  createdAt: Date;
  /** Evidence URL (PDB, PubMed, etc.) */
  evidenceUrl: string | null;
}

// ─── Connection ────────────────────────────────────────────────────────────────

let _pool: mysql.Pool | null = null;

function getPool(): mysql.Pool {
  if (_pool) return _pool;
  const url = process.env['DATABASE_URL'];
  if (!url) {
    throw new Error('[ttruthdeskBridge] DATABASE_URL env var is not set');
  }
  _pool = mysql.createPool(url);
  return _pool;
}

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Fetch verified claims from the ttruthdesk registry.
 *
 * Filters:
 *   - verdict IN ('Supported', 'Contradicted', 'Partially Supported')
 *   - confidenceScore >= 0.7
 *   - compositeTruthLabel IN ('verified_faithful', 'verified_distorted', 'contradicted')
 *   - createdAt > since (only new claims since last training run)
 *
 * Returns an empty array if the DB is unreachable (non-fatal).
 */
export async function fetchVerifiedClaims(since: Date): Promise<ClaimRecord[]> {
  try {
    const pool = getPool();
    const [rows] = await pool.execute<mysql.RowDataPacket[]>(
      `SELECT
         id,
         claimText,
         verdict,
         verdictRationale,
         confidenceScore,
         compositeTruthLabel,
         verticalDomain,
         createdAt,
         pdbEvidenceUrl AS evidenceUrl
       FROM claims
       WHERE verdict IN ('Supported', 'Contradicted', 'Partially Supported')
         AND confidenceScore >= 0.7
         AND compositeTruthLabel IN ('verified_faithful', 'verified_distorted', 'contradicted')
         AND createdAt > ?
       ORDER BY createdAt ASC`,
      [since]
    );
    return rows.map(mapRow);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.warn(`[ttruthdeskBridge] DB unreachable — returning empty array. Error: ${msg}`);
    return [];
  }
}

/**
 * Count verified claims created after `since`.
 * Used by CorpusWatcher to check if the threshold has been reached
 * without loading all rows.
 */
export async function countNewVerifiedClaims(since: Date): Promise<number> {
  try {
    const pool = getPool();
    const [rows] = await pool.execute<mysql.RowDataPacket[]>(
      `SELECT COUNT(*) AS cnt
       FROM claims
       WHERE verdict IN ('Supported', 'Contradicted', 'Partially Supported')
         AND confidenceScore >= 0.7
         AND compositeTruthLabel IN ('verified_faithful', 'verified_distorted', 'contradicted')
         AND createdAt > ?`,
      [since]
    );
    return Number((rows[0] as { cnt: number }).cnt ?? 0);
  } catch {
    return 0;
  }
}

/**
 * Close the connection pool. Call on process shutdown.
 */
export async function closeBridge(): Promise<void> {
  if (_pool) {
    await _pool.end();
    _pool = null;
  }
}

// ─── Internal ─────────────────────────────────────────────────────────────────

function mapRow(row: mysql.RowDataPacket): ClaimRecord {
  return {
    id: Number(row['id']),
    claimText: String(row['claimText'] ?? ''),
    verdict: String(row['verdict'] ?? ''),
    verdictRationale: row['verdictRationale'] != null ? String(row['verdictRationale']) : null,
    confidenceScore: row['confidenceScore'] != null ? Number(row['confidenceScore']) : null,
    compositeTruthLabel: row['compositeTruthLabel'] != null ? String(row['compositeTruthLabel']) : null,
    verticalDomain: row['verticalDomain'] != null ? String(row['verticalDomain']) : null,
    createdAt: row['createdAt'] instanceof Date ? row['createdAt'] : new Date(row['createdAt'] as string),
    evidenceUrl: row['evidenceUrl'] != null ? String(row['evidenceUrl']) : null,
  };
}
