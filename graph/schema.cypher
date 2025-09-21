// ChronoRAG-Lite minimal schema (optional)
// Nodes
CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Claim)  REQUIRE c.id IS UNIQUE;

// Suggested properties:
// Entity {id, kind, name}
// Claim  {id, text, valid_from, valid_to, observed_at, source_path}

// Indices for time queries (if using Neo4j retrieval):
CREATE BTREE INDEX IF NOT EXISTS FOR (c:Claim) ON (c.valid_from);
CREATE BTREE INDEX IF NOT EXISTS FOR (c:Claim) ON (c.valid_to);
