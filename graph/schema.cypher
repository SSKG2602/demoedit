// ChronoRAG-Lite — Minimal Graph Schema for History Texts

// Uniqueness
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (ch:Chapter)  REQUIRE ch.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (ev:Event)    REQUIRE ev.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (cl:Claim)    REQUIRE cl.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person)    REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (org:Org)     REQUIRE org.id IS UNIQUE;

// Suggested properties:
// Document {id, title, authors, source_path}
// Chapter  {id, title, ord, window_start, window_end, source_path}
// Event    {id, name, occurred_on, window_start, window_end, place}
// Claim    {id, text, valid_from, valid_to, observed_at, source_path}
// Person   {id, name}
// Org      {id, name}

// Helpful indexes for time queries
CREATE BTREE INDEX IF NOT EXISTS FOR (ev:Event) ON (ev.occurred_on);
CREATE BTREE INDEX IF NOT EXISTS FOR (cl:Claim) ON (cl.valid_from);
CREATE BTREE INDEX IF NOT EXISTS FOR (cl:Claim) ON (cl.valid_to);

// Relations:
// (Document)-[:HAS_CHAPTER]->(Chapter)
// (Chapter)-[:DESCRIBES_EVENT]->(Event)
// (Chapter)-[:ASSERTS]->(Claim)
// (Claim)-[:MENTIONS_PERSON]->(Person)
// (Claim)-[:MENTIONS_ORG]->(Org)
// (Chapter)-[:FROM_WORK]->(Document)
