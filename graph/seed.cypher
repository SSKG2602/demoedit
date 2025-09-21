// Seed — Independence history text (Bipan Chandra et al.), tailored to ./ingest/demo_dataset/independence.txt

// Document
MERGE (doc:Document {
  id: 'doc:independence:1857-1947',
  title: "India's Struggle for Independence (1857–1947)",
  authors: "Bipan Chandra; Mridula Mukherjee; Aditya Mukherjee; K N Panikkar; Sucheta Mahajan",
  source_path: "ingest/demo_dataset/independence.txt"
});

// Chapters (sample anchors; extend as needed)
MERGE (ch1:Chapter {
  id:'ch:rev_1857',
  title:'The Revolt of 1857',
  ord: 1,
  window_start:'1857-05-01T00:00:00Z',
  window_end:'1859-01-01T00:00:00Z',
  source_path: "ingest/demo_dataset/independence.txt"
})
MERGE (ch2:Chapter {
  id:'ch:santhal_1855',
  title:'Santhal Uprising',
  ord: 2,
  window_start:'1855-06-30T00:00:00Z',
  window_end:'1856-12-31T00:00:00Z',
  source_path: "ingest/demo_dataset/independence.txt"
})
MERGE (ch3:Chapter {
  id:'ch:munda_1899',
  title:'Munda Rebellion (Ulgulan)',
  ord: 3,
  window_start:'1899-12-24T00:00:00Z',
  window_end:'1900-06-30T00:00:00Z',
  source_path: "ingest/demo_dataset/independence.txt"
})
MERGE (ch4:Chapter {
  id:'ch:congress_foundation',
  title:'Foundation of Indian National Congress',
  ord: 4,
  window_start:'1885-12-28T00:00:00Z',
  window_end:'1886-12-31T00:00:00Z',
  source_path: "ingest/demo_dataset/independence.txt"
});

// Events (curated from the text)
MERGE (ev1:Event {
  id:'event:delhi_capture_1857_05_11',
  name:'Delhi seized by rebels',
  occurred_on:'1857-05-11',
  place:'Delhi',
  window_start:'1857-05-11T00:00:00Z',
  window_end:'1857-09-20T00:00:00Z'
})
MERGE (ev2:Event {
  id:'event:santhal_assembly_1855_06_30',
  name:'Santhal assembly at Bhagnadihi',
  occurred_on:'1855-06-30',
  place:'Bhagnadihi (Damin-i-koh)',
  window_start:'1855-06-30T00:00:00Z',
  window_end:'1856-12-31T00:00:00Z'
})
MERGE (ev3:Event {
  id:'event:munda_ulgulan_1899',
  name:'Munda Ulgulan (Birsa Munda Rebellion)',
  occurred_on:'1899-12-24',
  place:'Chhotanagpur',
  window_start:'1899-12-24T00:00:00Z',
  window_end:'1900-06-30T00:00:00Z'
})
MERGE (ev4:Event {
  id:'event:inc_foundation_1885',
  name:'Foundation of Indian National Congress',
  occurred_on:'1885-12-28',
  place:'Bombay',
  window_start:'1885-12-28T00:00:00Z',
  window_end:'1885-12-29T00:00:00Z'
});

// People / Orgs referenced
MERGE (p_bahadur:Person {id:'person:bahadur_shah_ii', name:'Bahadur Shah II'})
MERGE (p_rani:Person {id:'person:rani_lakshmibai', name:'Rani Lakshmibai'})
MERGE (p_birsa:Person {id:'person:birsa_munda', name:'Birsa Munda'})
MERGE (p_sido:Person {id:'person:sido', name:'Sido'})
MERGE (p_kanhu:Person {id:'person:kanhu', name:'Kanhu'})
MERGE (org_inc:Org {id:'org:inc', name:'Indian National Congress'})

// Claims (short, auditable snippets)
MERGE (c1:Claim {
  id:'claim:bahadur_shah_proclaimed_1857',
  text:'On May 11, 1857, soldiers from Meerut entered Delhi and proclaimed Bahadur Shah II as emperor.',
  valid_from:'1857-05-11T00:00:00Z',
  valid_to:'1857-09-20T00:00:00Z',
  observed_at:'1857-05-11T00:00:00Z',
  source_path:'independence.txt'
})
MERGE (c2:Claim {
  id:'claim:rani_lakshmibai_led_jhansi',
  text:'Rani Lakshmibai assumed leadership at Jhansi after annexation via Doctrine of Lapse and died fighting on 17 Jun 1858.',
  valid_from:'1857-06-01T00:00:00Z',
  valid_to:'1858-06-17T00:00:00Z',
  observed_at:'1858-06-17T00:00:00Z',
  source_path:'independence.txt'
})
MERGE (c3:Claim {
  id:'claim:santhal_call_1855',
  text:'On June 30, 1855, Santhal leaders resolved to expel exploiters and colonial authority, rallying tens of thousands.',
  valid_from:'1855-06-30T00:00:00Z',
  valid_to:'1856-12-31T00:00:00Z',
  observed_at:'1855-06-30T00:00:00Z',
  source_path:'independence.txt'
})
MERGE (c4:Claim {
  id:'claim:munda_ulgulan_1899',
  text:'Birsa Munda proclaimed rebellion (Ulgulan) in late 1899 to establish Munda rule; captured Feb 1900 and died June 1900.',
  valid_from:'1899-12-24T00:00:00Z',
  valid_to:'1900-06-30T00:00:00Z',
  observed_at:'1900-02-01T00:00:00Z',
  source_path:'independence.txt'
})
MERGE (c5:Claim {
  id:'claim:inc_founded_1885',
  text:'Indian National Congress was founded in December 1885 at Bombay as the first all-India expression of Indian nationalism.',
  valid_from:'1885-12-28T00:00:00Z',
  valid_to:'1886-01-31T00:00:00Z',
  observed_at:'1885-12-28T00:00:00Z',
  source_path:'independence.txt'
});

// Wire up structure
MERGE (doc)-[:HAS_CHAPTER]->(ch1)
MERGE (doc)-[:HAS_CHAPTER]->(ch2)
MERGE (doc)-[:HAS_CHAPTER]->(ch3)
MERGE (doc)-[:HAS_CHAPTER]->(ch4)

MERGE (ch1)-[:DESCRIBES_EVENT]->(ev1)
MERGE (ch2)-[:DESCRIBES_EVENT]->(ev2)
MERGE (ch3)-[:DESCRIBES_EVENT]->(ev3)
MERGE (ch4)-[:DESCRIBES_EVENT]->(ev4)

MERGE (ch1)-[:ASSERTS]->(c1)
MERGE (ch1)-[:ASSERTS]->(c2)
MERGE (ch2)-[:ASSERTS]->(c3)
MERGE (ch3)-[:ASSERTS]->(c4)
MERGE (ch4)-[:ASSERTS]->(c5)

MERGE (c1)-[:MENTIONS_PERSON]->(p_bahadur)
MERGE (c2)-[:MENTIONS_PERSON]->(p_rani)
MERGE (c3)-[:MENTIONS_PERSON]->(p_sido)
MERGE (c3)-[:MENTIONS_PERSON]->(p_kanhu)
MERGE (c4)-[:MENTIONS_PERSON]->(p_birsa)
MERGE (c5)-[:MENTIONS_ORG]->(org_inc)

// Link chapters back to the parent work
MERGE (ch1)-[:FROM_WORK]->(doc)
MERGE (ch2)-[:FROM_WORK]->(doc)
MERGE (ch3)-[:FROM_WORK]->(doc)
MERGE (ch4)-[:FROM_WORK]->(doc);
