-- seed_crisis_events.sql
-- Run ONCE manually (not part of create_tables.sql/init_schema, since this
-- is research data you curate, not schema). Safe to re-run — uses
-- INSERT IGNORE keyed on event_name to avoid duplicates.

INSERT IGNORE INTO crisis_events (event_name, crisis_type, start_date, end_date, description)
VALUES (
    '2024 Sri Lanka Presidential Election',
    'Political',
    '2024-09-21',
    '2024-11-30',
    'Anura Kumara Dissanayake elected president amid deep public anger over IMF-mandated austerity following the 2019-2024 economic crisis; his NPP alliance won a landslide parliamentary majority two months later. A scheduled, anticipated transition following a long build-up of economic/political discontent — contrasts with the 2026 event as a "known date, gradual build-up" case.'
);

INSERT IGNORE INTO crisis_events (event_name, crisis_type, start_date, end_date, description)
VALUES (
    '2026 Sri Lanka Fuel Crisis (Iran War Oil Shock)',
    'Economic',
    '2026-02-28',
    NULL,
    'Fuel and forex pressure triggered when the US-Israel war on Iran (began Feb 28, 2026) disrupted oil traffic through the Strait of Hormuz, through which Sri Lanka imports ~60% of its energy. Pump prices rose 33.8% in the first three weeks of the conflict. A sudden, unanticipated external shock — contrasts with the 2024 election as a "no advance warning" case.'
);