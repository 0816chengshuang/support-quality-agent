-- Fully synthetic data. UTC snapshot (exclusive): 2026-10-08 00:00:00.
-- Intentionally includes boundary events and future events for as-of testing.
-- A fresh database is required; no existing tables are dropped or overwritten.
CREATE TABLE support_tickets (
    id INTEGER PRIMARY KEY,
    category TEXT NOT NULL,
    created_at TEXT NOT NULL,
    due_at TEXT NOT NULL,
    resolved_at TEXT,
    cancelled_at TEXT,
    is_test INTEGER NOT NULL DEFAULT 0 CHECK (is_test IN (0, 1)),
    duplicate_of INTEGER REFERENCES support_tickets(id),
    CHECK (due_at >= created_at),
    CHECK (resolved_at IS NULL OR resolved_at >= created_at),
    CHECK (cancelled_at IS NULL OR cancelled_at >= created_at)
);

INSERT INTO support_tickets
    (id, category, created_at, due_at, resolved_at, cancelled_at, is_test, duplicate_of)
VALUES
    (1, 'account', '2026-10-01 08:00:00', '2026-10-02 08:00:00', '2026-10-01 10:00:00', NULL, 0, NULL),
    (2, 'account', '2026-10-02 08:00:00', '2026-10-03 08:00:00', '2026-10-03 10:00:00', NULL, 0, NULL),
    (3, 'account', '2026-10-03 08:00:00', '2026-10-04 08:00:00', NULL, NULL, 0, NULL),
    (4, 'delivery', '2026-10-04 08:00:00', '2026-10-05 08:00:00', '2026-10-05 08:00:00', NULL, 0, NULL),
    (5, 'delivery', '2026-10-05 08:00:00', '2026-10-06 08:00:00', '2026-10-06 10:00:00', NULL, 0, NULL),
    (6, 'delivery', '2026-10-07 08:00:00', '2026-10-08 08:00:00', NULL, '2026-10-08 00:00:00', 0, NULL),
    (7, 'billing', '2026-10-06 08:00:00', '2026-10-07 08:00:00', '2026-10-06 09:00:00', NULL, 0, NULL),
    (8, 'billing', '2026-10-07 20:00:00', '2026-10-08 20:00:00', NULL, '2026-10-09 00:00:00', 0, NULL),
    (9, 'account', '2026-10-07 10:00:00', '2026-10-07 22:00:00', '2026-10-08 10:00:00', NULL, 0, NULL),
    (10, 'delivery', '2026-10-01 00:00:00', '2026-10-02 00:00:00', '2026-10-01 01:00:00', NULL, 0, NULL),
    (11, 'account', '2026-10-01 08:00:00', '2026-10-02 08:00:00', NULL, NULL, 1, NULL),
    (12, 'account', '2026-10-02 08:00:00', '2026-10-03 08:00:00', NULL, NULL, 0, 2),
    (13, 'delivery', '2026-10-02 08:00:00', '2026-10-03 08:00:00', NULL, '2026-10-04 00:00:00', 0, NULL),
    (14, 'account', '2026-09-30 23:59:59', '2026-10-01 23:59:59', NULL, NULL, 0, NULL),
    (15, 'account', '2026-10-08 00:00:00', '2026-10-09 00:00:00', NULL, NULL, 0, NULL);
