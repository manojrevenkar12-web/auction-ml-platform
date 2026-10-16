-- Post-event data: how many bids the auction drew. Only exists after it ends.
SELECT
  auction_id              AS entity_key,
  ended_at                AS event_time,
  ended_at                AS known_at,
  n_bids::DOUBLE          AS n_bids
FROM auctions
WHERE ended_at IS NOT NULL AND ended_at >= TIMESTAMP '{start}' AND ended_at < TIMESTAMP '{end}'
