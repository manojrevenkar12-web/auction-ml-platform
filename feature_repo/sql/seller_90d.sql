-- Seller track record over auctions that ENDED in the 90 days before the snapshot.
-- event_time = snapshot (midnight after day d); known_at = latest ended_at used.
WITH days AS (
  SELECT d::DATE AS d FROM range(DATE '{start}', DATE '{end}', INTERVAL 1 DAY) t(d)
),
w AS (
  SELECT a.seller_id, days.d, a.sold, a.hammer_price, a.start_price, a.ended_at
  FROM days
  JOIN auctions a
    ON a.ended_at <  days.d + INTERVAL 1 DAY
   AND a.ended_at >= days.d + INTERVAL 1 DAY - INTERVAL 90 DAY
)
SELECT
  seller_id                                    AS entity_key,
  (d + INTERVAL 1 DAY)::TIMESTAMP              AS event_time,
  max(ended_at)                                AS known_at,
  count(*)::DOUBLE                             AS listings_90d,
  avg(sold::INT)::DOUBLE                       AS sell_through_90d,
  avg(CASE WHEN sold THEN hammer_price / start_price END)::DOUBLE AS price_to_start_90d
FROM w
GROUP BY seller_id, d
