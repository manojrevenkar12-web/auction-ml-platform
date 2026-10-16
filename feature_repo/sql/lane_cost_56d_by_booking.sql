-- WRONG ON PURPOSE (backward-window error): windows on booked_at, so a snapshot
-- includes costs of cars still on the truck, i.e. values not yet known.
-- known_at is still computed honestly from delivered_at, which is what lets the
-- review catch it (known_at > event_time).
WITH days AS (
  SELECT d::DATE AS d FROM range(DATE '{start}', DATE '{end}', INTERVAL 1 DAY) t(d)
),
t AS (
  SELECT t.origin_country || '>' || t.dest_country AS lane_key, t.booked_at, t.delivered_at, t.cost_eur,
         t.cost_eur / NULLIF(l.distance_km, 0) AS cost_per_km
  FROM transports t
  JOIN lanes l ON l.lane_key = t.origin_country || '>' || t.dest_country
  WHERE t.delivered_at IS NOT NULL
)
SELECT
  t.lane_key                              AS entity_key,
  (days.d + INTERVAL 1 DAY)::TIMESTAMP    AS event_time,
  max(t.delivered_at)                     AS known_at,
  median(t.cost_eur)::DOUBLE              AS median_cost_56d,
  median(t.cost_per_km)::DOUBLE           AS cost_per_km_56d,
  count(*)::DOUBLE                        AS volume_56d
FROM days JOIN t
  ON t.booked_at <  days.d + INTERVAL 1 DAY
 AND t.booked_at >= days.d + INTERVAL 1 DAY - INTERVAL 56 DAY
GROUP BY t.lane_key, days.d
