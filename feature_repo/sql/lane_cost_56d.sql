-- Transport cost per origin->destination lane. A cost is only known once the
-- car is DELIVERED, so the window is on delivered_at.
WITH days AS (
  SELECT d::DATE AS d FROM range(DATE '{start}', DATE '{end}', INTERVAL 1 DAY) t(d)
),
t AS (
  SELECT t.origin_country || '>' || t.dest_country AS lane_key, t.delivered_at, t.cost_eur,
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
  ON t.delivered_at <  days.d + INTERVAL 1 DAY
 AND t.delivered_at >= days.d + INTERVAL 1 DAY - INTERVAL 56 DAY
GROUP BY t.lane_key, days.d
