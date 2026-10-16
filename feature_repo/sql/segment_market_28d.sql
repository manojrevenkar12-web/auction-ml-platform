-- Segment x fuel market level over the last 28 days and its trend versus the
-- 28 days before that (1.0 = flat). Windows on ended_at, when prices are known.
WITH days AS (
  SELECT d::DATE AS d FROM range(DATE '{start}', DATE '{end}', INTERVAL 1 DAY) t(d)
),
ended AS (
  SELECT v.segment || '|' || v.fuel AS sf_key, a.ended_at, a.sold, a.hammer_price
  FROM auctions a JOIN vehicles v USING (vin)
  WHERE a.ended_at IS NOT NULL
),
recent AS (
  SELECT e.sf_key, days.d, max(e.ended_at) AS known_at,
         median(CASE WHEN e.sold THEN e.hammer_price END) AS p28,
         avg(e.sold::INT) AS st28
  FROM days JOIN ended e
    ON e.ended_at <  days.d + INTERVAL 1 DAY
   AND e.ended_at >= days.d + INTERVAL 1 DAY - INTERVAL 28 DAY
  GROUP BY e.sf_key, days.d
),
prior AS (
  SELECT e.sf_key, days.d, median(e.hammer_price) AS p_prior
  FROM days JOIN ended e
    ON e.sold
   AND e.ended_at <  days.d + INTERVAL 1 DAY - INTERVAL 28 DAY
   AND e.ended_at >= days.d + INTERVAL 1 DAY - INTERVAL 56 DAY
  GROUP BY e.sf_key, days.d
)
SELECT
  r.sf_key                                 AS entity_key,
  (r.d + INTERVAL 1 DAY)::TIMESTAMP        AS event_time,
  r.known_at                               AS known_at,
  r.p28::DOUBLE                            AS median_price_28d,
  (r.p28 / NULLIF(p.p_prior, 0))::DOUBLE   AS price_trend_28d,
  r.st28::DOUBLE                           AS sell_through_28d
FROM recent r LEFT JOIN prior p USING (sf_key, d)
