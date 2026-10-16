-- Recent clearing prices for the same make/model/fuel, from auctions that ENDED
-- in the 90 days before the snapshot.
WITH days AS (
  SELECT d::DATE AS d FROM range(DATE '{start}', DATE '{end}', INTERVAL 1 DAY) t(d)
),
sold AS (
  SELECT v.make || ' ' || v.model || '|' || v.fuel AS mm_fuel_key,
         a.ended_at, a.hammer_price, a.mileage_km,
         date_diff('month', v.first_registration, a.listed_at) AS age_months
  FROM auctions a JOIN vehicles v USING (vin)
  WHERE a.sold
)
SELECT
  s.mm_fuel_key                           AS entity_key,
  (days.d + INTERVAL 1 DAY)::TIMESTAMP    AS event_time,
  max(s.ended_at)                         AS known_at,
  median(s.hammer_price)::DOUBLE          AS median_price_90d,
  median(s.age_months)::DOUBLE            AS median_age_months_90d,
  median(s.mileage_km)::DOUBLE            AS median_km_90d,
  count(*)::DOUBLE                        AS volume_90d
FROM days
JOIN sold s
  ON s.ended_at <  days.d + INTERVAL 1 DAY
 AND s.ended_at >= days.d + INTERVAL 1 DAY - INTERVAL 90 DAY
GROUP BY s.mm_fuel_key, days.d
