-- Every listing is scored when it goes live. Label: did it sell at or above the
-- start price? Known when the auction ends (NULL while it is still running).
SELECT
  a.auction_id        AS entity_id,
  a.listed_at         AS as_of,
  CASE WHEN a.ended_at IS NOT NULL THEN a.sold::INT END AS label,
  a.ended_at          AS label_known_at,
  a.auction_id, v.make, v.model, v.fuel, v.segment, v.first_registration,
  a.mileage_km, a.damage_score, v.power_kw, a.start_price, a.seller_id,
  s.country           AS seller_country,
  a.relist_no
FROM auctions a
JOIN vehicles v USING (vin)
JOIN sellers  s USING (seller_id)
