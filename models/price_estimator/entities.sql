-- Every listing is scored when it goes live (as_of). The label (hammer price)
-- arrives when the auction ends, and only if the car sells; unsold or still-running
-- auctions have a NULL label and are dropped from training by the extract stage.
SELECT
  a.auction_id        AS entity_id,
  a.listed_at         AS as_of,
  CASE WHEN a.sold THEN a.hammer_price END AS label,
  a.ended_at          AS label_known_at,
  a.auction_id, v.make, v.model, v.fuel, v.segment, v.first_registration,
  a.mileage_km, a.damage_score, v.power_kw, a.start_price, a.seller_id,
  s.country           AS seller_country,
  a.relist_no
FROM auctions a
JOIN vehicles v USING (vin)
JOIN sellers  s USING (seller_id)
