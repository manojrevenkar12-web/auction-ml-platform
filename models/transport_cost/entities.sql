-- Every booked transport is quoted when the auction ends (booked_at). The true
-- cost is only known on delivery (NULL while the car is on the truck).
SELECT
  t.auction_id        AS entity_id,
  t.booked_at         AS as_of,
  t.cost_eur          AS label,
  t.delivered_at      AS label_known_at,
  t.origin_country, t.dest_country, v.segment, a.damage_score
FROM transports t
JOIN auctions a USING (auction_id)
JOIN vehicles v USING (vin)
