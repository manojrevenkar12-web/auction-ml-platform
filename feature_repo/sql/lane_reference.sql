-- Static reference data: road distance per lane. Valid from the beginning of time.
SELECT
  lane_key                         AS entity_key,
  TIMESTAMP '2000-01-01'           AS event_time,
  TIMESTAMP '2000-01-01'           AS known_at,
  distance_km::DOUBLE              AS distance_km
FROM lanes
