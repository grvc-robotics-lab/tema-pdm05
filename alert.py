import requests
import json
import hashlib
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
LD_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/ld+json",
    "Link": (
        "<https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context-v1.6.jsonld>; "
        'rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"'
    ),
}

# New polygon coordinates
NEW_POLYGON = [
        [
          6.996817325,
          50.521793116
        ],
        [
          6.982379622,
          50.517755746
        ],
        [
          6.976085293,
          50.513526712
        ],
        [
          6.980425116,
          50.506469699
        ],
        [
          6.995878527,
          50.511398138
        ],
        [
          6.996817325,
          50.521793116
        ],
        [
          6.996817325,
          50.521793116
        ]
      ]


def utc_now_iso() -> str:
    """Return the current UTC time in ISO‑8601 (ms precision, Z suffix)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Entity builders
# ---------------------------------------------------------------------------

def create_alert_entity(orion_url: str, entity_id: str):
    """Create an NGSI‑LD Alert entity with the new polygon."""
    bm_id_hash = hashlib.md5(entity_id.encode("utf-8")).hexdigest()
    sent_ts = utc_now_iso()
    expires_ts = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    entity = {
        "id": entity_id,
        "type": "Alert",
        "area": {
            "type": "GeoProperty",
            "value": {"type": "Polygon", "coordinates": [NEW_POLYGON]},
        },
        "areaDesc": {"type": "Property", "value": "Seville trial polygon"},
        "bm_id": {"type": "Property", "value": bm_id_hash},
        "category": {"type": "Property", "value": "Safety"},
        "certainty": {"type": "Property", "value": "Observed"},
        "effective": {"type": "Property", "value": sent_ts},
        "event": {"type": "Property", "value": "Flood"},
        "expires": {"type": "Property", "value": expires_ts},
        "ignitionPoints": {
            "type": "GeoProperty",
            "value": {"type": "Point", "coordinates": [-5.99823, 37.42509]},  # update if needed
        },
        "msgType": {"type": "Property", "value": "Alert"},
        "sender": {"type": "Property", "value": "alert@tema-project.eu"},
        "sent": {"type": "Property", "value": sent_ts},
        "severity": {"type": "Property", "value": "Severe"},
        "urgency": {"type": "Property", "value": "Immediate"},
        "status": {"type": "Property", "value": "Exercise"},
        "location": {
            "type": "GeoProperty",
            "value": {"type": "Polygon", "coordinates": [NEW_POLYGON]},
        },
        "scope": {"type": "Property", "value": "Private"},
    }

    response = requests.post(f"{orion_url}/ngsi-ld/v1/entities", headers=LD_HEADERS, json=entity, timeout=10)
    response.raise_for_status()
    return response


def create_ground_sensor_stats_entity(orion_url: str, id_date: str, date_modified: datetime | None = None):
    """Create a GroundSensorStats entity that uses the same polygon for the AOI."""
    date_modified = date_modified or datetime.now(timezone.utc)

    entity_id = f"urn:ngsi-ld:tema:DLR-KN:SummaryStatistics:GroundSensorStats:{id_date}"
    fname_value = f"{entity_id}.hdf5"

    entity = {
        "id": entity_id,
        "type": "GroundSensorStats",
        "description": {
            "type": "Property",
            "value": "NetCDF file containing sensor measurements from TEMA ground stations.",
        },
        "title": {"type": "Property", "value": "Summary Statistics for Flood Campaign"},
        "aoi": {
            "type": "Property",
            "value": {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [NEW_POLYGON],
                },
                "properties": {"description": "Polygon enclosing all points"},
            },
        },
        "bm_id": {"type": "Property", "value": id_date},
        "bucket": {"type": "Property", "value": "dlr"},
        "date": {"type": "Property", "value": utc_now_iso()},
        "dateModified": {
            "type": "Property",
            "value": date_modified.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        },
        "file_name": {
            "type": "Property",
            "value": "kn/summary_stats/sim_kahy_dataset.hdf5",
        },
        "fname": {"type": "Property", "value": fname_value},
        "minio_url": {
            "type": "Property",
            "value": "storage.tema.digital-enabler.eng.it/dlr/kn/summary_stats/sim_kahy_dataset.hdf5",
        },
    }

    response = requests.post(f"{orion_url}/ngsi-ld/v1/entities", headers=LD_HEADERS, json=entity, timeout=10)
    response.raise_for_status()
    return response


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ORION_URL = "https://orion.tema.digital-enabler.eng.it"

    # Use a new unique ID with timestamp
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ALERT_ENTITY_ID = f"urn:ngsi-ld:tema:use:test_{timestamp}:brk"
    # urn:ngsi-ld:tema:kamk:sv07:alert:brk:115
    alert_hash = hashlib.md5(ALERT_ENTITY_ID.encode("utf-8")).hexdigest()
    alert_resp = create_alert_entity(ORION_URL, ALERT_ENTITY_ID)
    print(f"✅ Alert created → {alert_resp.status_code}")

    # # --------------------------------------------------------------------
    # # 2) Create GroundSensorStats entity
    # # --------------------------------------------------------------------
    # stats_resp = create_ground_sensor_stats_entity(ORION_URL, id_date=alert_hash)
    # print(f"✅ GroundSensorStats created → {stats_resp.status_code}")