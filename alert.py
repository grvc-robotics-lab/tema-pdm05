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
# NEW_POLYGON = [
#         [
#           6.996817325,
#           50.521793116
#         ],
#         [
#           6.982379622,
#           50.517755746
#         ],
#         [
#           6.976085293,
#           50.513526712
#         ],
#         [
#           6.980425116,
#           50.506469699
#         ],
#         [
#           6.995878527,
#           50.511398138
#         ],
#         [
#           6.996817325,
#           50.521793116
#         ],
#         [
#           6.996817325,
#           50.521793116
#         ]
#       ]
# KML coordinates are (lon, lat)
# NEW_POLYGON = [
#     [8.629771764829439, 40.15859418187855],
#     [8.625061440063686, 40.15812980346134],
#     [8.622634553547057, 40.15561452675333],
#     [8.626133607094404, 40.15261485269694],
#     [8.632394881928512, 40.1533949156366],
#     [8.633083570598743, 40.15730750244064],
#     [8.629771764829439, 40.15859418187855],  # closed ring (same as first)
# ]

# NEW_POLYGON = [
#       [9.732311606, 40.627451875],
#       [9.716903069, 40.627095293],
#       [9.706001092, 40.619803824],
#       [9.703000808, 40.617262544],
#       [9.700696713, 40.611456144],
#       [9.698690347, 40.610015298],
#       [9.698271487, 40.6059604],
#       [9.710637112, 40.597243438],
#       [9.751027755, 40.596869824],
#       [9.750188491, 40.617371464],
#       [9.732311606, 40.627451875],
#   ]
#
NEW_POLYGON = [
      [27.70813229561291, 64.18763647770248],
      [27.70843288098518, 64.18705432734706],
      [27.71011139419683, 64.18683796457785],
      [27.71173590326671, 64.1867801133575],
      [27.71298094683969, 64.18681920397789],
      [27.71388445568453, 64.18704816421578],
      [27.71429071457188, 64.18765287719118],
      [27.71437122305278, 64.18831631333231],
      [27.71411059082378, 64.18877484367013],
      [27.71305967245402, 64.18903375037341],
      [27.71113889836703, 64.18905334250844],
      [27.7090518413021, 64.18892942675124],
      [27.70853502511614, 64.18869350510883],
      [27.70823110503597, 64.18843312412325],
      [27.70808620107882, 64.1880390521049],
      [27.70813229561291, 64.18763647770248],
  ]


# # SAETA
# NEW_POLYGON = [
#     [-6.000906549604413, 37.42710370215227],
#     [-6.000569285020273, 37.42800999203162],
#     [-6.00164888389621, 37.42839065946295],
#     [-6.002601870897303, 37.42810935796709],
#     [-6.003467440942791, 37.4279694581158],
#     [-6.004212396524713, 37.42770667134567],
#     [-6.004826796272501, 37.42733339335095],
#     [-6.005269269692375, 37.42696062327376],
#     [-6.005495969436032, 37.42644313735227],
#     [-6.00531508977907, 37.42609839842831],
#     [-6.004890047556412, 37.42571696753475],
#     [-6.000906549604413, 37.42710370215227],
# ]

#PLAZA DE AGUA
# NEW_POLYGON = [
#     [-6.002453130611205, 37.41045403994147],
#     [-6.002506329304445, 37.41037692251411],
#     [-6.002488888800814, 37.41032299739918],
#     [-6.002445738048923, 37.41028071629236],
#     [-6.002344171203778, 37.41027664986794],
#     [-6.002221662569371, 37.41031025381303],
#     [-6.002203644429422, 37.4103550456081],
#     [-6.002203441721191, 37.41040371456995],
#     [-6.002265397501597, 37.41048760532705],
#     [-6.002378073328086, 37.41048417598834],
#     [-6.002453130611205, 37.41045403994147],
# ]


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
    print(f"bm_id_hash {bm_id_hash}")
    expires_ts = (datetime.now(timezone.utc) + timedelta(minutes=60)).isoformat(timespec="milliseconds").replace("+00:00", "Z")

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
        "event": {"type": "Property", "value": "Fire"},
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
        "uav": {
            "type": "Property",
            "value": {
                "common_base": {
                    "type": "Point",
                    "coordinates": [9.7008307, 40.6057891],
                },
                "multi_uav": {
                    "enabled": True,
                },
                "sweep_m": 120,
            },
        },# 37.41047348,-6.00235670
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
        "title": {"type": "Property", "value": "Summary Statistics for Fire Campaign"},
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
    ALERT_ENTITY_ID = f"urn:ngsi-ld:tema:use:test_{timestamp}:USE_local_Saeta"
    # urn:ngsi-ld:tema:kamk:sv07:alert:brk:115
    alert_hash = hashlib.md5(ALERT_ENTITY_ID.encode("utf-8")).hexdigest()
    alert_resp = create_alert_entity(ORION_URL, ALERT_ENTITY_ID)
    print(f"✅ Alert created → {alert_resp.status_code}")

    # # --------------------------------------------------------------------
    # # 2) Create GroundSensorStats entity
    # # --------------------------------------------------------------------
    # stats_resp = create_ground_sensor_stats_entity(ORION_URL, id_date=alert_hash)
    # print(f"✅ GroundSensorStats created → {stats_resp.status_code}")
