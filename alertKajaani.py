import requests
import json
import hashlib
from datetime import datetime, timedelta, timezone


def create_alert_entity(orion_url, entity_id):

    # Generate bm_id as MD5 hash of the entity_id
    bm_id_hash = hashlib.md5(entity_id.encode("utf-8")).hexdigest()

    sent_timestamp = (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    expires_timestamp = (
        (datetime.now(timezone.utc) + timedelta(days=1))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    print(bm_id_hash)
    # Define the entity the sent filed is the real datetime, the expires filed is real datetime +1 day, effective field is the datetime of kajaani prescribed fire 2024-05-27
    entity = {
        "id": entity_id,
        "type": "Alert",
        "area": {
            "type": "GeoProperty",
            "value": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [8.631689086168455, 40.16335726020773],
                        [8.623206337720406, 40.16222291801559],
                        [8.619699230636662, 40.15612260191023],
                        [8.62482460751853, 40.15062652795407],
                        [8.635321541976319, 40.15143724846202],
                        [8.639270258224521, 40.15751529158516],
                        [8.631689086168455, 40.16335726020773]
                    ]
                ],
            },
        },
        "areaDesc": {"type": "Property", "value": "Sardinia hystorical data trial"},
        "bm_id": {"type": "Property", "value": bm_id_hash},
        "category": {"type": "Property", "value": "Safety"},
        "certainty": {"type": "Property", "value": "Observed"},
        "effective": {"type": "Property", "value": "2024-05-27T14:40:39.034Z"},
        "event": {"type": "Property", "value": "Fire"},
        "expires": {"type": "Property", "value": expires_timestamp},
        "ignitionPoints": {
            "type": "GeoProperty",
            "value": {"type": "Point", "coordinates": [29.128431, 64.382015]},
        },
        "msgType": {"type": "Property", "value": "Alert"},
        "sender": {"type": "Property", "value": "alert@tema-project.eu"},
        "sent": {"type": "Property", "value": "2024-05-27T14:40:39.034Z"},
        "severity": {"type": "Property", "value": "Severe"},
        "urgency": {"type": "Property", "value": "Immediate"},
        "status": {"type": "Property", "value": "Exercise"},
        "location": {
            "type": "GeoProperty",
            "value": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [8.631689086168455, 40.16335726020773],
                        [8.623206337720406, 40.16222291801559],
                        [8.619699230636662, 40.15612260191023],
                        [8.62482460751853, 40.15062652795407],
                        [8.635321541976319, 40.15143724846202],
                        [8.639270258224521, 40.15751529158516],
                        [8.631689086168455, 40.16335726020773]
                    ]
                ],
            },
        },
        "scope": {"type": "Property", "value": "Private"},
    }
    # Francesco Ip 29.130918, 64.383283  Abdal IP 29.132391, 64.382516, LAST 29.128431, 64.382015
    # test 29.125898, 64.379626   ***** 29.116815, 64.380469
    # Define the headers

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/ld+json",
        "Link": "<https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context-v1.6.jsonld>; "
        'rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"',
    }

    # Costruisci l'URL per la creazione dell'entità
    url = f"{orion_url}/ngsi-ld/v1/entities"

    # Effettua la POST all'endpoint di Orion-LD
    response = requests.post(url, headers=headers, data=json.dumps(entity))
    return response


def create_ground_sensor_stats_entity(
    orion_url, date: datetime, date_modified: datetime, id_date: str
):
    entity_id = f"urn:ngsi-ld:tema:DLR-KN:SummaryStatistics:GroundSensorStats:{id_date}"
    fname_value = f"{entity_id}.hdf5"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/ld+json",
        "Link": "<https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context-v1.6.jsonld>; "
        'rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"',
    }
    date = (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    date_modified = date_modified.isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )

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
                    "coordinates": [
                        [
                            [8.631689086168455, 40.16335726020773],
                            [8.623206337720406, 40.16222291801559],
                            [8.619699230636662, 40.15612260191023],
                            [8.62482460751853, 40.15062652795407],
                            [8.635321541976319, 40.15143724846202],
                            [8.639270258224521, 40.15751529158516],
                            [8.631689086168455, 40.16335726020773]
                        ]
                    ],
                },
                "properties": {"description": "Polygon enclosing all points"},
            },
        },
        "bm_id": {"type": "Property", "value": id_date},
        "bucket": {"type": "Property", "value": "dlr"},
        "date": {"type": "Property", "value": date},
        "dateModified": {"type": "Property", "value": date_modified},
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

    response = requests.post(
        f"{orion_url}/ngsi-ld/v1/entities", headers=headers, data=json.dumps(entity)
    )

    if response.status_code in [201, 204]:
        print("✅ Entity created successfully.")
    else:
        print(f"❌ Failed to create entity. Status: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    # Configurazione: URL del broker e ID dell'entità da creare
    ORION_URL = "https://orion.tema.digital-enabler.eng.it"
    ENTITY_ID = "urn:ngsi-ld:Alert:test20250519:Sardinia:012"
    bm_id_hash = hashlib.md5(ENTITY_ID.encode("utf-8")).hexdigest()
    # Creazione dell'entità
    resp = create_alert_entity(ORION_URL, ENTITY_ID)

    # Stampa la risposta del broker
    print(f"Response status: {resp.status_code}")
    print(resp.text)
    create_ground_sensor_stats_entity(
        orion_url="https://orion.tema.digital-enabler.eng.it",
        date=datetime(2025, 3, 20, 15, 49, 7),
        date_modified=datetime(2024, 5, 27, 14, 43, 25),
        id_date=bm_id_hash
    )

"""
<coordinates>
						[8.631689086168455,40.16335726020773]
						[8.623206337720406,40.16222291801559]
						[8.619699230636662,40.15612260191023] 
						[8.62482460751853,40.15062652795407] 
						[8.635321541976319,40.15143724846202]
						[8.639270258224521,40.15751529158516] 
						[8.631689086168455,40.16335726020773] 
"""