import requests
import json
import hashlib
from datetime import datetime, timedelta, timezone
import os

MAX_WORKERS = min(8, (os.cpu_count() or 4) * 2)
print(f"Max workers allocated ---> {MAX_WORKERS}")
# exit()

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
            "value": dict(type="Polygon", coordinates=[
                [
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
                    ]
                ]
            ]),
        },
        "areaDesc": {"type": "Property", "value": "Kaajani hystorical data trial"},
        "bm_id": {"type": "Property", "value": bm_id_hash},
        "category": {"type": "Property", "value": "Safety"},
        "certainty": {"type": "Property", "value": "Observed"},
        "effective": {"type": "Property", "value": "2024-05-27T14:40:39.034Z"},
        "event": {"type": "Property", "value": "Flood"},
        "expires": {"type": "Property", "value": expires_timestamp},
        "ignitionPoints": {
            "type": "GeoProperty",
            "value": {"type": "Point", "coordinates": [29.124676, 64.384618]},
        },
        "msgType": {"type": "Property", "value": "Alert"},
        "sender": {"type": "Property", "value": "alert@tema-project.eu"},
        "sent": {"type": "Property", "value": sent_timestamp},
        "severity": {"type": "Property", "value": "Severe"},
        "urgency": {"type": "Property", "value": "Immediate"},
        "status": {"type": "Property", "value": "Exercise"},
        "location": {
            "type": "GeoProperty",
            "value": {
                "type": "Polygon",
                "coordinates": [
      [
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
        ]
      ]
    ],
            },
        },
        "scope": {"type": "Property", "value": "Private"},
    }

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


if __name__ == "__main__":
    # Configurazione: URL del broker e ID dell'entità da creare
    ORION_URL = "https://orion.tema.digital-enabler.eng.it"
    ENTITY_ID = "urn:ngsi-ld:Alert:test20250512:Ahrtal:00001"

    # Creazione dell'entità
    resp = create_alert_entity(ORION_URL, ENTITY_ID)

    # Stampa la risposta del broker
    print(f"Response status: {resp.status_code}")
    print(resp.text)
