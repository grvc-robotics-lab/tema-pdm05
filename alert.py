import requests
import json
from datetime import datetime, timedelta, timezone

def create_entity():
    url = 'https://orion.tema.digital-enabler.eng.it/ngsi-ld/v1/entities'

    headers = {
        'Content-Type': 'application/ld+json',
        'Accept': 'application/ld+json'
    }

    data = {
        "@context": [
            "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
        ],
        "id": "urn:ngsi-ld:Alert:100",
        "type": "Alert",
        "areaDesc": {
            "type": "Property",
            "value": "TEMA Pilot exercise"
        },
        "category": {
            "type": "Property",
            "value": "Safety"
        },
        "certainty": {
            "type": "Property",
            "value": "Observed"
        },
        "effective": {
            "type": "Property",
            "value": "2024-12-18T11:27:00.343Z"
        },
        "event": {
            "type": "Property",
            "value": "Fire"
        },
        "expires": {
            "type": "Property",
            "value": "2024-12-28T11:27:00.343Z"
        },
        "location": {
            "type": "GeoProperty",
            "value": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [8.629761, 40.181759],
                        [8.702888, 40.194348],
                        [8.736877, 40.14004],
                        [8.651047, 40.098559],
                        [8.570709, 40.130591],
                        [8.598862, 40.171004],
                        [8.629761, 40.181759]
                    ]
                ]
            }
        },
        "ignitionPoints": {
            "type": "GeoProperty",
            "value": {
                "type": "MultiPoint",
                "coordinates": [
                    [8.67274, 40.165429],
                    [8.648016, 40.128889]
                ]
            }
        },
        "msgType": {
            "type": "Property",
            "value": "Alert"
        },
        "scope": {
            "type": "Property",
            "value": "Private"
        },
        "sender": {
            "type": "Property",
            "value": "alert@tema-project.eu"
        },
        "sent": {
            "type": "Property",
            "value": "2024-12-18T11:27:00.343Z"
        },
        "severity": {
            "type": "Property",
            "value": "Severe"
        },
        "status": {
            "type": "Property",
            "value": "Exercise"
        },
        "urgency": {
            "type": "Property",
            "value": "Immediate"
        },
        "area": {
            "type": "GeoProperty",
            "value": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [8.629761, 40.181759],
                        [8.702888, 40.194348],
                        [8.736877, 40.14004],
                        [8.651047, 40.098559],
                        [8.570709, 40.130591],
                        [8.598862, 40.171004],
                        [8.629761, 40.181759]
                    ]
                ]
            }
        }
    }

    response_ = requests.post(url, json=data, headers=headers)

    if response_.status_code == 201:
        print(f"Entity created successfully.")
        return response_
    elif response_.status_code == 409:
        print(f"Entity already exists.")
        return {"status": "Entity already exists"}, 409
    else:
        print(f"Failed to create entity: {response_.status_code} - {response_.text}")
        return {"status": "Error"}, 500

def update_entity(entity_id_):
    url_ = f'https://orion.tema.digital-enabler.eng.it/ngsi-ld/v1/entities/{entity_id_}/attrs'
    headers = {
        'Content-Type': 'application/ld+json',
        'Accept': 'application/ld+json'
    }
    data_to_send = {
        "effective": {
            "type": "Property",
            "value": datetime.now(timezone.utc).isoformat()
        },
        "severity": {
            "type": "Property",
            "value": "Moderate"  # Example of updating severity
        },
        "areaDesc": {
            "type": "Property",
            "value": "Updated TEMA Pilot exercise description"
        }
    }

    payload_with_context = {
        "@context": "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
        **data_to_send
    }

    try:
        response = requests.post(url_, json=payload_with_context, headers=headers, timeout=60)
        response.raise_for_status()

        print(f"Entity {entity_id_} updated successfully!")
        return response

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred while updating entity {entity_id_}: {e}. Response text: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while updating entity {entity_id_}: {e}")

    return None

create_entity()
update_entity("urn:ngsi-ld:Alert:100")
