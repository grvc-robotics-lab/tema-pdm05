import requests
from datetime import datetime, timezone

import requests
import uuid

def create_entity():
    url = 'https://orion.tema.digital-enabler.eng.it/ngsi-ld/v1/entities'
    headers = {
        'Content-Type': 'application/ld+json',
        'Accept': 'application/ld+json'
    }

    entity_id = f"urn:ngsi-ld:Alert:{uuid.uuid4()}"
    print(f"entity_id ---- {entity_id}")
    data = {
        "@context": [
            "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
        ],
        "id": entity_id,
        "type": "Alert",
        "area": {
            "type": "GeoProperty",
            "value": {
                "coordinates": [
                    [
                        [6.996817324763098, 50.52179311595503],
                        [6.982379621684984, 50.51775574602705],
                        [6.9760852934997, 50.51352671178176],
                        [6.980425116279783, 50.50646969878611],
                        [6.9958785266122, 50.51139813782763],
                        [6.996817324763098, 50.52179311595503]
                    ]
                ],
                "type": "Polygon"
            }
        },
        "areaDesc": {
            "type": "Property",
            "value": "TEMA Pilot exercise"
        },
        "bm_id": {
            "type": "Property",
            "value": "1dd6022df507af26f9388f2125564767"
        },
        "category": {
            "type": "Property",
            "value": "Met"
        },
        "certainty": {
            "type": "Property",
            "value": "Observed"
        },
        "effective": {
            "type": "Property",
            "value": "2025-02-12T14:40:35.727Z"
        },
        "event": {
            "type": "Property",
            "value": "Flood"
        },
        "expires": {
            "type": "Property",
            "value": "2025-12-31T10:57:55.325Z"
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
            "value": "2025-02-12T14:40:35.727Z"
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
        }
    }

    # Sending request
    response_ = requests.post(url, json=data, headers=headers)

    # Handling response
    if response_.status_code == 201:
        print("Entity created successfully.")
    elif response_.status_code == 409:
        print("Entity already exists.")
    else:
        print(f"Failed to create entity: {response_.status_code} - {response_.text}")

    return response_

# def create_entity():
#     url = 'https://orion.tema.digital-enabler.eng.it/ngsi-ld/v1/entities'
#     headers = {
#         'Content-Type': 'application/ld+json',
#         'Accept': 'application/ld+json'
#     }
#
#     data = {
#         # "@context": "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
#         "@context": [
#             "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
#         ],
#         "id": "urn:ngsi-ld:Alert:2025-02-12T15:40:35.727-e38f2abb-12fa-4d83-9c6d-7c7405fb2717",
#         "type": "Alert",
#         "area": {
#             "type": "GeoProperty",
#             "value": {
#                 "coordinates":
#                     [
#                         [
#                             [6.996817324763098, 50.52179311595503],
#                             [6.982379621684984, 50.51775574602705],
#                             [6.9760852934997, 50.51352671178176],
#                             [6.980425116279783, 50.50646969878611],
#                             [6.9958785266122, 50.51139813782763],
#                             [6.996817324763098, 50.52179311595503]
#                         ]
#                     ],
#
#                 "type": "Polygon"
#             }
#         },
#         "areaDesc": {
#             "type": "Property",
#             "value": "TEMA Pilot exercise"
#         },
#         "bm_id": {
#             "type": "Property",
#             "value": "1dd6022df507af26f9388f2125564767"
#         },
#         "category": {
#             "type": "Property",
#             "value": "Met"
#         },
#         "certainty": {
#             "type": "Property",
#             "value": "Observed"
#         },
#         "effective": {
#             "type": "Property",
#             "value": "2025-02-12T14:40:35.727Z"
#         },
#         "event": {
#             "type": "Property",
#             "value": "Flood"
#         },
#         "expires": {
#             "type": "Property",
#             "value": "2025-12-31T10:57:55.325Z"
#         },
#         "location": {
#             "type": "GeoProperty",
#             "value": {
#                                 "coordinates":
#                     [
#                         [
#                             [6.996817324763098, 50.52179311595503],
#                             [6.982379621684984, 50.51775574602705],
#                             [6.9760852934997, 50.51352671178176],
#                             [6.980425116279783, 50.50646969878611],
#                             [6.9958785266122, 50.51139813782763],
#                             [6.996817324763098, 50.52179311595503]
#                         ]
#                     ],
#
#                 "type": "Polygon"
#             }
#         },
#         "msgType": {
#             "type": "Property",
#             "value": "Alert"
#         },
#         "scope": {
#             "type": "Property",
#             "value": "Private"
#         },
#         "sender": {
#             "type": "Property",
#             "value": "alert@tema-project.eu"
#         },
#         "sent": {
#             "type": "Property",
#             "value": "2025-02-12T14:40:35.727Z"
#         },
#         "severity": {
#             "type": "Property",
#             "value": "Severe"
#         },
#         "status": {
#             "type": "Property",
#             "value": "Exercise"
#         },
#         "urgency": {
#             "type": "Property",
#             "value": "Immediate"
#         }
#     }
#
#     response_ = requests.post(url, json=data, headers=headers)
#
#     if response_.status_code == 201:
#         print("Entity created successfully.")
#         return response_
#     elif response_.status_code == 409:
#         print("Entity already exists.")
#         return {"status": "Entity already exists"}, 409
#     else:
#         print(f"Failed to create entity: {response_.status_code} - {response_.text}")
#         return {"status": "Error"}, 500


# def update_entity(entity_id_):
#     url_ = f'https://orion.tema.digital-enabler.eng.it/ngsi-ld/v1/entities/{entity_id_}/attrs'
#     headers = {
#         'Content-Type': 'application/ld+json',
#         'Accept': 'application/ld+json'
#     }
#
#     payload = {
#         "area": {
#             "type": "GeoProperty",
#             "value": {
#                                 "coordinates":
#                     [
#                         [
#                             [6.996817324763098, 50.52179311595503],
#                             [6.982379621684984, 50.51775574602705],
#                             [6.9760852934997, 50.51352671178176],
#                             [6.980425116279783, 50.50646969878611],
#                             [6.9958785266122, 50.51139813782763],
#                             [6.996817324763098, 50.52179311595503]
#                         ]
#                     ],
#
#                 "type": "Polygon"
#             }
#         },
#         "expires": {
#             "type": "Property",
#             "value": "2025-09-12T14:40:35.727Z"
#         },
#         "location": {
#             "type": "GeoProperty",
#             "value": {
#                                 "coordinates":
#                     [
#                         [
#                             [6.996817324763098, 50.52179311595503],
#                             [6.982379621684984, 50.51775574602705],
#                             [6.9760852934997, 50.51352671178176],
#                             [6.980425116279783, 50.50646969878611],
#                             [6.9958785266122, 50.51139813782763],
#                             [6.996817324763098, 50.52179311595503],
#                         ]
#                     ],
#
#                 "type": "Polygon"
#             }
#         },
#         "msgType": {
#             "type": "Property",
#             "value": "Alert"
#         },
#         "scope": {
#             "type": "Property",
#             "value": "Private"
#         },
#         "sender": {
#             "type": "Property",
#             "value": "alert@tema-project.eu"
#         },
#         "sent": {
#             "type": "Property",
#             "value": datetime.now(timezone.utc).isoformat()
#         },
#         "severity": {
#             "type": "Property",
#             "value": "Severe"
#         },
#         "status": {
#             "type": "Property",
#             "value": "Exercise"
#         },
#         "urgency": {
#             "type": "Property",
#             "value": "Immediate"
#         }
#     }
#
#     payload_with_context = {
#         "@context": "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
#         **payload
#     }
#
#     try:
#         response = requests.post(url_, json=payload_with_context, headers=headers, timeout=60)
#         response.raise_for_status()
#         print(f"Entity {entity_id_} updated successfully!")
#         return response
#
#     except requests.exceptions.HTTPError as e:
#         print(f"HTTP error occurred while updating entity {entity_id_}: {e}. Response text: {response.text}")
#     except requests.exceptions.RequestException as e:
#         print(f"Error occurred while updating entity {entity_id_}: {e}")
#
#     return None
import requests
from datetime import datetime, timezone


def update_entity(entity_id_):
    url_ = f'https://orion.tema.digital-enabler.eng.it/ngsi-ld/v1/entities/{entity_id_}/attrs'
    headers = {
        'Content-Type': 'application/ld+json',
        'Accept': 'application/ld+json'
    }

    payload = {
        "area": {
            "type": "GeoProperty",
            "value": {
                "coordinates": [
                    [
                        [6.996817324763098, 50.52179311595503],
                        [6.982379621684984, 50.51775574602705],
                        [6.9760852934997, 50.51352671178176],
                        [6.980425116279783, 50.50646969878611],
                        [6.9958785266122, 50.51139813782763],
                        [6.996817324763098, 50.52179311595503]
                    ]
                ],
                "type": "Polygon"
            }
        },
        "location": {
            "type": "GeoProperty",
            "value": {
                "coordinates": [
                    [
                        [6.996817324763098, 50.52179311595503],
                        [6.982379621684984, 50.51775574602705],
                        [6.9760852934997, 50.51352671178176],
                        [6.980425116279783, 50.50646969878611],
                        [6.9958785266122, 50.51139813782763],
                        [6.996817324763098, 50.52179311595503]
                    ]
                ],
                "type": "Polygon"
            }
        },
        "expires": {
            "type": "Property",
            "value": "2025-09-12T14:40:35.727Z"
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
            "value": datetime.now(timezone.utc).isoformat()
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
        }
    }

    payload_with_context = {
        "@context": ["https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"],
        # Ensuring correct NGSI-LD context format
        **payload
    }

    try:
        response = requests.post(url_, json=payload_with_context, headers=headers, timeout=60)
        response.raise_for_status()
        print(f"✅ Entity {entity_id_} updated successfully!")
        return response.json()

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error while updating entity {entity_id_}: {e}. Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error while updating entity {entity_id_}: {e}")

    return None


if __name__ == "__main__":
    # create_entity()
    update_entity("urn:ngsi-ld:Alert:c21f1c93-9392-452f-9ca6-42f2b78e2b87")

