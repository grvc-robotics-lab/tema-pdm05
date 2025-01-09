import requests

# Configuration
BROKER_URL = "https://orion.tema.digital-enabler.eng.it/ngsi-ld/v1/subscriptions"
BROKER_ENTITY_Maps4Fire_ID = "urn:ngsi-ld:USE:PDM-05:Maps4Fire:01"
CALLBACK_URL = "https://779e-193-147-160-155.ngrok-free.app/pdm05/notify"


def delete_maps4fire_subscriptions():
    try:
        # Fetch all subscriptions
        response = requests.get(BROKER_URL, headers={"Accept": "application/ld+json"})

        if response.status_code != 200:
            print(f"Failed to retrieve subscriptions: {response.status_code} - {response.text}")
            return

        subscriptions = response.json()

        # Filter subscriptions for Maps4Fire entity type with the specific callback URL
        maps4fire_subscriptions = [
            sub for sub in subscriptions
            if sub["notification"]["endpoint"]["uri"] == CALLBACK_URL
        ]

        if not maps4fire_subscriptions:
            print("No matching subscriptions found.")
            return

        # Delete matching subscriptions
        for sub in maps4fire_subscriptions:
            sub_id = sub["id"]
            delete_response = requests.delete(f"{BROKER_URL}/{sub_id}")

            if delete_response.status_code == 204:
                print(f"Successfully deleted subscription: {sub_id}")
            else:
                print(f"Failed to delete subscription {sub_id}: {delete_response.status_code} - {delete_response.text}")

    except Exception as e:
        print(f"An error occurred: {e}")


def delete_entity(entity_id):
    entity_to_be_deleted = f'https://orion.tema.digital-enabler.eng.it/ngsi-ld/v1/entities/{entity_id}'
    print(f"entity_to_be_deleted {entity_to_be_deleted}")
    response = requests.delete(entity_to_be_deleted)
    # Check the response status
    if response.status_code == 204:
        print('Entity deleted successfully.')
    else:
        print(f'Failed to delete entity. Status code: {response.status_code}, Response: {response.text}')


# Run the function
# delete_maps4fire_subscriptions()
# delete_entity("urn:ngsi-ld:USE:SV-01:DroneImages:01")
