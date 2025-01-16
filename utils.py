import requests
import config


def get_existing_subscriptions(url):
    """
    Fetches existing subscriptions from the NGSI-LD Context Broker.
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to retrieve subscriptions. Status code: {response.status_code}")


def delete_subscription(url, subscription_id):
    """
    Deletes a subscription from the NGSI-LD Context Broker.
    """
    response = requests.delete(f"{url}/{subscription_id}")
    if response.status_code == 204:
        return True
    else:
        raise Exception(f"Failed to delete subscription. Status code: {response.status_code}")


#####################################################
# Delete an entity from the CB
#####################################################
def delete_entity(broker_url_, entity_id):
    entity_to_be_deleted = f'{broker_url_}/ngsi-ld/v1/entities/{entity_id}'
    print(f"entity_to_be_deleted {entity_to_be_deleted}")
    response = requests.delete(entity_to_be_deleted)
    # Check the response status
    if response.status_code == 204:
        print('Entity deleted successfully.')
    else:
        print(f'Failed to delete entity. Status code: {response.status_code}, Response: {response.text}')


delete_entity(config.BROKER_URL, "urn:ngsi-ld:USE:PDM-05:Maps4Fire:01")
delete_entity(config.BROKER_URL, "urn:ngsi-ld:USE:PDM-05:Maps4Flood:01")
delete_entity(config.BROKER_URL, "urn:ngsi-ld:USE:PDM-05:Maps4Object:01")


