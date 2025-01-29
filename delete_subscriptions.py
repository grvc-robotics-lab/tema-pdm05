import requests

# Configure the Orion-LD broker URL and headers
ORION_URL = "https://orion.tema.digital-enabler.eng.it/ngsi-ld/v1/subscriptions"
HEADERS = {
    "Content-Type": "application/json"
}


def delete_subscription(subscription_id):
    """
    Delete subscription in Orion-LD by ID.

    Args:
        subscription_id (str): ID of the subscription to be deleted.

    Returns:
        dict: A dictionary containing the status and message of the operation.
    """
    url = f"{ORION_URL}/{subscription_id}"
    try:
        response = requests.delete(url, headers=HEADERS)

        if response.status_code == 204:
            return {"status": "success", "message": f"Subscription {subscription_id} deleted."}
        else:
            return {
                "status": "error",
                "message": f"Failed to delete subscription {subscription_id}",
                "details": f"HTTP {response.status_code}: {response.text}"
            }
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "message": f"Request failed for subscription {subscription_id}",
            "details": str(e)
        }


def delete_subscriptions(subscription_ids):
    """
    Delete a group of subscriptions from Orion-LD using delete_subscription.

    Args:
        subscription_ids (list): List of IDs to be deleted.
    """
    for sub_id in subscription_ids:
        result = delete_subscription(sub_id)
        print(f"{result['status'].upper()}: {result['message']}")
        if "details" in result:
            print(f"Details: {result['details']}")


# IDs to be deleted
subscriptions_to_delete = [
    ################################################
    # PDM05
    # ################################################
    "urn:ngsi-ld:tema:subscription:USE:PDM05:002",
    "urn:ngsi-ld:tema:subscription:USE:PDM05:003",
    "urn:ngsi-ld:tema:subscription:USE:PDM05:004",
    "urn:ngsi-ld:tema:subscription:USE:PDM05:005",
    "urn:ngsi-ld:tema:subscription:USE:PDM05:006",
    "urn:ngsi-ld:tema:subscription:USE:PDM05:007",
    "urn:ngsi-ld:tema:subscription:USE:PDM05:008",
    "urn:ngsi-ld:tema:subscription:USE:PDM05:009",
    "urn:ngsi-ld:tema:subscription:USE:PDM05:010",
    "urn:ngsi-ld:tema:subscription:USE:PDM05:011",
    ###############################################
    # SV01
    ################################################
    # "urn:ngsi-ld:tema:subscription:USE:SV01:001",
    # "urn:ngsi-ld:tema:subscription:USE:SV01:002"
]

# Delete Subscriptions
delete_subscriptions(subscriptions_to_delete)
