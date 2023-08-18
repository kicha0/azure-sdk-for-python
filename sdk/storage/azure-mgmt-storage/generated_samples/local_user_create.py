# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from azure.identity import DefaultAzureCredential
from azure.mgmt.storage import StorageManagementClient

"""
# PREREQUISITES
    pip install azure-identity
    pip install azure-mgmt-storage
# USAGE
    python local_user_create.py

    Before run the sample, please set the values of the client ID, tenant ID and client secret
    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,
    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:
    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal
"""


def main():
    client = StorageManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id="{subscription-id}",
    )

    response = client.local_users.create_or_update(
        resource_group_name="res6977",
        account_name="sto2527",
        username="user1",
        properties={
            "properties": {
                "hasSshPassword": True,
                "homeDirectory": "homedirectory",
                "permissionScopes": [
                    {"permissions": "rwd", "resourceName": "share1", "service": "file"},
                    {"permissions": "rw", "resourceName": "share2", "service": "file"},
                ],
                "sshAuthorizedKeys": [{"description": "key name", "key": "ssh-rsa keykeykeykeykey="}],
            }
        },
    )
    print(response)


# x-ms-original-file: specification/storage/resource-manager/Microsoft.Storage/stable/2023-01-01/examples/LocalUserCreate.json
if __name__ == "__main__":
    main()
