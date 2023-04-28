# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from azure.identity import DefaultAzureCredential
from azure.mgmt.qumulo import QumuloMgmtClient

"""
# PREREQUISITES
    pip install azure-identity
    pip install azure-mgmt-qumulo
# USAGE
    python file_systems_get_minimum_set_gen.py

    Before run the sample, please set the values of the client ID, tenant ID and client secret
    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,
    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:
    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal
"""


def main():
    client = QumuloMgmtClient(
        credential=DefaultAzureCredential(),
        subscription_id="aaaaaaa",
    )

    response = client.file_systems.get(
        resource_group_name="rgQumulo",
        file_system_name="aaaaaaaaaaaaaaaaa",
    )
    print(response)


# x-ms-original-file: specification/liftrqumulo/resource-manager/Qumulo.Storage/preview/2022-10-12-preview/examples/FileSystems_Get_MinimumSet_Gen.json
if __name__ == "__main__":
    main()
