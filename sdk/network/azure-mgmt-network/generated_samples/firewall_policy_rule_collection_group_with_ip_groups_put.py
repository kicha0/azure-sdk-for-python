# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from azure.identity import DefaultAzureCredential
from azure.mgmt.network import NetworkManagementClient

"""
# PREREQUISITES
    pip install azure-identity
    pip install azure-mgmt-network
# USAGE
    python firewall_policy_rule_collection_group_with_ip_groups_put.py

    Before run the sample, please set the values of the client ID, tenant ID and client secret
    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,
    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:
    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal
"""


def main():
    client = NetworkManagementClient(
        credential=DefaultAzureCredential(),
        subscription_id="subid",
    )

    response = client.firewall_policy_rule_collection_groups.begin_create_or_update(
        resource_group_name="rg1",
        firewall_policy_name="firewallPolicy",
        rule_collection_group_name="ruleCollectionGroup1",
        parameters={
            "properties": {
                "priority": 110,
                "ruleCollections": [
                    {
                        "action": {"type": "Deny"},
                        "name": "Example-Filter-Rule-Collection",
                        "ruleCollectionType": "FirewallPolicyFilterRuleCollection",
                        "rules": [
                            {
                                "destinationIpGroups": [
                                    "/subscriptions/subid/providers/Microsoft.Network/resourceGroup/rg1/ipGroups/ipGroups2"
                                ],
                                "destinationPorts": ["*"],
                                "ipProtocols": ["TCP"],
                                "name": "network-1",
                                "ruleType": "NetworkRule",
                                "sourceIpGroups": [
                                    "/subscriptions/subid/providers/Microsoft.Network/resourceGroup/rg1/ipGroups/ipGroups1"
                                ],
                            }
                        ],
                    }
                ],
            }
        },
    ).result()
    print(response)


# x-ms-original-file: specification/network/resource-manager/Microsoft.Network/stable/2022-11-01/examples/FirewallPolicyRuleCollectionGroupWithIpGroupsPut.json
if __name__ == "__main__":
    main()
