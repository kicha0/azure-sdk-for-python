# coding=utf-8
# pylint: disable=too-many-lines
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from typing import List, Optional, TYPE_CHECKING

from ... import _serialization

if TYPE_CHECKING:
    # pylint: disable=unused-import,ungrouped-imports
    from .. import models as _models


class Operation(_serialization.Model):
    """Microsoft.Resources operation.

    :ivar name: Operation name: {provider}/{resource}/{operation}.
    :vartype name: str
    :ivar display: The object that represents the operation.
    :vartype display: ~azure.mgmt.resource.links.v2016_09_01.models.OperationDisplay
    """

    _attribute_map = {
        "name": {"key": "name", "type": "str"},
        "display": {"key": "display", "type": "OperationDisplay"},
    }

    def __init__(self, *, name: Optional[str] = None, display: Optional["_models.OperationDisplay"] = None, **kwargs):
        """
        :keyword name: Operation name: {provider}/{resource}/{operation}.
        :paramtype name: str
        :keyword display: The object that represents the operation.
        :paramtype display: ~azure.mgmt.resource.links.v2016_09_01.models.OperationDisplay
        """
        super().__init__(**kwargs)
        self.name = name
        self.display = display


class OperationDisplay(_serialization.Model):
    """The object that represents the operation.

    :ivar provider: Service provider: Microsoft.Resources.
    :vartype provider: str
    :ivar resource: Resource on which the operation is performed: Profile, endpoint, etc.
    :vartype resource: str
    :ivar operation: Operation type: Read, write, delete, etc.
    :vartype operation: str
    :ivar description: Description of the operation.
    :vartype description: str
    """

    _attribute_map = {
        "provider": {"key": "provider", "type": "str"},
        "resource": {"key": "resource", "type": "str"},
        "operation": {"key": "operation", "type": "str"},
        "description": {"key": "description", "type": "str"},
    }

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        resource: Optional[str] = None,
        operation: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs
    ):
        """
        :keyword provider: Service provider: Microsoft.Resources.
        :paramtype provider: str
        :keyword resource: Resource on which the operation is performed: Profile, endpoint, etc.
        :paramtype resource: str
        :keyword operation: Operation type: Read, write, delete, etc.
        :paramtype operation: str
        :keyword description: Description of the operation.
        :paramtype description: str
        """
        super().__init__(**kwargs)
        self.provider = provider
        self.resource = resource
        self.operation = operation
        self.description = description


class OperationListResult(_serialization.Model):
    """Result of the request to list Microsoft.Resources operations. It contains a list of operations and a URL link to get the next set of results.

    :ivar value: List of Microsoft.Resources operations.
    :vartype value: list[~azure.mgmt.resource.links.v2016_09_01.models.Operation]
    :ivar next_link: URL to get the next set of operation list results if there are any.
    :vartype next_link: str
    """

    _attribute_map = {
        "value": {"key": "value", "type": "[Operation]"},
        "next_link": {"key": "nextLink", "type": "str"},
    }

    def __init__(self, *, value: Optional[List["_models.Operation"]] = None, next_link: Optional[str] = None, **kwargs):
        """
        :keyword value: List of Microsoft.Resources operations.
        :paramtype value: list[~azure.mgmt.resource.links.v2016_09_01.models.Operation]
        :keyword next_link: URL to get the next set of operation list results if there are any.
        :paramtype next_link: str
        """
        super().__init__(**kwargs)
        self.value = value
        self.next_link = next_link


class ResourceLink(_serialization.Model):
    """The resource link.

    Variables are only populated by the server, and will be ignored when sending a request.

    :ivar id: The fully qualified ID of the resource link.
    :vartype id: str
    :ivar name: The name of the resource link.
    :vartype name: str
    :ivar type: The resource link object.
    :vartype type: JSON
    :ivar properties: Properties for resource link.
    :vartype properties: ~azure.mgmt.resource.links.v2016_09_01.models.ResourceLinkProperties
    """

    _validation = {
        "id": {"readonly": True},
        "name": {"readonly": True},
        "type": {"readonly": True},
    }

    _attribute_map = {
        "id": {"key": "id", "type": "str"},
        "name": {"key": "name", "type": "str"},
        "type": {"key": "type", "type": "object"},
        "properties": {"key": "properties", "type": "ResourceLinkProperties"},
    }

    def __init__(self, *, properties: Optional["_models.ResourceLinkProperties"] = None, **kwargs):
        """
        :keyword properties: Properties for resource link.
        :paramtype properties: ~azure.mgmt.resource.links.v2016_09_01.models.ResourceLinkProperties
        """
        super().__init__(**kwargs)
        self.id = None
        self.name = None
        self.type = None
        self.properties = properties


class ResourceLinkFilter(_serialization.Model):
    """Resource link filter.

    All required parameters must be populated in order to send to Azure.

    :ivar target_id: The ID of the target resource. Required.
    :vartype target_id: str
    """

    _validation = {
        "target_id": {"required": True},
    }

    _attribute_map = {
        "target_id": {"key": "targetId", "type": "str"},
    }

    def __init__(self, *, target_id: str, **kwargs):
        """
        :keyword target_id: The ID of the target resource. Required.
        :paramtype target_id: str
        """
        super().__init__(**kwargs)
        self.target_id = target_id


class ResourceLinkProperties(_serialization.Model):
    """The resource link properties.

    Variables are only populated by the server, and will be ignored when sending a request.

    All required parameters must be populated in order to send to Azure.

    :ivar source_id: The fully qualified ID of the source resource in the link.
    :vartype source_id: str
    :ivar target_id: The fully qualified ID of the target resource in the link. Required.
    :vartype target_id: str
    :ivar notes: Notes about the resource link.
    :vartype notes: str
    """

    _validation = {
        "source_id": {"readonly": True},
        "target_id": {"required": True},
    }

    _attribute_map = {
        "source_id": {"key": "sourceId", "type": "str"},
        "target_id": {"key": "targetId", "type": "str"},
        "notes": {"key": "notes", "type": "str"},
    }

    def __init__(self, *, target_id: str, notes: Optional[str] = None, **kwargs):
        """
        :keyword target_id: The fully qualified ID of the target resource in the link. Required.
        :paramtype target_id: str
        :keyword notes: Notes about the resource link.
        :paramtype notes: str
        """
        super().__init__(**kwargs)
        self.source_id = None
        self.target_id = target_id
        self.notes = notes


class ResourceLinkResult(_serialization.Model):
    """List of resource links.

    Variables are only populated by the server, and will be ignored when sending a request.

    All required parameters must be populated in order to send to Azure.

    :ivar value: An array of resource links. Required.
    :vartype value: list[~azure.mgmt.resource.links.v2016_09_01.models.ResourceLink]
    :ivar next_link: The URL to use for getting the next set of results.
    :vartype next_link: str
    """

    _validation = {
        "value": {"required": True},
        "next_link": {"readonly": True},
    }

    _attribute_map = {
        "value": {"key": "value", "type": "[ResourceLink]"},
        "next_link": {"key": "nextLink", "type": "str"},
    }

    def __init__(self, *, value: List["_models.ResourceLink"], **kwargs):
        """
        :keyword value: An array of resource links. Required.
        :paramtype value: list[~azure.mgmt.resource.links.v2016_09_01.models.ResourceLink]
        """
        super().__init__(**kwargs)
        self.value = value
        self.next_link = None
