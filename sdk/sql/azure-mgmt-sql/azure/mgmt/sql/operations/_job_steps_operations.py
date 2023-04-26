# pylint: disable=too-many-lines
# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
import sys
from typing import Any, Callable, Dict, IO, Iterable, Optional, TypeVar, Union, overload

from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
    ResourceNotModifiedError,
    map_error,
)
from azure.core.paging import ItemPaged
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpResponse
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.core.utils import case_insensitive_dict
from azure.mgmt.core.exceptions import ARMErrorFormat

from .. import models as _models
from .._serialization import Serializer
from .._vendor import _convert_request, _format_url_section

if sys.version_info >= (3, 8):
    from typing import Literal  # pylint: disable=no-name-in-module, ungrouped-imports
else:
    from typing_extensions import Literal  # type: ignore  # pylint: disable=ungrouped-imports
T = TypeVar("T")
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]

_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False


def build_list_by_version_request(
    resource_group_name: str,
    server_name: str,
    job_agent_name: str,
    job_name: str,
    job_version: int,
    subscription_id: str,
    **kwargs: Any
) -> HttpRequest:
    _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
    _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

    api_version: Literal["2020-11-01-preview"] = kwargs.pop(
        "api_version", _params.pop("api-version", "2020-11-01-preview")
    )
    accept = _headers.pop("Accept", "application/json")

    # Construct URL
    _url = kwargs.pop(
        "template_url",
        "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Sql/servers/{serverName}/jobAgents/{jobAgentName}/jobs/{jobName}/versions/{jobVersion}/steps",
    )  # pylint: disable=line-too-long
    path_format_arguments = {
        "resourceGroupName": _SERIALIZER.url("resource_group_name", resource_group_name, "str"),
        "serverName": _SERIALIZER.url("server_name", server_name, "str"),
        "jobAgentName": _SERIALIZER.url("job_agent_name", job_agent_name, "str"),
        "jobName": _SERIALIZER.url("job_name", job_name, "str"),
        "jobVersion": _SERIALIZER.url("job_version", job_version, "int"),
        "subscriptionId": _SERIALIZER.url("subscription_id", subscription_id, "str"),
    }

    _url: str = _format_url_section(_url, **path_format_arguments)  # type: ignore

    # Construct parameters
    _params["api-version"] = _SERIALIZER.query("api_version", api_version, "str")

    # Construct headers
    _headers["Accept"] = _SERIALIZER.header("accept", accept, "str")

    return HttpRequest(method="GET", url=_url, params=_params, headers=_headers, **kwargs)


def build_get_by_version_request(
    resource_group_name: str,
    server_name: str,
    job_agent_name: str,
    job_name: str,
    job_version: int,
    step_name: str,
    subscription_id: str,
    **kwargs: Any
) -> HttpRequest:
    _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
    _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

    api_version: Literal["2020-11-01-preview"] = kwargs.pop(
        "api_version", _params.pop("api-version", "2020-11-01-preview")
    )
    accept = _headers.pop("Accept", "application/json")

    # Construct URL
    _url = kwargs.pop(
        "template_url",
        "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Sql/servers/{serverName}/jobAgents/{jobAgentName}/jobs/{jobName}/versions/{jobVersion}/steps/{stepName}",
    )  # pylint: disable=line-too-long
    path_format_arguments = {
        "resourceGroupName": _SERIALIZER.url("resource_group_name", resource_group_name, "str"),
        "serverName": _SERIALIZER.url("server_name", server_name, "str"),
        "jobAgentName": _SERIALIZER.url("job_agent_name", job_agent_name, "str"),
        "jobName": _SERIALIZER.url("job_name", job_name, "str"),
        "jobVersion": _SERIALIZER.url("job_version", job_version, "int"),
        "stepName": _SERIALIZER.url("step_name", step_name, "str"),
        "subscriptionId": _SERIALIZER.url("subscription_id", subscription_id, "str"),
    }

    _url: str = _format_url_section(_url, **path_format_arguments)  # type: ignore

    # Construct parameters
    _params["api-version"] = _SERIALIZER.query("api_version", api_version, "str")

    # Construct headers
    _headers["Accept"] = _SERIALIZER.header("accept", accept, "str")

    return HttpRequest(method="GET", url=_url, params=_params, headers=_headers, **kwargs)


def build_list_by_job_request(
    resource_group_name: str, server_name: str, job_agent_name: str, job_name: str, subscription_id: str, **kwargs: Any
) -> HttpRequest:
    _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
    _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

    api_version: Literal["2020-11-01-preview"] = kwargs.pop(
        "api_version", _params.pop("api-version", "2020-11-01-preview")
    )
    accept = _headers.pop("Accept", "application/json")

    # Construct URL
    _url = kwargs.pop(
        "template_url",
        "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Sql/servers/{serverName}/jobAgents/{jobAgentName}/jobs/{jobName}/steps",
    )  # pylint: disable=line-too-long
    path_format_arguments = {
        "resourceGroupName": _SERIALIZER.url("resource_group_name", resource_group_name, "str"),
        "serverName": _SERIALIZER.url("server_name", server_name, "str"),
        "jobAgentName": _SERIALIZER.url("job_agent_name", job_agent_name, "str"),
        "jobName": _SERIALIZER.url("job_name", job_name, "str"),
        "subscriptionId": _SERIALIZER.url("subscription_id", subscription_id, "str"),
    }

    _url: str = _format_url_section(_url, **path_format_arguments)  # type: ignore

    # Construct parameters
    _params["api-version"] = _SERIALIZER.query("api_version", api_version, "str")

    # Construct headers
    _headers["Accept"] = _SERIALIZER.header("accept", accept, "str")

    return HttpRequest(method="GET", url=_url, params=_params, headers=_headers, **kwargs)


def build_get_request(
    resource_group_name: str,
    server_name: str,
    job_agent_name: str,
    job_name: str,
    step_name: str,
    subscription_id: str,
    **kwargs: Any
) -> HttpRequest:
    _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
    _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

    api_version: Literal["2020-11-01-preview"] = kwargs.pop(
        "api_version", _params.pop("api-version", "2020-11-01-preview")
    )
    accept = _headers.pop("Accept", "application/json")

    # Construct URL
    _url = kwargs.pop(
        "template_url",
        "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Sql/servers/{serverName}/jobAgents/{jobAgentName}/jobs/{jobName}/steps/{stepName}",
    )  # pylint: disable=line-too-long
    path_format_arguments = {
        "resourceGroupName": _SERIALIZER.url("resource_group_name", resource_group_name, "str"),
        "serverName": _SERIALIZER.url("server_name", server_name, "str"),
        "jobAgentName": _SERIALIZER.url("job_agent_name", job_agent_name, "str"),
        "jobName": _SERIALIZER.url("job_name", job_name, "str"),
        "stepName": _SERIALIZER.url("step_name", step_name, "str"),
        "subscriptionId": _SERIALIZER.url("subscription_id", subscription_id, "str"),
    }

    _url: str = _format_url_section(_url, **path_format_arguments)  # type: ignore

    # Construct parameters
    _params["api-version"] = _SERIALIZER.query("api_version", api_version, "str")

    # Construct headers
    _headers["Accept"] = _SERIALIZER.header("accept", accept, "str")

    return HttpRequest(method="GET", url=_url, params=_params, headers=_headers, **kwargs)


def build_create_or_update_request(
    resource_group_name: str,
    server_name: str,
    job_agent_name: str,
    job_name: str,
    step_name: str,
    subscription_id: str,
    **kwargs: Any
) -> HttpRequest:
    _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
    _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

    api_version: Literal["2020-11-01-preview"] = kwargs.pop(
        "api_version", _params.pop("api-version", "2020-11-01-preview")
    )
    content_type: Optional[str] = kwargs.pop("content_type", _headers.pop("Content-Type", None))
    accept = _headers.pop("Accept", "application/json")

    # Construct URL
    _url = kwargs.pop(
        "template_url",
        "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Sql/servers/{serverName}/jobAgents/{jobAgentName}/jobs/{jobName}/steps/{stepName}",
    )  # pylint: disable=line-too-long
    path_format_arguments = {
        "resourceGroupName": _SERIALIZER.url("resource_group_name", resource_group_name, "str"),
        "serverName": _SERIALIZER.url("server_name", server_name, "str"),
        "jobAgentName": _SERIALIZER.url("job_agent_name", job_agent_name, "str"),
        "jobName": _SERIALIZER.url("job_name", job_name, "str"),
        "stepName": _SERIALIZER.url("step_name", step_name, "str"),
        "subscriptionId": _SERIALIZER.url("subscription_id", subscription_id, "str"),
    }

    _url: str = _format_url_section(_url, **path_format_arguments)  # type: ignore

    # Construct parameters
    _params["api-version"] = _SERIALIZER.query("api_version", api_version, "str")

    # Construct headers
    if content_type is not None:
        _headers["Content-Type"] = _SERIALIZER.header("content_type", content_type, "str")
    _headers["Accept"] = _SERIALIZER.header("accept", accept, "str")

    return HttpRequest(method="PUT", url=_url, params=_params, headers=_headers, **kwargs)


def build_delete_request(
    resource_group_name: str,
    server_name: str,
    job_agent_name: str,
    job_name: str,
    step_name: str,
    subscription_id: str,
    **kwargs: Any
) -> HttpRequest:
    _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

    api_version: Literal["2020-11-01-preview"] = kwargs.pop(
        "api_version", _params.pop("api-version", "2020-11-01-preview")
    )
    # Construct URL
    _url = kwargs.pop(
        "template_url",
        "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Sql/servers/{serverName}/jobAgents/{jobAgentName}/jobs/{jobName}/steps/{stepName}",
    )  # pylint: disable=line-too-long
    path_format_arguments = {
        "resourceGroupName": _SERIALIZER.url("resource_group_name", resource_group_name, "str"),
        "serverName": _SERIALIZER.url("server_name", server_name, "str"),
        "jobAgentName": _SERIALIZER.url("job_agent_name", job_agent_name, "str"),
        "jobName": _SERIALIZER.url("job_name", job_name, "str"),
        "stepName": _SERIALIZER.url("step_name", step_name, "str"),
        "subscriptionId": _SERIALIZER.url("subscription_id", subscription_id, "str"),
    }

    _url: str = _format_url_section(_url, **path_format_arguments)  # type: ignore

    # Construct parameters
    _params["api-version"] = _SERIALIZER.query("api_version", api_version, "str")

    return HttpRequest(method="DELETE", url=_url, params=_params, **kwargs)


class JobStepsOperations:
    """
    .. warning::
        **DO NOT** instantiate this class directly.

        Instead, you should access the following operations through
        :class:`~azure.mgmt.sql.SqlManagementClient`'s
        :attr:`job_steps` attribute.
    """

    models = _models

    def __init__(self, *args, **kwargs):
        input_args = list(args)
        self._client = input_args.pop(0) if input_args else kwargs.pop("client")
        self._config = input_args.pop(0) if input_args else kwargs.pop("config")
        self._serialize = input_args.pop(0) if input_args else kwargs.pop("serializer")
        self._deserialize = input_args.pop(0) if input_args else kwargs.pop("deserializer")

    @distributed_trace
    def list_by_version(
        self,
        resource_group_name: str,
        server_name: str,
        job_agent_name: str,
        job_name: str,
        job_version: int,
        **kwargs: Any
    ) -> Iterable["_models.JobStep"]:
        """Gets all job steps in the specified job version.

        :param resource_group_name: The name of the resource group that contains the resource. You can
         obtain this value from the Azure Resource Manager API or the portal. Required.
        :type resource_group_name: str
        :param server_name: The name of the server. Required.
        :type server_name: str
        :param job_agent_name: The name of the job agent. Required.
        :type job_agent_name: str
        :param job_name: The name of the job to get. Required.
        :type job_name: str
        :param job_version: The version of the job to get. Required.
        :type job_version: int
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: An iterator like instance of either JobStep or the result of cls(response)
        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.sql.models.JobStep]
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: Literal["2020-11-01-preview"] = kwargs.pop(
            "api_version", _params.pop("api-version", "2020-11-01-preview")
        )
        cls: ClsType[_models.JobStepListResult] = kwargs.pop("cls", None)

        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        def prepare_request(next_link=None):
            if not next_link:

                request = build_list_by_version_request(
                    resource_group_name=resource_group_name,
                    server_name=server_name,
                    job_agent_name=job_agent_name,
                    job_name=job_name,
                    job_version=job_version,
                    subscription_id=self._config.subscription_id,
                    api_version=api_version,
                    template_url=self.list_by_version.metadata["url"],
                    headers=_headers,
                    params=_params,
                )
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)

            else:
                request = HttpRequest("GET", next_link)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = "GET"
            return request

        def extract_data(pipeline_response):
            deserialized = self._deserialize("JobStepListResult", pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)  # type: ignore
            return deserialized.next_link or None, iter(list_of_elem)

        def get_next(next_link=None):
            request = prepare_request(next_link)

            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(  # pylint: disable=protected-access
                request, stream=_stream, **kwargs
            )
            response = pipeline_response.http_response

            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)

            return pipeline_response

        return ItemPaged(get_next, extract_data)

    list_by_version.metadata = {
        "url": "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Sql/servers/{serverName}/jobAgents/{jobAgentName}/jobs/{jobName}/versions/{jobVersion}/steps"
    }

    @distributed_trace
    def get_by_version(
        self,
        resource_group_name: str,
        server_name: str,
        job_agent_name: str,
        job_name: str,
        job_version: int,
        step_name: str,
        **kwargs: Any
    ) -> _models.JobStep:
        """Gets the specified version of a job step.

        :param resource_group_name: The name of the resource group that contains the resource. You can
         obtain this value from the Azure Resource Manager API or the portal. Required.
        :type resource_group_name: str
        :param server_name: The name of the server. Required.
        :type server_name: str
        :param job_agent_name: The name of the job agent. Required.
        :type job_agent_name: str
        :param job_name: The name of the job. Required.
        :type job_name: str
        :param job_version: The version of the job to get. Required.
        :type job_version: int
        :param step_name: The name of the job step. Required.
        :type step_name: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: JobStep or the result of cls(response)
        :rtype: ~azure.mgmt.sql.models.JobStep
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: Literal["2020-11-01-preview"] = kwargs.pop(
            "api_version", _params.pop("api-version", "2020-11-01-preview")
        )
        cls: ClsType[_models.JobStep] = kwargs.pop("cls", None)

        request = build_get_by_version_request(
            resource_group_name=resource_group_name,
            server_name=server_name,
            job_agent_name=job_agent_name,
            job_name=job_name,
            job_version=job_version,
            step_name=step_name,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            template_url=self.get_by_version.metadata["url"],
            headers=_headers,
            params=_params,
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)

        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(  # pylint: disable=protected-access
            request, stream=_stream, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        deserialized = self._deserialize("JobStep", pipeline_response)

        if cls:
            return cls(pipeline_response, deserialized, {})

        return deserialized

    get_by_version.metadata = {
        "url": "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Sql/servers/{serverName}/jobAgents/{jobAgentName}/jobs/{jobName}/versions/{jobVersion}/steps/{stepName}"
    }

    @distributed_trace
    def list_by_job(
        self, resource_group_name: str, server_name: str, job_agent_name: str, job_name: str, **kwargs: Any
    ) -> Iterable["_models.JobStep"]:
        """Gets all job steps for a job's current version.

        :param resource_group_name: The name of the resource group that contains the resource. You can
         obtain this value from the Azure Resource Manager API or the portal. Required.
        :type resource_group_name: str
        :param server_name: The name of the server. Required.
        :type server_name: str
        :param job_agent_name: The name of the job agent. Required.
        :type job_agent_name: str
        :param job_name: The name of the job to get. Required.
        :type job_name: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: An iterator like instance of either JobStep or the result of cls(response)
        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.sql.models.JobStep]
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: Literal["2020-11-01-preview"] = kwargs.pop(
            "api_version", _params.pop("api-version", "2020-11-01-preview")
        )
        cls: ClsType[_models.JobStepListResult] = kwargs.pop("cls", None)

        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        def prepare_request(next_link=None):
            if not next_link:

                request = build_list_by_job_request(
                    resource_group_name=resource_group_name,
                    server_name=server_name,
                    job_agent_name=job_agent_name,
                    job_name=job_name,
                    subscription_id=self._config.subscription_id,
                    api_version=api_version,
                    template_url=self.list_by_job.metadata["url"],
                    headers=_headers,
                    params=_params,
                )
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)

            else:
                request = HttpRequest("GET", next_link)
                request = _convert_request(request)
                request.url = self._client.format_url(request.url)
                request.method = "GET"
            return request

        def extract_data(pipeline_response):
            deserialized = self._deserialize("JobStepListResult", pipeline_response)
            list_of_elem = deserialized.value
            if cls:
                list_of_elem = cls(list_of_elem)  # type: ignore
            return deserialized.next_link or None, iter(list_of_elem)

        def get_next(next_link=None):
            request = prepare_request(next_link)

            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(  # pylint: disable=protected-access
                request, stream=_stream, **kwargs
            )
            response = pipeline_response.http_response

            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                raise HttpResponseError(response=response, error_format=ARMErrorFormat)

            return pipeline_response

        return ItemPaged(get_next, extract_data)

    list_by_job.metadata = {
        "url": "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Sql/servers/{serverName}/jobAgents/{jobAgentName}/jobs/{jobName}/steps"
    }

    @distributed_trace
    def get(
        self,
        resource_group_name: str,
        server_name: str,
        job_agent_name: str,
        job_name: str,
        step_name: str,
        **kwargs: Any
    ) -> _models.JobStep:
        """Gets a job step in a job's current version.

        :param resource_group_name: The name of the resource group that contains the resource. You can
         obtain this value from the Azure Resource Manager API or the portal. Required.
        :type resource_group_name: str
        :param server_name: The name of the server. Required.
        :type server_name: str
        :param job_agent_name: The name of the job agent. Required.
        :type job_agent_name: str
        :param job_name: The name of the job. Required.
        :type job_name: str
        :param step_name: The name of the job step. Required.
        :type step_name: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: JobStep or the result of cls(response)
        :rtype: ~azure.mgmt.sql.models.JobStep
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: Literal["2020-11-01-preview"] = kwargs.pop(
            "api_version", _params.pop("api-version", "2020-11-01-preview")
        )
        cls: ClsType[_models.JobStep] = kwargs.pop("cls", None)

        request = build_get_request(
            resource_group_name=resource_group_name,
            server_name=server_name,
            job_agent_name=job_agent_name,
            job_name=job_name,
            step_name=step_name,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            template_url=self.get.metadata["url"],
            headers=_headers,
            params=_params,
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)

        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(  # pylint: disable=protected-access
            request, stream=_stream, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        deserialized = self._deserialize("JobStep", pipeline_response)

        if cls:
            return cls(pipeline_response, deserialized, {})

        return deserialized

    get.metadata = {
        "url": "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Sql/servers/{serverName}/jobAgents/{jobAgentName}/jobs/{jobName}/steps/{stepName}"
    }

    @overload
    def create_or_update(
        self,
        resource_group_name: str,
        server_name: str,
        job_agent_name: str,
        job_name: str,
        step_name: str,
        parameters: _models.JobStep,
        *,
        content_type: str = "application/json",
        **kwargs: Any
    ) -> _models.JobStep:
        """Creates or updates a job step. This will implicitly create a new job version.

        :param resource_group_name: The name of the resource group that contains the resource. You can
         obtain this value from the Azure Resource Manager API or the portal. Required.
        :type resource_group_name: str
        :param server_name: The name of the server. Required.
        :type server_name: str
        :param job_agent_name: The name of the job agent. Required.
        :type job_agent_name: str
        :param job_name: The name of the job. Required.
        :type job_name: str
        :param step_name: The name of the job step. Required.
        :type step_name: str
        :param parameters: The requested state of the job step. Required.
        :type parameters: ~azure.mgmt.sql.models.JobStep
        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.
         Default value is "application/json".
        :paramtype content_type: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: JobStep or the result of cls(response)
        :rtype: ~azure.mgmt.sql.models.JobStep
        :raises ~azure.core.exceptions.HttpResponseError:
        """

    @overload
    def create_or_update(
        self,
        resource_group_name: str,
        server_name: str,
        job_agent_name: str,
        job_name: str,
        step_name: str,
        parameters: IO,
        *,
        content_type: str = "application/json",
        **kwargs: Any
    ) -> _models.JobStep:
        """Creates or updates a job step. This will implicitly create a new job version.

        :param resource_group_name: The name of the resource group that contains the resource. You can
         obtain this value from the Azure Resource Manager API or the portal. Required.
        :type resource_group_name: str
        :param server_name: The name of the server. Required.
        :type server_name: str
        :param job_agent_name: The name of the job agent. Required.
        :type job_agent_name: str
        :param job_name: The name of the job. Required.
        :type job_name: str
        :param step_name: The name of the job step. Required.
        :type step_name: str
        :param parameters: The requested state of the job step. Required.
        :type parameters: IO
        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.
         Default value is "application/json".
        :paramtype content_type: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: JobStep or the result of cls(response)
        :rtype: ~azure.mgmt.sql.models.JobStep
        :raises ~azure.core.exceptions.HttpResponseError:
        """

    @distributed_trace
    def create_or_update(
        self,
        resource_group_name: str,
        server_name: str,
        job_agent_name: str,
        job_name: str,
        step_name: str,
        parameters: Union[_models.JobStep, IO],
        **kwargs: Any
    ) -> _models.JobStep:
        """Creates or updates a job step. This will implicitly create a new job version.

        :param resource_group_name: The name of the resource group that contains the resource. You can
         obtain this value from the Azure Resource Manager API or the portal. Required.
        :type resource_group_name: str
        :param server_name: The name of the server. Required.
        :type server_name: str
        :param job_agent_name: The name of the job agent. Required.
        :type job_agent_name: str
        :param job_name: The name of the job. Required.
        :type job_name: str
        :param step_name: The name of the job step. Required.
        :type step_name: str
        :param parameters: The requested state of the job step. Is either a JobStep type or a IO type.
         Required.
        :type parameters: ~azure.mgmt.sql.models.JobStep or IO
        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.
         Default value is None.
        :paramtype content_type: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: JobStep or the result of cls(response)
        :rtype: ~azure.mgmt.sql.models.JobStep
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = case_insensitive_dict(kwargs.pop("headers", {}) or {})
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: Literal["2020-11-01-preview"] = kwargs.pop(
            "api_version", _params.pop("api-version", "2020-11-01-preview")
        )
        content_type: Optional[str] = kwargs.pop("content_type", _headers.pop("Content-Type", None))
        cls: ClsType[_models.JobStep] = kwargs.pop("cls", None)

        content_type = content_type or "application/json"
        _json = None
        _content = None
        if isinstance(parameters, (IO, bytes)):
            _content = parameters
        else:
            _json = self._serialize.body(parameters, "JobStep")

        request = build_create_or_update_request(
            resource_group_name=resource_group_name,
            server_name=server_name,
            job_agent_name=job_agent_name,
            job_name=job_name,
            step_name=step_name,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            content_type=content_type,
            json=_json,
            content=_content,
            template_url=self.create_or_update.metadata["url"],
            headers=_headers,
            params=_params,
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)

        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(  # pylint: disable=protected-access
            request, stream=_stream, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200, 201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        if response.status_code == 200:
            deserialized = self._deserialize("JobStep", pipeline_response)

        if response.status_code == 201:
            deserialized = self._deserialize("JobStep", pipeline_response)

        if cls:
            return cls(pipeline_response, deserialized, {})  # type: ignore

        return deserialized  # type: ignore

    create_or_update.metadata = {
        "url": "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Sql/servers/{serverName}/jobAgents/{jobAgentName}/jobs/{jobName}/steps/{stepName}"
    }

    @distributed_trace
    def delete(  # pylint: disable=inconsistent-return-statements
        self,
        resource_group_name: str,
        server_name: str,
        job_agent_name: str,
        job_name: str,
        step_name: str,
        **kwargs: Any
    ) -> None:
        """Deletes a job step. This will implicitly create a new job version.

        :param resource_group_name: The name of the resource group that contains the resource. You can
         obtain this value from the Azure Resource Manager API or the portal. Required.
        :type resource_group_name: str
        :param server_name: The name of the server. Required.
        :type server_name: str
        :param job_agent_name: The name of the job agent. Required.
        :type job_agent_name: str
        :param job_name: The name of the job. Required.
        :type job_name: str
        :param step_name: The name of the job step to delete. Required.
        :type step_name: str
        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: None or the result of cls(response)
        :rtype: None
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        error_map = {
            401: ClientAuthenticationError,
            404: ResourceNotFoundError,
            409: ResourceExistsError,
            304: ResourceNotModifiedError,
        }
        error_map.update(kwargs.pop("error_map", {}) or {})

        _headers = kwargs.pop("headers", {}) or {}
        _params = case_insensitive_dict(kwargs.pop("params", {}) or {})

        api_version: Literal["2020-11-01-preview"] = kwargs.pop(
            "api_version", _params.pop("api-version", "2020-11-01-preview")
        )
        cls: ClsType[None] = kwargs.pop("cls", None)

        request = build_delete_request(
            resource_group_name=resource_group_name,
            server_name=server_name,
            job_agent_name=job_agent_name,
            job_name=job_name,
            step_name=step_name,
            subscription_id=self._config.subscription_id,
            api_version=api_version,
            template_url=self.delete.metadata["url"],
            headers=_headers,
            params=_params,
        )
        request = _convert_request(request)
        request.url = self._client.format_url(request.url)

        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(  # pylint: disable=protected-access
            request, stream=_stream, **kwargs
        )

        response = pipeline_response.http_response

        if response.status_code not in [200, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response, error_format=ARMErrorFormat)

        if cls:
            return cls(pipeline_response, None, {})

    delete.metadata = {
        "url": "/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.Sql/servers/{serverName}/jobAgents/{jobAgentName}/jobs/{jobName}/steps/{stepName}"
    }
