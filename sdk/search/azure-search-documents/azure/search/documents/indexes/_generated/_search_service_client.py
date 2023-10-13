# coding=utf-8
# --------------------------------------------------------------------------
# Code generated by Microsoft (R) AutoRest Code Generator (autorest: 3.9.7, generator: @autorest/python@6.9.1)
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from copy import deepcopy
from typing import Any

from azure.core import PipelineClient
from azure.core.pipeline import policies
from azure.core.rest import HttpRequest, HttpResponse

from . import models as _models
from ._configuration import SearchServiceClientConfiguration
from ._serialization import Deserializer, Serializer
from .operations import (
    DataSourcesOperations,
    IndexersOperations,
    IndexesOperations,
    SearchServiceClientOperationsMixin,
    SkillsetsOperations,
    SynonymMapsOperations,
)


class SearchServiceClient(SearchServiceClientOperationsMixin):  # pylint: disable=client-accepts-api-version-keyword
    """Client that can be used to manage and query indexes and documents, as well as manage other
    resources, on a search service.

    :ivar data_sources: DataSourcesOperations operations
    :vartype data_sources: search_service_client.operations.DataSourcesOperations
    :ivar indexers: IndexersOperations operations
    :vartype indexers: search_service_client.operations.IndexersOperations
    :ivar skillsets: SkillsetsOperations operations
    :vartype skillsets: search_service_client.operations.SkillsetsOperations
    :ivar synonym_maps: SynonymMapsOperations operations
    :vartype synonym_maps: search_service_client.operations.SynonymMapsOperations
    :ivar indexes: IndexesOperations operations
    :vartype indexes: search_service_client.operations.IndexesOperations
    :param endpoint: The endpoint URL of the search service. Required.
    :type endpoint: str
    :keyword api_version: Api Version. Default value is "2023-11-01". Note that overriding this
     default value may result in unsupported behavior.
    :paramtype api_version: str
    """

    def __init__(  # pylint: disable=missing-client-constructor-parameter-credential
        self, endpoint: str, **kwargs: Any
    ) -> None:
        _endpoint = "{endpoint}"
        self._config = SearchServiceClientConfiguration(endpoint=endpoint, **kwargs)
        _policies = kwargs.pop("policies", None)
        if _policies is None:
            _policies = [
                policies.RequestIdPolicy(**kwargs),
                self._config.headers_policy,
                self._config.user_agent_policy,
                self._config.proxy_policy,
                policies.ContentDecodePolicy(**kwargs),
                self._config.redirect_policy,
                self._config.retry_policy,
                self._config.authentication_policy,
                self._config.custom_hook_policy,
                self._config.logging_policy,
                policies.DistributedTracingPolicy(**kwargs),
                policies.SensitiveHeaderCleanupPolicy(**kwargs) if self._config.redirect_policy else None,
                self._config.http_logging_policy,
            ]
        self._client: PipelineClient = PipelineClient(base_url=_endpoint, policies=_policies, **kwargs)

        client_models = {k: v for k, v in _models.__dict__.items() if isinstance(v, type)}
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self._serialize.client_side_validation = False
        self.data_sources = DataSourcesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.indexers = IndexersOperations(self._client, self._config, self._serialize, self._deserialize)
        self.skillsets = SkillsetsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.synonym_maps = SynonymMapsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.indexes = IndexesOperations(self._client, self._config, self._serialize, self._deserialize)

    def _send_request(self, request: HttpRequest, **kwargs: Any) -> HttpResponse:
        """Runs the network request through the client's chained policies.

        >>> from azure.core.rest import HttpRequest
        >>> request = HttpRequest("GET", "https://www.example.org/")
        <HttpRequest [GET], url: 'https://www.example.org/'>
        >>> response = client._send_request(request)
        <HttpResponse: 200 OK>

        For more information on this code flow, see https://aka.ms/azsdk/dpcodegen/python/send_request

        :param request: The network request you want to make. Required.
        :type request: ~azure.core.rest.HttpRequest
        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.
        :return: The response of your network call. Does not do error handling on your response.
        :rtype: ~azure.core.rest.HttpResponse
        """

        request_copy = deepcopy(request)
        path_format_arguments = {
            "endpoint": self._serialize.url("self._config.endpoint", self._config.endpoint, "str", skip_quote=True),
        }

        request_copy.url = self._client.format_url(request_copy.url, **path_format_arguments)
        return self._client.send_request(request_copy, **kwargs)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SearchServiceClient":
        self._client.__enter__()
        return self

    def __exit__(self, *exc_details: Any) -> None:
        self._client.__exit__(*exc_details)
