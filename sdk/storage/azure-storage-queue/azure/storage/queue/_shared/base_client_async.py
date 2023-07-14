# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from typing import (  # pylint: disable=unused-import
    Union, Optional, Any, Iterable, Dict, List, Type, Tuple,
    TYPE_CHECKING
)
import logging

from azure.core.credentials import AzureSasCredential
from azure.core.pipeline import AsyncPipeline
from azure.core.async_paging import AsyncList
from azure.core.exceptions import HttpResponseError
from azure.core.pipeline.policies import (
    AsyncBearerTokenCredentialPolicy,
    AsyncRedirectPolicy,
    AzureSasCredentialPolicy,
    ContentDecodePolicy,
    DistributedTracingPolicy,
    HttpLoggingPolicy,
)
from azure.core.pipeline.transport import AsyncHttpTransport

from .constants import CONNECTION_TIMEOUT, READ_TIMEOUT, STORAGE_OAUTH_SCOPE
from .authentication import SharedKeyCredentialPolicy
from .base_client import create_configuration
from .policies import (
    QueueMessagePolicy,
    StorageContentValidation,
    StorageHeadersPolicy,
    StorageHosts,
    StorageRequestHook,
)
from .policies_async import AsyncStorageResponseHook

from .response_handlers import process_storage_error, PartialBatchErrorException

if TYPE_CHECKING:
    from azure.core.pipeline.transport import HttpRequest
    from azure.core.configuration import Configuration
_LOGGER = logging.getLogger(__name__)


class AsyncStorageAccountHostsMixin(object):

    def __enter__(self):
        raise TypeError("Async client only supports 'async with'.")

    def __exit__(self, *args):
        pass

    async def __aenter__(self):
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self._client.__aexit__(*args)

    async def close(self):
        """ This method is to close the sockets opened by the client.
        It need not be used when using with a context manager.
        """
        await self._client.close()

    def _create_pipeline(self, credential, **kwargs):
        # type: (Any, **Any) -> Tuple[Configuration, AsyncPipeline]
        self._credential_policy: Optional[Union[AsyncBearerTokenCredentialPolicy, SharedKeyCredentialPolicy, AzureSasCredentialPolicy]] = None
        if hasattr(credential, 'get_token'):
            self._credential_policy = AsyncBearerTokenCredentialPolicy(credential, STORAGE_OAUTH_SCOPE)
        elif isinstance(credential, SharedKeyCredentialPolicy):
            self._credential_policy = credential
        elif isinstance(credential, AzureSasCredential):
            self._credential_policy = AzureSasCredentialPolicy(credential)
        elif credential is not None:
            raise TypeError(f"Unsupported credential: {credential}")
        config = kwargs.get('_configuration') or create_configuration(**kwargs)
        if kwargs.get('_pipeline'):
            return config, kwargs['_pipeline']
        transport = kwargs.get('transport')
        kwargs.setdefault("connection_timeout", CONNECTION_TIMEOUT)
        kwargs.setdefault("read_timeout", READ_TIMEOUT)
        if not transport:
            try:
                from azure.core.pipeline.transport import AioHttpTransport
            except ImportError:
                raise ImportError("Unable to create async transport. Please check aiohttp is installed.")
            transport = AioHttpTransport(**kwargs)
        if hasattr(self, '_hosts'):
            hosts = self._hosts
        policies = [
            QueueMessagePolicy(),
            config.headers_policy,
            config.proxy_policy,
            config.user_agent_policy,
            StorageContentValidation(),
            StorageRequestHook(**kwargs),
            self._credential_policy,
            ContentDecodePolicy(response_encoding="utf-8"),
            AsyncRedirectPolicy(**kwargs),
            StorageHosts(hosts=hosts, **kwargs),
            config.retry_policy,
            config.logging_policy,
            AsyncStorageResponseHook(**kwargs),
            DistributedTracingPolicy(**kwargs),
            HttpLoggingPolicy(**kwargs),
        ]
        if kwargs.get("_additional_pipeline_policies"):
            policies = policies + kwargs.get("_additional_pipeline_policies")  #type: ignore
        if hasattr(config, 'transport'):
            config.transport = transport
        return config, AsyncPipeline(transport, policies=policies)

    async def _batch_send(
        self,
        *reqs,  # type: HttpRequest
        **kwargs
    ):
        """Given a series of request, do a Storage batch call.
        """
        # Pop it here, so requests doesn't feel bad about additional kwarg
        raise_on_any_failure = kwargs.pop("raise_on_any_failure", True)
        if hasattr(self, '_client'):
            client = self._client
        if hasattr(self, 'scheme'):
            scheme = self.scheme
        if hasattr(self, 'primary_hostname'):
            primary_hostname = self.primary_hostname
        if hasattr(self, 'api_version'):
            api_version = self.api_version
        request = client._client.post(
            url=(
                f'{scheme}://{primary_hostname}/'
                f"{kwargs.pop('path', '')}?{kwargs.pop('restype', '')}"
                f"comp=batch{kwargs.pop('sas', '')}{kwargs.pop('timeout', '')}"
            ),
            headers={
                'x-ms-version': api_version
            }
        )

        policies = [StorageHeadersPolicy()]
        if self._credential_policy:
            policies.append(self._credential_policy)  # type: ignore

        request.set_multipart_mixed(
            *reqs,
            policies=policies,
            enforce_https=False
        )

        if hasattr(self, '_pipeline'):
            pipeline_response = await self._pipeline.run(
                request, **kwargs
            )
        response = pipeline_response.http_response

        try:
            if response.status_code not in [202]:
                raise HttpResponseError(response=response)
            parts = response.parts() # Return an AsyncIterator
            if raise_on_any_failure:
                parts_list = []
                async for part in parts:
                    parts_list.append(part)
                if any(p for p in parts_list if not 200 <= p.status_code < 300):
                    error = PartialBatchErrorException(
                        message="There is a partial failure in the batch operation.",
                        response=response, parts=parts_list
                    )
                    raise error
                return AsyncList(parts_list)
            return parts
        except HttpResponseError as error:
            process_storage_error(error)


class AsyncTransportWrapper(AsyncHttpTransport):
    """Wrapper class that ensures that an inner client created
    by a `get_client` method does not close the outer transport for the parent
    when used in a context manager.
    """
    def __init__(self, async_transport):
        self._transport = async_transport

    async def send(self, request, **kwargs):
        return await self._transport.send(request, **kwargs)

    async def open(self):
        pass

    async def close(self):
        pass

    async def __aenter__(self):
        pass

    async def __aexit__(self, *args):  # pylint: disable=arguments-differ
        pass
