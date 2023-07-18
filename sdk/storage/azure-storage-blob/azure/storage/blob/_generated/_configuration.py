# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

import sys
from typing import Any

from azure.core.configuration import Configuration
from azure.core.pipeline import policies

if sys.version_info >= (3, 8):
    from typing import Literal  # pylint: disable=no-name-in-module, ungrouped-imports
else:
    from typing_extensions import Literal  # type: ignore  # pylint: disable=ungrouped-imports

VERSION = "unknown"


class AzureBlobStorageConfiguration(Configuration):  # pylint: disable=too-many-instance-attributes
    """Configuration for AzureBlobStorage.

    Note that all parameters used to create this instance are saved as instance
    attributes.

    :param url: The URL of the service account, container, or blob that is the target of the
     desired operation. Required.
    :type url: str
    :keyword version: Specifies the version of the operation to use for this request. Default value
     is "2021-12-02". Note that overriding this default value may result in unsupported behavior.
    :paramtype version: str
    """

    def __init__(self, url: str, **kwargs: Any) -> None:
        super(AzureBlobStorageConfiguration, self).__init__(**kwargs)
        version: Literal["2021-12-02"] = kwargs.pop("version", "2021-12-02")

        if url is None:
            raise ValueError("Parameter 'url' must not be None.")

        self.url = url
        self.version = version
        kwargs.setdefault("sdk_moniker", "azureblobstorage/{}".format(VERSION))
        self._configure(**kwargs)

    def _configure(self, **kwargs: Any) -> None:
        self.user_agent_policy = kwargs.get("user_agent_policy") or policies.UserAgentPolicy(**kwargs)
        self.headers_policy = kwargs.get("headers_policy") or policies.HeadersPolicy(**kwargs)
        self.proxy_policy = kwargs.get("proxy_policy") or policies.ProxyPolicy(**kwargs)
        self.logging_policy = kwargs.get("logging_policy") or policies.NetworkTraceLoggingPolicy(**kwargs)
        self.http_logging_policy = kwargs.get("http_logging_policy") or policies.HttpLoggingPolicy(**kwargs)
        self.retry_policy = kwargs.get("retry_policy") or policies.RetryPolicy(**kwargs)
        self.custom_hook_policy = kwargs.get("custom_hook_policy") or policies.CustomHookPolicy(**kwargs)
        self.redirect_policy = kwargs.get("redirect_policy") or policies.RedirectPolicy(**kwargs)
        self.authentication_policy = kwargs.get("authentication_policy")
