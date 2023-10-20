# --------------------------------------------------------------------------
#
# Copyright (c) Microsoft Corporation. All rights reserved.
#
# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the ""Software""), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
# --------------------------------------------------------------------------
"""
This module is the requests implementation of Pipeline ABC
"""
from __future__ import annotations
import json
import inspect
import logging
import os
import platform
import xml.etree.ElementTree as ET
import types
import re
from typing import IO, cast, Union, Optional, AnyStr, Dict, Any, Set, Mapping, TYPE_CHECKING

from ... import __version__ as core_version
from ...exceptions import DecodeError

from ._base import SansIOHTTPPolicy

from ...rest import HttpRequest
from ...rest._rest_py3 import _HttpResponseBase as SansIOHttpResponse

if TYPE_CHECKING:
    from ...runtime.pipeline import PipelineRequest, PipelineResponse

_LOGGER = logging.getLogger(__name__)

HTTPRequestType = HttpRequest
HTTPResponseType = SansIOHttpResponse


class HeadersPolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """A simple policy that sends the given headers with the request.

    This will overwrite any headers already defined in the request. Headers can be
    configured up front, where any custom headers will be applied to all outgoing
    operations, and additional headers can also be added dynamically per operation.

    :param dict base_headers: Headers to send with the request.
    """

    def __init__(
        self, base_headers: Optional[Dict[str, str]] = None, **kwargs: Any
    ) -> None:  # pylint: disable=super-init-not-called
        self._headers: Dict[str, str] = base_headers or {}
        self._headers.update(kwargs.pop("headers", {}))

    @property
    def headers(self) -> Dict[str, str]:
        """The current headers collection.

        :rtype: dict[str, str]
        :return: The current headers collection.
        """
        return self._headers

    def add_header(self, key: str, value: str) -> None:
        """Add a header to the configuration to be applied to all requests.

        :param str key: The header.
        :param str value: The header's value.
        """
        self._headers[key] = value

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        """Updates with the given headers before sending the request to the next policy.

        :param request: The PipelineRequest object
        :type request: ~corehttp.runtime.pipeline.PipelineRequest
        """
        request.http_request.headers.update(self.headers)
        additional_headers = request.context.options.pop("headers", {})
        if additional_headers:
            request.http_request.headers.update(additional_headers)


class _Unset:
    pass


class UserAgentPolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """User-Agent Policy. Allows custom values to be added to the User-Agent header.

    :param str base_user_agent: Sets the base user agent value.

    :keyword bool user_agent_overwrite: Overwrites User-Agent when True. Defaults to False.
    :keyword bool user_agent_use_env: Gets user-agent from environment. Defaults to True.
    :keyword str user_agent: If specified, this will be added in front of the user agent string.
    :keyword str sdk_moniker: If specified, the user agent string will be
        azsdk-python-[sdk_moniker] Python/[python_version] ([platform_version])
    """

    _USERAGENT = "User-Agent"
    _ENV_ADDITIONAL_USER_AGENT = "CORE_HTTP_USER_AGENT"

    def __init__(
        self, base_user_agent: Optional[str] = None, **kwargs: Any
    ) -> None:  # pylint: disable=super-init-not-called
        self.overwrite: bool = kwargs.pop("user_agent_overwrite", False)
        self.use_env: bool = kwargs.pop("user_agent_use_env", True)
        application_id: Optional[str] = kwargs.pop("user_agent", None)
        sdk_moniker: str = kwargs.pop("sdk_moniker", "core/{}".format(core_version))

        if base_user_agent:
            self._user_agent = base_user_agent
        else:
            self._user_agent = "python-{} Python/{} ({})".format(
                sdk_moniker, platform.python_version(), platform.platform()
            )

        if application_id:
            self._user_agent = "{} {}".format(application_id, self._user_agent)

    @property
    def user_agent(self) -> str:
        """The current user agent value.

        :return: The current user agent value.
        :rtype: str
        """
        if self.use_env:
            add_user_agent_header = os.environ.get(self._ENV_ADDITIONAL_USER_AGENT, None)
            if add_user_agent_header is not None:
                return "{} {}".format(self._user_agent, add_user_agent_header)
        return self._user_agent

    def add_user_agent(self, value: str) -> None:
        """Add value to current user agent with a space.
        :param str value: value to add to user agent.
        """
        self._user_agent = "{} {}".format(self._user_agent, value)

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        """Modifies the User-Agent header before the request is sent.

        :param request: The PipelineRequest object
        :type request: ~corehttp.runtime.pipeline.PipelineRequest
        """
        http_request = request.http_request
        options_dict = request.context.options
        if "user_agent" in options_dict:
            user_agent = options_dict.pop("user_agent")
            if options_dict.pop("user_agent_overwrite", self.overwrite):
                http_request.headers[self._USERAGENT] = user_agent
            else:
                user_agent = "{} {}".format(user_agent, self.user_agent)
                http_request.headers[self._USERAGENT] = user_agent

        elif self.overwrite or self._USERAGENT not in http_request.headers:
            http_request.headers[self._USERAGENT] = self.user_agent


class NetworkTraceLoggingPolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):

    """The logging policy in the pipeline is used to output HTTP network trace to the configured logger.

    This accepts both global configuration, and per-request level with "enable_http_logger"

    :param bool logging_enable: Use to enable per operation. Defaults to False.
    """

    def __init__(self, logging_enable: bool = False, **kwargs: Any):  # pylint: disable=unused-argument
        self.enable_http_logger = logging_enable

    def on_request(
        self, request: PipelineRequest[HTTPRequestType]
    ) -> None:  # pylint: disable=too-many-return-statements
        """Logs HTTP request to the DEBUG logger.

        :param request: The PipelineRequest object.
        :type request: ~corehttp.runtime.pipeline.PipelineRequest
        """
        http_request = request.http_request
        options = request.context.options
        logging_enable = options.pop("logging_enable", self.enable_http_logger)
        request.context["logging_enable"] = logging_enable
        if logging_enable:
            if not _LOGGER.isEnabledFor(logging.DEBUG):
                return

            try:
                log_string = "Request URL: '{}'".format(http_request.url)
                log_string += "\nRequest method: '{}'".format(http_request.method)
                log_string += "\nRequest headers:"
                for header, value in http_request.headers.items():
                    log_string += "\n    '{}': '{}'".format(header, value)
                log_string += "\nRequest body:"

                # We don't want to log the binary data of a file upload.
                if isinstance(http_request.content, types.GeneratorType):
                    log_string += "\nFile upload"
                    _LOGGER.debug(log_string)
                    return
                try:
                    if isinstance(http_request.content, types.AsyncGeneratorType):
                        log_string += "\nFile upload"
                        _LOGGER.debug(log_string)
                        return
                except AttributeError:
                    pass
                if http_request.content:
                    log_string += "\n{}".format(str(http_request.content))
                    _LOGGER.debug(log_string)
                    return
                log_string += "\nThis request has no body"
                _LOGGER.debug(log_string)
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.debug("Failed to log request: %r", err)

    def on_response(
        self,
        request: PipelineRequest[HTTPRequestType],
        response: PipelineResponse[HTTPRequestType, HTTPResponseType],
    ) -> None:
        """Logs HTTP response to the DEBUG logger.

        :param request: The PipelineRequest object.
        :type request: ~corehttp.runtime.pipeline.PipelineRequest
        :param response: The PipelineResponse object.
        :type response: ~corehttp.runtime.pipeline.PipelineResponse
        """
        http_response = response.http_response
        try:
            logging_enable = response.context["logging_enable"]
            if logging_enable:
                if not _LOGGER.isEnabledFor(logging.DEBUG):
                    return

                log_string = "Response status: '{}'".format(http_response.status_code)
                log_string += "\nResponse headers:"
                for res_header, value in http_response.headers.items():
                    log_string += "\n    '{}': '{}'".format(res_header, value)

                # We don't want to log binary data if the response is a file.
                log_string += "\nResponse content:"
                pattern = re.compile(r'attachment; ?filename=["\w.]+', re.IGNORECASE)
                header = http_response.headers.get("content-disposition")

                if header and pattern.match(header):
                    filename = header.partition("=")[2]
                    log_string += "\nFile attachments: {}".format(filename)
                elif http_response.headers.get("content-type", "").endswith("octet-stream"):
                    log_string += "\nBody contains binary data."
                elif http_response.headers.get("content-type", "").startswith("image"):
                    log_string += "\nBody contains image data."
                else:
                    if response.context.options.get("stream", False):
                        log_string += "\nBody is streamable."
                    else:
                        log_string += "\n{}".format(http_response.text())
                _LOGGER.debug(log_string)
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.debug("Failed to log response: %s", repr(err))


class _HiddenClassProperties(type):
    # Backward compatible for DEFAULT_HEADERS_WHITELIST
    # https://github.com/Azure/azure-sdk-for-python/issues/26331

    @property
    def DEFAULT_HEADERS_WHITELIST(cls) -> Set[str]:
        return cls.DEFAULT_HEADERS_ALLOWLIST

    @DEFAULT_HEADERS_WHITELIST.setter
    def DEFAULT_HEADERS_WHITELIST(cls, value: Set[str]) -> None:
        cls.DEFAULT_HEADERS_ALLOWLIST = value


class ContentDecodePolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """Policy for decoding unstreamed response content.

    :param response_encoding: The encoding to use if known for this service (will disable auto-detection)
    :type response_encoding: str
    """

    # Accept "text" because we're open minded people...
    JSON_REGEXP = re.compile(r"^(application|text)/([0-9a-z+.-]+\+)?json$")

    # Name used in context
    CONTEXT_NAME = "deserialized_data"

    def __init__(
        self, response_encoding: Optional[str] = None, **kwargs: Any  # pylint: disable=unused-argument
    ) -> None:
        self._response_encoding = response_encoding

    @classmethod
    def deserialize_from_text(
        cls,
        data: Optional[Union[AnyStr, IO[AnyStr]]],
        mime_type: Optional[str] = None,
        response: Optional[HTTPResponseType] = None,
    ) -> Any:
        """Decode response data according to content-type.

        Accept a stream of data as well, but will be load at once in memory for now.
        If no content-type, will return the string version (not bytes, not stream)

        :param data: The data to deserialize.
        :type data: str or bytes or file-like object
        :param response: The HTTP response.
        :type response: ~corehttp.rest.HttpResponse
        :param str mime_type: The mime type. As mime type, charset is not expected.
        :param response: If passed, exception will be annotated with that response
        :type response: any
        :raises ~corehttp.exceptions.DecodeError: If deserialization fails
        :returns: A dict (JSON), XML tree or str, depending of the mime_type
        :rtype: dict[str, Any] or xml.etree.ElementTree.Element or str
        """
        if not data:
            return None

        if hasattr(data, "read"):
            # Assume a stream
            data = cast(IO, data).read()

        if isinstance(data, bytes):
            data_as_str = data.decode(encoding="utf-8-sig")
        else:
            # Explain to mypy the correct type.
            data_as_str = cast(str, data)

        if mime_type is None:
            return data_as_str

        if cls.JSON_REGEXP.match(mime_type):
            try:
                return json.loads(data_as_str)
            except ValueError as err:
                raise DecodeError(
                    message="JSON is invalid: {}".format(err),
                    response=response,
                    error=err,
                ) from err
        elif "xml" in (mime_type or []):
            try:
                return ET.fromstring(data_as_str)  # nosec
            except ET.ParseError as err:
                # It might be because the server has an issue, and returned JSON with
                # content-type XML....
                # So let's try a JSON load, and if it's still broken
                # let's flow the initial exception
                def _json_attemp(data):
                    try:
                        return True, json.loads(data)
                    except ValueError:
                        return False, None  # Don't care about this one

                success, json_result = _json_attemp(data)
                if success:
                    return json_result
                # If i'm here, it's not JSON, it's not XML, let's scream
                # and raise the last context in this block (the XML exception)
                # The function hack is because Py2.7 messes up with exception
                # context otherwise.
                _LOGGER.critical("Wasn't XML not JSON, failing")
                raise DecodeError("XML is invalid", response=response) from err
        elif mime_type.startswith("text/"):
            return data_as_str
        raise DecodeError("Cannot deserialize content-type: {}".format(mime_type))

    @classmethod
    def deserialize_from_http_generics(
        cls,
        response: HTTPResponseType,
        encoding: Optional[str] = None,
    ) -> Any:
        """Deserialize from HTTP response.

        Headers will tested for "content-type"

        :param response: The HTTP response
        :type response: any
        :param str encoding: The encoding to use if known for this service (will disable auto-detection)
        :raises ~corehttp.exceptions.DecodeError: If deserialization fails
        :returns: A dict (JSON), XML tree or str, depending of the mime_type
        :rtype: dict[str, Any] or xml.etree.ElementTree.Element or str
        """
        # Try to use content-type from headers if available
        if response.content_type:
            mime_type = response.content_type.split(";")[0].strip().lower()
        # Ouch, this server did not declare what it sent...
        # Let's guess it's JSON...
        # Also, since Autorest was considering that an empty body was a valid JSON,
        # need that test as well....
        else:
            mime_type = "application/json"

        # Rely on transport implementation to give me "text()" decoded correctly
        if hasattr(response, "read"):
            # since users can call deserialize_from_http_generics by themselves
            # we want to make sure our new responses are read before we try to
            # deserialize. Only read sync responses since we're in a sync function
            #
            # Technically HttpResponse do not contain a "read()", but we don't know what
            # people have been able to pass here, so keep this code for safety,
            # even if it's likely dead code
            if not inspect.iscoroutinefunction(response.read):  # type: ignore
                response.read()  # type: ignore
        return cls.deserialize_from_text(response.text(encoding), mime_type, response=response)

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        options = request.context.options
        response_encoding = options.pop("response_encoding", self._response_encoding)
        if response_encoding:
            request.context["response_encoding"] = response_encoding

    def on_response(
        self,
        request: PipelineRequest[HTTPRequestType],
        response: PipelineResponse[HTTPRequestType, HTTPResponseType],
    ) -> None:
        """Extract data from the body of a REST response object.
        This will load the entire payload in memory.
        Will follow Content-Type to parse.
        We assume everything is UTF8 (BOM acceptable).

        :param request: The PipelineRequest object.
        :type request: ~corehttp.runtime.pipeline.PipelineRequest
        :param response: The PipelineResponse object.
        :type response: ~corehttp.runtime.pipeline.PipelineResponse
        :raises JSONDecodeError: If JSON is requested and parsing is impossible.
        :raises UnicodeDecodeError: If bytes is not UTF8
        :raises xml.etree.ElementTree.ParseError: If bytes is not valid XML
        :raises ~corehttp.exceptions.DecodeError: If deserialization fails
        """
        # If response was asked as stream, do NOT read anything and quit now
        if response.context.options.get("stream", True):
            return

        response_encoding = request.context.get("response_encoding")

        response.context[self.CONTEXT_NAME] = self.deserialize_from_http_generics(
            response.http_response, response_encoding
        )


class ProxyPolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """A proxy policy.

    Dictionary mapping protocol or protocol and host to the URL of the proxy
    to be used on each Request.

    :param dict proxies: Maps protocol or protocol and hostname to the URL
     of the proxy.
    """

    def __init__(
        self, proxies: Optional[Mapping[str, str]] = None, **kwargs: Any
    ):  # pylint: disable=unused-argument,super-init-not-called
        self.proxies = proxies

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        ctxt = request.context.options
        if self.proxies and "proxies" not in ctxt:
            ctxt["proxies"] = self.proxies
