# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
from copy import deepcopy
from typing import Any, TYPE_CHECKING, overload, Callable, Union, Optional, Dict
import sys
import logging
import time
import threading

import websocket
import urllib.parse

from ._models import (
    WebPubSubClientOptions,
    OnConnectedArgs,
    OnDisconnectedArgs,
    OnServerDataMessageArgs,
    OnGroupDataMessageArgs,
    OnRejoinGroupFailedArgs,
    SendEventOptions,
    JoinGroupOptions,
    LeaveGroupOptions,
    SendToGroupOptions,
    WebPubSubRetryOptions,
    WebPubSubJsonReliableProtocol,
    SequenceId,
    RetryPolicy,
    WebPubSubGroup,
    SendMessageErrorOptions,
    WebPubSubMessage,
    SendMessageError,
    SendEventMessage,
    SendToGroupMessage,
    AckMessage,
    ConnectedMessage,
    SequenceAckMessage,
    CloseEvent,
    OnRestoreGroupFailedArgs,
    DisconnectedMessage,
    GroupDataMessage,
    ServerDataMessage,
)
from ._enums import WebPubSubDataType, WebPubSubClientState, CallBackType

_LOGGER = logging.getLogger(__name__)

if sys.version_info >= (3, 8):
    from typing import Literal  # pylint: disable=no-name-in-module, ungrouped-imports
else:
    from typing_extensions import Literal  # type: ignore  # pylint: disable=ungrouped-imports


class WebPubSubClientCredential:
    @overload
    def __init__(self, client_access_url_provider: str) -> None:
        ...

    @overload
    def __init__(self, client_access_url_provider: Callable[[Any], str]) -> None:
        ...

    def __init__(self, client_access_url_provider: Union[str, Callable[[Any], str]]) -> None:
        if isinstance(client_access_url_provider, str):
            self._client_access_url_provider = lambda: client_access_url_provider
        else:
            self._client_access_url_provider = client_access_url_provider

    def get_client_access_url(self) -> str:
        return self._client_access_url_provider()


class WebPubSubClient:  # pylint: disable=client-accepts-api-version-keyword
    """WebPubSubClient."""

    @overload
    def __init__(
        self,
        credential: WebPubSubClientCredential,
        options: Optional[WebPubSubClientOptions] = None,
        **kwargs: Any,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        client_access_url: str,
        options: Optional[WebPubSubClientOptions] = None,
        **kwargs: Any,
    ) -> None:
        ...

    def __init__(
        self,
        credential: Optional[WebPubSubClientCredential] = None,
        client_access_url: Optional[str] = None,
        options: Optional[WebPubSubClientOptions] = None,
        **kwargs: Any,
    ) -> None:
        if credential:
            self._credential = credential
        elif client_access_url:
            self._credential = WebPubSubClientCredential(client_access_url)
        else:
            raise TypeError("Please input parameter credential or client_access_url")

        if options is None:
            options = WebPubSubClientOptions()
        self._build_default_options(options)
        self._options = options
        self._message_retry_policy = RetryPolicy(self._options.message_retry_options)
        self._reconnect_retry_policy = RetryPolicy(
            WebPubSubRetryOptions(max_retries=sys.maxint, retry_delay_in_ms=1000, mode="Fixed")
        )
        self._protocol = self._options.protocol
        self._group_map: Dict[str, WebPubSubGroup] = {}
        self._ack_map: Dict[int, SendMessageErrorOptions] = {}
        self._sequence_id = SequenceId()
        self._state = WebPubSubClientState.STOPPED
        self._ack_id = 0
        self._url = None
        self._ws = None
        self._handler = {
            CallBackType.CONNECTED: [],
            CallBackType.DISCONNECTED: [],
            CallBackType.REJOIN_GROUP_FAILED: [],
            CallBackType.GROUP_MESSAGE: [],
            CallBackType.SERVER_MESSAGE: [],
            CallBackType.STOPPED: [],
        }
        self._last_disconnected_message = None
        self._connection_id = None
        self._is_initial_connected = False
        self._is_stopping = False
        self._last_close_event = None
        self._reconnection_token = None

    def _next_ack_id(self) -> int:
        self._ack_id = self._ack_id + 1
        return self._ack_id

    def _send_message(self, message: WebPubSubMessage):
        pay_load = self._protocol.write_message(message)
        if self._ws is None or not self._ws.connected:
            raise Exception("The connection is not connected.")

        self._ws.send(pay_load)

    def _send_message_with_ack_id(
        self,
        message_provider: Callable[[int], WebPubSubMessage],
        ack_id: Optional[int] = None,
    ):
        if ack_id is None:
            ack_id = self._next_ack_id()

        message = message_provider(ack_id)
        if ack_id not in self._ack_map:
            self._ack_map[ack_id] = None
        try:
            self._send_message(message)
        except Exception as e:
            self._ack_map.pop(ack_id)
            raise e

        # wait for ack from service
        while True:
            self.switch_thread()
            if self._ack_map[ack_id]:
                options = self._ack_map.pop(ack_id)
                if options.error_detail is not None:
                    raise SendMessageError(message="Failed to send message.", options=options)
                break

    def _get_or_add_group(self, name: str) -> WebPubSubGroup:
        if name not in self._group_map:
            self._group_map[name] = WebPubSubGroup(name=name)
        return self._group_map[name]

    def join_group(self, group_name: str, options: Optional[JoinGroupOptions] = None):
        self._retry(self._join_group_attempt, group_name, options)

    def _join_group_attempt(self, group_name: str, options: Optional[JoinGroupOptions] = None):
        group = self._get_or_add_group(group_name)
        self._join_group_core(group_name, options)
        group.is_joined = True

    def _join_group_core(self, group_name: str, options: Optional[JoinGroupOptions] = None):
        self._send_message_with_ack_id(
            message_provider=lambda id: JoinGroupMessage(group=group_name, ack_id=id),
            ack_id=options.ack_id if options else None,
        )

    def leave_group(self, group_name: str, options: Optional[LeaveGroupOptions] = None):
        self._retry(self._leave_group_attempt, group_name, options)

    def _leave_group_attempt(self, group_name: str, options: Optional[LeaveGroupOptions] = None):
        group = self._get_or_add_group(group_name)
        self._send_message_with_ack_id(
            message_provider=lambda id: LeaveGroupMessage(group=group_name, ack_id=id),
            ack_id=options.ack_id if options else None,
        )
        group.is_joined = False

    def send_event(
        self,
        event_name: str,
        content: Any,
        data_type: WebPubSubDataType,
        options: Optional[SendEventOptions] = None,
    ):
        self._retry(self._send_event_attempt, event_name, content, data_type, options)

    def _send_event_attempt(
        self,
        event_name: str,
        content: Any,
        data_type: WebPubSubDataType,
        options: Optional[SendEventOptions] = None,
    ):
        fire_and_forget = options.fire_and_forget if options else False
        if not fire_and_forget:
            self._send_message_with_ack_id(
                message_provider=lambda id: SendEventMessage(
                    data_type=data_type, data=content, ack_id=id, event=event_name
                )
            )
        else:
            self._send_message(message=SendEventMessage(data_type=data_type, data=content, event=event_name))

    def send_to_group(
        self,
        group_name: str,
        content: Any,
        data_type: WebPubSubDataType,
        options: Optional[SendToGroupOptions] = None,
    ):
        self._retry(self._send_to_group_attempt, group_name, content, data_type, options)

    def _send_to_group_attempt(
        self,
        group_name: str,
        content: Any,
        data_type: WebPubSubDataType,
        options: Optional[SendToGroupOptions] = None,
    ):
        fire_and_forget = options.fire_and_forget if options else False
        no_echo = options.no_echo if options else False
        if not fire_and_forget:
            self._send_message_with_ack_id(
                message_provider=lambda id: SendToGroupMessage(
                    group=group_name, data_type=data_type, data=content, ack_id=id, no_echo=no_echo
                )
            )
        else:
            self._send_message(
                message=SendToGroupMessage(group=group_name, data_type=data_type, data=content, no_echo=no_echo)
            )

    def _retry(self, func: Any, *args, **argv):
        retry_attempt = 0
        while True:
            try:
                func(self, *args, **argv)
            except Exception as e:
                retry_attempt = retry_attempt + 1
                delay_in_ms = self._message_retry_policy.next_retry_delay_in_ms(retry_attempt)
                if delay_in_ms is None:
                    raise e

            time.sleep(float(delay_in_ms) / 1000.0)

    @staticmethod
    def switch_thread():
        time.sleep(0.000001)

    def on(self, type: str, listener: Callable[[Any], None]):
        self._handler[type].append(listener)

    def _call_back(self, type: CallBackType, *args):
        for func in self._handler[type]:
            if self._state == WebPubSubClientState.CONNECTED:
                func(args)

    def on_message(self, data: str):
        def handle_ack_message(message: AckMessage):
            if message.ack_id in self._ack_map:
                if message.success or (message.error and message.error.name == "Duplicate"):
                    self._ack_map[message.ack_id] = SendMessageErrorOptions()
                else:
                    self._ack_map[message.ack_id] = SendMessageErrorOptions(
                        ack_id=message.ack_id, error_detail=message.error
                    )

        def handle_connected_message(message: ConnectedMessage):
            self._connection_id = message.connection_id

            if not self._is_initial_connected:
                self._is_initial_connected = True
                for group_name, group in self._group_map.items():
                    if group.is_joined:
                        try:
                            self._join_group_core(group_name)
                        except Exception as e:
                            self._call_back(
                                CallBackType.REJOIN_GROUP_FAILED,
                                OnRestoreGroupFailedArgs(group=group_name, error=e),
                            )

                connected_args = OnConnectedArgs(connection_id=message.connection_id, user_id=message.user_id)
                self._call_back(CallBackType.CONNECTED, connected_args)

        def handle_disconnected_message(message: DisconnectedMessage):
            self._last_disconnected_message = message

        def handle_group_data_message(message: GroupDataMessage):
            if message.sequence_id is not None:
                if not self._sequence_id.try_update(message.sequence_id):
                    # // drop duplicated message
                    return

            self._call_back(CallBackType.GROUP_MESSAGE, OnGroupDataMessageArgs(message))

        def handle_server_data_message(message: ServerDataMessage):
            if message.sequence_id is not None:
                if not self._sequence_id.try_update(message.sequence_id):
                    # // drop duplicated message
                    return

            self._call_back(CallBackType.SERVER_MESSAGE, OnServerDataMessageArgs(message))

        parsed_message = self._protocol.parse_messages(data)
        type_handler = {
            "connected": handle_connected_message,
            "disconnected": handle_disconnected_message,
            "ack": handle_ack_message,
            "groupData": handle_group_data_message,
            "serverData": handle_server_data_message,
        }
        if parsed_message.kind in type_handler:
            type_handler[parsed_message.kind](parsed_message)
        else:
            raise Exception(f"unknown message type: {parsed_message.kind}")

    def _get_close_args(self, close_frame):
        """
        _get_close_args extracts the close code and reason from the close body
        if it exists (RFC6455 says WebSocket Connection Close Code is optional)
        """
        # Need to catch the case where close_frame is None
        # Otherwise the following if statement causes an error
        # Extract close frame status code
        if close_frame is None:
            return [None, None]
        if close_frame.data and len(close_frame.data) >= 2:
            close_status_code = 256 * close_frame.data[0] + close_frame.data[1]
            reason = close_frame.data[2:].decode("utf-8")
            return [close_status_code, reason]
        else:
            # Most likely reached this because len(close_frame_data.data) < 2
            return [None, None]

    def _start_from_restarting(self):
        if self._state != WebPubSubClientState.DISCONNECTED:
            _LOGGER.warn("Client can be only restarted when it's Disconnected")
            return

        try:
            self._start_core()
        except Exception as e:
            self._state = WebPubSubClientState.DISCONNECTED
            raise e

    def _auto_reconnect(self):
        success = True
        attempt = 0
        while not self._is_stopping:
            try:
                self._start_from_restarting()
                success = True
                break
            except Exception as e:
                _LOGGER.warn("An attempt to reconnect connection failed", e)
                attempt = attempt + 1
                delay_in_ms = self._reconnect_retry_policy.next_retry_delay_in_ms(attempt)
                if not delay_in_ms:
                    break
                time.sleep(float(delay_in_ms) / 1000.0)
        if not success:
            self._handle_connection_stopped()

    def _handle_connection_stopped(self):
        self._is_stopping = False
        self._state = WebPubSubClientState.STOPPED
        self._call_back(CallBackType.STOPPED)

    def _handle_connection_close_and_no_recovery(self):
        self._state = WebPubSubClientState.DISCONNECTED
        self._call_back(
            CallBackType.DISCONNECTED,
            OnDisconnectedArgs(connection_id=self._connection_id, message=self._last_disconnected_message),
        )
        if self._options.auto_reconnect:
            self._auto_reconnect()
        else:
            self._handle_connection_stopped()

    def _build_recovery_url(self):
        if self._connection_id and self._reconnection_token and self._url:
            params = {"awps_connection_id": self._connection_id, "awps_reconnection_token": self._reconnection_token}
            url_parse = urllib.parse.urlparse(self._url)
            url_dict = dict(urllib.parse.parse_qsl(url_parse.query))
            url_dict.update(params)
            new_query = urllib.parse.urlencode(url_dict)
            url_parse = url_parse._replace(query=new_query)
            new_url = urllib.parse.urlunparse(url_parse)
            return new_url
        return None

    def _handle_connection_close(self):
        # clean ack cache
        self._ack_map.clear()

        if self._is_stopping:
            _LOGGER.warn("The client is stopping state. Stop recovery.")
            self._handle_connection_close_and_no_recovery()
            return

        if self._last_close_event and self._last_close_event.close_status_code == 1008:
            _LOGGER.warn("The websocket close with status code 1008. Stop recovery.")
            self._handle_connection_close_and_no_recovery()
            return

        if not self._protocol.is_reliable_sub_protocol:
            _LOGGER.warn("The protocol is not reliable, recovery is not applicable")
            self._handle_connection_close_and_no_recovery()
            return

        recovery_url = self._build_recovery_url()
        if not recovery_url:
            _LOGGER.warn("Connection id or reconnection token is not available")
            self._handle_connection_close_and_no_recovery()
            return

        self._state = WebPubSubClientState.RECOVERING
        while i < 30 or self._is_stopping:
            try:
                self._connect(recovery_url)
                return
            except:
                time.sleep(1)
            i = i + 1

        _LOGGER.warn("Recovery attempts failed more then 30 seconds or the client is stopping")
        self._handle_connection_close_and_no_recovery()

    def _listen(self):
        while self._state == WebPubSubClientState.CONNECTED:
            try:
                op_code, frame = self._ws.recv_data_frame(True)
            except (
                websocket.WebSocketConnectionClosedException,
                websocket.KeyboardInterrupt,
            ) as e:
                if self._state == WebPubSubClientState.CONNECTED:
                    raise e
                pass
            else:
                if op_code == websocket.ABNF.OPCODE_CLOSE:
                    close_status_code, close_reason = self._get_close_args(frame)
                    if self._state == WebPubSubClientState.CONNECTED:
                        self._last_close_event = CloseEvent(
                            close_status_code=close_status_code, close_reason=close_reason
                        )
                        self._handle_connection_close()
                elif op_code == websocket.ABNF.OPCODE_TEXT or op_code == websocket.ABNF.OPCODE_BINARY:
                    data = frame.data
                    if op_code == websocket.ABNF.OPCODE_TEXT:
                        data = data.decode("utf-8")
                    self.on_message(data)

    def sequence_id_ack_periodically(self):
        while self._state == WebPubSubClientState.CONNECTED:
            try:
                is_updated, seq_id = self._sequence_id.try_get_sequence_id()
                if is_updated:
                    self._send_message(SequenceAckMessage(sequence_id=seq_id))
            finally:
                time.sleep(1)

    def _connect(self, url: str):
        self._ws = websocket.WebSocket()
        self._ws.connect(url, subprotocols=[self._protocol.name])

        if self._is_stopping:
            try:
                self._ws.close()
            finally:
                return

        self._state = WebPubSubClientState.CONNECTED
        if self._protocol.is_reliable_sub_protocol:
            self._thread_seq_ack = threading.Thread(target=self.sequence_id_ack_periodically, daemon=True)
            self._thread_seq_ack.start()

        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _start_core(self):
        self._state = WebPubSubClientState.CONNECTING
        _LOGGER.info("Staring a new connection")

        # Reset before a pure new connection
        self._sequence_id.reset()
        self._is_initial_connected = False
        self._last_close_event = None
        self._last_disconnected_message = None
        self._connection_id = None
        self._reconnection_token = None
        self._url = None

        self._url = self._credential.get_client_access_url()
        self._connect(self._url)

    def start(self):
        if self._is_stopping:
            _LOGGER.error("Can't start a client during stopping")
            return
        if self._state != WebPubSubClientState.STOPPED:
            _LOGGER.warn("Client can be only started when it's Stopped")
            return

        try:
            self._start_core()
        except Exception as e:
            self._state = WebPubSubClientState.STOPPED
            self._is_stopping = False
            raise e

    def stop(self):
        if self._state == WebPubSubClientState.STOPPED or self._is_stopping:
            return
        self._is_stopping = True
        if self._ws:
            self._ws.close()

    @staticmethod
    def _build_default_options(self, options: WebPubSubClientOptions):
        if options.auto_reconnect is None:
            options.auto_reconnect = True
        if options.auto_restore_groups is None:
            options.auto_restore_groups = True
        if options.protocol is None:
            options.protocol = WebPubSubJsonReliableProtocol()

        self._build_message_retry_options(options)

    @staticmethod
    def _build_message_retry_options(options: WebPubSubClientOptions):
        if options.message_retry_options is None:
            options.message_retry_options = WebPubSubRetryOptions()
        if options.message_retry_options.max_retries is None or options.message_retry_options.max_retries < 0:
            options.message_retry_options.max_retries = 3
        if (
            options.message_retry_options.retry_delay_in_ms is None
            or options.message_retry_options.retry_delay_in_ms < 0
        ):
            options.message_retry_options.retry_delay_in_ms = 1000
        if (
            options.message_retry_options.max_retry_delay_in_ms is None
            or options.message_retry_options.max_retry_delay_in_ms < 0
        ):
            options.message_retry_options.max_retry_delay_in_ms = 30000
        if options.message_retry_options.mode is None:
            options.message_retry_options.mode = "Fixed"

    @overload
    def on(self, event: Literal["connected"], listener: Callable[[OnConnectedArgs], None]) -> None:
        """"""

    @overload
    def on(self, event: Literal["disconnected"], listener: Callable[[OnDisconnectedArgs], None]) -> None:
        """"""

    @overload
    def on(event: Literal["stopped"], listener: Callable[[], None]) -> None:
        """"""

    @overload
    def on(self, event: Literal["server-message"], listener: Callable[[OnServerDataMessageArgs], None]) -> None:
        """"""

    @overload
    def on(self, event: Literal["group-message"], listener: Callable[[OnGroupDataMessageArgs], None]) -> None:
        """"""

    @overload
    def on(self, event: Literal["rejoin-group-failed"], listener: Callable[[OnRejoinGroupFailedArgs], None]) -> None:
        """"""

    def on(self, event: CallBackType, listener: Callable[[Any], None]) -> None:
        if event in self._handler:
            self._handler[event].append(listener)
        else:
            _LOGGER.error(f"wrong event type: {event}")

    @overload
    def off(self, event: Literal["connected"], listener: Callable[[OnConnectedArgs], None]) -> None:
        """"""

    @overload
    def off(self, event: Literal["disconnected"], listener: Callable[[OnDisconnectedArgs], None]) -> None:
        """"""

    @overload
    def on(event: Literal["stopped"], listener: Callable[[], None]) -> None:
        """"""

    @overload
    def off(self, event: Literal["server-message"], listener: Callable[[OnServerDataMessageArgs], None]) -> None:
        """"""

    @overload
    def off(self, event: Literal["group-message"], listener: Callable[[OnGroupDataMessageArgs], None]) -> None:
        """"""

    @overload
    def off(self, event: Literal["rejoin-group-failed"], listener: Callable[[OnRejoinGroupFailedArgs], None]) -> None:
        """"""

    def off(self, event: CallBackType, listener: Callable[[Any], None]) -> None:
        if event in self._handler:
            if listener in self._handler[event]:
                self._handler[event].remove(listener)
            else:
                _LOGGER.info(f"target listener does not exist")
        else:
            _LOGGER.error(f"wrong event type: {event}")
