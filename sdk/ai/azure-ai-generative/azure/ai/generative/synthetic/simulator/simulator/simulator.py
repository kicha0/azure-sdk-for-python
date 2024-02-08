# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from typing import Callable, Dict, List, Union, Optional, Sequence, Any
from azure.ai.generative.synthetic.simulator._conversation import (
    ConversationBot,
    ConversationRole,
    ConversationTurn,
    simulate_conversation,
)
from azure.ai.generative.synthetic.simulator._model_tools.models import (
    OpenAIChatCompletionsModel,
    AsyncHTTPClientWithRetry,
)
from azure.ai.generative.synthetic.simulator import _template_dir as template_dir
from azure.ai.generative.synthetic.simulator._model_tools import (
    OpenAIChatCompletionsModel,
    LLMBase,
    ManagedIdentityAPITokenManager,
)
from azure.ai.generative.synthetic.simulator.templates.simulator_templates import (
    SimulatorTemplates,
)
from azure.ai.generative.synthetic.simulator.simulator._simulation_request_dto import (
    SimulationRequestDTO,
)
from azure.ai.generative.synthetic.simulator.simulator._token_manager import PlainTokenManager, TokenScope

from azure.ai.generative.synthetic.simulator.simulator._proxy_completion_model import ProxyChatCompletionsModel

from azure.ai.generative.synthetic.simulator.simulator._callback_conversation_bot import CallbackConversationBot

from azure.ai.generative.synthetic.simulator._rai_rest_client.rai_client import RAIClient

from azure.ai.generative.synthetic.simulator.simulator._utils import JsonLineList

import logging
import os
import asyncio
import threading
import json
import random

BASIC_MD = os.path.join(template_dir, "basic.md")  # type: ignore[has-type]
USER_MD = os.path.join(template_dir, "user.md")  # type: ignore[has-type]


class Simulator:
    def __init__(
        self,
        simulator_connection: "AzureOpenAIModelConfiguration" = None,  # type: ignore[name-defined]
        ai_client: "AIClient" = None,  # type: ignore[name-defined]
        simulate_callback: Optional[
            Callable[
                [str, Sequence[Union[Dict, ConversationTurn]], Optional[Dict]], str
            ]
        ] = None,
    ):
        self.ai_client = ai_client
        self.simulator_connection = self._to_openai_chat_completion_model(simulator_connection)
        self.adversarial = False
        self.rai_client = None
        if ai_client:
            self.ml_client = ai_client._ml_client
            self.token_manager = ManagedIdentityAPITokenManager(
                token_scope=TokenScope.DEFAULT_AZURE_MANAGEMENT,
                logger=logging.getLogger("acli token manager"),
            )
            self.rai_client = RAIClient(self.ml_client, self.token_manager)
        self.template_handler = SimulatorTemplates(self.rai_client)

        if self.simulator_connection is None:
            self.adversarial = True

        self.simulate_callback = simulate_callback

    def get_user_proxy_completion_model(self, tkey, tparam):     
        return ProxyChatCompletionsModel(
            name="raisvc_proxy_model",
            template_key=tkey,
            template_parameters=tparam,
            endpoint_url=self.rai_client.simulation_submit_endpoint,
            token_manager=self.token_manager,
            api_version="2023-07-01-preview",
            max_tokens=1200,
            temperature=0.0,
        )

    def _to_openai_chat_completion_model(self, config: "AzureOpenAIModelConfiguration"):
        if config is None:
            return None
        token_manager = PlainTokenManager(
            openapi_key=config.api_key,
            auth_header="api-key",
            logger=logging.getLogger(f"{config.deployment_name}_bot_token_manager"),
        )
        return OpenAIChatCompletionsModel(
            endpoint_url=f"{config.api_base}openai/deployments/{config.deployment_name}/chat/completions",
            token_manager=token_manager,
            api_version=config.api_version,
            name=config.model_name,
            **config.model_kwargs,
        )

    def create_bot(
        self,
        role: ConversationRole,
        conversation_template: str,
        instantiation_parameters: dict,
        adversarial_template_key: str = None,
        model: Union[LLMBase, OpenAIChatCompletionsModel] = None,  # type: ignore[arg-type]
    ):
        if role == ConversationRole.USER and self.adversarial:
            model = self.get_user_proxy_completion_model(
                tkey=adversarial_template_key, tparam=instantiation_parameters
            )

            return ConversationBot(
                role=role,
                model=model,
                conversation_template=conversation_template,
                instantiation_parameters=instantiation_parameters,
            )
        if role == ConversationRole.ASSISTANT and self.simulate_callback:
            dummy_model = lambda: None
            dummy_model.name = "dummy_model"
            return CallbackConversationBot(
                callback=self.simulate_callback,
                role=role,
                model=dummy_model,
                user_template=conversation_template,
                user_template_parameters=instantiation_parameters,
                conversation_template="",
                instantiation_parameters={},
        )

        return ConversationBot(
            role=role,
            model=model,
            conversation_template=conversation_template,
            instantiation_parameters=instantiation_parameters,
        )

    def setup_bot(
        self, role: Union[str, ConversationRole], template: "Template", parameters: dict
    ):
        if role == ConversationRole.ASSISTANT:
            return self.create_bot(
                role,
                template,
                parameters
            )
        elif role == ConversationRole.USER:
            if self.adversarial:
                return self.create_bot(
                    role, str(template), parameters, template.template_name
                )

            return self.create_bot(
                role, str(template), parameters, model=self.simulator_connection
            )

    def _ensure_service_dependencies(self):
        if self.rai_client is None:
            raise ValueError("Simulation options require rai services but ai client is not provided.")

    def _join_conversation_starter(self, parameters, to_join):
        ckey = "conversation_starter"
        if ckey in parameters.keys():
            parameters[ckey] = f"{to_join} {parameters[ckey]}"
        else:
            parameters[ckey] = to_join

        return parameters

    async def simulate_async(
        self,
        template: "Template",
        max_conversation_turns: int,
        parameters: list[dict] = [],
        jailbreak: bool = False,
        api_call_retry_max_count: int = 3,
        api_call_retry_sleep_sec: int = 1,
        api_call_delay_sec: float = 0,
        concurrent_async_task: int = 3,
        simulation_result_limit: int = 3
    ):
        if template.content_harm:
            self._ensure_service_dependencies()
            templates = await self.template_handler._get_ch_template_collections(template.template_name)
        else:
            template.template_parameters = parameters
            templates = [template]

        semaphore = asyncio.Semaphore(concurrent_async_task)
        sim_results = []
        tasks = []

        for t in templates:
            for p in t.template_parameters:
                if jailbreak:
                    self._ensure_service_dependencies()
                    jailbreak_dataset = await self.rai_client.get_jailbreaks_dataset()
                    p = self._join_conversation_starter(p, random.choice(jailbreak_dataset))

                tasks.append(
                    asyncio.create_task(
                        self._simulate_async(
                        template=template,
                        parameters=p,
                        max_conversation_turns=max_conversation_turns,
                        api_call_retry_max_count=api_call_retry_max_count,
                        api_call_delay_sec=api_call_delay_sec,
                        sem=semaphore
                        )
                    )
                )

                if len(tasks) >= simulation_result_limit:
                    break

            if len(tasks) >= simulation_result_limit:
                break
        
        sim_results = await asyncio.gather(*tasks)
        
        return JsonLineList(sim_results)


    async def _simulate_async(
        self,
        template: "Template",
        max_conversation_turns: int,
        parameters: dict = {},
        api_call_retry_max_count: int = 3,
        api_call_retry_sleep_sec: int = 1,
        api_call_delay_sec: float = 0,
        sem: "asyncio.Semaphore" = asyncio.Semaphore(3)
    ):
        # create user bot
        user_bot = self.setup_bot(ConversationRole.USER, template, parameters)
        system_bot = self.setup_bot(
            ConversationRole.ASSISTANT, template, parameters
        )

        bots = [user_bot, system_bot]
            
        # simulate the conversation

        asyncHttpClient = AsyncHTTPClientWithRetry(
            n_retry=api_call_retry_max_count,
            retry_timeout=api_call_retry_sleep_sec,
            logger=logging.getLogger(),
        )
        async with sem:
            async with asyncHttpClient.client as session:
                conversation_id, conversation_history = await simulate_conversation(
                    bots=bots,
                    session=session,
                    turn_limit=max_conversation_turns,
                    api_call_delay_sec=api_call_delay_sec,
                )

        return self._to_chat_protocol(template, conversation_history, parameters)

    def _get_citations(self, parameters, context_keys, turn_num=None):
        citations = []
        for c_key in context_keys:
            if isinstance(parameters[c_key], dict):
                if "callback_citation_key" in parameters[c_key]:
                    callback_citation_key = parameters[c_key]["callback_citation_key"]
                    callback_citations = self._get_callback_citations(
                        parameters[c_key][callback_citation_key], turn_num
                    )
                else:
                    callback_citations = []
                if callback_citations:
                    citations.extend(callback_citations)
                else:
                    for k, v in parameters[c_key].items():
                        if k not in ["callback_citations", "callback_citation_key"]:
                            citations.append(
                                {"id": k, "content": self._to_citation_content(v)}
                            )
            else:
                citations.append(
                    {
                        "id": c_key,
                        "content": self._to_citation_content(parameters[c_key]),
                    }
                )

        return {"citations": citations}

    def _to_citation_content(self, obj):
        if isinstance(obj, str):
            return obj
        else:
            return json.dumps(obj)

    def _get_callback_citations(
        self, callback_citations: dict, turn_num: Optional[int] = None
    ):
        if turn_num == None:
            return []
        current_turn_citations = []
        current_turn_str = "turn_" + str(turn_num)
        if current_turn_str in callback_citations.keys():
            citations = callback_citations[current_turn_str]
            if isinstance(citations, dict):
                for k, v in citations.items():
                    current_turn_citations.append(
                        {"id": k, "content": self._to_citation_content(v)}
                    )
            else:
                current_turn_citations.append(
                    {
                        "id": current_turn_str,
                        "content": self._to_citation_content(citations),
                    }
                )
        return current_turn_citations

    def _to_chat_protocol(self, template, conversation_history, template_parameters):
        messages = []

        for i, m in enumerate(conversation_history):
            message = {
                    "content": m.message,
                    "role": m.role.value
            }
            if len(template.context_key) > 0:
                citations = self._get_citations(
                    template_parameters, template.context_key, i
                )
                message["context"] = citations
            messages.append(message)

        return {
            "template_parameters": template_parameters,
            "messages": messages,
            "$schema": "http://azureml/sdk-2-0/ChatConversation.json",
        }

    def wrap_async(
        self,
        results,
        template: "Template",
        max_conversation_turns: int,
        parameters: dict = {},
        jailbreak: bool = False,
        api_call_retry_max_count: int = 3,
        api_call_retry_sleep_sec: int = 1,
        api_call_delay_sec: float = 0,
        concurrent_async_task: int = 1
    ):
        result = asyncio.run(
            self.simulate_async(
                template=template,
                parameters=parameters,
                max_conversation_turns=max_conversation_turns,
                jailbreak=jailbreak,
                api_call_retry_max_count=api_call_retry_max_count,
                api_call_retry_sleep_sec=api_call_retry_sleep_sec,
                api_call_delay_sec=api_call_delay_sec,
                concurrent_async_task=concurrent_async_task
            )
        )
        results[0] = result

    def simulate(
        self,
        template: "Template",
        max_conversation_turns: int,
        parameters: List[dict] = [],
        jailbreak: bool = False,
        api_call_retry_max_count: int = 3,
        api_call_retry_sleep_sec: int = 1,
        api_call_delay_sec: float = 0,
    ):
        results = [None]
        concurrent_async_task = 1

        thread = threading.Thread(
            target=self.wrap_async,
            args=(
                results,
                template,
                max_conversation_turns,
                parameters,
                jailbreak,
                api_call_retry_max_count,
                api_call_retry_sleep_sec,
                api_call_delay_sec,
                concurrent_async_task
            ),
        )

        thread.start()
        thread.join()

        return results[0]

    @staticmethod
    def from_fn(fn: Callable[[Any], dict], simulator_connection=None, ai_client=None, **kwargs):
        if hasattr(fn, "__wrapped__"):
            func_module = fn.__wrapped__.__module__
            func_name = fn.__wrapped__.__name__
            if func_module == "openai.resources.chat.completions" and func_name == "create":
                return Simulator._from_openai_chat_completions(fn, simulator_connection, ai_client, **kwargs)
    
        return Simulator(
            simulator_connection = simulator_connection,
            ai_client = ai_client,
            simulate_callback = fn
        )

    @staticmethod
    def _from_openai_chat_completions(fn: Callable[[Any], dict], simulator_connection=None, ai_client=None, **kwargs):
        return Simulator(
            simulator_connection = simulator_connection,
            ai_client = ai_client,
            simulate_callback = Simulator._wrap_openai_chat_completion(fn, **kwargs)
        )

    @staticmethod
    def _wrap_openai_chat_completion(fn, **kwargs):
        async def callback(chat_protocol_message):
            response = await fn(
                messages=chat_protocol_message["messages"],
                **kwargs
            )

            message = response.choices[0].message

            formatted_response = {  
                "role": message.role,  
                "content": message.content
            }

            chat_protocol_message["messages"].append(formatted_response)

            return chat_protocol_message
        return callback

    @staticmethod
    def from_pf_path(pf_path, simulator_connection=None, ai_client=None, **kwargs):
        try:
            from promptflow import load_flow
        except:
            raise EnvironmentError("Unable to import from promptflow. Have you installed promptflow in the python environment?")
        flow = load_flow(pf_path)
        return Simulator(
            simulator_connection = simulator_connection,
            ai_client = ai_client,
            simulate_callback = Simulator._wrap_pf(flow)
        )

    @staticmethod
    def _wrap_pf(flow):
        flow_ex = flow._init_executable()
        for k, v in flow_ex.inputs.items():
            if v.is_chat_history:
                chat_history_key = k
                if v.type.value != "list":
                    raise TypeError(f"Chat history {k} not a list.")
            
            if v.is_chat_input:
                chat_input_key = k
                if v.type.value != "string":
                    raise TypeError(f"Chat input {k} not a string.")

        for k, v in flow_ex.outputs.items():
            if v.is_chat_output:
                chat_output_key = k
                if v.type.value != "string":
                    raise TypeError(f"Chat output {k} not a string.")

        if chat_output_key is None or chat_input_key is None:
            raise ValueError("Prompflow has no required chat input and/or chat output.")

        async def callback(chat_protocol_message):
            all_messages = chat_protocol_message["messages"]
            input_data = {
                chat_input_key: all_messages[-1]
            }
            if chat_history_key:
                input_data[chat_history_key] = all_messages

            response = flow.invoke(input_data).output
            chat_protocol_message["messages"].append(
                {
                    "role": "assistant",
                    "content": response[chat_output_key]
                }
            )
            
            return chat_protocol_message
        return callback

    @staticmethod
    def create_template(name: [str], template: Optional[str], template_path: Optional[str], context_key: Optional[list[str]]):
        if (template is None and template_path is None) or (template is not None and template_path is not None):
            raise ValueError("One and only one of the parameters [template, template_path] needs to be set.")
    
        if template is not None:
            return Template(
                template_name=name,
                text=template,
                context_key=context_key
            )

        if template_path is not None:
            with open(template_path, "r") as f:
                tc = f.read()
            
            return Template(
                template_name=name,
                text=tc,
                context_key=context_key
            )

        raise ValueError("Condition not met for creating template, please check examples and parameter list.")

    @staticmethod
    def get_template(template_name):
        st = SimulatorTemplates()
        return st.get_template(template_name)