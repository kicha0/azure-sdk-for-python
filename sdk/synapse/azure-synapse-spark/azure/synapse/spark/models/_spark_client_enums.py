# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from enum import Enum
from azure.core import CaseInsensitiveEnumMeta


class LivyStatementStates(str, Enum, metaclass=CaseInsensitiveEnumMeta):

    WAITING = "waiting"
    RUNNING = "running"
    AVAILABLE = "available"
    ERROR = "error"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"

class LivyStates(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The batch state
    """

    NOT_STARTED = "not_started"
    STARTING = "starting"
    IDLE = "idle"
    BUSY = "busy"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"
    DEAD = "dead"
    KILLED = "killed"
    SUCCESS = "success"
    RUNNING = "running"
    RECOVERING = "recovering"

class PluginCurrentState(str, Enum, metaclass=CaseInsensitiveEnumMeta):

    PREPARATION = "Preparation"
    RESOURCE_ACQUISITION = "ResourceAcquisition"
    QUEUED = "Queued"
    SUBMISSION = "Submission"
    MONITORING = "Monitoring"
    CLEANUP = "Cleanup"
    ENDED = "Ended"

class SchedulerCurrentState(str, Enum, metaclass=CaseInsensitiveEnumMeta):

    QUEUED = "Queued"
    SCHEDULED = "Scheduled"
    ENDED = "Ended"

class SparkBatchJobResultType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The Spark batch job result.
    """

    UNCERTAIN = "Uncertain"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    CANCELLED = "Cancelled"

class SparkErrorSource(str, Enum, metaclass=CaseInsensitiveEnumMeta):

    SYSTEM = "System"
    USER = "User"
    UNKNOWN = "Unknown"
    DEPENDENCY = "Dependency"

class SparkJobType(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """The job type.
    """

    SPARK_BATCH = "SparkBatch"
    SPARK_SESSION = "SparkSession"

class SparkSessionResultType(str, Enum, metaclass=CaseInsensitiveEnumMeta):

    UNCERTAIN = "Uncertain"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    CANCELLED = "Cancelled"

class SparkStatementLanguageType(str, Enum, metaclass=CaseInsensitiveEnumMeta):

    SPARK = "spark"
    PY_SPARK = "pyspark"
    DOT_NET_SPARK = "dotnetspark"
    SQL = "sql"
