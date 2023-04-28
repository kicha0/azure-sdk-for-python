# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from ._app_service_certificate_orders_operations import AppServiceCertificateOrdersOperations
from ._certificate_orders_diagnostics_operations import CertificateOrdersDiagnosticsOperations
from ._certificate_registration_provider_operations import CertificateRegistrationProviderOperations
from ._domains_operations import DomainsOperations
from ._top_level_domains_operations import TopLevelDomainsOperations
from ._domain_registration_provider_operations import DomainRegistrationProviderOperations
from ._app_service_environments_operations import AppServiceEnvironmentsOperations
from ._app_service_plans_operations import AppServicePlansOperations
from ._certificates_operations import CertificatesOperations
from ._container_apps_operations import ContainerAppsOperations
from ._container_apps_revisions_operations import ContainerAppsRevisionsOperations
from ._deleted_web_apps_operations import DeletedWebAppsOperations
from ._diagnostics_operations import DiagnosticsOperations
from ._global_operations_operations import GlobalOperations
from ._kube_environments_operations import KubeEnvironmentsOperations
from ._provider_operations import ProviderOperations
from ._recommendations_operations import RecommendationsOperations
from ._resource_health_metadata_operations import ResourceHealthMetadataOperations
from ._web_site_management_client_operations import WebSiteManagementClientOperationsMixin
from ._static_sites_operations import StaticSitesOperations
from ._web_apps_operations import WebAppsOperations
from ._workflows_operations import WorkflowsOperations
from ._workflow_runs_operations import WorkflowRunsOperations
from ._workflow_run_actions_operations import WorkflowRunActionsOperations
from ._workflow_run_action_repetitions_operations import WorkflowRunActionRepetitionsOperations
from ._workflow_run_action_repetitions_request_histories_operations import (
    WorkflowRunActionRepetitionsRequestHistoriesOperations,
)
from ._workflow_run_action_scope_repetitions_operations import WorkflowRunActionScopeRepetitionsOperations
from ._workflow_triggers_operations import WorkflowTriggersOperations
from ._workflow_trigger_histories_operations import WorkflowTriggerHistoriesOperations
from ._workflow_versions_operations import WorkflowVersionsOperations

from ._patch import __all__ as _patch_all
from ._patch import *  # pylint: disable=unused-wildcard-import
from ._patch import patch_sdk as _patch_sdk

__all__ = [
    "AppServiceCertificateOrdersOperations",
    "CertificateOrdersDiagnosticsOperations",
    "CertificateRegistrationProviderOperations",
    "DomainsOperations",
    "TopLevelDomainsOperations",
    "DomainRegistrationProviderOperations",
    "AppServiceEnvironmentsOperations",
    "AppServicePlansOperations",
    "CertificatesOperations",
    "ContainerAppsOperations",
    "ContainerAppsRevisionsOperations",
    "DeletedWebAppsOperations",
    "DiagnosticsOperations",
    "GlobalOperations",
    "KubeEnvironmentsOperations",
    "ProviderOperations",
    "RecommendationsOperations",
    "ResourceHealthMetadataOperations",
    "WebSiteManagementClientOperationsMixin",
    "StaticSitesOperations",
    "WebAppsOperations",
    "WorkflowsOperations",
    "WorkflowRunsOperations",
    "WorkflowRunActionsOperations",
    "WorkflowRunActionRepetitionsOperations",
    "WorkflowRunActionRepetitionsRequestHistoriesOperations",
    "WorkflowRunActionScopeRepetitionsOperations",
    "WorkflowTriggersOperations",
    "WorkflowTriggerHistoriesOperations",
    "WorkflowVersionsOperations",
]
__all__.extend([p for p in _patch_all if p not in __all__])
_patch_sdk()
