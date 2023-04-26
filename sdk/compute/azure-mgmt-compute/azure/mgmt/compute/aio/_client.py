# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is
# regenerated.
# --------------------------------------------------------------------------

from typing import Any, Optional, TYPE_CHECKING

from azure.mgmt.core import AsyncARMPipelineClient
from azure.profiles import KnownProfiles, ProfileDefinition
from azure.profiles.multiapiclient import MultiApiClientMixin

from .._serialization import Deserializer, Serializer
from ._configuration import ComputeManagementClientConfiguration

if TYPE_CHECKING:
    # pylint: disable=unused-import,ungrouped-imports
    from azure.core.credentials_async import AsyncTokenCredential


from .operations import (
    AvailabilitySetsOperations,
    UsageOperations,
    VirtualMachineExtensionImagesOperations,
    VirtualMachineExtensionsOperations,
    VirtualMachineImagesOperations,
    VirtualMachineScaleSetVMsOperations,
    VirtualMachineScaleSetsOperations,
    VirtualMachineSizesOperations,
    VirtualMachinesOperations,
    DisksOperations,
    ImagesOperations,
    SnapshotsOperations,
    ResourceSkusOperations,
    VirtualMachineRunCommandsOperations,
    VirtualMachineScaleSetExtensionsOperations,
    VirtualMachineScaleSetRollingUpgradesOperations,
    LogAnalyticsOperations,
    Operations,
    ProximityPlacementGroupsOperations,
    GalleriesOperations,
    GalleryImageVersionsOperations,
    GalleryImagesOperations,
    DedicatedHostGroupsOperations,
    DedicatedHostsOperations,
    GalleryApplicationVersionsOperations,
    GalleryApplicationsOperations,
    DiskEncryptionSetsOperations,
    VirtualMachineScaleSetVMExtensionsOperations,
    SshPublicKeysOperations,
    DiskAccessesOperations,
    VirtualMachineScaleSetVMRunCommandsOperations,
    DiskRestorePointOperations,
    GallerySharingProfileOperations,
    SharedGalleriesOperations,
    SharedGalleryImageVersionsOperations,
    SharedGalleryImagesOperations,
    CloudServiceRoleInstancesOperations,
    CloudServiceRolesOperations,
    CloudServicesOperations,
    CloudServicesUpdateDomainOperations,
    VirtualMachineImagesEdgeZoneOperations,
    CloudServiceOperatingSystemsOperations,
    RestorePointCollectionsOperations,
    RestorePointsOperations,
    CapacityReservationGroupsOperations,
    CapacityReservationsOperations,
    CommunityGalleriesOperations,
    CommunityGalleryImageVersionsOperations,
    CommunityGalleryImagesOperations,
)
from .._validation import api_version_validation
from .. import models
class _SDKClient(object):
    def __init__(self, *args, **kwargs):
        """This is a fake class to support current implemetation of MultiApiClientMixin."
        Will be removed in final version of multiapi azure-core based client
        """
        pass

class ComputeManagementClient(MultiApiClientMixin, _SDKClient):
    """Compute Client.

    This ready contains multiple API versions, to help you deal with all of the Azure clouds
    (Azure Stack, Azure Government, Azure China, etc.).
    By default, it uses the latest API version available on public Azure.
    For production, you should stick to a particular api-version and/or profile.
    The profile sets a mapping between an operation group and its API version.
    The api-version parameter sets the default API version if the operation
    group is not described in the profile.

    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials_async.AsyncTokenCredential
    :param subscription_id: Subscription credentials which uniquely identify Microsoft Azure subscription. The subscription ID forms part of the URI for every service call. Required.
    :type subscription_id: str
    :param api_version: API version to use if no profile is provided, or if missing in profile.
    :type api_version: str
    :param base_url: Service URL
    :type base_url: str
    :param profile: A profile definition, from KnownProfiles to dict.
    :type profile: azure.profiles.KnownProfiles
    :keyword int polling_interval: Default waiting time between two polls for LRO operations if no Retry-After header is present.
    """

    DEFAULT_API_VERSION = '2022-11-01'
    _PROFILE_TAG = "azure.mgmt.compute.ComputeManagementClient"
    LATEST_PROFILE = ProfileDefinition({
        _PROFILE_TAG: {
            None: DEFAULT_API_VERSION,
            'cloud_service_operating_systems': '2022-09-04',
            'cloud_service_role_instances': '2022-09-04',
            'cloud_service_roles': '2022-09-04',
            'cloud_services': '2022-09-04',
            'cloud_services_update_domain': '2022-09-04',
            'community_galleries': '2022-03-03',
            'community_gallery_image_versions': '2022-03-03',
            'community_gallery_images': '2022-03-03',
            'disk_accesses': '2022-07-02',
            'disk_encryption_sets': '2022-07-02',
            'disk_restore_point': '2022-07-02',
            'disks': '2022-07-02',
            'galleries': '2022-03-03',
            'gallery_application_versions': '2022-03-03',
            'gallery_applications': '2022-03-03',
            'gallery_image_versions': '2022-03-03',
            'gallery_images': '2022-03-03',
            'gallery_sharing_profile': '2022-03-03',
            'resource_skus': '2021-07-01',
            'shared_galleries': '2022-03-03',
            'shared_gallery_image_versions': '2022-03-03',
            'shared_gallery_images': '2022-03-03',
            'snapshots': '2022-07-02',
        }},
        _PROFILE_TAG + " latest"
    )

    def __init__(
        self,
        credential: "AsyncTokenCredential",
        subscription_id: str,
        api_version: Optional[str] = None,
        base_url: str = "https://management.azure.com",
        profile: KnownProfiles = KnownProfiles.default,
        **kwargs: Any
    ) -> None:
        self._config = ComputeManagementClientConfiguration(credential, subscription_id, **kwargs)
        self._client = AsyncARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        super(ComputeManagementClient, self).__init__(
            api_version=api_version,
            profile=profile
        )



    @classmethod
    def _models_dict(cls):
        return {k: v for k, v in models.__dict__.items() if isinstance(v, type)}


    @property
    @api_version_validation(
        api_versions=['2015-06-15', '2016-03-30', '2016-04-30-preview', '2017-03-30', '2017-12-01', '2018-04-01', '2018-06-01', '2018-10-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def availability_sets(self):
        api_version = self._get_api_version("availability_sets")
        return AvailabilitySetsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2015-06-15', '2016-03-30', '2016-04-30-preview', '2017-03-30', '2017-12-01', '2018-04-01', '2018-06-01', '2018-10-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def usage(self):
        api_version = self._get_api_version("usage")
        return UsageOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2015-06-15', '2016-03-30', '2016-04-30-preview', '2017-03-30', '2017-12-01', '2018-04-01', '2018-06-01', '2018-10-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def virtual_machine_extension_images(self):
        api_version = self._get_api_version("virtual_machine_extension_images")
        return VirtualMachineExtensionImagesOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2015-06-15', '2016-03-30', '2016-04-30-preview', '2017-03-30', '2017-12-01', '2018-04-01', '2018-06-01', '2018-10-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def virtual_machine_extensions(self):
        api_version = self._get_api_version("virtual_machine_extensions")
        return VirtualMachineExtensionsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2015-06-15', '2016-03-30', '2016-04-30-preview', '2017-03-30', '2017-12-01', '2018-04-01', '2018-06-01', '2018-10-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def virtual_machine_images(self):
        api_version = self._get_api_version("virtual_machine_images")
        return VirtualMachineImagesOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2015-06-15', '2016-03-30', '2016-04-30-preview', '2017-03-30', '2017-12-01', '2018-04-01', '2018-06-01', '2018-10-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def virtual_machine_scale_set_vms(self):
        api_version = self._get_api_version("virtual_machine_scale_set_vms")
        return VirtualMachineScaleSetVMsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2015-06-15', '2016-03-30', '2016-04-30-preview', '2017-03-30', '2017-12-01', '2018-04-01', '2018-06-01', '2018-10-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def virtual_machine_scale_sets(self):
        api_version = self._get_api_version("virtual_machine_scale_sets")
        return VirtualMachineScaleSetsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2015-06-15', '2016-03-30', '2016-04-30-preview', '2017-03-30', '2017-12-01', '2018-04-01', '2018-06-01', '2018-10-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def virtual_machine_sizes(self):
        api_version = self._get_api_version("virtual_machine_sizes")
        return VirtualMachineSizesOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2015-06-15', '2016-03-30', '2016-04-30-preview', '2017-03-30', '2017-12-01', '2018-04-01', '2018-06-01', '2018-10-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def virtual_machines(self):
        api_version = self._get_api_version("virtual_machines")
        return VirtualMachinesOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2016-04-30-preview', '2017-03-30', '2018-04-01', '2018-06-01', '2018-09-30', '2019-03-01', '2019-07-01', '2019-11-01', '2020-05-01', '2020-06-30', '2020-09-30', '2020-12-01', '2021-04-01', '2021-08-01', '2021-12-01', '2022-03-02', '2022-07-02']
    )
    def disks(self):
        api_version = self._get_api_version("disks")
        return DisksOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2016-04-30-preview', '2017-03-30', '2017-12-01', '2018-04-01', '2018-06-01', '2018-10-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def images(self):
        api_version = self._get_api_version("images")
        return ImagesOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2016-04-30-preview', '2017-03-30', '2018-04-01', '2018-06-01', '2018-09-30', '2019-03-01', '2019-07-01', '2019-11-01', '2020-05-01', '2020-06-30', '2020-09-30', '2020-12-01', '2021-04-01', '2021-08-01', '2021-12-01', '2022-03-02', '2022-07-02']
    )
    def snapshots(self):
        api_version = self._get_api_version("snapshots")
        return SnapshotsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2017-03-30', '2017-09-01', '2019-04-01', '2021-07-01']
    )
    def resource_skus(self):
        api_version = self._get_api_version("resource_skus")
        return ResourceSkusOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2017-03-30', '2017-12-01', '2018-04-01', '2018-06-01', '2018-10-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def virtual_machine_run_commands(self):
        api_version = self._get_api_version("virtual_machine_run_commands")
        return VirtualMachineRunCommandsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2017-03-30', '2017-12-01', '2018-04-01', '2018-06-01', '2018-10-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def virtual_machine_scale_set_extensions(self):
        api_version = self._get_api_version("virtual_machine_scale_set_extensions")
        return VirtualMachineScaleSetExtensionsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2017-03-30', '2017-12-01', '2018-04-01', '2018-06-01', '2018-10-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def virtual_machine_scale_set_rolling_upgrades(self):
        api_version = self._get_api_version("virtual_machine_scale_set_rolling_upgrades")
        return VirtualMachineScaleSetRollingUpgradesOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2017-12-01', '2018-04-01', '2018-06-01', '2018-10-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def log_analytics(self):
        api_version = self._get_api_version("log_analytics")
        return LogAnalyticsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2017-12-01', '2018-04-01', '2018-06-01', '2018-10-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def operations(self):
        api_version = self._get_api_version("operations")
        return Operations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2018-04-01', '2018-06-01', '2018-10-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def proximity_placement_groups(self):
        api_version = self._get_api_version("proximity_placement_groups")
        return ProximityPlacementGroupsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2018-06-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-09-30', '2021-07-01', '2021-10-01', '2022-01-03', '2022-03-03']
    )
    def galleries(self):
        api_version = self._get_api_version("galleries")
        return GalleriesOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2018-06-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-09-30', '2021-07-01', '2021-10-01', '2022-01-03', '2022-03-03']
    )
    def gallery_image_versions(self):
        api_version = self._get_api_version("gallery_image_versions")
        return GalleryImageVersionsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2018-06-01', '2019-03-01', '2019-07-01', '2019-12-01', '2020-09-30', '2021-07-01', '2021-10-01', '2022-01-03', '2022-03-03']
    )
    def gallery_images(self):
        api_version = self._get_api_version("gallery_images")
        return GalleryImagesOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def dedicated_host_groups(self):
        api_version = self._get_api_version("dedicated_host_groups")
        return DedicatedHostGroupsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2019-03-01', '2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def dedicated_hosts(self):
        api_version = self._get_api_version("dedicated_hosts")
        return DedicatedHostsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2019-03-01', '2019-07-01', '2019-12-01', '2020-09-30', '2021-07-01', '2021-10-01', '2022-01-03', '2022-03-03']
    )
    def gallery_application_versions(self):
        api_version = self._get_api_version("gallery_application_versions")
        return GalleryApplicationVersionsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2019-03-01', '2019-07-01', '2019-12-01', '2020-09-30', '2021-07-01', '2021-10-01', '2022-01-03', '2022-03-03']
    )
    def gallery_applications(self):
        api_version = self._get_api_version("gallery_applications")
        return GalleryApplicationsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2019-07-01', '2019-11-01', '2020-05-01', '2020-06-30', '2020-09-30', '2020-12-01', '2021-04-01', '2021-08-01', '2021-12-01', '2022-03-02', '2022-07-02']
    )
    def disk_encryption_sets(self):
        api_version = self._get_api_version("disk_encryption_sets")
        return DiskEncryptionSetsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2019-07-01', '2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def virtual_machine_scale_set_vm_extensions(self):
        api_version = self._get_api_version("virtual_machine_scale_set_vm_extensions")
        return VirtualMachineScaleSetVMExtensionsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2019-12-01', '2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def ssh_public_keys(self):
        api_version = self._get_api_version("ssh_public_keys")
        return SshPublicKeysOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2020-05-01', '2020-06-30', '2020-09-30', '2020-12-01', '2021-04-01', '2021-08-01', '2021-12-01', '2022-03-02', '2022-07-02']
    )
    def disk_accesses(self):
        api_version = self._get_api_version("disk_accesses")
        return DiskAccessesOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2020-06-01', '2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def virtual_machine_scale_set_vm_run_commands(self):
        api_version = self._get_api_version("virtual_machine_scale_set_vm_run_commands")
        return VirtualMachineScaleSetVMRunCommandsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2020-09-30', '2020-12-01', '2021-04-01', '2021-08-01', '2021-12-01', '2022-03-02', '2022-07-02']
    )
    def disk_restore_point(self):
        api_version = self._get_api_version("disk_restore_point")
        return DiskRestorePointOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2020-09-30', '2021-07-01', '2021-10-01', '2022-01-03', '2022-03-03']
    )
    def gallery_sharing_profile(self):
        api_version = self._get_api_version("gallery_sharing_profile")
        return GallerySharingProfileOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2020-09-30', '2021-07-01', '2022-01-03', '2022-03-03']
    )
    def shared_galleries(self):
        api_version = self._get_api_version("shared_galleries")
        return SharedGalleriesOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2020-09-30', '2021-07-01', '2022-01-03', '2022-03-03']
    )
    def shared_gallery_image_versions(self):
        api_version = self._get_api_version("shared_gallery_image_versions")
        return SharedGalleryImageVersionsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2020-09-30', '2021-07-01', '2022-01-03', '2022-03-03']
    )
    def shared_gallery_images(self):
        api_version = self._get_api_version("shared_gallery_images")
        return SharedGalleryImagesOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2020-10-01-preview', '2021-03-01', '2022-04-04', '2022-09-04']
    )
    def cloud_service_role_instances(self):
        api_version = self._get_api_version("cloud_service_role_instances")
        return CloudServiceRoleInstancesOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2020-10-01-preview', '2021-03-01', '2022-04-04', '2022-09-04']
    )
    def cloud_service_roles(self):
        api_version = self._get_api_version("cloud_service_roles")
        return CloudServiceRolesOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2020-10-01-preview', '2021-03-01', '2022-04-04', '2022-09-04']
    )
    def cloud_services(self):
        api_version = self._get_api_version("cloud_services")
        return CloudServicesOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2020-10-01-preview', '2021-03-01', '2022-04-04', '2022-09-04']
    )
    def cloud_services_update_domain(self):
        api_version = self._get_api_version("cloud_services_update_domain")
        return CloudServicesUpdateDomainOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2020-12-01', '2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def virtual_machine_images_edge_zone(self):
        api_version = self._get_api_version("virtual_machine_images_edge_zone")
        return VirtualMachineImagesEdgeZoneOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2021-03-01', '2022-04-04', '2022-09-04']
    )
    def cloud_service_operating_systems(self):
        api_version = self._get_api_version("cloud_service_operating_systems")
        return CloudServiceOperatingSystemsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def restore_point_collections(self):
        api_version = self._get_api_version("restore_point_collections")
        return RestorePointCollectionsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2021-03-01', '2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def restore_points(self):
        api_version = self._get_api_version("restore_points")
        return RestorePointsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def capacity_reservation_groups(self):
        api_version = self._get_api_version("capacity_reservation_groups")
        return CapacityReservationGroupsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2021-04-01', '2021-07-01', '2021-11-01', '2022-03-01', '2022-08-01', '2022-11-01']
    )
    def capacity_reservations(self):
        api_version = self._get_api_version("capacity_reservations")
        return CapacityReservationsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2021-07-01', '2022-01-03', '2022-03-03']
    )
    def community_galleries(self):
        api_version = self._get_api_version("community_galleries")
        return CommunityGalleriesOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2021-07-01', '2022-01-03', '2022-03-03']
    )
    def community_gallery_image_versions(self):
        api_version = self._get_api_version("community_gallery_image_versions")
        return CommunityGalleryImageVersionsOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    @property
    @api_version_validation(
        api_versions=['2021-07-01', '2022-01-03', '2022-03-03']
    )
    def community_gallery_images(self):
        api_version = self._get_api_version("community_gallery_images")
        return CommunityGalleryImagesOperations(
            self._client,
            self._config,
            Serializer(self._models_dict()),
            Deserializer(self._models_dict()),
            api_version=api_version,
        )

    async def close(self):
        await self._client.close()
    async def __aenter__(self):
        await self._client.__aenter__()
        return self
    async def __aexit__(self, *exc_details):
        await self._client.__aexit__(*exc_details)
