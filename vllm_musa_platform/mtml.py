# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MTML (Moore Threads Management Library) wrapper for vLLM.

This module provides a wrapper around pymtml to access MUSA device information
without initializing the MUSA runtime context.
"""

from typing import Any

# Try to import pymtml
try:
    import pymtml
    MTML_AVAILABLE = True
except ImportError:
    MTML_AVAILABLE = False
    pymtml = None  # type: ignore[assignment]

# Type alias for device handle (opaque type from pymtml)
DeviceHandle = Any

__all__ = [
    # Availability
    "MTML_AVAILABLE",
    "is_mtml_available",
    # Exception
    "MTMLError",
    # Lifecycle
    "mtmlInit",
    "mtmlShutdown",
    # Device queries
    "mtmlDeviceGetCount",
    "mtmlDeviceGetHandleByIndex",
    "mtmlDeviceGetName",
    "mtmlDeviceGetUUID",
    "mtmlDeviceGetMemoryInfo",
    # MtLink
    "mtmlDeviceGetMtLinkSpec",
    "mtmlDeviceGetMtLinkState",
    "mtmlDeviceGetMtLinkRemoteDevice",
    # Constants
    "MTML_MTLINK_STATE_DOWN",
    "MTML_MTLINK_STATE_UP",
    "MTML_MTLINK_STATE_DOWNGRADE",
]


class MTMLError(Exception):
    """Exception raised for MTML errors."""
    pass


def mtmlInit() -> None:
    """Initialize the MTML library.

    Must be called before any other MTML functions.
    Can be called multiple times; use reference counting for nested calls.

    Raises:
        MTMLError: If pymtml is not available.
    """
    if not MTML_AVAILABLE:
        raise MTMLError("pymtml is not available")
    pymtml.mtmlLibraryInit()


def mtmlShutdown() -> None:
    """Shutdown the MTML library.

    Should be called when done with MTML operations.

    Raises:
        MTMLError: If pymtml is not available.
    """
    if not MTML_AVAILABLE:
        raise MTMLError("pymtml is not available")
    pymtml.mtmlLibraryShutDown()


def mtmlDeviceGetCount() -> int:
    """Get the number of MUSA devices.

    Returns:
        The number of MUSA devices available.

    Raises:
        MTMLError: If pymtml is not available.
    """
    if not MTML_AVAILABLE:
        raise MTMLError("pymtml is not available")
    return pymtml.mtmlLibraryCountDevice()


def mtmlDeviceGetHandleByIndex(index: int) -> DeviceHandle:
    """Get a device handle by index.

    Args:
        index: The device index (0-based).

    Returns:
        An opaque device handle for use with other MTML functions.

    Raises:
        MTMLError: If pymtml is not available.
    """
    if not MTML_AVAILABLE:
        raise MTMLError("pymtml is not available")
    return pymtml.mtmlLibraryInitDeviceByIndex(index)


def mtmlDeviceGetName(handle: DeviceHandle) -> str:
    """Get the name of a device.

    Args:
        handle: Device handle from mtmlDeviceGetHandleByIndex.

    Returns:
        The device name (e.g., "MTT S80", "MTT S5000").

    Raises:
        MTMLError: If pymtml is not available.
    """
    if not MTML_AVAILABLE:
        raise MTMLError("pymtml is not available")
    return pymtml.mtmlDeviceGetName(handle)


def mtmlDeviceGetUUID(handle: DeviceHandle) -> str:
    """Get the UUID of a device.

    Args:
        handle: Device handle from mtmlDeviceGetHandleByIndex.

    Returns:
        The device UUID string.

    Raises:
        MTMLError: If pymtml is not available.
    """
    if not MTML_AVAILABLE:
        raise MTMLError("pymtml is not available")
    return pymtml.mtmlDeviceGetUUID(handle)


def mtmlDeviceGetMemoryInfo(handle: DeviceHandle) -> int:
    """Get the total memory of a device in bytes.

    Args:
        handle: Device handle from mtmlDeviceGetHandleByIndex.

    Returns:
        Total device memory in bytes.

    Raises:
        MTMLError: If pymtml is not available.
    """
    if not MTML_AVAILABLE:
        raise MTMLError("pymtml is not available")
    memory = pymtml.mtmlDeviceInitMemory(handle)
    return pymtml.mtmlMemoryGetTotal(memory)


def mtmlDeviceGetMtLinkSpec(handle: DeviceHandle) -> Any:
    """Get the MtLink specification for a device.

    Args:
        handle: Device handle from mtmlDeviceGetHandleByIndex.

    Returns:
        MtLink specification object with linkNum attribute.

    Raises:
        MTMLError: If pymtml is not available.
    """
    if not MTML_AVAILABLE:
        raise MTMLError("pymtml is not available")
    return pymtml.mtmlDeviceGetMtLinkSpec(handle)


def mtmlDeviceGetMtLinkState(handle: DeviceHandle, link_index: int) -> int:
    """Get the MtLink state for a specific link.

    Args:
        handle: Device handle from mtmlDeviceGetHandleByIndex.
        link_index: The link index to query.

    Returns:
        Link state (MTML_MTLINK_STATE_DOWN, MTML_MTLINK_STATE_UP,
        or MTML_MTLINK_STATE_DOWNGRADE).

    Raises:
        MTMLError: If pymtml is not available.
    """
    if not MTML_AVAILABLE:
        raise MTMLError("pymtml is not available")
    return pymtml.mtmlDeviceGetMtLinkState(handle, link_index)


def mtmlDeviceGetMtLinkRemoteDevice(handle: DeviceHandle, link_index: int) -> DeviceHandle:
    """Get the remote device connected via MtLink.

    Args:
        handle: Device handle from mtmlDeviceGetHandleByIndex.
        link_index: The link index to query.

    Returns:
        Device handle of the remote device.

    Raises:
        MTMLError: If pymtml is not available.
    """
    if not MTML_AVAILABLE:
        raise MTMLError("pymtml is not available")
    return pymtml.mtmlDeviceGetMtLinkRemoteDevice(handle, link_index)


# MtLink state constants
MTML_MTLINK_STATE_DOWN = 0
MTML_MTLINK_STATE_UP = 1
MTML_MTLINK_STATE_DOWNGRADE = 2


def is_mtml_available() -> bool:
    """Check if MTML is available.

    Returns:
        True if pymtml is installed and can be imported.
    """
    return MTML_AVAILABLE

