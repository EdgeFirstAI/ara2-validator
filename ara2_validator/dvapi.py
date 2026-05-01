#  Copyright (c) 2018-25, Kinara, Inc. All rights reserved.
#  Kinara  Proprietary. This software is owned or controlled by Kinara and may only be used strictly in accordance with the applicable license terms.
"""
DV inference proxy python APIS

Examples
--------
>>> import numpy as np
>>> from dvapi import *
>>> ret, connection = DVSession.create_via_unix_socket("/var/run/proxy.sock")
>>> with connection as conn:
...     ret, connected_endpoints = conn.get_endpoint_list()
...     ret, loaded_model = conn.load_model_from_file("./my_model.dvm")
...     input_param = loaded_model.input_param[0]
...     my_image_data = np.fromfile("./my_preprocessed_image", dtype=np.int8)
...     input_tensor = DVTensor(my_input, input_param)
...     ret, response = loaded_model.infer_sync(
...         [input_tensor],
...         timeout=50000,
...         endpoint=connected_endpoints.get(0))
"""

from ctypes import *
from enum import IntEnum
from typing import Callable
import logging
import os
import errno
import sys
import functools
import numpy as np
import platform
import mmap
import typing

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger('dvapi')
logger.setLevel(logging.INFO)

dvapi_handler = logging.StreamHandler()
dvapi_formatter = logging.Formatter('[%(levelname)s] - DVAPI: %(message)s')
dvapi_handler.setFormatter(dvapi_formatter)

logger.addHandler(dvapi_handler)
logger.propagate = False

def dv_is_null(pv):
    return (cast(pv, c_void_p).value is None)

class DVClientLibNotFoundError(FileNotFoundError):
    pass

@functools.lru_cache(maxsize=10)
def _dvApiObj():
    DV_API_OBJ = None
    def load_dvinfclient_lib(libpath):
        nonlocal DV_API_OBJ
        try:
            if os.path.exists(libpath):
                DV_API_OBJ = CDLL(libpath)
                logger.info("loaded dvinfclient lib: %s", libpath)
                return True
            else:
                logger.info("failed to load dvinfclient lib: %s", libpath)
                return False
        except:
            logger.info("failed to load dvinfclient lib: %s", libpath)
            return False

    lib_loaded= False
    cwd=os.path.dirname(os.path.abspath(__file__))
    curr_platform = platform.system()
    env=os.environ.get('DV_TGT_ROOT')
    if curr_platform == "Windows":
        if 'DV_TGT_ROOT' in os.environ:
            if os.path.exists(os.path.join(env, "art/windows/x86/client/md_release/araclient.dll")):
                lib_loaded = load_dvinfclient_lib(os.path.join(env, "art/windows/x86/client/md_release/araclient.dll"))
        else:
            lib_loaded = load_dvinfclient_lib(os.path.abspath(os.path.join(cwd,"../client/md_release/araclient.dll")))
    if curr_platform == "Linux":
        os_info = os.uname()
        if "x86_64" in os_info[4]:
            if 'DV_TGT_ROOT' in os.environ:
                if os.path.exists(os.path.join(env, "art/linux/x86/lib/libaraclient_x86_64.so")):
                    lib_loaded = load_dvinfclient_lib(os.path.join(env, "art/linux/x86/lib/libaraclient_x86_64.so"))
            else:
                lib_loaded = load_dvinfclient_lib(os.path.abspath(os.path.join(cwd,"../lib/libaraclient_x86_64.so")))
        else:
            if 'DV_TGT_ROOT' in os.environ:
                if os.path.exists(os.path.join(env, "art/linux/aarch64/lib/libaraclient_aarch64.so")):
                    lib_loaded = load_dvinfclient_lib(os.path.join(env, "art/linux/aarch64/lib/libaraclient_aarch64.so"))
            else:
                lib_loaded = load_dvinfclient_lib(os.path.abspath(os.path.join(cwd,"../lib/libaraclient_aarch64.so")))
      
    # Fallback: try system-installed library via ldconfig/soname lookup.
    # On imx8mp-frdm the library is at /usr/lib/libaraclient_aarch64.so
    # but the SDK-relative paths above won't find it.
    if not lib_loaded:
        import ctypes.util
        soname = None
        if curr_platform == "Linux":
            os_info = os.uname()
            if "x86_64" in os_info[4]:
                soname = ctypes.util.find_library("araclient_x86_64")
            else:
                soname = ctypes.util.find_library("araclient_aarch64")
                # find_library may return None if ldconfig doesn't index it;
                # try well-known system paths directly
                if not soname:
                    for candidate in ["/usr/lib/libaraclient_aarch64.so",
                                      "/usr/local/lib/libaraclient_aarch64.so"]:
                        if os.path.exists(candidate):
                            soname = candidate
                            break
        if soname:
            try:
                DV_API_OBJ = CDLL(soname)
                lib_loaded = True
                logger.info("loaded dvinfclient lib (system fallback): %s", soname)
            except Exception:
                pass

    if not lib_loaded:
        raise DVClientLibNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "unable to load dvinfclient library")
    return DV_API_OBJ


###############################################################################
# dvapi enum wrappers based on ctypes
###############################################################################
class dv_DV_ERROR_CATEGORY(IntEnum):
    DV_ERROR_CATEGORY_SUCCESS                   = 0,
    DV_ERROR_CATEGORY_RETRY                     = 100,
    DV_ERROR_CATEGORY_INVALID                   = 200,
    DV_ERROR_CATEGORY_SW_CLIENT_FATAL           = 300,
    DV_ERROR_CATEGORY_SW_SERVER_FATAL           = 400,
    DV_ERROR_CATEGORY_HW_FATAL                  = 500,


class dv_status_code(IntEnum):
    # DV_ERROR_CATEGORY_SUCCESS
    DV_ERROR_CATEGORY_SUCCESS_START = 0,
    DV_SUCCESS = 0,
    DV_ERROR_CATEGORY_SUCCESS_END = 0,
    # TODO KARTHIK only change name to DV_STATUS_UNKNOWN
    DV_FAILURE_UNKOWN = 1,

    # DV_ERROR_CATEGORY_RETRY
    DV_ERROR_CATEGORY_RETRY_START = 100,
    DV_ENDPOINT_OUT_OF_MEMORY = 100,  # Not enough DDR memory available on endpoint; use
                                        # dv_endpoint_get_dram_statistics to get available memory
                                        # inference queue (Internal to Kinara)
    DV_ERROR_CATEGORY_RETRY_END = 199,

    # DV_ERROR_CATEGORY_INVALID
    DV_ERROR_CATEGORY_INVALID_START = 200,
    DV_RESOURCE_NOT_FOUND = 200,  # Model file given as input not accessible
    DV_INVALID_VALUE = 201,       # Invalid pointer value encountered in proxy (Internal to Kinara)
    DV_INVALID_HOST_PTR = 202,    # Invalid pointer value encountered in clientlib
    DV_INVALID_OPERATION = 203,   # One of the below invalid requests sent
                                    #   1. SHM/SFD request over TCP/IP socket
    #   2. Snapshot dump recieved invalid endpoint ID
    #   3. Proxy received invalid request ID (Internal to Kinara)

    DV_OPERATION_NOT_PERMITTED = 204,            # Reserved (unused currently)
    DV_OPERATION_NOT_SUPPORTED = 205,            # Reserved (unused currently)
    DV_SESSION_UNIX_SOCKET_FILE_TOO_LONG = 220,  # Unix socket file exceeded 108 bytes
    DV_SESSION_INVALID_TCP_IPV4_ADDR = 221,      # Invalid TCP Address
    DV_SESSION_INVALID_TCP_IPV4_PORT = 222,      # Invalid TCP port value
    DV_SESSION_INVALID_HANDLE = 223,             # Request sent over session that does not exist
    DV_ENDPOINT_INVALID_HANDLE = 230,            # Request sent for endpoint list with no active endpoints
    DV_ENDPOINT_INVALID_PARAMS = 231,            # Invalid endpoint param sent for stats collection/fault recovery
    DV_ENDPOINT_NOT_FOUND = 232,                 # Reserved (currently unused)
    DV_ENDPOINT_NOT_AVAILABLE = 233,             # No valid endpoints received in dv_endpoint_get_list
    DV_ENDPOINT_GROUP_INVALID = 234,             # Reserved (unused currently)
    DV_ENDPOINT_POWER_SWITCH_FAILURE = 235,      # Endpoint power switch failed
    DV_ENDPOINT_POWER_GATED = 236,               # Endpoint cannot run inferences since it is in power gated mode
    DV_ENDPOINT_INVALID_POWER_STATE = 237,       # Invalid power state switch requested
    DV_ENDPOINT_GET_LIST_FAILED = 238,           # Generic error if get endpoint list fails.
    DV_ENDPOINT_GET_STATS_FAILED = 239,
    DV_MODEL_INVALID_PARAMS = 240,  # One of the below errors occurred
    #  1. Model context not found/ invalid on device (Internal to
    #  Kinara)
    #  2. Inference request sent on a device with no model loaded
    #  3. Invalid input size sent for model
    DV_MODEL_INVALID_HANDLE = 241,  # One of the below errors occurred
    #  1. Null/invalid model handle passed for inference
    #  2. Invalid model handle passed to dv_model_unload
    DV_MODEL_INVALID_MODEL_FILE = 242,   # Model of size 0 passed
    DV_MODEL_PARSE_FAILURE = 243,        # Invalid file format of model.dvm
    DV_MODEL_UNSUPPORTED_VERSION = 244,  # Deprecated model.dvm file provided
    DV_MODEL_CACHING_FAILURE = 245,      # One of the below errors occurred
    #  1. Model cache path inaccessible for model write
    #  2. Model cache path not writable
    DV_MODEL_CACHE_FETCH_FAILURE = 246,  # One of the below errors occurred
    #  1. Model cache path inaccessible for model read
    #  2. Model not readable from cache
    #  3. Checksum comparison failed after model read from cache
    #  (Internal to Kinara)
    DV_INFER_INVALID_SHM_BUFFERS = 250,     # Unable to read/write SHM buffers
    DV_INFER_REQUEST_INVALID_HANDLE = 251,  # dv_wait_for_completion called on invalid inference request object
                                            # (or) error in clientlib while processing inference
    DV_FIRMWARE_VERSION_INVALID = 252,      # invalid firmware version for firmware binaries
    DV_MODEL_LOAD_SUBMITTED = 253,
    DV_MODEL_ALREADY_LOADED = 254,
    DV_MODEL_UNLOAD_SUBMITTED = 255,
    DV_MODEL_LOAD_SUBMIT_FAILED = 256,
    DV_MODEL_LOAD_ABORTED = 257,
    DV_MODEL_UNLOAD_SUBMIT_FAILED = 258,
    DV_ERROR_CATEGORY_INVALID_END = 299,

    # DV_ERROR_CATEGORY_SW_CLIENT_FATAL
    DV_ERROR_CATEGORY_SW_CLIENT_FATAL_START = 300,
    DV_CLIENT_VERSION_MISMATCH = 300,  # Incorrect version of clientlib used; to
                                        # be matched with Proxy version
    DV_CONNECTION_ERROR = 301,         # Connection is faulty/closed
    DV_HIF_PUSH_FAILED = 302,
    DV_HIF_ERROR = 303,
    DV_HIF_TIMEOUT = 304,
    DV_HIF_POP_FAILED = 305,
    DV_ERROR_CATEGORY_SW_CLIENT_FATAL_END = 399,

    # DV_ERROR_CATEGORY_SW_SERVER_FATAL
    DV_ERROR_CATEGORY_SW_SERVER_FATAL_START = 400,
    DV_HOST_OUT_OF_MEMORY = 400,        # Host is out of memory (calloc/malloc call fails)
    DV_INTERNAL_ERROR = 401,            # One of the below errors occurred (Internal to Kinara)
                                        # 1. Mismatch in expected payload sizes
                                        # 2. Failed to add request to host list
    DV_REQUEST_TIMEDOUT = 402,          # Request timed out; default values of timeouts
                                        # are API specific and mentioned in dvapi.h
    DV_SHMBUF_NOT_PERMITTED = 403,      # One of the below errors occurred
                                        # 1. Mmap of shm file failed
                                        # 2. Invalid fd sent in sfd request
                                        # 3. No more free shm buffers present
    DV_REQUEST_SEND_FAILED = 404,       # failed to send the request
    DV_REQUEST_PROCESSING_FAILED = 405,  # failed to processing incoming request
                                        # 1. happens at on_request()

    DV_ERROR_CATEGORY_SW_SERVER_FATAL_END = 499,

    # DV_ERROR_CATEGORY_HW_FATAL (Endpoint to be reset)
    DV_ERROR_CATEGORY_HW_FATAL_START = 500,
    DV_ENDPOINT_DMA_FAILED = 500,             # DMA from endpoint to host failed in case of PCIe
    DV_ENDPOINT_FIRMWARE_LOAD_FAILURE = 501,  # Failed to load firmware
    DV_ENDPOINT_FIRMWARE_BOOT_FAILURE = 502,  # Failed to boot firmware
    DV_ENDPOINT_NO_FIRMWARE = 503,            # Reserved (unused currently)
    DV_ENDPOINT_NOT_REACHABLE = 504,          # Interface gone bad or device exception occurred (Refer Fault
                                                # Handling document for further details)
    DV_ENDPOINT_MODEL_BINDING_FAILURE = 505,  # Model binding not present in the  device.
    DV_TENSOR_FREE_ERROR = 506,               # Failed to free the allocated tensors

    DV_MODEL_LOAD_FAILURE = 520,    # Model load request failed on all endpoints in list due to one of
                                    # the following
                                    #  1. Model write to endpoints failed
                                    #  2. Model context write to endpoints failed
                                    #  3. Unable to perform model integrity check after model write
                                    #  (Internal to Kinara)
                                    #  4. Model checksum compare failure (Internal to Kinara)
                                    #  5. Invalid model object received in request
                                    #  6. No endpoints active in endpoint list to load model
    DV_MODEL_RELOAD_FAILURE = 521,  # Device in faulty state on trying to reload model from cache
    DV_MODEL_UNLOAD_FAILURE = 522,
    DV_PARTIAL_SUCCESS = 523,                 # failed to load/unload model on some of the given devices
    DV_TENSOR_WRITE_FAILURE = 541,            # Failed to write buffer to endpoint (Internal to Kinara)
    DV_TENSOR_READ_FAILURE = 542,             # Failed to read buffer from endpoint  (Internal to Kinara)
    DV_TENSOR_CREATE_FAILURE = 543,           # Failed to create tensor
    DV_TENSOR_ALLOCTAION_FAILURE = 545,       # Failed to allocate device memory for a tensor.
    DV_TENSOR_INTEGRITY_CHECK_FAILURE = 546,  # Integrity check failed.

    DV_PROXY_DEVICE_INIT_FAILURE = 551,        # Proxy not able to initialize device


    DV_INFER_TIME_OUT = 560,         # Inference request failed to complete within the timeout limit
    DV_INFER_FAILURE = 561,          # inference failure
    DV_INFER_INVALID_INPUT = 562,    # inference request received with invalid input
    DV_INFER_QUEUE_FULL = 563,       # Inference queue on host side is full; current inferences in queue
                                    # to be processed to accept more requests
    DV_INFER_QUEUE_EMPTY = 564,      # No inferences scheduled on endpoint host
    DV_INFER_MODEL_NOT_FOUND = 565,  # Model not loaded on for inference request
    DV_INFER_ABORTED = 566,          # Inference aborted before submitting to device
    DV_INFER_SUBMIT_FAILURE = 567,   # Infer request submission to device failed
    DV_INFER_TIME_OUT_THERMAL_RUNAWAY = 568,  # Inference request failed due to increase in temperature

    DV_ERROR_CATEGORY_HW_FATAL_END = 599,

    DV_CLIENT_TXRX_WRITE_FAILURE = 600,         # UV write failue
    DV_CLIENT_TXRX_READ_FAILURE = 601,          # UV read failure
    DV_CLIENT_TXRX_ASYNC_SEND_FAILURE = 602,    # UV read failure
    DV_CLIENT_TXRX_FD_COUNT_MISMATCH = 603,     # fd count mismatch at proxy client
    DV_CLIENT_TXRX_DISCONNECT_ERROR = 604,      # client disconnected from server
    DV_CLIENT_RECEIVED_UNKNOWN_RESPONSE = 605,  # proxy client received unknown response
    DV_CLIENT_TO_TXRX_CONNECTION_ERROR = 606,   # client not connnected to transreceiver

    DV_FLOW_CREATE_FAILED = 700,  # Failed to submit the control flow
    DV_FLOW_SUBMIT_FAILED = 701,  # Failed to submit the control flow
    DV_FLOW_ABORTED = 702,        # Flow is aborted explicitly

    DV_CP_INDEX_ALLOC_FAILURE = 800,  # Failed to get the CP index.

class dv_session_socket_type(IntEnum):
    DV_SESSION_SOCKET_TYPE_UNIX    = 0,     # unix domain socket
    DV_SESSION_SOCKET_TYPE_TCPIPv4 = 1      # tcp ipv4 socket


class dv_endpoint_host_interface(IntEnum):
    DV_ENDPOINT_HOST_INTERFACE_PCIE = 1,    # host and dv connected via pcie interface
    DV_ENDPOINT_HOST_INTERFACE_USB  = 2     # host and dv connected via usb interface


class dv_endpoint_default_group(IntEnum):
    DV_ENDPOINT_DEFAULT_GROUP_ALL   = 0,    # default group for all the endpoint(s) connected to inference proxy server
    DV_ENDPOINT_DEFAULT_GROUP_PCIE  = 1,    # default group for all the pcie endpoint(s) connected to inference proxy server
    DV_ENDPOINT_DEFAULT_GROUP_USB   = 2,    # default group for all the usb endpoint(s) connected to inference proxy server

class dv_endpoint_state(IntEnum):
    DV_ENDPOINT_STATE_INIT              = 0,    # endpoint reset done but firmware not loaded yet(this is a transient state)
    DV_ENDPOINT_STATE_IDLE              = 1,    # firmware loaded and endpoint is ready for inference execution
    DV_ENDPOINT_STATE_ACTIVE            = 2,    # inference execution ongoing
    DV_ENDPOINT_STATE_ACTIVE_SLOW       = 3,    # endpoint is operating at reduced frequency
    DV_ENDPOINT_STATE_ACTIVE_BOOSTED    = 4,    # endpoint is operating at reduced frequency
    DV_ENDPOINT_STATE_THERMAL_INACTIVE  = 5,    # endpoint is in thermal Inactive state
    DV_ENDPOINT_STATE_THERMAL_UNKNOWN   = 6,    # endpoint is in unown thermal state
    DV_ENDPOINT_STATE_INACTIVE          = 7,    # endpoint is in Inactive state
    DV_ENDPOINT_STATE_FAULT             = 8,    # endpoint is in faulty state
    DV_ENDPOINT_STATE_BAD_INTERFACE     = 1001, # [unsupported]
    DV_ENDPOINT_STATE_RECOVERY          = 1003, # [unsupported]
    DV_ENDPOINT_STATE_DEAD              = 1004, # [unsupported]
    DV_ENDPOINT_STATE_DRAIN             = 1005, # [unsupported]
    DV_ENDPOINT_STATE_POWER_GATED       = 1006, # [unsupported]
    DV_ENDPOINT_STATE_CLOSED            = 1007, # [unsupported]

class dv_endpoint_power_state(IntEnum):
    DV_POWER_STATE_L0   = 0,    # state with maximum power and performance (default state when endpoint is initialized)
    DV_POWER_STATE_L1   = 1,    # endpoint operating at a sys clock of 300 MHz
    DV_POWER_STATE_L1A  = 2,    # endpoint operating at a sys clock of 150 MHz
    DV_POWER_STATE_L2   = 3     # endpoint set to a sys clock of 150 MHz and all subsystems are power gated 

class dv_endpoint_group_type(IntEnum):
    DV_ENDPOINT_GROUP_TYPE_NONE     = 0,    # endpoint group type none
    DV_ENDPOINT_GROUP_TYPE_ALL      = 1,    # endpoint group type all
    DV_ENDPOINT_GROUP_TYPE_PCIE     = 2,    # endpoint group type pcie
    DV_ENDPOINT_GROUP_TYPE_USB      = 3,    # endpoint group type usb
    DV_ENDPOINT_GROUP_TYPE_CUSTOM   = 4,    # endpoint group type custom


#  Model Network type
class dv_layer_output_type(IntEnum):
    DV_LAYER_OUTPUT_TYPE_CLASSIFICATION            = 0,  # represent classification type of network
    DV_LAYER_OUTPUT_TYPE_DETECTION                 = 1,  # represent detection type of network
    DV_LAYER_OUTPUT_TYPE_SEMANTIC_SEGMENTATION     = 2,  # represent semantic segmentation type of network
    DV_LAYER_OUTPUT_TYPE_RAW                       = 3   # represents all other network types which can't be determined


class dv_model_priority_level(IntEnum):
    DV_MODEL_PRIORITY_LEVEL_LOW       = 0,  # model priority low
    DV_MODEL_PRIORITY_LEVEL_MEDIUM    = 1,  # model priority medium
    DV_MODEL_PRIORITY_LEVEL_DEFAULT   = 1,  # model priority default
    DV_MODEL_PRIORITY_LEVEL_HIGH      = 2   # model priority high


class dv_inference_status(IntEnum):
    DV_INFERENCE_STATUS_QUEUED    = 0,    # Inference is in queued state
    DV_INFERENCE_STATUS_RUNNING   = 1,    # Inference is in running/executing state
    DV_INFERENCE_STATUS_COMPLETED = 2,    # Inference is in completed state
    DV_INFERENCE_STATUS_FAILED    = 4,    # Inference is in failed state
    DV_INFERENCE_STATUS_UNKNOWN   = 5     # Inference information is not available


class dv_blob_type(IntEnum):
    DV_BLOB_TYPE_RAW_POINTER      = 0,    # represents blob backed by raw pointer
    DV_BLOB_TYPE_SHM_DESCRIPTOR   = 1,    # represents blob backed by registered shared memory descriptor
    DV_BLOB_TYPE_FD               = 2     # represents blob backed by non-registered file descriptor


###############################################################################
# dvapi_private enum wrappers based on ctypes
###############################################################################

class dv_client_log_level(IntEnum):
    DV_CLIENT_LOG_LEVEL_OFF      = 0,     # turn of client logs
    DV_CLIENT_LOG_LEVEL_ERROR    = 1,     # dump only error logs
    DV_CLIENT_LOG_LEVEL_WARN     = 2,     # dump error and warning logs
    DV_CLIENT_LOG_LEVEL_INFO     = 3      # dump error, warning and info logs


# ************** dvapi structs **************


class dv_blob(Structure):
    _fields_ = [ ("handle",     c_void_p),          # blob handle (raw pointer or shared file id returned by server)
                 ("offset",     c_uint64),          # blob offset
                 ("size",       c_uint64),          # blob size
                 ("blob_type",  c_int) ]            # blob type as represented in enum DV_BLOB_TYPE

class dv_session_options(Structure):
    _fields_ = [ ("timeout_ms", c_int) ]    # global default timeout


class dv_session(Structure):
    _fields_ = [ ("handle",      c_void_p),                      # session private handle, managed by client library
                 ("socket_str",  c_char_p),                      # NULL terminated socket connection string
                 ("socket_type", c_int) ]                         # socket types: Unix domain socket/TCPIPv4


class dv_endpoint_chip_info(Structure):
    _fields_ = [ ("id",                         c_char_p),      # dv chip id
                 ("rev",                        c_char_p),      # dv chip revision
                 ("control_processor_count",    c_int),         # dv chip control processor count
                 ("neural_processor_count",     c_int),         # dv chip neural processor count
                 ("l2_memory_size",             c_uint)]        # dv chip internal L2 memory size in bytes


class dv_endpoint_dram_info(Structure):
    _fields_ = [("vendor_id",   c_uint),        # dv dram vendor id
                ("vendor_name", c_char_p),      # dv dram vendor name
                ("size",        c_uint),        # dv dram memory size in bytes
                ("rev_id1",     c_ubyte),       # dv dram revision id 1
                ("rev_id2",     c_ubyte),       # dv dram revision id 2
                ("density",     c_ubyte),       # dv dram density
                ("io_width",    c_ubyte)]       # dv dram io width


class dv_endpoint_iface_info(Structure):
    class dv_endpoint_iface_info_sysfs_path(Union):
        _fields_ = [("pcie_dir", c_char_p)]

    _fields_ = [("type",        c_int),     # dv module physical interface (pcie, usb) with host
                ("bus_num",     c_int),     # host interface bus number on which dv device is connected
                ("device_num",  c_int),     # host interface device number on which dv device is connected
                ("sysfs_path",  dv_endpoint_iface_info_sysfs_path),
                ("port_num",    c_int)]     # port number for on which USB device is connected


class dv_endpoint_info(Structure):
    _fields_ = [("device_id",       c_uint),                            # dv endpoint device id
                ("vendor_id",       c_uint),                            # dv endpoint vendor id
                ("chip",            POINTER(dv_endpoint_chip_info)),    # dv endpoint chip informantion
                ("dram",            POINTER(dv_endpoint_dram_info)),    # dv endpoint external dram information
                ("iface",           POINTER(dv_endpoint_iface_info)),   # dv endpoint interface information
                ("module_name",     c_char_p),                          # dv physical module name connected to server
                ("gpio0",           c_uint),                            # dv endpoint GPIO value0
                ("gpio1",           c_uint),                            # dv endpoint GPIO value1
                ("device_uid",      c_uint)]                            # dv endpoint GPIO id


class dv_endpoint(Structure):
    _fields_ = [("handle",          c_void_p),                              # endpoint private handle, managed by client library
                ("session",         POINTER(dv_session)),                   # session object
                ("num_ep",          c_int),                                 # number of endpoints in the group
                ("grp_type",        c_int),                                 # endpoint group type
                ("ep_info_list",    POINTER(POINTER(dv_endpoint_info)))]    # list of configuration for all the endpoint(s) in the group

class dv_endpoint_dram_statistics(Structure):
    _fields_ = [("ep",                                  POINTER(dv_endpoint)),    # endpoint handle
                ("ep_total_dram_size",                  c_uint32),                # endpoint dram size in bytes
                ("ep_total_dram_occupancy_size",        c_uint32),                # endpoint dram memory occupied in bytes
                ("ep_total_free_size",                  c_uint32),                # endpoint dram memory free in bytes
                ("ep_total_reserved_occupancy_size",    c_uint32),                # endpoint dram reserved memory in bytes for firmware
                ("ep_total_model_occupancy_size",       c_uint32),                # endpoint dram memory occupied by all the active model artefacts in bytes
                ("ep_total_tensor_occupancy_size",      c_uint32)]                # endpoint dram memory occupied by all the active model tensors in bytes


class dv_inference_queue_statistics(Structure):
    _fields_ = [("occupancy_count", c_int),         # Number of inference queue slots occupied with inference request for the endpoint
                ("length",          c_int),         # length of the inference queue for the endpoint
                ("wait_time",       c_float)]       # waiting time in mili secs for the new inference request to get picked up by endpoint


class dv_model_statistics(Structure):
    _fields_ = [("model",                                   c_uint32),        # model handle
                ("active_input_tensors_count",              c_uint32),        # number for active model input tensor(s) present in an endpoint
                ("active_output_tensors_count",             c_uint32),        # number for active model output tensor(s) present in an endpoint
                ("active_inferences_count",                 c_uint32),        # number for active model inference request queued in an endpoint
                ("model_total_dram_occupancy_size",         c_uint32),        # total endpoint dram occupancy in bytes by model artefacts
                ("model_total_input_tensor_occupancy_size", c_uint32),        # total endpoint dram occupancy in bytes by model input tensors
                ("model_total_output_tensor_occupancy_size",c_uint32),        # total endpoint dram occupancy in bytes by model output tensors
                ("model_handle",                            c_void_p)]      # since r5.3 void* handle which can be compared to `handle` member of dv_model_t


class dv_endpoint_stats(Structure):
    _fields_ = [("ep",                              POINTER(dv_endpoint)),                          # endpoint handle
                ("state",                           c_int),                                         # endpoint state
                ("ep_sys_clk",                      c_int),                                         # endpoint system core clock in MHz
                ("ep_dram_clk",                     c_int),                                         # endpoint dram clock in MHz
                ("ep_core_voltage",                 c_float),                                       # average of endpoint core voltage across all measurement point in hardware in volts
                ("ep_temp",                         c_float),                                       # average of endpoint temperature across all measurement point in hardware in degree celsius
                ("num_inference_queues",            c_int),                                         # number of inference queues available for the endpoint
                ("ep_infq_stats",                   POINTER(dv_inference_queue_statistics)),        # inference queue statistics for the endpoint
                ("num_active_models",               c_int),                                         # number of active models present in endpoint
                ("model_stats",                     POINTER(dv_model_statistics)),                  # statistics for all models active on the endpoint
                ("ep_dram_stats",                   dv_endpoint_dram_statistics),                   # endpoint dram statistics
                ("ep_power_state",                  c_int),                                         # endpoint power state
                ("ep_soft_reset_count",             c_uint32)]                                      # endpoint soft reset count, non zero for usb devices


class dv_model_input_preprocess_param(Structure):
    _fields_ = [("qn",              c_float),           # input quantization
                ("scale",           POINTER(c_float)),  # per channel scale
                ("mean",            POINTER(c_float)),  # per channel mean
                ("aspect_resize",   c_bool),            # aspect ratio based resize
                ("mirror",          c_bool),            # mirror effect
                ("center_crop",     c_bool),            # center crop
                ("bgr_to_rgb",      c_bool),            # convert BGR to RGB
                ("interpolation",   c_int),             # interpolation method supported by OpenCV
                ("is_signed",        c_bool),           # input range ((-128) - 128)/(0 - 255)
                ("bpp",             c_int),             # bytes per pixel
                ("output_scale",    c_float),           # outputscale for asymmetric
                ("aspect_resize_scale", c_float),       # aspect resize scaling factor
                ("offset",   c_int),             # output offset
                ("qmode",           c_int)]


class dv_model_input_param(Structure):
    _fields_ = [("preprocess_param", POINTER(dv_model_input_preprocess_param)),
                ("layer_id",        c_int),         # input layer id
                ("blob_id",         c_int),         # input blob id
                ("layer_name",      c_char_p),      # input layer name
                ("blob_name",       c_char_p),      # input blob name
                ("layer_type",      c_char_p),      # input layer type
                ("layout",          c_char_p),      # input layer type
                ("size",            c_int),         # tensor size in bytes
                ("width",           c_int),         # tensor width
                ("height",          c_int),         # tensor height
                ("depth",           c_int),         # depth dimension
                ("nch",             c_int),         # number of channels
                ("bpp",             c_int),         # bytes per pixel
                ("batch_size",      c_int),         # batch size
                ("num",             c_int),         # num
                ("src_graph_layer_name", c_char_p)] # src_graph_layer_name


class dv_model_output_postprocess_param(Structure):
    _fields_ = [("qn",                  c_float),       # output quantization parameter
                ("is_struct_format",    c_bool),        # output is structured or not
                ("is_float",            c_bool),        # output is float type
                ("is_signed",           c_bool),        # output is signed ot not
                ("output_scale",        c_float),       # outputscale for asymmetric
                ("offset",              c_int)]         # offset for asymmetric


class  dv_model_output_param(Structure):
    _fields_ = [("postprocess_param", POINTER(dv_model_output_postprocess_param)),
                ("layer_id",                c_int),         # layer id
                ("blob_id",                 c_int),         # blob id
                ("fused_parent_id",         c_int),         # layer fused parent id
                ("layer_name",              c_char_p),      # layer name
                ("blob_name",               c_char_p),      # blob name
                ("layer_fused_parent_name", c_char_p),      # layer fused parent name
                ("layer_type",              c_char_p),      # layer type
                ("layout",                  c_char_p),      # layout
                ("size",                    c_int),         # layer size in bytes
                ("width",                   c_int),         # layer width in pixels
                ("height",                  c_int),         # layer height in pixels
                ("depth",                   c_int),         # layer depth in pixels
                ("nch",                     c_int),         # number of channels
                ("bpp",                     c_int),         # bytes per pixel
                ("num_classes",             c_int),         # number of classes for which model is trained on
                ("layer_output_type",       c_int),         # output type of layer
                ("num",                     c_int),         # num
                ("max_dynamic_id",          c_int),         # max batch id
                ("src_graph_layer_name",    c_char_p)]      # src_graph_layer_name


class dv_compiler_statistics(Structure):
    _fields_ = [("config_name",c_char_p),              # DV1 config name, governed on ep system core clock
                ("cycles",c_float),                    # total cycles estimated by compiler
                ("ips",c_float),                       # inference per seconds estimated by compiler
                ("ddr_bandwidth",c_float)]             # ep dram estimated by compiler


class dv_model_type(IntEnum):
    DV_MODEL_TYPE_ARA1_CNN = 0,       # for ara1 cnn models
    DV_MODEL_TYPE_ARA2_CNN = 1,       # for ara2 cnn models
    DV_MODEL_TYPE_ARA2_LLM = 2,       # for ara2 dyn quant v1 llm models
    DV_MODEL_TYPE_ARA2_LLM_DYN_V2 = 3 # for ara2 v2 llm models

class dv_model_options(Structure):
     _fields_ = [("model_name",c_char_p),              # model name (unused)
                 ("priority",c_int),                   # priority of the model
                 ("cache",c_bool),                     # if true, the model is cached on disk
                 ("async",c_bool),                     # if true, the model load API immediately return \see dv_model_load_wait_for_completion
                 ("model_type", c_int)]

class dv_infer_type(IntEnum):
    DV_INFER_TYPE_ARA1_CNN = 0,
    DV_INFER_TYPE_ARA2_CNN = 1,
    DV_INFER_TYPE_LLM_PROMPT_PROCESSING = 2,
    DV_INFER_TYPE_LLM_FOLLOWUP_PROMPT_PROCESSING = 3,
    DV_INFER_TYPE_LLM_TOKEN_GENERATION = 4,

class dv_infer_options(Structure):
    _fields_ = [ ("enable_stats", c_bool),
                ("infer_type",    c_int),
                ("active_tokens", c_uint64),
                ("valid_tokens",  c_uint32),
                ("tokens_to_skip", c_uint32)]

class dv_model_llm_params(Structure):
    _fields_ = [("vocab_size",        c_uint32),
                ("embedding_size",    c_uint32),
                ("input_precision",   c_uint32),
                ("output_precision",  c_uint32),
                ("max_num_tokens",    c_uint32),
                ("is_dynamic",        c_uint32),
                ("num_inputs",        c_uint32),
                ("pad_token_id",      c_uint32),
                ("eos_token_id",      c_uint32),
                ("bos_token_id",      c_uint32)]
    
    def __str__(self):
        return 'dv_model_llm_params <vocab_size={}, embedding_size={}, input_precision={}, output_precision={}, max_num_tokens={}, is_dynamic={}, num_inputs={}, pad_token_id={}, eos_token_id={}, bos_token_id={}>'.format(self.vocab_size, self.embedding_size, self.input_precision, self.output_precision, self.max_num_tokens, self.is_dynamic, self.num_inputs, self.pad_token_id, self.eos_token_id, self.bos_token_id)


class dv_llm_cfg_upd_req(Structure):
    _fields_ = [("top_k",                   c_uint32),
                ("top_p",                   c_float),
                ("temperature",             c_float),
                ("repetition_penalty",      c_float),
                ("target_token_post_mcp",   c_uint32),
                ("target_token_pre_mcp",    c_uint32),
                ("target_prompt_post_mcp",  c_uint32),
                ("target_prompt_pre_mcp",   c_uint32),
                ("draft_token_post_mcp",    c_uint32),
                ("draft_token_pre_mcp",     c_uint32),
                ("draft_prompt_post_mcp",   c_uint32),
                ("draft_prompt_pre_mcp",    c_uint32)]

def __str__(self):
        return 'dv_model_llm_params <top_k={}, top_p={:.3f}, temperature={:.3f}, repetition_penalty={}, target_token_post_mcp={}, target_token_pre_mcp={}, target_prompt_post_mcp={}, target_prompt_pre_mcp={}, draft_token_post_mcp={}, draft_token_pre_mcp={}, draft_prompt_post_mcp={}, draft_prompt_pre_mcp={}, >'.format(self.top_k, self.top_p, self.temperature, self.repetition_penalty, self.target_token_post_mcp, self.target_token_pre_mcp, self.target_prompt_post_mcp, self.target_prompt_pre_mcp, self.draft_token_post_mcp, self.draft_token_pre_mcp, self.draft_prompt_post_mcp, self.draft_prompt_pre_mcp)


class dv_model(Structure):
    _fields_ = [("handle",              c_void_p),                              # private handle, managed by client library
                ("session",             POINTER(dv_session)),                   # session object
                ("endpoint",            POINTER(dv_endpoint)),                  # endpoint object
                ("version",             c_uint),                                # model version as generated by compiler
                ("name",                c_char_p),                              # internal model name as generated by compiler
                ("model_type",          c_int),
                ("internal_name",       c_char_p),                              # internal model name as generated by compiler
                ("num_inputs",          c_int),                                 # number of inputs needed by model
                ("num_outputs",         c_int),                                 # number of output produced by model
                ("priority",            c_int),                                 # model priority as set by user
                ("input_param",         POINTER(dv_model_input_param)),         # list of input params
                ("output_param",        POINTER(dv_model_output_param)),        # list of output params
                ("llm_params",          POINTER(dv_model_llm_params)),          # list of llm params
                ("num_compiler_config", c_int),                                 # number of compiler stats config
                ("compiler_stats",      POINTER(dv_compiler_statistics)),       # list of compiler stats
                ("model_load_options",  POINTER(dv_model_options)),             # @since r6.0 options used when loading the model
                ("cp_layer",            c_bool)]                        

class c_timespec(Structure):
    _fields_ = [("tv_sec", c_long),
                ("tv_nsec", c_long)]

class dv_infer_statistics(Structure):
    _fields_ = [("ep_hw_sys_clk",                   c_int),                 # ep hardware system core clock in MHz
                ("ep_hw_nnp_clk" ,                  c_int ),                # ep hardware external nnp clock in MHz 
                ("ep_hw_sbp_clk" ,                  c_int ),                # ep hardware external sbp clock in MHz 
                ("ep_hw_dram_clk",                  c_int),                 # ep hardware external dram clock in MHz
                ("ep_hw_total_inference_cycles",    c_uint),                # total cycles taken to compute inference in hardware, including floating point computation
                ("ep_hw_fp_cycles",                 c_uint),                # cycles taken to compute floating point operation in hardware
                ("input_transfer_time",             c_float),               # time taken in mili-seconds to transfer input(s) from host dram to ep hardware dram
                ("output_transfer_time",            c_float),               # time taken in mili-seconds to transfer output(s) from ep hardware dram to host dram
                ("ep_queue_submission_time",        c_float),               # time taken in mili-seconds to submit inference request to ep hardw
                ("cum_replay_cnt",                  c_uint32),
                ("cur_replay_cnt",                  c_uint32),
                ("input_transfer_start_time_stamp" ,    c_timespec ),       # time stamp when input transfer started
                ("output_transfer_start_time_stamp",    c_timespec ),       # time stamp when output transfer started
                ("inference_start_time_stamp",          c_timespec ),       # time stamp when inference went into NNP queue 
                ("inference_execution_time",        c_float),               # time taken for inference execution in microseconds
                ("input_ddr_address",               c_int32),               # input ddr address
                ("output_ddr_address",              c_int32)]               # output ddr address


class dv_infer_llm_info(Structure):
    _fields_ = [("llm_infer_resp_num_valid_tokens", c_uint32)]


class dv_infer_request(Structure):
    _fields_ = [ ("handle",         c_void_p),                          # private handle, managed by client library
                 ("session",        POINTER(dv_session)),               # session for which inference is submitted
                 ("ep_queued",      POINTER(dv_endpoint)),              # endpoint for which inference is queued.
                 ("ep_submitted",   POINTER(dv_endpoint)),              # when inference request is queued on group of endpoints, this provide endpoint info on which inference is submitted.
                 ("model",          POINTER(dv_model)),                 # model handle for inference request
                 ("ip_blob_list",   POINTER(dv_blob)),                  # input blob list
                 ("op_blob_list",   POINTER(dv_blob)),                  # output blob list
                 ("status",         c_int),                             # inference run status
                 ("stats",          POINTER(dv_infer_statistics)),      # inference stats
                 ("llm_infer_info", POINTER(dv_infer_llm_info))]

# ************** dvapi private structs **************

class dv_endpoint_hw_statistics(Structure):
    _fields_ = [("ep",              POINTER(dv_endpoint)),                  # endpoint handle
                ("ep_sys_clk",      c_int),                                 # endpoint system core clock in MHz
                ("ep_dram_clk",     c_int),                                 # endpoint dram clock in MHz
                ("ep_dram_size",    c_uint),                                # endpoint total dram size in bytes
                ("num_volt",        c_int),                                 # number of voltage measurement points available in hardware
                ("ep_core_volt",    POINTER(c_float)),                      # list of endpoint voltage readings for all measurement points in volts
                ("num_temp",        c_int),                                 # number of temperature measurement points available in hardware
                ("ep_temp",         POINTER(c_float))]                      # list of endpoint temperature readings for all measurement points in degree celsius


class dv_request_options(Structure):
     _fields_ = [ ("timeout_ms", c_int) ]   # timeout in milliseconds for synchronous request


class dv_endpoint_takedown_options(Structure):
    _fields_ = [ ("request_options", dv_request_options)]  # proxy API request options


###############################################################################
# dvapi function wrappers based on ctypes
###############################################################################
def dv_stringify_status_code(status_code):
    """
    Parameters
    ----------
    dv_status_code

    Returns
    -------
    string
        stringified status code
    """
    fun = _dvApiObj().dv_stringify_status_code
    fun.argtypes = [c_int]
    fun.restype = c_char_p
    s = fun(status_code)
    return s.decode("utf8")



def _glibc_free(ctypes_ptr):
    fun = _dvApiObj().free
    fun.argtypes = [c_void_p]
    fun.restype = None
    fun(ctypes_ptr)

def dv_model_set_llm_cfg_params(session : dv_session, endpoint : dv_endpoint, model : dv_model, llm_cfg_update : dv_llm_cfg_upd_req) -> dv_status_code:
    fun = _dvApiObj().dv_model_set_llm_cfg_params
    fun.argtypes = [POINTER(dv_session), POINTER(dv_endpoint), POINTER(dv_model), POINTER(dv_llm_cfg_upd_req)]
    fun.restype = dv_status_code
    ret = fun(session, endpoint, model, llm_cfg_update)
    return ret

def dv_session_create_via_unix_socket(socket_path):
    """ connects to the proxy via unix socket

    Parameters
    ----------
    socket_path : str
        path to the unix socket

    Returns
    -------
    (dv_status_code, dv_session)
        DV_SUCCESS, dv_session if the connection was successfully established
        if the status is not DV_SUCCESS, dv_session is invalid
    """
    fun = _dvApiObj().dv_session_create_via_unix_socket
    fun.argtypes = [c_char_p, POINTER(POINTER(dv_session))]
    fun.restype = dv_status_code
    session = POINTER(dv_session)()
    ret = fun(socket_path.encode('utf-8'), byref(session))
    return ret, session


def dv_session_create_via_tcp_ipv4_socket(ipv4_addr, port):
    """ connects to the proxy over the network

    Parameters
    ----------
    ip_addr : str
        IPv4 address to connect to

    port : int
        Port number proxy is listening on

    Returns
    -------
    (dv_status_code, dv_session)
        DV_SUCCESS, dv_session if the connection was successfully established
        if the status is not DV_SUCCESS, dv_session is invalid
    """
    fun = _dvApiObj().dv_session_create_via_tcp_ipv4_socket
    fun.argtypes = [c_char_p, c_uint, POINTER(POINTER(dv_session))]
    fun.restype = dv_status_code
    session = POINTER(dv_session)()
    ret = fun(ipv4_addr.encode('utf-8'), c_uint(port), byref(session))
    return ret, session


def dv_session_close(session):
    """ closes connection `dv_session` to the proxy

    Parameters
    ----------

    session : dv_session
        dv_session object returned by initial dv_session_create_* call

    Returns
    -------
    dv_status_code
        dv_status_code.DV_SUCCESS if successful
    """
    fun = _dvApiObj().dv_session_close
    fun.argtypes = [POINTER(dv_session)]
    fun.restype = dv_status_code
    return fun(session)



def dv_endpoint_get_list(session):
    """ returns a list of endpoints that are enumerated by the connected proxy

    Parameters
    ----------
    session : dv_session
        client session

    Returns
    -------
    (dv_status_code, dv_endpoint, count)
        error status, list of endpoints connected, num connected endpoints
        if error status is not DV_SUCCESS, the list will be None
    """
    fun = _dvApiObj().dv_endpoint_get_list
    fun.argtypes = [POINTER(dv_session), POINTER(POINTER(dv_endpoint)), POINTER(c_int)]
    fun.restype = dv_status_code

    count = c_int()
    eplist = POINTER(dv_endpoint)()
    ret = fun(session, byref(eplist), byref(count))

    if ret != dv_status_code.DV_SUCCESS:
        return ret, None, 0

    return ret, eplist, count


def dv_endpoint_get_default_group(session, ep_default_grp):
    """ get default endpoint group created by client

    Parameters
    ----------
    session : dv_session
        session

    ep_default_grp : dv_endpoint

    Returns
    -------
    (dv_status_code, dv_endpoint)
        status code and endpoint handle of the created default endpoint group
    """

    fun = _dvApiObj().dv_endpoint_get_default_group
    fun.argtypes = [POINTER(dv_session), c_int, POINTER(POINTER(dv_endpoint))]
    fun.restype = dv_status_code

    ep_grp = POINTER(dv_endpoint)()
    ret = fun(session, ep_default_grp, byref(ep_grp))

    if ret != dv_status_code.DV_SUCCESS:
        return ret, None

    return ret, ep_grp


def dv_endpoint_create_group(session, ep_list, ep_count):
    """ create a custom group from the endpoint list

    Parameters
    ----------
    session : dv_session
        session
    ep_list : dv_endpoint
        list of endpoints
    ep_count : dv_endpoint
        number of endpoints in the list

    Returns
    -------
    (dv_status_code, dv_endpoint)
        status code and endpoint handle of the created custom endpoint group
    """

    fun = _dvApiObj().dv_endpoint_create_group
    fun.argtypes = [POINTER(dv_session), POINTER(POINTER(dv_endpoint)), c_int, POINTER(POINTER(dv_endpoint))]
    fun.restype = dv_status_code
    ep_grp = POINTER(dv_endpoint)()
    ep_list_array = (POINTER(dv_endpoint) * ep_count)(*[pointer(s) for s in ep_list])
    ret = fun(session, ep_list_array, c_int(ep_count), byref(ep_grp))

    if ret != dv_status_code.DV_SUCCESS:
        return ret, None

    return ret, ep_grp


def dv_endpoint_free_group(ep_grp):
    """ free the endpoint group created from dv_endpoint_create_group

    Parameters
    ----------
    ep_grp : dv_endpoint
        endpoint group

    Returns
    -------
    dv_status_code
        status == DV_SUCCESS on pass
    """
    fun = _dvApiObj().dv_endpoint_free_group
    fun.argtypes = [POINTER(dv_endpoint)]
    fun.restype = dv_status_code

    return fun(ep_grp)


def dv_model_load_from_file(session, ep, model_path, model_name, priority):
    """ loads a model onto the device from the provided file path

    Parameters
    ----------

    session : dv_session
        connected client session

    ep : dv_endpoint
        endpoint on which to load the model

    model_path : str
        path to the model on filesystem

    model_name : str
        a reference name to be assigned to the model

    priority : int
        priority of the model

    Returns
    -------
    (dv_status_code, dv_model)
        an error status and the model handle. if the error status is not
        DV_SUCCESS, dv_model_handle is invalid
    """
    fun = _dvApiObj().dv_model_load_from_file
    fun.argtypes = [
        POINTER(dv_session),
        POINTER(dv_endpoint),
        c_char_p,
        c_char_p,
        c_int,
        POINTER(POINTER(dv_model))]

    fun.restype = dv_status_code

    model = POINTER(dv_model)()

    ret = fun(session,
              ep,
              model_path.encode('utf-8'),
              model_name.encode('utf-8'),
              priority,
              byref(model))

    if ret != dv_status_code.DV_SUCCESS:
        return ret, None

    return ret, model


def dv_model_load_from_blob(session, ep, model_blob, model_name, priority):
    """ loads a model onto the device from the provided file path

    Parameters
    ----------

    session : dv_session
        connected client session

    ep : dv_endpoint
        endpoint on which to load the model

    model_blob : dv_blob
        model blob

    model_name : str
        a reference name to be assigned to the model

    priority : int
        priority of the model

    enable_caching : bool
        indicaes if the model should be cached by the server.
        API call fails if the model could not be cached by the server

    Returns
    -------
    (dv_status_code, dv_model_handle)
        an error status and the model handle. if the error status is not
        DV_SUCCESS, dv_model_handle is invalid
    """

    fun = _dvApiObj().dv_model_load_from_blob
    fun.argtypes = [
        POINTER(dv_session),
        POINTER(dv_endpoint),
        POINTER(dv_blob),
        c_char_p,
        c_int,
        POINTER(POINTER(dv_model))]

    fun.restype = dv_status_code

    model = POINTER(dv_model)()

    ret = fun(session,
              ep,
              model_blob,
              model_name.encode('utf-8'),
              priority,
              byref(model))

    if ret != dv_status_code.DV_SUCCESS:
        return ret, None

    return ret, model


def dv_model_unload(model):
    """ unloads model from the dvinfproxy

    Parameters
    ----------
    model : dv_model
        model handle returned by dv_model_load*

    Returns
    -------
    dv_status_code
        DV_SUCCESS if the unload was successful
    """
    fun = _dvApiObj().dv_model_unload
    fun.argtypes = [POINTER(dv_model)]
    fun.restype = dv_status_code
    return fun(model)


class dvBoxOutput(Structure):
    """ ctypes mapping to `struct BoxOutput` defined in `dvoutstruct.h` """
    _fields_ = [("xmin", c_float),
                ("ymin", c_float),
                ("xmax", c_float),
                ("ymax", c_float),
                ("imageId",c_uint),
                ("label", c_uint),
                ("score", c_float)]

    def __str__(self):
        return 'dvBoxOutput<xmin={:.3f}, ymin={:.3f}, xmax={:.3f}, ymax={:.3f}, imageId={}, label={}, score={:.3f}>'.format(self.xmin, self.ymin, self.xmax, self.ymax, self.imageId, self.label, self.score)


class dvDetectionOutput(Structure):
    """ ctypes mapping to `struct DetectionOutput` defined in `dvoutstruct.h` """

    _fields_ = [("numBoxesDetected", c_uint),
                ("detectedBoxes", dvBoxOutput * 200)]


class dvClassificationItem(Structure):
    """ ctypes mapping to `struct ClassificationItem` defined in `dvoutstruct.h` """

    _fields_ = [("label", c_uint),
                ("score", c_float)]

    def __str__(self):
        return 'dvClassificationItem<label={}, score={:.3f}>'.format(self.label, self.score)


class dvClassificationOutput(Structure):
    """ ctypes mapping to `struct ClassificationOutput` defined in `dvoutstruct.h` """

    _fields_ = [("numOutputs", c_uint),
                ("rows", dvClassificationItem * 10)]


"""
Deprecated since, 7.2.
Instead use DVModel level API i.e., infer_sync.
"""
def dv_infer_sync(session, endpoint, model, inputs, outputs, timeout, enable_stats=True):
    """ submits a synchronous inference request with timeout

    Parameters
    ----------
    session : dv_session

    endpoint : dv_endpoint

    model : dv_model

    inputs : contiguous dvIoTensorBlob
        pointers buffers that contain the inputs, this must contain pointers
        to each input in order, pointer at index 0 will be used to read data
        for the 1st input etc...

    outputs : contiguous dvIoTensorBlob
        ptrs to where outputs of the inference are written, this list must be
        in order, ie 1st output is written to the pointer at index 0, 2nd output
        is written to the pointer at index 1 etc..

    timeout : int
        timeout in ms

    Warnings
    --------
    Please ensure that the object backing the ctypes.c_void_p are not allowed to
    be refcounted or garbage collected - if the pointer becomes invalid during
    operation, the runtime will segfault

    Returns
    -------
    (dv_status_code, inf_request_id)
        DV_SUCCESS, inf_request_id if the inference was completed successfully
        else, inf_request_id is invalid
    """
    fun = _dvApiObj().dv_infer_sync
    fun.argtypes = [POINTER(dv_session),
                    POINTER(dv_endpoint),
                    POINTER(dv_model),
                    POINTER(dv_blob),
                    POINTER(dv_blob),
                    c_int,
                    c_bool,
                    POINTER(POINTER(dv_infer_request))]

    fun.restype = dv_status_code

    inf_request = POINTER(dv_infer_request)()

    ret = fun(session,
              endpoint,
              model,
              cast(inputs, POINTER(dv_blob)),
              cast(outputs, POINTER(dv_blob)),
              c_int(timeout),
              c_bool(enable_stats),
              byref(inf_request))

    if ret != dv_status_code.DV_SUCCESS:
        return ret, None

    return ret, inf_request


def dv_infer_async(session, endpoint, model, inputs, outputs, enable_stats=True):
    """ submits an asynchronous inference request with timeout

    Parameters
    ----------
    session : dv_session

    endpoint : dv_endpoint

    model : dv_model

    inputs : list of ctypes.c_void_p
        pointers buffers that contain the inputs, this must contain pointers
        to each input in order, pointer at index 0 will be used to read data
        for the 1st input etc...

    outputs : list of ctypes.c_void_p
        ptrs to where outputs of the inference are written, this list must be
        in order, ie 1st output is written to the pointer at index 0, 2nd output
        is written to the pointer at index 1 etc..

    Warnings
    --------
    Please ensure that the object backing the ctypes.c_void_p are not allowed to
    be refcounted and freed or garbage collected - if the pointer becomes invalid
    while a response is being written to the ptr the runtime will segfault

    Returns
    -------
    (dv_status_code, inf_request_id)
        DV_SUCCESS, inf_request_id if the inference was submitted successfully
        to the inference proxy else, inf_request_id is invalid
    """
    fun = _dvApiObj().dv_infer_async
    fun.argtypes = [POINTER(dv_session),
                    POINTER(dv_endpoint),
                    POINTER(dv_model),
                    POINTER(dv_blob),
                    POINTER(dv_blob),
                    c_bool,
                    POINTER(POINTER(dv_infer_request))]
    fun.restype = dv_status_code
    inf_request = POINTER(dv_infer_request)()
    ret = fun(session,
              endpoint,
              model,
              cast(inputs, POINTER(dv_blob)),
              cast(outputs, POINTER(dv_blob)),
              c_bool(enable_stats),
              byref(inf_request))

    if ret != dv_status_code.DV_SUCCESS:
        return ret, None

    return ret, inf_request

def dv_infer_sync_with_flags(session, endpoint, model, inputs, outputs, timeout, options, enable_stats=True):
    fun = _dvApiObj().dv_infer_sync_with_flags
    fun.argtypes = [POINTER(dv_session),
                    POINTER(dv_endpoint),
                    POINTER(dv_model),
                    POINTER(dv_blob),
                    POINTER(dv_blob),
                    c_int,
                    c_bool,
                    POINTER(POINTER(dv_infer_request)),
                    c_uint16]

    fun.restype = dv_status_code
    inf_request = POINTER(dv_infer_request)()
    ret = fun(session,
              endpoint,
              model,
              cast(inputs, POINTER(dv_blob)),
              cast(outputs, POINTER(dv_blob)),
              c_int(timeout),
              c_bool(enable_stats),
              byref(inf_request),
              c_uint16(options))
    if ret != dv_status_code.DV_SUCCESS:
        return ret, None
    return ret, inf_request

# def dv_infer_wait_for_completion(session, inf_list, timeout) -> tuple[dv_status_code ,dv_infer_request] : # fixme::typing hint giving error
def dv_infer_wait_for_completion(session, inf_list, timeout):
    """ blocking call that waits for inference to reach a final state

    Parameters
    ----------
    session : dv_session

    inf_id : dv_infer_request

    timeout : int
        timeout in milliseconds

    Returns
    -------
    dv_status_code

    """
    fun = _dvApiObj().dv_infer_wait_for_completion
    fun.argtypes = [POINTER(dv_session),
                    POINTER(POINTER(dv_infer_request)),
                    c_int,
                    c_int,
                    POINTER(POINTER(dv_infer_request))]
    fun.restype = dv_status_code

    completed_req = POINTER(dv_infer_request)()

    status = fun(session, byref(inf_list), c_int(1), c_int(timeout), byref(completed_req))

    return status, completed_req


def dv_infer_get_req_id(infer_req):
    """
    Returns the status code and request id of an inference

    If return value is DV_SUCCESS req_id will contain the request id of the 
    inference request sent to kinara inference proxy

    Returns
    -------
    dv_status_code : SUCCESS or FAILURE code
    int : req_id
    """
    fun = _dvApiObj().dv_infer_get_req_id
    fun.argtypes = [POINTER(dv_infer_request), POINTER(c_uint)]
    fun.restype = dv_status_code
    req_id = c_uint()
    status = fun(infer_req, byref(req_id))
    return status, req_id


def dv_infer_free (inf_req_obj):
    fun = _dvApiObj().dv_infer_free
    fun.argtypes = [POINTER(dv_infer_request)]
    fun.restype = dv_status_code

    return fun(inf_req_obj)


def dv_infer_get_inflight_count(session):
    """ Returns the number of inflight inference requests for the session
    object for  which the client library has not recieved a response from
    the proxy server

    Parameters
    ----------
    session : dv_session

    Returns
    -------
    dv_status_code

    count: c_int
        number of inference requests in flight

    """
    fun = _dvApiObj().dv_infer_get_inflight_count
    fun.argtypes = [POINTER(dv_session), POINTER(c_int)]
    fun.restype = dv_status_code
    count = c_int()
    status = fun(session,byref(count))
    return status,count

def dv_infer_sync_with_options(session : dv_session, endpoint : dv_endpoint, model : dv_model, inputs, outputs, timeout_ms : int, infer_options : dv_infer_options):
    """ submits an asynchronous inference request with timeout

    Parameters
    ----------
    session : dv_session

    endpoint : dv_endpoint

    model : dv_model
    inputs : list of DVBlob
        pointers buffers that contain the inputs, this must contain pointers
        to each input in order, pointer at index 0 will be used to read data
        for the 1st input etc...

    outputs : list of DVBlob
        ptrs to where outputs of the inference are written, this list must be
        in order, ie 1st output is written to the pointer at index 0, 2nd output
        is written to the pointer at index 1 etc..

    Warnings
    --------
    Please ensure that the object backing the ctypes.c_void_p are not allowed to
    be refcounted and freed or garbage collected - if the pointer becomes invalid
    while a response is being written to the ptr the runtime will segfault

    Returns
    -------
    (dv_status_code, inf_request_id)
        DV_SUCCESS, inf_request_id if the inference was submitted successfully
        to the inference proxy else, inf_request_id is invalid
    """

    fun = _dvApiObj().dv_infer_sync_with_options
    fun.argtypes = [POINTER(dv_session),
                    POINTER(dv_endpoint),
                    POINTER(dv_model),
                    POINTER(dv_blob),
                    POINTER(dv_blob),
                    c_int,
                    POINTER(POINTER(dv_infer_request)),
                    POINTER(dv_infer_options)]

    fun.restype = dv_status_code
    inf_request = POINTER(dv_infer_request)()
    ret = fun(session,
              endpoint,
              model,
              cast(inputs, POINTER(dv_blob)),
              cast(outputs, POINTER(dv_blob)),
              timeout_ms,
              byref(inf_request), 
              infer_options)

    if ret != dv_status_code.DV_SUCCESS:
        return ret, None

    return ret, inf_request

###############################################################################
# dvapi_private function wrappers based on ctypes
###############################################################################
def dv_client_set_logger(logger: Callable[[str], None]):
    """
    Register a logger plugin with client

    Parameters
    ----------
    logger : Callable[[str], None]

    Returns
    -------
    None
    """
    pfn_logger = CFUNCTYPE(None, c_char_p)

    fun = _dvApiObj().dv_client_set_logger
    fun.argtypes = [pfn_logger]
    fun.restype = None

    global logger_ptr
    logger_ptr = pfn_logger(logger)
    fun(logger_ptr)


def dv_client_set_log_level(log_level):
    """
    Set log level for the client to dump desired logs

    Parameters
    ----------
    log_level

    Returns
    -------
    dv_status_code
    """
    fun = _dvApiObj().dv_client_set_log_level
    fun.argtype = [dv_client_log_level]
    fun.restype = dv_status_code

    return fun(log_level)


def dv_session_create_via_unix_socket_with_options(socket_path, timeout):
    """ connects to the proxy via unix socket

    Parameters
    ----------
    socket_path : str
        path to the unix socket
    
    timeout : int
        timeout value in msec

    Returns
    -------
    (dv_status_code, dv_session)
        DV_SUCCESS, dv_session if the connection was successfully established
        if the status is not DV_SUCCESS, dv_session is invalid
    """
    fun = _dvApiObj().dv_session_create_via_unix_socket_with_options
    fun.argtypes = [c_char_p, POINTER(POINTER(dv_session)), POINTER(dv_session_options)]
    fun.restype = dv_status_code
    session = POINTER(dv_session)()
    options = dv_session_options()
    options.timeout_ms = c_int(timeout)
    ret = fun(socket_path.encode('utf-8'), byref(session), byref(options))
    return ret, session


def dv_session_create_via_tcp_ipv4_socket_with_options(ipv4_addr, port, timeout):
    """ connects to the proxy over the network

    Parameters
    ----------
    ip_addr : str
        IPv4 address to connect to

    port : int
        Port number proxy is listening on

    timeout : int
        timeout value in msec

    Returns
    -------
    (dv_status_code, dv_session)
        DV_SUCCESS, dv_session if the connection was successfully established
        if the status is not DV_SUCCESS, dv_session is invalid
    """
    fun = _dvApiObj().dv_session_create_via_tcp_ipv4_socket_with_options
    fun.argtypes = [c_char_p, c_uint, POINTER(POINTER(dv_session)), POINTER(dv_session_options)]
    fun.restype = dv_status_code
    session = POINTER(dv_session)()
    options = dv_session_options()
    options.timeout_ms = c_int(timeout)
    ret = fun(ipv4_addr.encode('utf-8'), c_uint(port), byref(session), byref(options))
    return ret, session

def dv_endpoint_get_hw_statistics(session, ep):
    """
    Parameters
    ----------
    session : dv_session

    ep : dv_endpoint
        endpoint/endpoint_group

    Returns
    -------
    (dv_status_code, endpoint_hw_stats, dv_endpoint)

        endpoint/endpoint_group count

    """
    fun = _dvApiObj().dv_endpoint_get_hw_statistics
    fun.argtypes = [POINTER(dv_session),
                    POINTER(dv_endpoint),
                    POINTER(POINTER(dv_endpoint_hw_statistics)),
                    POINTER(c_int)]
    fun.restype = dv_status_code
    endpoint_hw_stats = POINTER(dv_endpoint_hw_statistics)()
    ep_count = c_int()
    status = fun(session,ep,byref(endpoint_hw_stats),byref(ep_count))
    return status, endpoint_hw_stats, ep_count


def dv_endpoint_free_hw_statistics(ep_hw_stats,count):
    """
    Parameters
    ----------
    ep_hw_stats : dv_endpoint_hw_statistics
        calloc pointer created by dv_endpoint_get_hw_statistics for endpoint_hw_stats

    count : c_int
        number of endpoints in endpoint/endpoint_group

    Returns
    -------
    dv_status_code

    """
    fun = _dvApiObj().dv_endpoint_free_hw_statistics
    fun.argtypes = [POINTER(dv_endpoint_hw_statistics),
                    c_int]
    fun.restype = dv_status_code
    return fun(ep_hw_stats,count)


def dv_endpoint_get_dram_statistics(session,ep):
    """
    Parameters
    ----------
    session : dv_session

    ep : dv_endpoint
        endpoint/endpoint_group handle for particular endpoint or endpoint group
        or NULL for all endpoints

    Returns
    -------
    (dv_status_code, endpoint_dram_statistics, endpoint/endpoint_group count)
        status code, list of endpoint_dram_statistics for the endpoint/endpoint group
    """
    fun = _dvApiObj().dv_endpoint_get_dram_statistics
    fun.argtypes = [POINTER(dv_session),
                    POINTER(dv_endpoint),
                    POINTER(POINTER(dv_endpoint_dram_statistics)),
                    POINTER(c_int)]
    fun.restype = dv_status_code
    ep_count = c_int()
    endpoint_dram_statistics = POINTER(dv_endpoint_dram_statistics)()
    status = fun(session,ep,byref(endpoint_dram_statistics),byref(ep_count))
    return status, endpoint_dram_statistics, ep_count


def dv_endpoint_free_dram_statistics(ep_dram_stats,count):
    """
    Parameters
    ----------
    ep_dram_stats : dv_endpoint_dram_statistics
        calloc pointer created by dv_endpoint_get_dram_statistics for endpoint_dram_statistics

    count : c_int
        number of endpoints in endpoint/endpoint_group

    Returns
    -------
    dv_status_code

    """
    fun = _dvApiObj().dv_endpoint_free_dram_statistics
    fun.argtypes = [POINTER(dv_endpoint_dram_statistics),
                    c_int]
    fun.restype = dv_status_code
    return fun(ep_dram_stats,count)


def dv_endpoint_get_statistics(session,ep):
    """
    Parameters
    ----------
    session : dv_session

    ep : dv_endpoint
        endpoint/endpoint_group

    Returns
    -------
    (dv_status_code, endpoint_statistics, endpoint/endpoint_group count)

    """
    fun = _dvApiObj().dv_endpoint_get_statistics
    fun.argtypes = [POINTER(dv_session),
                    POINTER(dv_endpoint),
                    POINTER(POINTER(dv_endpoint_stats)),
                    POINTER(c_int)]
    fun.restype = dv_status_code
    ep_stats = POINTER(dv_endpoint_stats)()
    ep_count = c_int()
    status = fun(session,ep,byref(ep_stats),byref(ep_count))
    return status, ep_stats, ep_count


def dv_endpoint_free_statistics(ep_stats,count):
    """
    Parameters
    ----------
    ep_stats : dv_endpoint_stats
        calloc pointer created by dv_endpoint_get_statistics for ep_stats

    count : c_int
        number of endpoints in endpoint/endpoint_group

    Returns
    -------
    dv_status_code

    """
    fun = _dvApiObj().dv_endpoint_free_statistics
    fun.argtypes = [POINTER(dv_endpoint_stats),
                    c_int]
    fun.restype = dv_status_code
    return fun(ep_stats,count)


def dv_register_fault_callback(session, fault_handler, user_data=None):
    """ Subscribe to notification event in case of a fault
    Returns DV_SUCCESS on success, else error
    Endpoint handle dv_endpoint_t received as input to the callback function
    denotes the endpoint that has gone faulty Only one client per proxy needs to
    register to this callback to avoid multiple resets of the same endpoint

    Parameters
    ----------
    session : dv_session handle
    fault_handler : pointer to function that does platform-specific hard reset of device
    user_data : pointer to custom user data which will be passed as an argument to the fault handler, default is NULL
    """
    dv_endpoint_fault_callback_fun  = CFUNCTYPE(None, POINTER(dv_endpoint_stats), c_void_p)
    fun = _dvApiObj().dv_register_fault_callback
    fun.argtypes = [POINTER(dv_session), dv_endpoint_fault_callback_fun, c_void_p]
    fun.restype = dv_status_code
    fault_handler_ptr = dv_endpoint_fault_callback_fun(fault_handler)
    return fun(session, fault_handler_ptr, user_data)


def dv_endpoint_bringup(session,ep):
    """ Proxy will attempt to reload all models which were cached earlier.
    If this API returns `DV_MODEL_CACHE_FETCH_FAILURE`, loading of cached
    models have failed, but device can be assumed to be in good state and
    `dv_model_load*` APIs can be used to load the same again.

    Parameters
    ----------
    session : dv_session
    ep : dv_endpoint
        endpoint/endpoint_group
    Returns
    -------
    dv_status_code
    """
    fun = _dvApiObj().dv_endpoint_bringup
    fun.argtypes = [POINTER(dv_session), POINTER(dv_endpoint)]
    fun.restype = dv_status_code
    return fun(session,ep)


def dv_endpoint_takedown(session,ep):
    """ Takes down ANY active endpoint. If the endpoint is part of an
    active endpoint group, requests will be scheduled by Proxy to other
    active endpoints in the group

    Parameters
    ----------
    session : dv_session

    ep : dv_endpoint
        endpoint/endpoint_group

    Returns
    -------
    dv_status_code

    """
    fun = _dvApiObj().dv_endpoint_takedown
    fun.argtypes = [POINTER(dv_session),
                    POINTER(dv_endpoint)]
    fun.restype = dv_status_code
    return fun(session,ep)


def dv_endpoint_takedown_with_options(session,ep,options):
    """
    Takes down ANY active endpoint. If the endpoint is part of an active endpoint group, 
    requests will be scheduled by Proxy to other active endpoints in the group.

    Parameters
    ----------
    session : dv_session

    ep : dv_endpoint
        endpoint/endpoint_group

    options: dv_endpoint_takedown_options

    Returns
    -------
    dv_status_code
    """

    fun = _dvApiObj().dv_endpoint_takedown_with_options
    fun.argtypes = [POINTER(dv_session),
                    POINTER(dv_endpoint),
                    POINTER(dv_endpoint_takedown_options)]
    fun.restype = dv_status_code
    return fun(session,ep,options)


def dv_endpoint_check_status(session,ep):
    """checks the state of the endpoint

    Parameters
    ----------
    session : dv_session

    ep : dv_endpoint
        endpoint/endpoint_group

    Returns
    -------
    (dv_status_code, state)
        Endpoint/endpoint_group current state

    """
    fun = _dvApiObj().dv_endpoint_check_status
    fun.argtypes = [POINTER(dv_session),
                    POINTER(dv_endpoint),
                    POINTER(c_int)]
    fun.restype = dv_status_code
    state = c_int()
    ret = fun(session,ep,byref(state))
    return ret,state


def dv_endpoint_enter_power_state(session, ep, power_state):
    """Switches endpoint to a different supported power state.
    Returns DV_SUCCESS when endpoint is moved to same state as current state; does not perform any action on the endpoint.

    Parameters
    ----------
    session : dv_session

    ep : dv_endpoint
        endpoint/endpoint_group

    power_state : c_int
        state to switch endpoint to
    """
    fun = _dvApiObj().dv_endpoint_enter_power_state
    fun.argtypes = [POINTER(dv_session), POINTER(dv_endpoint), POINTER(c_int)]
    fun.restype = dv_status_code
    p_state = c_int(power_state)
    return fun(session,ep,byref(p_state))


def dv_register_endpoint_power_state_callback(session, power_state_handler, user_data=None):
    """
    Registers a callback function to be called when an endpoint changes power state
    Endpoint handle dv_endpoint_t received as input to the callback function denotes 
    the endpoint which has switched states
    
    Parameters
    ----------
    session : dv_session

    power_state_handler : Callable[[POINTER(dv_endpoint_t), dv_endpoint_power_state, dv_endpoint_power_state, c_void_p], None]
        callback function to be called when an endpoint changes power state

    user_data : c_void_p
        user data pointer to be passed to the callback function

    Returns
    -------
    dv_status_code
    """
    dv_endpoint_power_state_callback = CFUNCTYPE(None, POINTER(dv_endpoint), c_int, c_int, c_void_p)
    fun = _dvApiObj().dv_register_endpoint_power_state_callback
    fun.argtypes = [POINTER(dv_session), dv_endpoint_power_state_callback, c_void_p]
    fun.restype = dv_status_code
    power_state_callback_ptr = dv_endpoint_power_state_callback(power_state_handler)
    return fun(session, power_state_callback_ptr, user_data)


def dv_model_load_from_file_cache(session, ep, model_path, model_name, priority):
    """ loads a model onto the device from the provided file path and caches the model onto disk

    Parameters
    ----------

    session : dv_session
        connected client session

    ep : dv_endpoint
        endpoint on which to load the model

    model_path : str
        path to the model on filesystem

    model_name : str
        a reference name to be assigned to the model

    priority : int
        priority of the model

    Returns
    -------
    (dv_status_code, dv_model)
        an error status and the model handle. if the error status is not
        DV_SUCCESS, dv_model_handle is invalid
    """
    fun = _dvApiObj().dv_model_load_from_file_cache
    fun.argtypes = [
        POINTER(dv_session),
        POINTER(dv_endpoint),
        c_char_p,
        c_char_p,
        c_int,
        POINTER(POINTER(dv_model))]

    fun.restype = dv_status_code

    model = POINTER(dv_model)()

    ret = fun(session,
              ep,
              model_path.encode('utf-8'),
              model_name.encode('utf-8'),
              priority,
              byref(model))

    if ret != dv_status_code.DV_SUCCESS:
        return ret, None

    return ret, model


def dv_model_load_from_blob_cache(session, ep, model_blob, model_name, priority):
    """ loads a model onto the device from the provided file path and caches model onto disk

    Parameters
    ----------
    session : dv_session
        connected client session

    ep : dv_endpoint
        endpoint on which to load the model

    model_blob : dv_blob
        model blob

    model_name : str
        a reference name to be assigned to the model

    priority : int
        priority of the model

    enable_caching : bool
        indicaes if the model should be cached by the server.
        API call fails if the model could not be cached by the server

    Returns
    -------
    (dv_status_code, dv_model_handle)
        an error status and the model handle. if the error status is not
        DV_SUCCESS, dv_model_handle is invalid
    """

    fun = _dvApiObj().dv_model_load_from_blob_cache
    fun.argtypes = [
        POINTER(dv_session),
        POINTER(dv_endpoint),
        POINTER(dv_blob),
        c_char_p,
        c_int,
        POINTER(POINTER(dv_model))]

    fun.restype = dv_status_code

    model = POINTER(dv_model)()

    ret = fun(session,
              ep,
              model_blob,
              model_name.encode('utf-8'),
              priority,
              byref(model))

    if ret != dv_status_code.DV_SUCCESS:
        return ret, None

    return ret, model


def dv_model_load_from_file_with_options(session : dv_session, ep : dv_endpoint, model_file_path : str, options: dv_model_options):
    """Creates a model object and load model contents from blob and transfer it to endpoint

    Notes
    -----
    Model object contains the model handle and model parameters.
    `dv_model_get_loaded_endpoint_list` returns the list of endpoints (individual device dv_endpoint_t handles)
    on which the model was successfully loaded.
    If model load fails on all individual devices representing `endpt`, the API call will error out

    Parameters
    ----------
    session : dv_session
        session handle
    ep : dv_endpoint
        endpoint handle
    model_file_path : str
        path to model file
    options : dv_model_options
        model load option

    Returns
    -------
    (dv_status_code, dv_model)
        an error status and the model handle.
    """

    fun = _dvApiObj().dv_model_load_from_file_with_options
    fun.argtypes = [
        POINTER(dv_session),
        POINTER(dv_endpoint),
        c_char_p,
        POINTER(POINTER(dv_model)),
        POINTER(dv_model_options)]

    fun.restype = dv_status_code

    model = POINTER(dv_model)()

    ret = fun(session,
              ep,
              model_file_path.encode('utf-8'),
              byref(model),
              options)

    if ret != dv_status_code.DV_SUCCESS:
        return ret, None

    return ret, model

def dv_model_load_from_file_with_flags(session : dv_session, ep : dv_endpoint, model_file_path : str, flags : int, model_name : str, priority : dv_model_priority_level, cache : bool, async_flag : bool):
    fun = _dvApiObj().dv_model_load_from_file_with_flags
    fun.argtypes = [
        POINTER(dv_session),
        POINTER(dv_endpoint),
        c_char_p,
        c_char_p,
        c_int,
        c_bool,
        c_bool,
        POINTER(POINTER(dv_model)),
        c_uint16]
    fun.restype = dv_status_code
    model = POINTER(dv_model)()
    ret = fun(session,
              ep,
              model_file_path.encode('utf-8'),
              model_name.encode('utf-8'),
              priority,
              cache,
              async_flag,
              byref(model),
              flags)

    if ret != dv_status_code.DV_SUCCESS:
        return ret, None

    return ret, model

def dv_model_load_from_blob_with_options(session, ep, blob, options):
    """Creates a model object and load model contents from blob and transfer it to endpoint

    Notes
    -----
    Model object contains the model handle and model parameters.
    `dv_model_get_loaded_endpoint_list` returns the list of endpoints (individual device dv_endpoint_t handles)
    on which the model was successfully loaded.
    If model load fails on all individual devices representing `endpt`, the API call will error out

    Parameters
    ----------
    session : dv_session
        session handle
    ep : dv_endpoint
        endpoint handle
    blob : dv_blob_t
        blob of the model
    options : dv_model_options
        model load option

    Returns
    -------
    (dv_status_code, dv_model)
        an error status and the model handle.
    """

    fun = _dvApiObj().dv_model_load_from_blob_with_options
    fun.argtypes = [
        POINTER(dv_session),
        POINTER(dv_endpoint),
        POINTER(dv_blob),
        POINTER(POINTER(dv_model)),
        POINTER(dv_model_options)]

    fun.restype = dv_status_code

    model = POINTER(dv_model)()

    ret = fun(session,
              ep,
              blob,
              byref(model),
              options)

    if ret != dv_status_code.DV_SUCCESS:
        return ret, None

    return ret, model


def dv_model_load_wait_for_completion(model, timeout):
    """Wait for asynchronous model load API call to complete. If the model was loaded specifying
    `async = true`, this API will wait for the model load result to become availalbe. If the model is
    already done loading, the API will return immediately.
    """
    fun = _dvApiObj().dv_model_load_wait_for_completion
    fun.argtypes = [POINTER(dv_model),
                    c_int]

    fun.restype = dv_status_code

    ret = fun(model, timeout)

    return ret

def dv_model_get_parameters_from_file(model_path):
    """ get model parameters without loading into device

    Parameters
    ----------
    model_path : str
        path to the model on filesystem

    Returns
    -------
    (dv_status_code, dv_model)
        DV_SUCCESS if the model was successful
        unloaded model object with the parameters
    """

    fun = _dvApiObj().dv_model_get_parameters_from_file
    fun.argtypes = [c_char_p, POINTER(POINTER(dv_model))]
    fun.restype = dv_status_code

    model = POINTER(dv_model)()

    ret = fun(model_path.encode('utf-8'), byref(model))

    if ret != dv_status_code.DV_SUCCESS:
        return ret, None
    return ret, model


def dv_model_get_parameters_from_blob(model_blob):
    """ get model parameters without loading into device

    Parameters
    ----------
    model_blob : dv_blob
        model blob

    Returns
    -------
    (dv_status_code, dv_model)
        DV_SUCCESS if the unload was successful
        unloaded model object with the parameters
    """

    fun = _dvApiObj().dv_model_get_parameters_from_blob
    fun.argtypes = [POINTER(dv_blob), POINTER(POINTER(dv_model))]
    fun.restype = dv_status_code

    model = POINTER(dv_model)()

    ret = fun(model_blob, byref(model))

    if ret != dv_status_code.DV_SUCCESS:
        return ret, None

    return ret, model


def dv_model_free_parameters(model):
    """ free model parameters

    Parameters
    ----------
    model : dv_model
        model object

    Returns
    -------
    dv_status_code
        DV_SUCCESS if the unload was successful
    """

    fun = _dvApiObj().dv_model_free_parameters
    fun.argtypes = [POINTER(dv_model)]
    fun.restype = dv_status_code
    return fun(model)


def dv_model_get_loaded_endpoint_list(session, model):
    """Returns a list of all connected endpoints + list of indexes in the aforementioned list on which the specified model is loaded.

    Notes
    -----
    `endpt_list` and `endpt_list_size` is populated with the same output as `dv_endpoint_get_list`
    and is managed by the client library - `endpt_list` is not to be freed / deallocated by the caller.
    `loaded_endpt_idxes` is not managed by the client library and must be freed by the caller.

    Parameters
    ----------
    session : dv_session
        connected client session
    model : dv_model
        model object

    Returns
    -------
    (dv_status_code,list(dv_endpoint),endpt_list_size,list(loaded_endpt_idxes),loaded_endpt_list_size)
        dv_status_code          : status code

        endpt_list              : list of all endpoints as returned by dv_endpoint_get_list

        endpt_list_size         : length of endpt_list as returned by dv_endpoint_get_list

        loaded_endpt_idxes      : list of indexes of items in endpt_list which have model loaded

        loaded_endpt_list_size  : length of loaded_endpt_idxes

    """

    fun = _dvApiObj().dv_model_get_loaded_endpoint_list
    fun.argtypes = [POINTER(dv_session),
                    POINTER(dv_model),
                    POINTER(POINTER(dv_endpoint)),
                    POINTER(c_int),
                    POINTER(POINTER(c_int)),
                    POINTER(c_int)]
    fun.restype = dv_status_code

    endpt_list = POINTER(dv_endpoint)()
    endpt_list_size = c_int()
    loaded_endpt_idxes = POINTER(c_int)()
    loaded_endpt_list_size = c_int()

    ret = fun(session, model, byref(endpt_list), byref(endpt_list_size), byref(loaded_endpt_idxes), byref(loaded_endpt_list_size))
    if ret != dv_status_code.DV_SUCCESS:
        return ret, None, None, None, None

    return ret, endpt_list, endpt_list_size, loaded_endpt_idxes, loaded_endpt_list_size


def dv_model_compare_handles(model1, model2):
    """ Compare model handles
    Used to compare if model objects returned by different APIs point to the same model file
    Ex: model handle returned by dv_endpoint_get_statistics and dv_model_load* APIs can be compared
    Returns "true" if they're the same, else false

    Parameters
    ----------
    model1: dv_model_handle_t handle of model to be compared
    model2: dv_model_handle_t handle of model to be compared
    """
    fun = _dvApiObj().dv_model_compare_handles
    fun.argtype = [c_void_p, c_void_p]
    fun.restype = c_bool
    return fun(model1, model1)


def dv_get_dynamic_op_size(model, op_blob_list, op_index):
    """ returns the index and size of a valid output for networks that produce a dynamic output

    Parameters
    ----------
    model: model for which outputs are generated

    op_blob_list: list of outputs for the given model

    op_index: index for which output size is to be returned

    Returns
    -------
    dv_status_code

    op_index: c_int
        size of op chunk to be read

    size_index: c_int
        index at which data is to read
    """
    fun = _dvApiObj().dv_get_dynamic_op_size
    fun.argtypes = [POINTER(dv_model), POINTER(dv_blob), c_int, POINTER(c_int), POINTER(c_int)]
    fun.restype = dv_status_code

    op_size = c_int()
    size_index = c_int()

    status = fun(model, cast(op_blob_list, POINTER(dv_blob)), op_index, byref(op_size), byref(size_index))

    return status, op_size, size_index

def dv_session_create_via_named_pipe(named_pipe, session):
    """ Create a session to the server using windows named pipe.
        Returns DV_SUCCESS on success, else error

    Parameters
    ----------
    named_pipe: server pipe

    session : dv_session
        session handle

    Returns
    -------
    dv_status_code

    """
    fun = _dvApiObj().dv_session_create_via_named_pipe
    fun.argtypes = [c_char_p, POINTER(POINTER(dv_session))]
    fun.restype = dv_status_code
    session = POINTER(dv_session)()
    return fun(named_pipe.encode('utf-8'), byref(session))

def dv_infer_wait_for_all_completion(session, inf_list, timeout):
    """ Monitor multiple inference request object and wait until, all of the
    request updates/changes run status If the list is empty it waits until,
    atleast one inference request submitted in the session changes run status and
    API keeps track of status change for a inference request and run status is
    reported only once.
   
    For non-empty request list, application needs to remove the completed
    inference object from the list. Request completion status can be reported
    multiple times.

    Parameters
    ----------
    session : dv_session
        session handle

    inf_list : dv_infer_request
        inference request object list

    timeout : int
        maximum time in mili seconds to wait for inference request to change status
        (defaults to 60 seconds in case timeout is passed as -1)

    Returns
    -------
    dv_status_code

    completed_inf_list : list of completed inferences

    completed_inf_count : count of completed inferences

    """
    fun = _dvApiObj().dv_infer_wait_for_all_completion
    fun.argtypes = [POINTER(dv_session),
                    POINTER(POINTER(dv_infer_request)),
                    c_int,
                    c_int,
                    POINTER(POINTER(dv_infer_request)),
                    POINTER(c_int)]
    fun.restype = dv_status_code
    completed_inf_list = POINTER(dv_infer_request)()
    completed_inf_count = c_int()
    status = fun(session, byref(inf_list), c_int(1), c_int(timeout), byref(completed_inf_list), byref(completed_inf_count))
    if status != dv_status_code.DV_SUCCESS:
        return status, None, None
    return status, completed_inf_list, completed_inf_count

def dv_get_endpoint_busyness(session, endpoint):
    """ Get the busyness for the endpoint.
    Server provides busyness of any one endpoint at the time of request 
    Memory for is_busy bool variable should be provided by client app 
    Returns DV_SUCCESS on success, else error

    Parameters
    ----------
    session : dv_session
        session handle
    endpoint : dv_endpoint
        endpoint handle for particular endpoint

    Returns
    -------
    dv_status_code

    is_busy : bool
    """
    fun = _dvApiObj().dv_get_endpoint_busyness
    fun.argtypes = [POINTER(dv_session),
                    POINTER(c_bool)]
    fun.restype = dv_status_code
    is_busy = c_bool()
    status = fun(session, endpoint, byref(is_busy))
    return status, is_busy

def dv_infer_async_with_options(session, endpoint, model, inputs, outputs, infer_options):
    """submits an asynchronous inference request.

    Parameters
    ----------
    session : dv_session
        session handle

    endpoint : dv_endpoint
        endpoint handle for particular endpoint

    model : dv_model

    inputs : list of DVBlob
        pointers buffers that contain the inputs, this must contain pointers
        to each input in order, pointer at index 0 will be used to read data
        for the 1st input etc...

    outputs : list of DVBlob
        ptrs to where outputs of the inference are written, this list must be
        in order, ie 1st output is written to the pointer at index 0, 2nd output
        is written to the pointer at index 1 etc..

    infer_options : dv_infer_options

    Warnings
    --------
    Please ensure that the object backing the ctypes.c_void_p are not allowed to
    be refcounted and freed or garbage collected - if the pointer becomes invalid
    while a response is being written to the ptr the runtime will segfault

    Returns
    -------
    (dv_status_code, dv_infer_request)
        DV_SUCCESS, inf_request_id if the inference was submitted successfully
        to the inference proxy else, inf_request_id is invalid
    """
    fun = _dvApiObj().dv_infer_async_with_options
    fun.argtypes = [POINTER(dv_session),
                    POINTER(dv_endpoint),
                    POINTER(dv_model),
                    POINTER(dv_blob),
                    POINTER(dv_blob),
                    POINTER(POINTER(dv_infer_request)),
                    POINTER(dv_infer_options)]
    fun.restype = dv_status_code
    inf_obj = POINTER(dv_infer_request)()
    status = fun(session,
                 endpoint,
                 model,
                 cast(inputs, POINTER(dv_blob)),
                 cast(outputs, POINTER(dv_blob)),
                 byref(inf_obj),
                 infer_options)

    if status != dv_status_code.DV_SUCCESS:
        return status, None

    return status, inf_obj


def dv_fetch_outputs_by_layer_name(inf_obj : dv_infer_request, src_op_layer_name : str) -> typing.Tuple[dv_status_code, typing.List[dv_blob]]:
    op_blobs = POINTER(dv_blob)()
    num_op = c_int()
    fun = _dvApiObj().dv_fetch_outputs_by_layer_name
    fun.argtypes = [POINTER(dv_infer_request), c_char_p, POINTER(POINTER(dv_blob)), POINTER(c_int)]
    fun.restype = dv_status_code

    ret = fun(inf_obj, src_op_layer_name.encode('utf-8'), byref(op_blobs), byref(num_op))
    output_blobs = []
    if ret == dv_status_code.DV_SUCCESS:
        for i in range(num_op.value):
            output_blobs.append(op_blobs[i])
    return ret, output_blobs


def dv_reg_read(session, endpoint, address, value):
    """ reads the register at the address and returns the value

    Parameters
    ----------
    session : dv_session
        session handle

    endpoint : dv_endpoint
        endpoint handle for particular endpoint

    address : register address

    value: the value saved at the provided address

    """
    fun = _dvApiObj().dv_reg_read
    fun.argtypes = [POINTER(dv_session), POINTER(dv_endpoint), c_uint32, POINTER(c_uint32)]
    fun.restype = dv_status_code
    return fun(session, endpoint, address, value)

def dv_reg_write(session, endpoint, address, value):
    """ writes the value at the address

    Parameters
    ----------
    session : dv_session
        session handle

    endpoint : dv_endpoint
        endpoint handle for particular endpoint

    address : register address

    value: the value to be written at the provided address

    """
    fun = _dvApiObj().dv_reg_write
    fun.argtypes = [POINTER(dv_session), POINTER(dv_endpoint), c_uint32, c_uint32]
    fun.restype = dv_status_code
    return fun(session, endpoint, address, value)

def dv_mem_read(session, endpoint, address, value):
    """ reads from the memory address and returns the value

    Parameters
    ----------
    session : dv_session
        session handle

    endpoint : dv_endpoint
        endpoint handle for particular endpoint

    address : memory address

    value: the value saved at the provided address

    """
    fun = _dvApiObj().dv_mem_read
    fun.argtypes = [POINTER(dv_session), POINTER(dv_endpoint), c_uint32, POINTER(c_uint32)]
    fun.restype = dv_status_code
    return fun(session, endpoint, address, value)

def dv_mem_write(session, endpoint, address, value):
    """ writes the value at the memory address

    Parameters
    ----------
    session : dv_session
        session handle

    endpoint : dv_endpoint
        endpoint handle for particular endpoint

    address : memory address

    value: the value to be written at the provided address

    """
    fun = _dvApiObj().dv_mem_write
    fun.argtypes = [POINTER(dv_session), POINTER(dv_endpoint), c_uint32, c_uint32]
    fun.restype = dv_status_code
    return fun(session, endpoint, address, value)

def dv_bulk_read(session, endpoint, address, size, buffer, perf_mode=False, completion_time=None):
    """ Read in bulk(size) from the endpoint memory address to the buffer
    
    Parameters
    ----------
    session : dv_session
        session handle

    endpoint : dv_endpoint
        endpoint handle for particular endpoint

    address : endPoint memory address

    size : size to read

    buffer: buffer to be filled

    perf_mode : enable perf mode in the interface, data will not be copied

    completion_time : time to complete the operation
    
    """
    fun = _dvApiObj().dv_bulk_read
    fun.argtypes = [POINTER(dv_session), POINTER(dv_endpoint), c_uint32, c_uint32, POINTER(c_uint32), c_bool, POINTER(c_float)]
    fun.restype = dv_status_code
    return fun(session, endpoint, address, size, buffer, perf_mode, completion_time)


###############################################################################
# Object oriented APIs
###############################################################################

def _one_time_callable_cleanup_function(f):
    """ a utility decorator that ensures that member function that must be called
    only once (like connection close, deallocate memory etc..) are called just
    once, for cases where the function may be called on garbage collection or by
    the users themselves

    Methods marked with this decorator are definitely not idempotent and calling
    them repeatedly might cause issues
    """
    @functools.wraps(f)
    def wrapper(target_self):
        attr = getattr(target_self, '_dv_is_deleted', None)
        if attr is None:
            retval = f(target_self)
            target_self._dv_is_deleted = retval
        else:
            return target_self._dv_is_deleted
    return wrapper

class DVTensor:
    """ a convenience wrapper that encapsulates tensor along with its
    parameters for convenience

    Attributes
    ----------
    numpy_data : numpy array of type int8
        actual tensor data

    params : DVIpTensorParam | DVOpTensorParam
        tensor parameter object, currently this will
        only contain DVOpTensorParam
    """
    def __init__(self, numpy_data, params, mem_desc=None, offset=0):
        self.numpy_data : np.ndarray = numpy_data
        self.params     = params
        self.mem_desc   = mem_desc
        self.offset     = offset

    def as_classification_output(self):
        """ treat the underlying int8 data as the output of a classification
        network. this operation is valid if `params.is_structure_output` and
        output_type on the DVLoadedModel is `DV_LAYER_OUTPUT_TYPE.DV_MODEL_NW_CLASSIFICATION`

        At most 10 rows will be returned.

        Returns
        -------
        dvClassificationOutput
        """
        ptr = self.numpy_data.ctypes.data_as(c_char_p)
        return  cast(ptr, POINTER(dvClassificationOutput)).contents

    def as_fp32softmax_classification(self, n=10):
        """ treats the underlying data as if it the channels are pure probabilities
        given by a softmax layer, and return a DVClassificationOutput

        Parameters
        ----------
        n : int
            number of rows to return - this must be less than the number
            of channels in the output layer

        Returns
        -------
        dvClassificationOutput
        """
        channels = self.params.nch
        buf = self.numpy_data.astype(np.float32)
        items = (dvClassificationItem * n)()

        for c in range(0, channels):
            for topk in range(0, n):
                if items[topk].score < buf[c]:
                    for swap in reversed(range(topk, n)):
                        items[swap].score = items[swap-1].score
                        items[swap].label = items[swap-1].label
                    items[topk].label = c
                    items[topk].score = buf[c]
                    break

        return dvClassificationOutput(n, items)

    def as_detection_output(self):
        """ treat the underlying data as structured output of a detection type
        network. this operation is valid if `params.is_structured_output`
        and the output_type is `DV_LAYER_OUTPUT_TYPE.DV_MODEL_NW_DETECTION`

        At most 200 detection boxes will be returned

        Returns
        -------
        dvDetectionOutput
        """
        ptr = self.numpy_data.ctypes.data_as(c_char_p)
        return cast(ptr, POINTER(dvDetectionOutput)).contents

class TimeStamp:
    def __init__(self, timestamp) -> None:
        self.tv_sec = timestamp.tv_sec
        self.tv_nsec = timestamp.tv_nsec

    def __str__(self) -> str:
        return (" {} seconds, {} nanoseconds").format(self.tv_sec, self.tv_nsec)
class DVInferStatistics:
    """ Inference performance stats

    Attributes
    ----------
    ep_hw_sys_clk : int
    
    ep_hw_dram_clk : int
        DRAM clock at the time of inference

    ep_hw_total_inference_cycles: int
        total cycles taken to compute inference in hardware, including floating point computation

    ep_hw_fp_cycles: int
        cycles taken to compute floating point operation in hardware
    
    tensor_read_time : float
        time taken in miliseconds to transfer output(s) from ep hardware dram to host dram

    tensor_write_time : float
        time taken in miliseconds to submit inference request to ep hardware
    """
    def __init__(self, inf_stats):
        self.ep_hw_sys_clk                  = inf_stats[0].ep_hw_sys_clk
        self.ep_hw_dram_clk                 = inf_stats[0].ep_hw_dram_clk
        self.ep_hw_total_inference_cycles   = inf_stats[0].ep_hw_total_inference_cycles
        self.ep_hw_fp_cycles                = inf_stats[0].ep_hw_fp_cycles
        self.input_transfer_time            = inf_stats[0].input_transfer_time
        self.output_transfer_time           = inf_stats[0].output_transfer_time
        self.ep_queue_submission_time       = inf_stats[0].ep_queue_submission_time
        self.input_transfer_start_time_stamp = TimeStamp(inf_stats[0].input_transfer_start_time_stamp)
        self.output_transfer_start_time_stamp = TimeStamp(inf_stats[0].output_transfer_start_time_stamp)
        self.inference_start_time_stamp = TimeStamp(inf_stats[0].inference_start_time_stamp)
        # inference_execution_time is reported in microseconds by the firmware
        self.inference_execution_time       = inf_stats[0].inference_execution_time

    @property
    def npu_compute_ms(self):
        """NPU core compute time in milliseconds.

        Prefers inference_execution_time (firmware-reported microseconds)
        when available, falling back to cycles / clock derivation.
        """
        if self.inference_execution_time and self.inference_execution_time > 0:
            return self.inference_execution_time / 1000.0
        if self.ep_hw_sys_clk > 0:
            return self.ep_hw_total_inference_cycles / (self.ep_hw_sys_clk * 1000)
        return 0.0

    def __str__(self):
        return ('dvInfStats: HwInfTime={:.2f}ms, HwFpComputationTime={:.2f}ms, HwPerfStats(sclk={} MHz, dclk={} MHz, TotalCycles={}, FPCycles={}), ' + \
            'ipXferTime={:.2f}ms, opXferTime={:.2f}ms, queueSubmissionTime={:.2f}ms, input_transfer_start_time_stamp={}, output_transfer_start_time_stamp={}, inference_start_time_stamp={}').format(
                self.ep_hw_total_inference_cycles/(self.ep_hw_sys_clk * 1000),
                self.ep_hw_fp_cycles/(self.ep_hw_sys_clk * 1000),
                self.ep_hw_sys_clk,
                self.ep_hw_dram_clk,
                self.ep_hw_total_inference_cycles,
                self.ep_hw_fp_cycles,
                self.input_transfer_time,
                self.output_transfer_time,
                self.ep_queue_submission_time,
                self.input_transfer_start_time_stamp,
                self.output_transfer_start_time_stamp,
                self.inference_start_time_stamp
                )

class DVLLMInferInfo:
    def __init__(self, llm_info : dv_infer_llm_info):
        self.llm_infer_resp_num_valid_tokens = llm_info[0].llm_infer_resp_num_valid_tokens


class DVInferRequest:
    """ response that represents a response that is yet to
    reach a terminal state

    Attributes
    ----------
    request_id
        request_id returned by the proxy for the enqueued
        inference request

    inputs: [DVTensor]
        original inputs passed to the sync or async inference APIs

    outputs : [DVTensor]
        outputs of the inference

    model_obj : DVModel
        dv_model object

    stats : DVInferStatistics
        statistics collected for the inference request. this information
        is available only after `wait_for_completion` is invoked on the inference
        which ran succesfully

    status: dv_inference_status
    """

    def __init__(self, inputs : typing.List[DVTensor], outputs : typing.List[DVTensor], inf_req : dv_infer_request, model_obj):
        self._inf_req       = inf_req
        self.handle         = inf_req[0].handle
        self.session        = inf_req[0].session
        self._model          = inf_req[0].model
        self.ep_queued      = DVEndpoint(inf_req[0].ep_submitted[0])
        self.ep_submitted   = None
        self.model          = model_obj
        self.inputs         = inputs
        self.outputs        = outputs
        self.status         = inf_req[0].status
        self.stats          = None
        self.llm_infer_info = None

        if self.status == dv_inference_status.DV_INFERENCE_STATUS_COMPLETED or self.status == dv_inference_status.DV_INFERENCE_STATUS_FAILED:
            self.stats = DVInferStatistics(inf_req[0].stats)
            try:
                self.ep_submitted = DVEndpoint(inf_req[0].ep_submitted[0])
            except:
                self.ep_submitted = None
            
            try:
                self.llm_infer_info = DVLLMInferInfo(self._inf_req[0].llm_infer_info)
            except:
                self.llm_infer_info = None


    @_one_time_callable_cleanup_function
    def mark_complete(self):
        """DVInferRequest method that finish the life cycle for the inference request.
        API will free up the associated memory for the inference request and will no longer be accessible.
        After completion, any operation on the inference request will be invalid.

        Returns
        -------
        dv_status_code

        """
        return dv_infer_free(self._inf_req)
    
    def __del__(self):
        self.mark_complete()

    def wait_for_completion(self, timeout=50000):
        """ blocks on the current thread till either inference completes
        or the timeout supplied in the initial inference call is exceeded
        
        if the request has already reached a terminal state, `DV_SUCCESS` is returned

        Parameters
        ----------
        timeout : int
            timeout to wait for inference response

        Returns
        -------
        dv_status_code
        """

        # return immediately if we already have success
        if self.status == dv_inference_status.DV_INFERENCE_STATUS_COMPLETED or self.status == dv_inference_status.DV_INFERENCE_STATUS_FAILED:
            return dv_status_code.DV_SUCCESS

        status, completed_req = dv_infer_wait_for_completion(self.session, self._inf_req, timeout)
        if status != dv_status_code.DV_SUCCESS:
            return status

        self.stats = DVInferStatistics(completed_req[0].stats)
        self.status = completed_req[0].status
        try:
            self.ep_submitted = DVEndpoint(completed_req[0].ep_submitted[0])
        except:
            self.ep_submitted = None
        
        try:
            self.llm_infer_info = DVLLMInferInfo(self._inf_req[0].llm_infer_info)
        except:
            self.llm_infer_info = None
        return status

    def get_output_tensors(self):
        return self.outputs

    def get_infer_req_id(self):
        """ 
        Returns the status code and request id of an inference

        If return value is DV_SUCCESS req_id will contain the request id of the 
        inference request sent to kinara inference proxy

        Returns
        -------
        dv_status_code : SUCCESS or FAILURE code
        int : req_id
        """
        return dv_infer_get_req_id(self._inf_req)


    def get_performance_statistics(self):
        """ 
        returns the inference statistics for a completed inference request
        stats are generated only for inference that complete successfully
        
        Returns
        -------
        (dv_status_code, DVInferStatistics)


        Notes
        -----
        The perf stats pertaining to an inference are not held by the
        server for a long duration and will be overwritten when it needs
        more memory, in such scenario the dv_status_code returned will be
        DV_INF_REQUEST_INVALID_HANDLE
        """
        return dv_status_code.DV_SUCCESS, self.stats

    def fetch_outputs_for_layer(self, src_op_layer_name : str) -> typing.Tuple[dv_status_code, typing.List[dv_blob]]:
        return dv_fetch_outputs_by_layer_name(self._inf_req, src_op_layer_name)

class DVModelInputPreProcessParam:
    """ input image preprocessing as specified by the model

    Attributes
    ----------

    aspect_resize : bool
    mirror : bool
    center_crop : bool
    bgr2rgb : bool
    mean : float
    scale : float
    qn : float
    aspect_resize_scale : float
    interpolation : int
        corresponds to the following enum (check See Also)
        NN = 0
        LINEAR = 1
        CUBIC = 2
        AREA = 3
        LANCZOS4 = 4

    See Also
    --------
    dutils.preprocessor (preprocessor.py script shipped in debug utils)
    """
    def __init__(self, input_preproc_param, nch):
        self.qn             = input_preproc_param.qn
        self.aspect_resize  = input_preproc_param.aspect_resize
        self.mirror         = input_preproc_param.mirror
        self.center_crop    = input_preproc_param.center_crop
        self.bgr_to_rgb     = input_preproc_param.bgr_to_rgb
        self.interpolation  = input_preproc_param.interpolation
        self.is_signed      = input_preproc_param.is_signed
        self.bpp            = input_preproc_param.bpp
        self.output_scale   = input_preproc_param.output_scale
        self.aspect_resize_scale = input_preproc_param.aspect_resize_scale
        self.mean           = []
        self.scale          = []
        self.offset      = input_preproc_param.offset
        self.qmode              = input_preproc_param.qmode

        if nch <= 3:
            for i in range(0,nch):
                self.mean.append(input_preproc_param.mean[i])
                self.scale.append(input_preproc_param.scale[i])
    def __str__(self):
        ipp_str = 'qn={}, scale={}, mean={}, aspect_resize={}, mirror={}, center_crop={}, bgr_to_rgb={} interpolation={}, offset:{}, qmode:{},'.format(
            self.qn, self.scale, self.mean, self.aspect_resize, self.mirror, self.center_crop, self.bgr_to_rgb, self.interpolation, self.offset, self.qmode)
        return ipp_str


class DVModelInputParam:
    """ Input tensor params

    Attributes
    ----------
    id : int
        layer_id

    size : int
        size in bytes of the input tensor

    width : int
    height : int
    depth : int
    channels : int
    num : int
    input_format : int

    layer_name : str
        name of the layer as per the generated caffe.prototxt

    layer_type : str
        type of the neural network layer

    preproc_params : DVIpTensorPreProcParams
        represents the preprocessing that needs to be done on raw JPEG images
        before they can be quantized and supplied to the model
    """
    def __init__(self, input_param):
        self.layer_id           = input_param.layer_id
        self.blob_id            = input_param.blob_id
        self.layer_name         = input_param.layer_name.decode('utf-8')
        self.blob_name          = input_param.blob_name.decode('utf-8')
        self.layer_type         = input_param.layer_type.decode('utf-8')
        self.layout             = input_param.layout.decode('utf-8')
        self.size               = input_param.size
        self.width              = input_param.width
        self.height             = input_param.height
        self.depth              = input_param.depth
        self.nch                = input_param.nch
        self.bpp                = input_param.bpp
        self.batch_size         = input_param.batch_size
        self.num                = input_param.num
        self.src_graph_layer_name = input_param.src_graph_layer_name.decode('utf-8')
        self.preprocess_param   = DVModelInputPreProcessParam(input_param.preprocess_param[0], input_param.nch)
    def __str__(self):
        ipparam_str = 'layer_id={}, blob_id={}, layer_name={}, blob_name={}, layer_type={}, layout={}, size={}, width={}, height={}, depth={}, nch={}, bpp={}, batch_size={}, num:{}, src_graph_layer_name:{}, \npreprocess_params:{}'.format(
            self.layer_id, self.blob_id, self.layer_name, self.blob_name, self.layer_type, self.layout,
            self.size, self.width, self.height, self.depth,
            self.nch, self.bpp, self.batch_size, self.num, self.src_graph_layer_name, self.preprocess_param)
        return ipparam_str


class DVModelOutputPostProcessParam:
    def __init__(self, op_postproc_param):
        self.qn                 = op_postproc_param.qn
        self.is_struct_format   = op_postproc_param.is_struct_format
        self.is_float           = op_postproc_param.is_float
        self.is_signed          = op_postproc_param.is_signed
        self.output_scale       = op_postproc_param.output_scale
        self.offset             = op_postproc_param.offset

    def __str__(self):
        ip_preproc_str = 'qn={}, is_struct_format={}, is_float={}, is_signed={}, output_scale = {}, offset = {}'.format(
            self.qn, self.is_struct_format, self.is_float, self.is_signed, self.output_scale, self.offset)
        return ip_preproc_str


class DVModelOutputParam:
    """ describes an output tensor

    Attributes
    ----------
    id : str
        layer_id

    size : int
        size in bytes of the tensor

    width : int
    height : int
    depth : int
    channels : int
    num : int
    ncl : int

    bpp : int
        bytes per pixel

    is_struct_format : bool
        specifies if the output must be cast to a C struct type specified
        in `dvoutstruct.h`

    is_output_signed : bool
        specifies if the output is signed

    is_output_float : bool
        specifies if the output is to be treated as 32 bit floating point

    output_qn : float

    fused_layer_name : str

    layer_name : str
        name of the neural network layer

    layer_type : str
        type of the neural network layer that produces this output

    Notes
    -----

    usage of `bpp` and `is_output_signed`

    =========  ================  ===========
    bpp value  is_output_signed  numpy dtype
    =========  ================  ===========
    1          False             np.uint8
    1          True              np.int8
    2          False             np.uint16
    2          True              np.int16
    4          False             np.uint32
    4          True              np.int32
    =========  ================  ===========

    """
    def __init__(self, output_param):
        self.layer_id                   = output_param.layer_id
        self.blob_id                    = output_param.blob_id
        self.fused_parent_id            = output_param.fused_parent_id
        self.layer_name                 = output_param.layer_name.decode('utf-8')
        self.blob_name                  = output_param.blob_name.decode('utf-8')
        self.layer_fused_parent_name    = output_param.layer_fused_parent_name.decode('utf-8')
        self.layer_type                 = output_param.layer_type.decode('utf-8')
        self.layout                     = output_param.layout.decode('utf-8')
        self.size                       = output_param.size
        self.width                      = output_param.width
        self.height                     = output_param.height
        self.depth                      = output_param.depth
        self.nch                        = output_param.nch
        self.bpp                        = output_param.bpp
        self.num_classes                = output_param.num_classes
        self.layer_output_type          = output_param.layer_output_type
        self.num                        = output_param.num
        self.max_dynamic_id             = output_param.max_dynamic_id
        self.src_graph_layer_name       = output_param.src_graph_layer_name.decode('utf-8')
        self.postprocess_param          = DVModelOutputPostProcessParam(output_param.postprocess_param[0])

    def __str__(self):
        opparam_str = 'layer_id={}, blob_id={}, fused_parent_id={},layer_name={}, blob_name={}, layer_fused_parent_name={}, layer_type={}, layout={}, size={}, width={}, height={}, depth={}, nch={}, bpp={}, num_classes={}, layer_output_type={}, num={}, max_dynamic_id={}, src_graph_layer_name={}, postprocess_param={}'.format(
            self.layer_id, self.blob_id, self.fused_parent_id, self.layer_name, self.blob_name, self.layer_fused_parent_name, self.layer_type, self.layout,
            self.size, self.width, self.height, self.depth,
            self.nch, self.bpp, self.num_classes, self.layer_output_type, self.num, self.max_dynamic_id, self.src_graph_layer_name, self.postprocess_param)

        return opparam_str
class DVModelLoadOptions:
    def __init__(self,model_load_options):
        self._model_load_options    = model_load_options
        self.model_name             = model_load_options[0].model_name.decode('utf-8')
        self.priority               = model_load_options[0].priority
        self.cache                  = model_load_options[0].cache
        self.model_type             = model_load_options[0].model_type
        self.model_async            = getattr(model_load_options[0],"async")

    @classmethod
    def model_load_option(cls, model_name = "model.dvm", priority = 1, cache = False, model_async = False, model_type : dv_model_type = dv_model_type.DV_MODEL_TYPE_ARA2_CNN):
        m_options = dv_model_options(model_name.encode('utf-8'), c_int(priority), c_bool(cache), c_bool(model_async), dv_model_type(model_type))
        m_options_pointer = pointer(m_options)
        dv_model_load_options = cls(m_options_pointer)
        return dv_model_load_options

    def __str__(self):
        model_load_options = "model_name={}, priority={}, cache={}, async={}, model_type={}".format(self.model_name, self.priority, self.cache, self.model_async, self.model_type)
        return model_load_options


class DVLLMModelCfgUpdateParams:
    def __init__(self,llm_config_update_params: dv_llm_cfg_upd_req):
        self._llm_config_options = llm_config_update_params
        self.top_k = llm_config_update_params[0].top_k
        self.top_p = llm_config_update_params[0].top_p
        self.temperature = llm_config_update_params[0].temperature
        self.repetition_penalty = llm_config_update_params[0].repetition_penalty
        self.target_token_post_mcp = llm_config_update_params[0].target_token_post_mcp
        self.target_token_pre_mcp = llm_config_update_params[0].target_token_pre_mcp
        self.target_prompt_post_mcp = llm_config_update_params[0].target_prompt_post_mcp
        self.target_prompt_pre_mcp = llm_config_update_params[0].target_prompt_pre_mcp
        self.draft_token_post_mcp = llm_config_update_params[0].draft_token_post_mcp
        self.draft_token_pre_mcp = llm_config_update_params[0].draft_token_pre_mcp
        self.draft_prompt_post_mcp = llm_config_update_params[0].draft_prompt_post_mcp
        self.draft_prompt_pre_mcp = llm_config_update_params[0].draft_prompt_pre_mcp

    @classmethod
    def llm_model_cfg_options(cls, top_k = 1, top_p = 0.9, temperature = 0.7, repetition_penalty = 1.12, target_token_post_mcp = 0, target_token_pre_mcp = 0, target_prompt_post_mcp = 0, target_prompt_pre_mcp = 0, draft_token_post_mcp = 1, draft_token_pre_mcp = 1, draft_prompt_post_mcp = 1, draft_prompt_pre_mcp = 1):
        cfg_options = dv_llm_cfg_upd_req(c_uint32(top_k),c_float(top_p),c_float(temperature),c_float(repetition_penalty),c_uint32(target_token_post_mcp),c_uint32(target_token_pre_mcp),c_uint32(target_prompt_post_mcp),c_uint32(target_prompt_pre_mcp),c_uint32(draft_token_post_mcp),c_uint32(draft_token_pre_mcp),c_uint32(draft_prompt_post_mcp),c_uint32(draft_prompt_pre_mcp))
        cfg_options_pointer = pointer(cfg_options)
        dv_llm_cfg_options = cls(cfg_options_pointer)
        return dv_llm_cfg_options


class DVInferOptions:
    def __init__(self, infer_options_handle : dv_infer_options):
        self._infer_options_handle = infer_options_handle
        self.enable_stats : bool = infer_options_handle[0].enable_stats
        self.infer_type : dv_infer_type = infer_options_handle[0].infer_type
        self.active_tokens : c_int64 = infer_options_handle[0].active_tokens
        self.valid_tokens : c_int64 = infer_options_handle[0].valid_tokens
        self.tokens_to_skip : c_int32 = infer_options_handle[0].tokens_to_skip

    @classmethod
    def dv_infer_options(cls, enable_stats : bool, infer_type : dv_infer_type, active_tokens: int, valid_tokens : int, tokens_to_skip: int):
        m_infer_options = dv_infer_options(c_bool(enable_stats), c_int(infer_type), c_uint64(active_tokens), c_uint32(valid_tokens), c_uint32(tokens_to_skip))
        m_infer_options_ptr = pointer(m_infer_options)
        return cls(m_infer_options_ptr)

    def __str__(self):
        # todo ::
        return "DVInferOptions, this need to be filled "

class DVBlob(Structure):
    """ specification of a tensor

    Attributes
    ----------

    handle : c_void_p
        void pointer to the underlying buffer or memory mapped file

    offset : int
        applicable for memory mapped files - offset at which the tensor
        is located in the file

    size : int
        size of the tensor in a memory mapped file starting for offset

    blob_type : dv_blob_type
        type of blob.
    """
    _fields_ = [("handle", c_void_p),
                ("offset", c_uint64),
                ("size", c_uint64),
                ("blob_type", c_int)]

    def for_shm_desc(self, shm_desc, offset, size):
        """ utility that sets the attribute values for a mmapped file

        Parameters
        ----------
        shm_desc : dv_shm_descriptor
            shm_desc as returned by dv_shmfile_register

        offset : int

        size : int
        """
        ptr = cast(shm_desc, c_void_p)
        self.handle = ptr
        self.offset = c_uint64(offset)
        self.size = c_uint64(size)
        self.blob_type = c_int(dv_blob_type.DV_BLOB_TYPE_SHM_DESCRIPTOR)

    def for_raw_pointer(self, ctypes_void_ptr, size):
        """ set attribute values for a raw ctypes buffer

        Parameters
        ----------
        ctypes_void_ptr : c_void_p
            pointer to a buffer, normally we pass np.array().ctypes.data_as(c_void_p)

        size : int
        """
        self.handle = ctypes_void_ptr
        self.offset = c_uint64(0)
        self.size = c_uint64(size)
        self.blob_type = c_int(dv_blob_type.DV_BLOB_TYPE_RAW_POINTER)

class DVModel:
    """ a model that has been loaded onto an endpoint

    Attributes
    ----------

    model : DVModel
        Underlying model buffer that was parsed - this is stateless

    connection : DVConnection
        The proxy connection which was used to load the model

    """
    def __init__(self, model):
        self._model               = model
        self.handle               = model.handle
        self.session              = model.session
        self.endpoint             = model.endpoint
        self.version              = model.version
        self.name                 = model.name.decode('utf-8')
        self.internal_name        = model.internal_name.decode('utf-8')
        self.priority             = model.priority
        self.num_inputs           = model.num_inputs
        self.num_outputs          = model.num_outputs
        self.input_param          = []
        self.output_param         = []
        self.num_compiler_config  = model.num_compiler_config
        self.compiler_stats       = []
        self.cp_layer             = model.cp_layer
        self.model_type           = dv_model_type(model.model_type)
        if dv_is_null(model.model_load_options):
            self.model_load_options   = None
        else:
            self.model_load_options   = DVModelLoadOptions(model.model_load_options)
        if (self.model_type is not (dv_model_type.DV_MODEL_TYPE_ARA2_LLM_DYN_V2)) and (self.model_type is not (dv_model_type.DV_MODEL_TYPE_ARA2_LLM)):
            for i in range (0, self.num_compiler_config):
                self.compiler_stats.append(DVCompilerStats(model.compiler_stats[i]))

            for i in range(0, self.num_inputs):
                if model.input_param[i] != None:
                    self.input_param.append(DVModelInputParam(model.input_param[i]))
                else :
                    print("DVModelInputParam idx:{} is none", i)

            for j in range(0, self.num_outputs):
                self.output_param.append(DVModelOutputParam(model.output_param[j]))

    def __str__(self):
        compiler_stats_str = ', '.join(map(str, self.compiler_stats))
        input_param_str    = ', '.join(map(str, self.input_param))
        output_param_str   = ', '.join(map(str, self.output_param))
        model_str = 'handle={}, version={}, name={}, internal_name={}, priority={}, num_inputs={}, num_outputs={}, num_compiler_config={}, model_load_options={}, cp_layer={}, \ninput_parameters=[{}], \noutput_parameters=[{}], \ncompiler_statistics=[{}]'.format(
            self.handle, self.version, self.name, self.internal_name, self.priority, self.num_inputs, self.num_outputs, self.num_compiler_config, self.model_load_options, self.cp_layer, input_param_str, output_param_str, compiler_stats_str)
        return model_str

    @staticmethod
    def get_parameters_from_file(model_path):
        """Takes the (.dvm) model from the specified path and get parameters

        Parameters
        ----------
        model_path : str

        Returns
        -------
        (dv_status_code, DVModel)
            DV_SUCCESS on successfully getting parameters from the specified path

        """
        ret, model = dv_model_get_parameters_from_file(model_path)
        if ret != dv_status_code.DV_SUCCESS:
            return ret, None

        mdl = DVModel(model[0])
        mdl.unload = lambda : None
        dv_model_free_parameters(model)

        return ret, mdl

    @staticmethod
    def get_parameters_from_blob(model_blob):
        """
        Takes model blob as parameters
        
        Parameters
        ---------
        model_blob: dv_blob()

        Returns
        ---------
        (dv_status_code, DVModel)
        """
        ret, model = dv_model_get_parameters_from_blob(model_blob)
        if ret != dv_status_code.DV_SUCCESS:
            return ret, None

        mdl = DVModel(model[0])
        mdl.unload = lambda : None
        dv_model_free_parameters(model)

        return ret, mdl
        

    @_one_time_callable_cleanup_function
    def unload(self):
        """ unloads this model from the endpoint on which
        it was loaded. This is a blocking call

        Returns
        -------
        dv_status_code
            DV_SUCCESS if the unload was successful

        Notes
        -----
        Call this with caution!
        This is a blocking call.
        It is automatically called by class destructor.

        """
        return dv_model_unload(self._model)

    def _allocate_output_tensors(self):
        """ allocates space for all output tensors and returns
        a map of tensor_name -> tensor
        """
        # fixme:: this api may break for llm models
        if (self.model_type is dv_model_type.DV_MODEL_TYPE_ARA2_LLM_DYN_V2) or (self.model_type is dv_model_type.DV_MODEL_TYPE_ARA2_LLM) :
            logging.warning("_allocate_output_tensors may break with llm models")
        output_size_total = 0
        offsets = []
        for param in self.output_param:
            output_size_total += param.size
            offsets.append((param.size, param.layer_name))

        output_tensor_contig = np.zeros((output_size_total,), dtype=np.int8)

        slices = dict()

        start_offset = 0

        for idx, (offset, tensor_name) in enumerate(offsets):
            tensor_slice =\
                output_tensor_contig[start_offset : offset + start_offset]

            slices[idx] =\
                DVTensor(tensor_slice, self.output_param[idx])
            start_offset += offset
        return slices

    def _output_tensors_to_blobs(self, output_tensors):
        output_tensor_blobs = (DVBlob * len(output_tensors))()
        # fixme:: this api may break for llm models
        if (self.model_type is dv_model_type.DV_MODEL_TYPE_ARA2_LLM_DYN_V2) or (self.model_type is dv_model_type.DV_MODEL_TYPE_ARA2_LLM) :
            logging.warning("_output_tensors_to_blobs may break with llm models")
        for idx, param in enumerate(self.output_param):
            tensor = output_tensors[idx]
            if tensor.mem_desc is not None:
                output_tensor_blobs[idx].for_shm_desc(tensor.mem_desc._shm_desc,
                                                      tensor.offset,
                                                      param.size)
            else:
                ptr = tensor.numpy_data.ctypes.data_as(c_void_p)
                output_tensor_blobs[idx].for_raw_pointer(ptr,param.size)

        return output_tensor_blobs

    def _input_tensors_to_blobs(self, input_tensors):
        input_tensor_blobs = (DVBlob * len(input_tensors))()
        # fixme:: this api may break for llm models
        if (self.model_type is dv_model_type.DV_MODEL_TYPE_ARA2_LLM_DYN_V2) or (self.model_type is dv_model_type.DV_MODEL_TYPE_ARA2_LLM) :
            logging.warning("_input_tensors_to_blobs may break with llm models")
        for idx, param in enumerate(self.input_param):
            tensor = input_tensors[idx]
            if tensor.mem_desc is not None:
                input_tensor_blobs[idx].for_shm_desc(tensor.mem_desc._shm_desc,
                                                     tensor.offset,
                                                     param.size)
                tensor.mem_desc.mmap_buf.flush()
            else:
                ptr = tensor.numpy_data.ctypes.data_as(c_void_p)
                input_tensor_blobs[idx].for_raw_pointer(ptr, param.size)

        return input_tensor_blobs

    def infer_sync(self, input_tensors, output_tensors=None, endpoint=None, timeout=50000):
        status, async_inf_response = self.infer_async(input_tensors, output_tensors, endpoint)
        if status != dv_status_code.DV_SUCCESS:
            return status, None

        status = async_inf_response.wait_for_completion(timeout)
        return status, async_inf_response

    def infer_sync_with_options(self, input_tensors : typing.List[DVTensor], output_tensors : typing.List[DVTensor], endpoint : dv_endpoint, timeout_ms : int, infer_options : DVInferOptions):
        def get_blobs_frm_tensors_list(tensors : typing.List[DVTensor]):
            tensor_blobs = (DVBlob * len(tensors))()
            for idx,tensor in enumerate(tensors):
                size = tensor.numpy_data.nbytes
                ptr = tensor.numpy_data.ctypes.data_as(c_void_p)
                tensor_blobs[idx].for_raw_pointer(ptr, size)
            return tensor_blobs
        input_tensor_blobs = get_blobs_frm_tensors_list(input_tensors)
        output_tensor_blobs = get_blobs_frm_tensors_list(output_tensors)
        status, inf_req_obj =\
            dv_infer_sync_with_options(self.session,
                               self.endpoint,
                               self._model,
                               input_tensor_blobs,
                               output_tensor_blobs,
                               timeout_ms,
                               infer_options._infer_options_handle)
        if status != dv_status_code.DV_SUCCESS:
            return status, None
        # todo:: return DVInferRequest from here
        return status, inf_req_obj

    def infer_async(self, input_tensors, output_tensors=None, endpoint=None, desc=None):
        """ send an inference request for the given list of input
        tensors

        Parameters
        ----------
        input_tensors : [DVTensor]
            dict that specifies the tensor to be loaded onto input whose
            name is indicated by the key

        output_tensors : Optional[DVTensor]
            optional - specify output tensor where the output should be written, if none
            is provided, numpy arrays are created automatically to which outputs will
            be written

        endpoint : int
            endpoint to run the inference on - by default the inference runs
            on the default endpoint group created by the proxy connected to.

        Returns
        -------
        (dv_status_code, DVInferRequest)
            DV_SUCCESS, A future like object which can be waited on in a blocking
            manner pending inference request reaching a final state

            error status, None if the inference could not be initiated
        """
        if output_tensors is None:
            output_tensors = self._allocate_output_tensors()
        input_tensor_blobs = self._input_tensors_to_blobs(input_tensors)
        output_tensor_blobs = self._output_tensors_to_blobs(output_tensors)
        # form the infernce request and add to map before initiating
        # the call to prevent any racy behavior

        status, inf_req_obj =\
                dv_infer_async(self.session,
                               self.endpoint,
                               self._model,
                               input_tensor_blobs,
                               output_tensor_blobs)

        if status != dv_status_code.DV_SUCCESS:
            return status, None

        inf_req = DVInferRequest(input_tensors, output_tensors, inf_req_obj, self)
        return status, inf_req

    def infer_sync_with_flags(self, input_tensors : typing.List[DVTensor] , output_tensors : typing.List[DVTensor]= None, flags = None, timeout = 60000, endpoint=None, desc=None ):
        if output_tensors is None:
            output_tensors = self._allocate_output_tensors()

        def get_blobs_frm_tensors_list(tensors : typing.List[DVTensor]):
            tensor_blobs = (DVBlob * len(tensors))()
            for idx,tensor in enumerate(tensors):
                size = tensor.numpy_data.nbytes
                ptr = tensor.numpy_data.ctypes.data_as(c_void_p)
                tensor_blobs[idx].for_raw_pointer(ptr, size)
            return tensor_blobs

        input_tensor_blobs = get_blobs_frm_tensors_list(input_tensors)
        output_tensor_blobs = get_blobs_frm_tensors_list(output_tensors)

        status, inf_req_obj =\
                dv_infer_sync_with_flags(self.session,
                                         self.endpoint,
                                         self._model,
                                         input_tensor_blobs,
                                         output_tensor_blobs,
                                         timeout,
                                         flags)
        if status != dv_status_code.DV_SUCCESS:
            return status, None

    def __del__(self):
        self.unload()

class DVCompilerStats:
    def __init__(self,compiler_stats):
        self.config_name    = compiler_stats.config_name.decode('utf-8')
        self.cycles         = compiler_stats.cycles
        self.ips            = compiler_stats.ips
        self.ddr_bandwidth  = compiler_stats.ddr_bandwidth

    def __str__(self):
        compiler_stats_str = "config_name={}, cycles={}, ips={}, ddr_bandwidth={}".format(self.config_name, self.cycles, self.ips, self.ddr_bandwidth)
        return compiler_stats_str

class DVEndpointDramInfo:
    def __init__(self, dram_info):
        self.vendor_id      = dram_info.vendor_id
        self.vendor_name    = dram_info.vendor_name.decode('utf-8')
        self.size           = dram_info.size
        self.rev_id1        = dram_info.rev_id1
        self.rev_id2        = dram_info.rev_id2
        self.density        = dram_info.density
        self.io_widths      = dram_info.io_width

    def __str__(self):
        dram_info = 'vendor_id={}, vendor_name={}, size={}, rev_id1={}, rev_id2={}, density={}, io_widths={}'.format(
            self.vendor_id, self.vendor_name, self.size, self.rev_id1, self.rev_id2, self.density, self.io_widths)
        return dram_info

class DVEndpointIfaceInfo:
    def  __init__(self, iface_info):
        self.type       = iface_info.type
        self.bus_num    = iface_info.bus_num
        self.device_num = iface_info.device_num
        self.sysfs_path = ""
        if self.type == dv_endpoint_host_interface.DV_ENDPOINT_HOST_INTERFACE_PCIE:
            self.sysfs_path = iface_info.sysfs_path.pcie_dir.decode('utf-8')

    def __str__(self):
        iface_info = 'type={}, bus_num={}, device_num={}, sysfs_path={}'.format(
            self.type, self.bus_num, self.device_num, self.sysfs_path)
        return iface_info

class DVEndpointChipInfo:
    def  __init__(self, chip_info):
        self.id                         = chip_info.id.decode('utf-8')
        self.rev                        = chip_info.rev.decode('utf-8')
        self.control_processor_count    = chip_info.control_processor_count
        self.neural_processor_count     = chip_info.neural_processor_count
        self.l2_memory_size             = chip_info.l2_memory_size

    def __str__(self):
        chip_info_str = 'id={}, rev={}, control_processor_count={}, neural_processor_count={}, l2_memory_size={}'.format(
            self.id, self.rev, self.control_processor_count, self.neural_processor_count, self.l2_memory_size)
        return chip_info_str

class DVEndpointInfo:
    def __init__(self, endpoint_info):
        self.gpio0          = endpoint_info.gpio0
        self.gpio1          = endpoint_info.gpio1
        self.device_uid     = endpoint_info.device_uid
        self.device_id      = endpoint_info.device_id
        self.vendor_id      = endpoint_info.vendor_id
        self.module_name    = "" 
        self.chip           = DVEndpointChipInfo(endpoint_info.chip[0])
        self.dram           = DVEndpointDramInfo(endpoint_info.dram[0])
        self.iface          = DVEndpointIfaceInfo(endpoint_info.iface[0])

    def __str__(self):
        ep_info_str = 'gpio0={}, gpio1={}, device_uid={}, device_id={}, vendor_id={}, module_name={}, chip={}, dram={}, ifaace={}'.format(
            self.gpio0, self.gpio1, self.device_uid, self.device_id, self.vendor_id, self.module_name,
            str(self.chip), str(self.dram), str(self.iface))
        return ep_info_str

class DVEndpoint:
    """ a wrapper around a C buffer that stores a list of endpoint handles
    """
    def __init__(self, endpoint):
        self._endpoint  = endpoint
        self.handle     = endpoint.handle
        self.session    = endpoint.session
        self.num_ep     = endpoint.num_ep
        self.ep_info    = []

        for i in range(0, self.num_ep):
            self.ep_info.append(DVEndpointInfo(endpoint.ep_info_list[i][0]))

    def __str__(self):
        ep_info_str = ', '.join(map(str, self.ep_info))
        ep_str = 'handle={}, num_ep={}, ep_info[]={}'.format(self.handle, self.num_ep, ep_info_str)
        return ep_str


class DVSession:
    """ base class for connections. instantiate `DVProxyUnixConnection` or
    `DVProxyTCPConnection` to communicate with the proxy.
    """

    def __init__(self, session):
        self._session                   = session
        self.handle                     = session[0].handle
        self.socket_type                = session[0].socket_type
        self.socket_str                 = session[0].socket_str.decode('utf-8')
        self.options                    = None
        self.fault_callback_ptr         = None
        self.power_state_callback_ptr   = None
        self._ep_list                   = None
        self.ep_list                    = []
        self.inf_map                    = {}

    @_one_time_callable_cleanup_function
    def close(self):
        """closes the connection, existing requests that hold a reference
        to this connection will become invalidated or will never receive their
        respective responses

        Returns
        -------
        dv_status_code
            DV_SUCCESS if the connection was closed correctly
        """
        return dv_session_close(self._session)

    def __str__(self):
        session_str = 'handle={}, socket_type={}, socket_str={}, options={}'.format(self.handle, self.socket_type, self.socket_str, self.options)
        return session_str

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()
        return False

    def __del__(self):
        self.close()

    @classmethod
    def create_via_unix_socket(cls, path):
        """DVSession @classmethod that connects to the proxy via unix socket

        Parameters
        ----------
        path : str
            path to the unix socket

        Returns
        -------
        (dv_status_code, DVSession)
            DV_SUCCESS, DVSession if the connection was successfully established else returns the Error Code

        """
        ret, session = dv_session_create_via_unix_socket(socket_path=path)
        if ret != dv_status_code.DV_SUCCESS:
            return ret, None

        s = cls(session)
        return ret, s
    
    @classmethod
    def create_via_tcp_ipv4_socket(cls, ip, port):
        """DVSession @classmethod that connects to the proxy over the network

        Parameters
        ----------
        ip : str
            IPv4 address to connect to

        port : int
            Port number proxy is listening on

        Returns
        -------
        (dv_status_code, DVSession)
            DV_SUCCESS, DVSession if the connection was successfully established else returns the Error Code

        """
        ret, session = dv_session_create_via_tcp_ipv4_socket(ipv4_addr=ip, port=port)
        if ret != dv_status_code.DV_SUCCESS:
            return ret, None

        s = cls(session)
        return ret, s

    def get_endpoint_list(self):
        """DVSession method that lists all endpoints connected to the inference proxy

        Returns
        -------
        (dv_status_code, list(DVEndpoint))
            DVEndpoints will be None if the dv_status_code != DV_SUCCESS

        """

        ret, epl, count = dv_endpoint_get_list(self._session)

        if ret != dv_status_code.DV_SUCCESS:
            return ret, None

        self._ep_list = epl

        for i in range(0, count.value):
            self.ep_list.append(DVEndpoint(epl[i]))

        return ret, self.ep_list
    
    def get_loaded_endpoint_list(self, model):
        """DVSession method that returns a list of all connected endpoints + list of indexes in the aforementioned list
        on which the specified model is loaded.

        Parameters
        ----------
        model : DVModel

        Returns
        -------
        (dv_status_code, list(DVEndpoint), list(loaded_ep_indexs)
            dv_status_code   : DV_SUCCESS in case of success, else the error code
            ep_list          : list of DVEndpoint object
            loaded_ep_indexs : list of indexes in the aforementioned list on which the specified model is loaded
        """
        ret, endpt_list, endpt_list_size, loaded_endpt_idxes, loaded_endpt_list_size = dv_model_get_loaded_endpoint_list(self._session, model._model)
        if ret != dv_status_code.DV_SUCCESS:
            return ret, None, None, None, None

        ep_list = []
        for i in range(0, endpt_list_size.value):
            ep_list.append(DVEndpoint(endpt_list[i]))

        loaded_ep_indexs = []
        for i in range(0, loaded_endpt_list_size.value):
            loaded_ep_indexs.append(loaded_endpt_idxes[i])

        return ret, ep_list, loaded_ep_indexs

    def get_endpoint_default_group(self, default_grp_enum):
        """DVSession method to get default endpoint group created by client

        Parameters
        ----------
        default_grp_enum
            dv_endpoint_default_group

        Returns
        -------
        (dv_status_code, DVEndpoint)

        """
        ret, ep_grp = dv_endpoint_get_default_group(self._session, default_grp_enum)

        if ret != dv_status_code.DV_SUCCESS:
            return ret, None
        ep_default_grp = DVEndpoint(ep_grp[0])
        return ret, ep_default_grp

    def infer_inflight_count(self):
        """DVSession method that returns the number of inflight inference requests for the session
        object for  which the client library has not recieved a response from
        the proxy server

        Returns
        -------
        dv_status_code

        count : number of inference requests in flight

        """
        ret, count = dv_infer_get_inflight_count(self._session)
        if ret != dv_status_code.DV_SUCCESS:
            return ret, None

        return ret, count.value

    def free_endpoint_group(self, ep_group):
        """DVSession method free endpoint group

        Parameters
        ----------
        ep_group : DVEndpoint

        Returns
        -------
        (dv_status_code)

        """
        return dv_endpoint_free_group(ep_group._endpoint)

    def load_model_from_file(self,endpoint,model_path,model_name = "model.dvm",priority = dv_model_priority_level.DV_MODEL_PRIORITY_LEVEL_DEFAULT,cache_model = False):
        """DVSession method that loads and parses a `model.dvm` file present at path onto the default
           endpoint group of the connected proxy.

        Parameters
        ----------
        endpoint : DVEndpoint
            endpoint on which we wish to load the model

        path : str
            path to a compiled model.dvm file

        model_name : str
            name of the model

        priority : DV_MODEL_PRIORITY_LEVEL
            priority associated with the model, by default it is
            DV_MODEL_PRIORITY_LEVEL.DV_MODEL_PRIORITY_LEVEL_DEFAULT

        cache_model : bool
            indicates if the model should be cached by the server.
            if the caching of the model fails, the model load API call with fail

        Returns
        -------
        (dv_status_code, DVModel)
            DV_SUCCESS, DVModel object if load call succeeded

        (dv_status_code, None)
            if the load call could not succeed

        """
        if cache_model:
            pass
        else :
            ret, model_handle = dv_model_load_from_file(self._session,
                                    endpoint._endpoint,
                                    model_path,
                                    model_name,
                                    priority)

        if ret != dv_status_code.DV_SUCCESS:
            return ret, None

        model = DVModel(model_handle[0])
        return ret, model

    def load_model_from_file_with_flags(self, endpoint,model_path,flags,model_name = "model.dvm",priority = dv_model_priority_level.DV_MODEL_PRIORITY_LEVEL_DEFAULT,cache_model = False, load_async=True):
        ret, model_handle = dv_model_load_from_file_with_flags(self._session,
                                    endpoint._endpoint,
                                    model_path,
                                    flags,
                                    model_name,
                                    priority,
                                    cache_model,
                                    load_async)

        if ret != dv_status_code.DV_SUCCESS:
            return ret, None

        model = DVModel(model_handle[0])
        return ret, model

    def load_model_from_file_with_options(self, endpoint : dv_endpoint, model_path : str, options : DVModelLoadOptions):
        """DVSession method that creates a model object and load model contents from blob and transfer it to endpoint

        Notes
        -----
        If model load fails on all individual devices representing `endpt`, the API call will error out

        Parameters
        ----------
        endpoint : DVEndpoint
            endpoint on which we wish to load the model

        model_path
            path to model file

        options : DVModelLoadOptions
            model load option

        """
        ret, model_handle = dv_model_load_from_file_with_options(self._session,
                            endpoint._endpoint,
                            model_path,
                            options._model_load_options)

        if ret != dv_status_code.DV_SUCCESS:
            return ret, None

        model = DVModel(model_handle[0])
        return ret, model
