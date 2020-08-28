from __future__ import print_function
__original_author__      = "Zhi-Qiang Zhou"
__original_copyright__   = "Copyright 2018"
__author__      = "Rahul Raj"
__copyright__   = "Copyright 2020"

import pycuda.autoinit
import pycuda.driver as cuda

def cudaInfo():
    """Return CUDA information.
    """
    deviceCount = cuda.Device.count()
    output = {}
    output['CUDA device(s) detected']='{:d}'.format(deviceCount)
    for device_id in range(deviceCount):
        device_info_dict = getDeviceInfo(device_id)
        for key_k in device_info_dict:
            output[key_k] = device_info_dict[key_k]
    return output

def convertSMVer2Cores(major, minor):
    """GPU Architecture definitions.

    Refs:
        https://en.wikipedia.org/wiki/CUDA
        https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h

    Args:
        major (int): major version
        minor (int): minor version

    Returns:
        int: the # of cores per Streaming Multiprocessor (SM)

    """
    sm = str(major) + str(minor)
    if   sm == '30': return 192 # Kepler Generation (SM 3.0) GK10x class
    elif sm == '32': return 192 # Kepler Generation (SM 3.2) GK10x class
    elif sm == '35': return 192 # Kepler Generation (SM 3.5) GK11x class
    elif sm == '37': return 192 # Kepler Generation (SM 3.7) GK21x class
    elif sm == '50': return 128 # Maxwell Generation (SM 5.0) GM10x class
    elif sm == '52': return 128 # Maxwell Generation (SM 5.2) GM20x class
    elif sm == '53': return 128 # Maxwell Generation (SM 5.3) GM20x class
    elif sm == '60': return 64  # Pascal Generation (SM 6.0) GP100 class
    elif sm == '61': return 128 # Pascal Generation (SM 6.1) GP10x class
    elif sm == '62': return 128 # Pascal Generation (SM 6.2) GP10x class
    elif sm == '70': return 64  # Volta Generation (SM 7.0) GV100 class
    elif sm == '72': return 64  # Volta Generation (SM 7.2) GV10B class
    elif sm == '75': return 64  # Turing Generation (SM 7.5) TU10x class

def getDeviceInfo(device_id, prefix=''):
    """Return CUDA device information.

    Args:
        device_id (int): Device ID
        prefix (str): Prefix string
    """
    device = cuda.Device(device_id)
    attributes = device.get_attributes()
    attrs = {}
    for (key,value) in attributes.items():
        attrs[str(key)] = value

    major = attrs['COMPUTE_CAPABILITY_MAJOR']
    minor = attrs['COMPUTE_CAPABILITY_MINOR']
    cores = convertSMVer2Cores(major, minor)

    runtimeVersion = pycuda.driver.get_version()
    driverVersion = pycuda.driver.get_driver_version()
    output = {}
    output['Device ID'= 'Device {:d}'.format(device_id)
    output['Device name'= device.name()
    output['CUDA Driver Version / Runtime Version']= '{:d}.{:d} / {:d}.{:d}'.format(
            driverVersion // 1000, (driverVersion % 100) // 10,
            runtimeVersion[0], runtimeVersion[1])
    output['CUDA Capability Major/Minor version number']= str(major) + '.' + str(minor)
    output['Total amount of global memory']= '{} MBytes'.format(device.total_memory() // 1024**2)
    output['Multiprocessors Count']= attrs['MULTIPROCESSOR_COUNT']
    output['CUDA Cores/MP'] =  cores
    output['Total Cuda Cores']= '{} CUDA Cores'.format(attrs['MULTIPROCESSOR_COUNT'] * cores)
    output['GPU Clock rate']= '{} MHz'.format(attrs['CLOCK_RATE'] / 1e3)
    output['Memory Clock rate']= '{} MHz'.format(attrs['MEMORY_CLOCK_RATE'] / 1e3)
    output['Memory Bus Width']= '{:d}-bit'.format(attrs['GLOBAL_MEMORY_BUS_WIDTH'])
    output['L2 Cache Size']= '{} MBytes'.format(attrs['L2_CACHE_SIZE'] / 1024**2)
    output['Max Texture Dimension Size (x)']= '1D = ({:d})'.format(attrs['MAXIMUM_TEXTURE1D_WIDTH'])
    output['Max Texture Dimension Size (x,y)']= '2D = ({:d}, {:d})'.format(attrs['MAXIMUM_TEXTURE2D_WIDTH'], 
                                   attrs['MAXIMUM_TEXTURE2D_HEIGHT'])
    output['Max Texture Dimension Size (x,y,z)']= '3D = ({:d}, {:d}, {:d})'.format(attrs['MAXIMUM_TEXTURE3D_WIDTH'], 
                                         attrs['MAXIMUM_TEXTURE3D_HEIGHT'], 
                                         attrs['MAXIMUM_TEXTURE3D_DEPTH'])
    output['Max Layered Texture Size (1D) x layers']= '1D = ({:d}) x {:d}'.format(
            attrs['MAXIMUM_TEXTURE1D_LAYERED_WIDTH'], 
            attrs['MAXIMUM_TEXTURE1D_LAYERED_LAYERS'])
    output['Max Layered Texture Size (2D) x layers']= '2D = ({:d}, {:d}) x {:d}'.format(
            attrs['MAXIMUM_SURFACE2D_LAYERED_WIDTH'], 
            attrs['MAXIMUM_SURFACE2D_LAYERED_HEIGHT'], 
            attrs['MAXIMUM_SURFACE2D_LAYERED_LAYERS'])
    output['Total amount of constant memory']= '{} Bytes'.format(attrs['TOTAL_CONSTANT_MEMORY'])
    output['Total amount of shared memory per block']='{} Bytes'.format(attrs['MAX_SHARED_MEMORY_PER_BLOCK'])
    output['Total number of registers available per block']= attrs['MAX_REGISTERS_PER_BLOCK']
    output['Warp size']= attrs['WARP_SIZE']
    output['Maximum number of threads per multiprocessor']= attrs['MAX_THREADS_PER_MULTIPROCESSOR']
    output['Maximum number of threads per block']= attrs['MAX_THREADS_PER_BLOCK']
    output['Maximum sizes of each dimension of a block']= '{:d} x {:d} x {:d}'.format(attrs['MAX_BLOCK_DIM_X'], 
                                    attrs['MAX_BLOCK_DIM_Y'], 
                                    attrs['MAX_BLOCK_DIM_Z'])
    output['Maximum sizes of each dimension of a grid']= '{:d} x {:d} x {:d}'.format(attrs['MAX_GRID_DIM_X'], 
                                    attrs['MAX_GRID_DIM_Y'], 
                                    attrs['MAX_GRID_DIM_Z'])
    output['Maximum memory pitch']= 'Bytes'.format(attrs['MAX_PITCH'])
    output['Texture alignment']= 'Bytes'.format(attrs['TEXTURE_ALIGNMENT'])
    output['Concurrent copy and kernel execution']= '{} with {:d} copy engine(s)'.format(
            bool(attrs['CONCURRENT_KERNELS']), 
            attrs['ASYNC_ENGINE_COUNT'])
    output['Run time limit on kernels']= bool(attrs['KERNEL_EXEC_TIMEOUT'])
    output['Integrated GPU sharing Host Memory']= bool(attrs['INTEGRATED'])
    output['Support host page-locked memory mapping']= bool(attrs['CAN_MAP_HOST_MEMORY'])
    output['Alignment requirement for Surfaces']= bool(attrs['SURFACE_ALIGNMENT'])
    output['Device has ECC support']= bool(attrs['ECC_ENABLED'])
    output['Device supports Unified Addressing (UVA)']= bool(attrs['UNIFIED_ADDRESSING'])
    output['Device PCI Bus ID / PCI location ID']= '{} / {}'.format(attrs['PCI_BUS_ID'], attrs['PCI_DEVICE_ID'])
    output['Compute Mode']= attrs['COMPUTE_MODE']
    return output

if __name__ == '__main__':
    print(cudaInfo())
