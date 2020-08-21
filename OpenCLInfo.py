from __future__ import print_function
__original_author__      = "Zhi-Qiang Zhou"
__original_copyright__   = "Copyright 2017"
__author__      = "Rahul Raj"
__copyright__   = "Copyright 2020"

import pyopencl as cl

def openclInfo():
    """Print OpenCL information.
    """
    platforms = cl.get_platforms()
    output = {}
    output['Platform(s) detected']='{}'.format(len(platforms))
    for platform_id, platform in enumerate(platforms):
        output['Platform']= platform.name
        output['Vendor']=platform.vendor
        output['Version']=platform.version
        output['Number of devices']=len(platform.get_devices())
        for device_id, device in enumerate(platform.get_devices()):
            device_info_dict = getDeviceInfo(platform_id, device_id)
            for key_k in device_info_dict:
                output[key_k]=device_info_dict[key_k]

def getDeviceInfo(platform_id, device_id, prefix=''):
    """Print OpenCL device information.

    Args:
        platform_id (int): Platform ID
        device_id (int): Device ID
        prefix (str): Prefix string
    """
    platforms = cl.get_platforms()
    devices = platforms[platform_id].get_devices()
    device = devices[device_id]
    output={}
    output['Device {:d}'.format(device_id)]='{} - {}'.format(str(device.name), 
        '[Type: {:s}]'.format(cl.device_type.to_string(device.type)))
    output['Device version']=device.version
    output['Driver version']=device.driver_version
    output['Vendor']=device.vendor
    output['Available']=bool(device.available)
    output['Address bits']=device.address_bits
    output['Max compute units']=device.max_compute_units
    output['Max clock frequency']='{} MHz'.format(device.max_clock_frequency)
    output['Global memory']={} MB'.format(int(device.global_mem_size / 1024**2))
    output['Global cache memory']={} B'.format(int(device.global_mem_cache_size))
    output['Local memory']='{} KB'.format(int(device.local_mem_size / 1024))
    output['Max allocable memory']='{} MB'.format(int(device.max_mem_alloc_size / 1024**2))
    output['Max constant args']=device.max_constant_args
    output['Max work group siz']=device.max_work_group_size
    output['Max work item dimensions']=device.max_work_item_dimensions
    output['Max work item size']=device.max_work_item_sizes
    output['Image support']=bool(device.image_support)
    output['Max Image2D size (H x W)']='{}x{}'.format(device.image2d_max_height,
        device.image2d_max_width)
    output['Max Image3D size (D x H x W)']={}x{}x{}'.format(device.image3d_max_depth,
        device.image3d_max_height, device.image3d_max_width)
    return output

if __name__ == '__main__':

    print(openclInfo())
