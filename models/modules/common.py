# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import collections
from enum import Enum

from lib.layers import MinkowskiSwitchNorm, MinkowskiLayerNorm

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF


class NormType(Enum):
  BATCH_NORM = 0
  SPARSE_LAYER_NORM = 1
  SPARSE_INSTANCE_NORM = 2
  SPARSE_SWITCH_NORM = 3


class NonlinearityType(Enum):
  ReLU = 0
  LeakyReLU = 1
  PReLU = 2
  CELU = 3
  SELU = 4


def get_norm(norm_type, n_channels, D, bn_momentum=0.1):
  if norm_type == NormType.BATCH_NORM:
    return ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
  elif norm_type == NormType.SPARSE_LAYER_NORM:
    return MinkowskiLayerNorm(n_channels, D=D)
  elif norm_type == NormType.SPARSE_INSTANCE_NORM:
    return ME.MinkowskiInstanceNorm(n_channels, dimension=D)
  elif norm_type == NormType.SPARSE_SWITCH_NORM:
    return MinkowskiSwitchNorm(n_channels, D=D)
  else:
    raise ValueError(f'Norm type: {norm_type} not supported')


str_to_nonlinearity_dict = {m.name: m for m in NonlinearityType}


def get_nonlinearity_fn(nonlinearity_type, input, *args, **kwargs):
  nonlinearity_type = str_to_nonlinearity_dict[nonlinearity_type]
  if nonlinearity_type == NonlinearityType.ReLU:
    return MEF.relu(input, *args, **kwargs)
  elif nonlinearity_type == NonlinearityType.ReLU:
    return MEF.leaky_relu(input, *args, **kwargs)
  elif nonlinearity_type == NonlinearityType.PReLU:
    return MEF.prelu(input, *args, **kwargs)
  elif nonlinearity_type == NonlinearityType.CELU:
    return MEF.celu(input, *args, **kwargs)
  elif nonlinearity_type == NonlinearityType.SELU:
    return MEF.selu(input, *args, **kwargs)
  else:
    raise ValueError(f'Norm type: {nonlinearity_type} not supported')


class ConvType(Enum):
  """
  Define the kernel region type
  """
  HYPERCUBE = 0, 'HYPERCUBE'
  SPATIAL_HYPERCUBE = 1, 'SPATIAL_HYPERCUBE'
  SPATIO_TEMPORAL_HYPERCUBE = 2, 'SPATIO_TEMPORAL_HYPERCUBE'
  HYPERCROSS = 3, 'HYPERCROSS'
  SPATIAL_HYPERCROSS = 4, 'SPATIAL_HYPERCROSS'
  SPATIO_TEMPORAL_HYPERCROSS = 5, 'SPATIO_TEMPORAL_HYPERCROSS'
  SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS = 6, 'SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS '

  def __new__(cls, value, name):
    member = object.__new__(cls)
    member._value_ = value
    member.fullname = name
    return member

  def __int__(self):
    return self.value


# Covert the ConvType var to a RegionType var
conv_to_region_type = {
    # kernel_size = [k, k, k, 1]
    ConvType.HYPERCUBE: ME.RegionType.HYPERCUBE,
    ConvType.SPATIAL_HYPERCUBE: ME.RegionType.HYPERCUBE,
    ConvType.SPATIO_TEMPORAL_HYPERCUBE: ME.RegionType.HYPERCUBE,
    ConvType.HYPERCROSS: ME.RegionType.HYPERCROSS,
    ConvType.SPATIAL_HYPERCROSS: ME.RegionType.HYPERCROSS,
    ConvType.SPATIO_TEMPORAL_HYPERCROSS: ME.RegionType.HYPERCROSS,
    ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS: ME.RegionType.HYBRID
}

int_to_region_type = {m.value: m for m in ME.RegionType}


def convert_region_type(region_type):
  """
  Convert the integer region_type to the corresponding RegionType enum object.
  """
  return int_to_region_type[region_type]


def convert_conv_type(conv_type, kernel_size, D):
  assert isinstance(conv_type, ConvType), "conv_type must be of ConvType"
  region_type = conv_to_region_type[conv_type]
  axis_types = None
  if conv_type == ConvType.SPATIAL_HYPERCUBE:
    # No temporal convolution
    if isinstance(kernel_size, collections.Sequence):
      kernel_size = kernel_size[:3]
    else:
      kernel_size = [
          kernel_size,
      ] * 3
    if D == 4:
      kernel_size.append(1)
  elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCUBE:
    # conv_type conversion already handled
    assert D == 4
  elif conv_type == ConvType.HYPERCUBE:
    # conv_type conversion already handled
    pass
  elif conv_type == ConvType.SPATIAL_HYPERCROSS:
    if isinstance(kernel_size, collections.Sequence):
      kernel_size = kernel_size[:3]
    else:
      kernel_size = [
          kernel_size,
      ] * 3
    if D == 4:
      kernel_size.append(1)
  elif conv_type == ConvType.HYPERCROSS:
    # conv_type conversion already handled
    pass
  elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCROSS:
    # conv_type conversion already handled
    assert D == 4
  elif conv_type == ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS:
    # Define the CUBIC conv kernel for spatial dims and CROSS conv for temp dim
    axis_types = [
        ME.RegionType.HYPERCUBE,
    ] * 3
    if D == 4:
      axis_types.append(ME.RegionType.HYPERCROSS)
  return region_type, axis_types, kernel_size


def conv(in_planes,
         out_planes,
         kernel_size,
         stride=1,
         dilation=1,
         bias=False,
         conv_type=ConvType.HYPERCUBE,
         D=-1):
  assert D > 0, 'Dimension must be a positive integer'
  region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
  kernel_generator = ME.KernelGenerator(
      kernel_size, stride, dilation, region_type=region_type, axis_types=axis_types, dimension=D)

  return ME.MinkowskiConvolution(
      in_channels=in_planes,
      out_channels=out_planes,
      kernel_size=kernel_size,
      stride=stride,
      dilation=dilation,
      has_bias=bias,
      kernel_generator=kernel_generator,
      dimension=D)


def conv_tr(in_planes,
            out_planes,
            kernel_size,
            upsample_stride=1,
            dilation=1,
            bias=False,
            conv_type=ConvType.HYPERCUBE,
            D=-1):
  assert D > 0, 'Dimension must be a positive integer'
  region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
  kernel_generator = ME.KernelGenerator(
      kernel_size,
      upsample_stride,
      dilation,
      region_type=region_type,
      axis_types=axis_types,
      dimension=D)

  return ME.MinkowskiConvolutionTranspose(
      in_channels=in_planes,
      out_channels=out_planes,
      kernel_size=kernel_size,
      stride=upsample_stride,
      dilation=dilation,
      has_bias=bias,
      kernel_generator=kernel_generator,
      dimension=D)


def avg_pool(kernel_size,
             stride=1,
             dilation=1,
             conv_type=ConvType.HYPERCUBE,
             in_coords_key=None,
             D=-1):
  assert D > 0, 'Dimension must be a positive integer'
  region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
  kernel_generator = ME.KernelGenerator(
      kernel_size, stride, dilation, region_type=region_type, axis_types=axis_types, dimension=D)

  return ME.MinkowskiAvgPooling(
      kernel_size=kernel_size,
      stride=stride,
      dilation=dilation,
      kernel_generator=kernel_generator,
      dimension=D)


def avg_unpool(kernel_size, stride=1, dilation=1, conv_type=ConvType.HYPERCUBE, D=-1):
  assert D > 0, 'Dimension must be a positive integer'
  region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
  kernel_generator = ME.KernelGenerator(
      kernel_size, stride, dilation, region_type=region_type, axis_types=axis_types, dimension=D)

  return ME.MinkowskiAvgUnpooling(
      kernel_size=kernel_size,
      stride=stride,
      dilation=dilation,
      kernel_generator=kernel_generator,
      dimension=D)


def sum_pool(kernel_size, stride=1, dilation=1, conv_type=ConvType.HYPERCUBE, D=-1):
  assert D > 0, 'Dimension must be a positive integer'
  region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
  kernel_generator = ME.KernelGenerator(
      kernel_size, stride, dilation, region_type=region_type, axis_types=axis_types, dimension=D)

  return ME.MinkowskiSumPooling(
      kernel_size=kernel_size,
      stride=stride,
      dilation=dilation,
      kernel_generator=kernel_generator,
      dimension=D)