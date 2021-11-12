import tensorflow as tf
import numpy as np

import sys
import struct

# TODO: make sure this takes data from the Accelerate compiler. Right now it
# has the following assumptions:
#   1. There are exactly 2 input arrays
#   2. The shape of both arrays is [100]
#   3. The representative dataset is random floats in range [0.0, 10.0)
def representative_data_gen(rng):
    for _ in range(0, 10):
        x = rng.random([100], dtype=np.float32) * 10.0
        y = rng.random([100], dtype=np.float32) * 10.0

        yield [x, y]

# This script might need some additional data to work.
# TODO
#
# Format specified in pseudo-BNF:
# representative_data_file -> tensor_count dataset_count dataset+
# tensor_count -> i32
# dataset_count -> i32
# dataset -> shape data
# shape -> i32 i32+ (the amount of i32s is 1 + the value of the first)
# data -> f32+ (the amount of f32s is the product of the dimensions from the shape)
def parse_representative_data_file(file_path):
    with open(file_path, "rb") as f:
        # Parse the amount of input tensors
        tensor_count = read_int32(f)

        # Parse the amount of representative datasets
        dataset_count = read_int32(f)

        for _ in range(dataset_count):
            tensors = []
            for _ in range(tensor_count):
                shape = read_shape(f)
                # Parse the data (TODO assuming f32 data?)
                data = read_data(f, shape)
                np_data = np.array(data, dtype=np.float32)
                tensors.append(np_data)

            #end
            yield tensors

        #end

    #end (I hate python's indentation rules...)


def read_int32(f):
    b = bytearray(4)
    f.readinto(b)
    return struct.unpack("i", b)[0]


# Reads a 32-bit integer which describes the amount of dimensions the data has,
# followed by that amount of 32-bit integers
def read_shape(f):
    dims = read_int32(f)
    b = bytearray(dims * 4)
    f.readinto(b)
    return struct.unpack("i" * dims, b)


# Reads f32 data, the amount of which is the product of all dimensions in the shape
def read_data(f, shape):
    # Apparently functional style is not "pythonic" so here we go, have a
    # foldl (*) 1:
    res = 1
    for x in shape:
        res *= x

    #end

    data = bytearray(res * 4)
    f.readinto(data)
    return struct.unpack("f" * res, data)


# Parses command-line arguments
# Options are:
#   --graph_def_file= pb_file
#   --output_file= tflite_file
#   --input_arrays=  comma-separated list of names of  input arrays
#   --output_arrays= comma-separated list of names of output arrays
#   --data_path= representative_data_file (see parse_representative_data_file)
# TODO: the following changes need to happen at some point:
#    1. sanitize the data in in_arrs and out_arrs before setting their values.
def parse_args(args):
    graph_def = None
    outfile = None
    in_arrs = None
    out_arrs = None
    data_path = None

    for arg in args:
        if arg.startswith('--graph_def_file='):
            graph_def = arg[17:]
            pass
        elif arg.startswith('--output_file='):
            outfile = arg[14:]
            pass
        elif arg.startswith('--input_arrays='):
            in_arrs = parse_array_arg(arg[15:])
            pass
        elif arg.startswith('--output_arrays='):
            out_arrs = parse_array_arg(arg[16:])
            pass
        elif arg.startswith('--data_path='):
            data_path = arg[12:]
            pass
        #end
    #end

    return (graph_def, outfile, in_arrs, out_arrs, data_path)


def parse_array_arg(arg):
    return arg.split(",")


def main():
    in_file, out_file, inputs, outputs, data_path = parse_args(sys.argv)

    rng = np.random.default_rng()

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=in_file
      , input_arrays=inputs
      , output_arrays=outputs
      )

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if data_path:
        converter.representative_dataset = tf.lite.RepresentativeDataset(
                lambda: parse_representative_data_file(data_path))
    else:
        # TODO make this an error instead? It's likely to give an error otherwise, anyway...
        converter.representative_dataset = tf.lite.RepresentativeDataset(
                lambda: representative_data_gen(rng))
    #end

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]

    tflite_model = converter.convert()
    with tf.io.gfile.GFile(out_file, 'wb') as f:
      f.write(tflite_model)


if __name__ == "__main__":
    main()

