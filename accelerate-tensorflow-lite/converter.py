import tensorflow as tf
import numpy as np

import sys
import struct


# TODO: This script might need some additional data to work.
#
# Format specified in pseudo-BNF:

#   representative_data_file -> dataset_count tensor_count array+
#   dataset_count -> word8
#   tensor_count -> word8
#   array -> shape component_count array_data+
#   shape -> rank dimension*
#   rank -> word8
#   dimension -> i64
#   componont_count -> word8
#   array_data -> datatype databuffer
#   datatype -> word8 in [0..10]
#   databuffer -> (the amount of values is the product of the dimensions from the shape)
#
def parse_representative_data_file(file_path):
    with open(file_path, "rb") as f:
        # Parse the number of representative datasets
        dataset_count = read_word8(f)

        for _ in range(dataset_count):
            # Parse the number of input tensors per representative dataset
            tensor_count = read_word8(f)

            tensors = []
            for _ in range(tensor_count):
                tensors.extend(read_array(f))
            #end
            yield tensors
        #end
    #end (I hate python's indentation rules...)


def read_word8(f):
    b = bytearray(1)
    f.readinto(b)
    return struct.unpack("B", b)[0]

def read_int64(f):
    b = bytearray(8)
    f.readinto(b)
    return struct.unpack("q", b)[0]

def read_datatype(f):
    # NOTE: Must match encoding used by 'tagOfType'
    #
    # Using a dictionary here because python doesn't have the standard
    # switch-case construct
    #
    switch = {
        0:  (1, "b", np.byte),      # Type.Int8
        1:  (2, "h", np.short),     # Type.Int16
        2:  (4, "i", np.intc),      # Type.Int32
        3:  (8, "q", np.longlong),  # Type.Int64
        4:  (1, "B", np.ubyte),     # Type.Word8
        5:  (2, "H", np.ushort),    # Type.Word16
        6:  (4, "I", np.uintc),     # Type.Word32
        7:  (8, "Q", np.ulonglong), # Type.Word64
        8:  (2, "h", np.float16),   # Type.Float16
        9:  (4, "f", np.float32),   # Type.Float32
        10: (8, "d", np.float64),   # Type.Float64
    }
    t = read_word8(f)
    return switch.get(t)


def read_shape(f):
    rank = read_word8(f)
    b = bytearray(rank * 8)
    f.readinto(b)
    shape = struct.unpack("q" * rank, b)

    # Apparently functional style is not "pythonic" so here we go, have a foldl (*) 1:
    size = 1
    for x in shape:
        size *= x
    #end

    return (rank, shape, size)


# Surely there is a better way to interpret binary data in python??!
#
def read_array(f):
    (rank, shape, size) = read_shape(f)
    component_count = read_word8(f)

    array_data = []
    for _ in range(component_count):
        (bytes_per_elem, str_format, numpy_dtype) = read_datatype(f)
        data = bytearray(size * bytes_per_elem)
        f.readinto(data)
        array_data.append(np.array(struct.unpack(str_format * size, data), dtype=numpy_dtype))
    #end

    return array_data


# Parses command-line arguments
#
# Options are:
#   --graph_def_file= pb_file
#   --output_file= tflite_file
#   --input_arrays=  comma-separated list of names of  input arrays
#   --output_arrays= comma-separated list of names of output arrays
#   --data_file= representative_data_file (see parse_representative_data_file)
#
# TODO: the following changes need to happen at some point:
#    1. sanitize the data in in_arrs and out_arrs before setting their values.
#
def parse_args(args):
    graph_def = ""
    outfile   = ""
    data_file = ""
    in_arrs   = ""
    out_arrs  = ""

    for arg in args:
        if arg.startswith('--graph_def_file='):
            graph_def = arg[17:]
            pass
        elif arg.startswith('--output_file='):
            outfile = arg[14:]
            pass
        elif arg.startswith('--data_file='):
            data_file = arg[12:]
            pass
        elif arg.startswith('--input_arrays='):
            in_arrs = parse_array_arg(arg[15:])
            pass
        elif arg.startswith('--output_arrays='):
            out_arrs = parse_array_arg(arg[16:])
            pass
        #end
    #end

    return (graph_def, outfile, in_arrs, out_arrs, data_file)


def parse_array_arg(arg):
    return arg.split(",")


def main():
    in_file, out_file, inputs, outputs, data_file = parse_args(sys.argv)

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=in_file
      , input_arrays=inputs
      , output_arrays=outputs
      )

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8] # or tf.uint8

    converter.representative_dataset = tf.lite.RepresentativeDataset(
        lambda: parse_representative_data_file(data_file))

    tflite_model = converter.convert()
    with tf.io.gfile.GFile(out_file, 'wb') as f:
      f.write(tflite_model)


if __name__ == "__main__":
    main()

