import tensorflow as tf
import numpy as np

import sys
import struct


# Format for the "representative data" file, in pseudo-BNF.
#
# A "list" is a word64LE length, followed by that many elements.
#
# representative_data_file -> list[dataset]
# dataset -> list[array]                      # list of arguments for one model invocation
# array -> numelements list[array_data]       # one argument array in SoA format
# numelements -> word64LE                     # the number of elements in this array
# array_data -> datatype databuffer           # one tuple component (of the SoA form)
# datatype -> word8 in [0..10]
# databuffer -> (values of specified data type; count is the numelements field for this array)
def parse_representative_data_file(file_path):
    with open(file_path, "rb") as f:
        for dataset in read_list(f, read_dataset, lazy=True):
            yield dataset

def read_dataset(f):
    return read_list(f, read_array)

def read_array(f):
    num_elems = read_word64le(f)
    return read_list(f, lambda f: read_array_data(num_elems, f))

def read_array_data(num_elems, f):
    (bytes_per_elem, str_format, numpy_dtype) = read_datatype(f)
    data = bytearray(num_elems * bytes_per_elem)
    f.readinto(data)
    return np.array(struct.unpack("<" + str_format * size, data), dtype=numpy_dtype)

def read_datatype(f):
    # NOTE: Must match encoding used by 'tagOfType'
    switch = {
        0:  (1, "b", np.int8),
        1:  (2, "h", np.int16),
        2:  (4, "i", np.int32),
        3:  (8, "q", np.int64),
        4:  (1, "B", np.uint8),
        5:  (2, "H", np.uint16),
        6:  (4, "I", np.uint32),
        7:  (8, "Q", np.uint64),
        8:  (2, "h", np.float16),
        9:  (4, "f", np.float32),
        10: (8, "d", np.float64),
    }
    t = read_word8(f)
    return switch.get(t)

def read_word64le(f):
    b = bytearray(8)
    f.readinto(b)
    return struct.unpack("<q", b)[0]

def read_word8(f):
    b = bytearray(1)
    f.readinto(b)
    return b[0]

def read_list(f, reader, lazy=False):
    n = read_word64le(f)
    result_iter = (reader(f) for _ in range(n))
    if lazy: return result_iter
    else: return list(result_iter)


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
        if arg.startswith('--graph_def_file='):  graph_def = arg[17:]
        elif arg.startswith('--output_file='):   outfile = arg[14:]
        elif arg.startswith('--data_file='):     data_file = arg[12:]
        elif arg.startswith('--input_arrays='):  in_arrs = arg[15:].split(",")
        elif arg.startswith('--output_arrays='): out_arrs = arg[16:].split(",")

    return (graph_def, outfile, in_arrs, out_arrs, data_file)


def main():
    in_file, out_file, inputs, outputs, data_file = parse_args(sys.argv)

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=in_file,
        input_arrays=inputs,
        output_arrays=outputs
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
