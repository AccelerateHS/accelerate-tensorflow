import tensorflow as tf
import numpy as np

import sys, os
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
    return [arr for tup in read_list(f, read_array, lazy=True) for arr in tup]

def read_array(f):
    num_elems = read_word64le(f)
    return read_list(f, lambda f: read_array_data(num_elems, f), lazy=True)

def read_array_data(num_elems, f):
    (bytes_per_elem, str_format, numpy_dtype) = read_datatype(f)
    data = bytearray(num_elems * bytes_per_elem)
    f.readinto(data)
    return np.array(struct.unpack("<" + str_format * num_elems, data), dtype=numpy_dtype)

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

# command -> (1::word8) job_spec          # run a job
#          | (2::word8)                   # exit
# job_spec -> graph_def_file input_arrays output_arrays data_file
# graph_def_file -> filename              # pb_file
# input_arrays -> list[node_name]         # input arrays of the model
# output_arrays -> list[node_name]        # output arrays of the model
# data_file -> filename                   # representative data file (see parse_representative_data_file)
# filename -> list[word8]                 # filename interpreted in UTF8
# node_name -> list[word8]                # node name interpreted in ASCII
#
# On completion of a job, the tflite file is written to the output FD as a list[word8].
def read_stdin_command(f):
    command = read_word8(f)
    if command == 1:
        return ("job", read_job_spec(f))
    elif command == 2:
        return ("exit", None)
    else:
        sys.exit(1)

def read_job_spec(f):
    graph_def_file = read_filename(f)
    input_arrays = read_list(f, read_node_name)
    output_arrays = read_list(f, read_node_name)
    data_file = read_filename(f)
    return (graph_def_file, input_arrays, output_arrays, data_file)

def read_filename(f):
    s = read_list(f, read_word8)
    return bytearray(s).decode("utf8")

def read_node_name(f):
    s = read_list(f, read_word8)
    return bytearray(s).decode("ascii")

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


def build_word64le(x):
    return struct.pack("<q", x)


def handle_job(outfd, in_file, inputs, outputs, data_file):
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

    # Write the tflite model as a list[word8] to stdout; use sys.stdout.buffer to write binary data
    outfd.write(build_word64le(len(tflite_model)))
    outfd.write(tflite_model)
    outfd.flush()

def handle_command(outfd):
    (command, cmdval) = read_stdin_command(sys.stdin.buffer)
    if command == "job":
        handle_job(outfd, *cmdval)
    elif command == "exit":
        sys.exit(0)
    else:
        assert False

def main():
    if len(sys.argv) != 2:
        print("Usage: converter.py <output FD>", file=sys.stderr)
        sys.exit(1)

    with os.fdopen(int(sys.argv[1]), "wb") as outfd:
        while True:
            handle_command(outfd)


if __name__ == "__main__":
    main()
