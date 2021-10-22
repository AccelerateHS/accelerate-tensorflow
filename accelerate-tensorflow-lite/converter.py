import tensorflow as tf
import numpy as np

import sys

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

# Parses command-line arguments
# Options are:
#   --graph_def_file=" pb_file
#   --output_file=" tflite_file
#   --input_arrays="  comma-separated list of names of  input arrays
#   --output_arrays=" comma-separated list of names of output arrays
# TODO: the following changes need to happen at some point:
#    1. sanitize the data in in_arrs and out_arrs before setting their values.
#    2. Accept a cli-arg that describes what representative data there is, how
#       to get it, and how to use it.
def parse_args(args):
    graph_def = None
    outfile = None
    in_arrs = None
    out_arrs = None

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

    return (graph_def, outfile, in_arrs, out_arrs)


def parse_array_arg(arg):
    return arg.split(",")


def main():
    in_file, out_file, inputs, outputs = parse_args(sys.argv)

    rng = np.random.default_rng()

    # TODO: use the inputs and outputs values
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=in_file
      , input_arrays=inputs
      , output_arrays=outputs
      )

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = tf.lite.RepresentativeDataset(lambda: representative_data_gen(rng))

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]

    tflite_model = converter.convert()
    with tf.io.gfile.GFile(out_file, 'wb') as f:
      f.write(tflite_model)

if __name__ == "__main__":
    main()

