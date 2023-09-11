# Environment variables

The backends respond to a few environment variables that can be helpful for debugging when developing the backends.

`accelerate-tensorflow`:
- `ACCELERATE_TF_PRINT_TFGRAPH`: when compiling a model, print GraphViz DOT representations of the generated TensorFlow graphs to stdout.

`accelerate-tensorflow-lite`:
- `ACCELERATE_TFLITE_PRINT_TFGRAPH`: when compiling a model, print GraphViz DOT representations of the generated TensorFlow graphs to stdout.
- `ACCELERATE_TFLITE_PRESERVE_TEMP`: do not remove the temporary working directories (in `/tmp`) containing inputs and outputs of the TFLite conversion process (`converter.py` as well as `edgetpu_compiler`).
