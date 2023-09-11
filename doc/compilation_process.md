# Overview of the compilation process

In this file you can find a brief description of the various stages of the conversion process from an Accelerate internal AST to running a mode on a TPU.


## 1. Frontend and overview

As can be seen in the haddocks of the `Data.Array.Accelerate.TensorFlow.Lite` (henceforth abbreviated as `DAA.TF.Lite`) module, compilation (conversion to a TFLite model that can be run using `libedgetpu`) and execution (running the model using `libedgetpu`) are split in the API.
This is not a strict requirement -- such separate compilation and execution happens for most Accelerate backends, yet the others have a single `runN`-style API.
Here, for transparency, we have chosen to separate the steps.

When `compile`ing an Accelerate program, the program first goes through the standard Accelerate optimisation and fusion pipeline (`convertAfunWith` and `simplifyAfun`).
Afterwards, we interpret the program into a TensorFlow graph.


## 2. Vectorisation into a TensorFlow graph

This interpretation is essentially a _vectorisation_ process, and happens in [`DAA.TF.CodeGen`](accelerate-tensorflow/src/Data/Array/Accelerate/TensorFlow/CodeGen.hs) and [`DAA.TF.CodeGen.Exp`](accelerate-tensorflow/src/Data/Array/Accelerate/TensorFlow/CodeGen/Exp.hs).
In `DAA.TF.CodeGen` we interpret the array (`Acc`) primitives of Accelerate, and the expressions (`Exp`) contained therein are interpreted in `DAA.TF.CodeGen.Exp`.

Because the TensorFlow surface language is first-order (i.e. does not have a second-order `map`, for example), we have to do work to convert something like `generate` or `map` to TensorFlow.
This work is vectorisation: in essence, an Accelerate program looking e.g. as follows:
```hs
generate sh (\i -> 2 * (a ! i) + 1)
```
is converted as if it was written as:
```hs
generate sh (\i -> 2) * backpermute sh a (\i -> i) + generate sh (\i -> 2)
```
where the `*` and `+` apply element-wise.
Thus, the two environments used in `DAA.TF.CodeGen.Exp.buildOpenExp`, the `Val` and the `Aval`, are both environments of _tensors_, and the whole building process happens in a "context" (the shape of the array that we are building).
Subexpressions are built as tensors the size of the array that we are building (the context).

### Tensor representation

The TensorFlow Haskell bindings have a `Tensor` data type that represents a term (an expression).
Because in our compilation process we often need to have not only the _value_ of a tensor available but also its _shape_, we wrap this data type with our own `Tensor` data type in `DAA.TF.CodeGen.Tensor`.
This data type contains:
1. a singleton (`ArrayR`) for the type of the tensor (this usage of singletons is pervasive in the entire Accelerate compilation pipeline, and indeed comes from the main `accelerate` package);
2. a tuple of zero-dimensional TensorFlow tensors (i.e. scalars) of type `Int`, containing the shape of this tensor;
3. a tuple of TensorFlow tensors containing the actual value of the (struct-of-arrays-transformed) array.

After this wrapping, we wrap `Tensor` _again_ using `DAA.TF.CodeGen.Tensor.Shim`; this second wrapper additionally tracks the graph that we are building separate from the internal TensorFlow graph representation, allowing us to render a DOT graph of the generated TensorFlow graph after it has been built.
The TF Native backend prints this graph if the `ACCELERATE_TF_PRINT_TFGRAPH` environment variable is non-empty; for the TFLite (TPU) backend, this is `ACCELERATE_TFLITE_PRINT_TFGRAPH`.
(See also [`environment_variables.md`](environment_variables.md).)

This shim means that uses of TF tensor operations should be wrapped in `Sh.wrap*` in the code.


## 3. Conversion to a TFLite model

Having a TensorFlow graph, this needs to be converted to a TFLite model in order to be able to pass it to the `edgetpu_compiler`.
This happens in `DAA.TF.Lite.Compile.compileTfunIn`.
Unfortunately, the API for doing so is only available in Python as far as we know (`tf.compat.v1.lite.TFLiteConverter`), and hence there is a Python script just for doing this.
This is represented in the user-level API of the TFLite backend as a `ConverterPy`.

Here, care must be taken with exactly which inputs from the sample data we supply to the converter: supplying inputs that are not present in the actual TF graph will result in an error from the TFLite converter.
Inputs not being present in the TF graph can happen because either Accelerate optimised those inputs out due to being unused, or if not, TF can still optimise the graph further and (e.g. with a more sophisticated analysis) determine that a particular input is unused, and hence remove it from the graph.
Therefore, in `compileTfunIn` we query which input nodes actually exist in the graph (by their names, which we gave them), and only pass precisely those inputs to `converter.py`.

In order to run on the TPU hardware, a TFLite model needs to be passed through Coral's `edgetpu_compiler`.
This is also done in `DAA.TF.Lite.Compile`.

Finally, the converted TFLite model is read into memory again from the file system, and returned to the user wrapped as a `Model`.


## 4. Execution

Execution consists of two phases: preparing all the inputs (in Haskell) and running the model on the TPU hardware (in C++).

In Haskell, we take the (already flat, courtesy of `accelerate`) input array buffers and allocate new output array buffers, and wrap all of those in something that the TensorFlow Haskell bindings can understand.
Then we call out to [`accelerate-tensorflow-lite/cbits/edgetpu.cc`](accelerate-tensorflow-lite/cbits/edgetpu.cc), which does the actual execution.
