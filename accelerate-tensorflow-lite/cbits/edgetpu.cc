
#include <iostream>
#include <stdint.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "edgetpu.h"

std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* edgetpu_context)
{
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
    std::cerr << "Failed to build TPU interpreter" << std::endl;
  }

  // Bind given context with interpreter
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  interpreter->SetNumThreads(1);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate TPU tensors" << std::endl;
  }

  return interpreter;
}

extern "C" void edgetpu_run(const char* model_path, const char** tensor_name, uint8_t** tensor_data, size_t* tensor_size_bytes, size_t tensor_count)
{
  // Load the TPU model
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);

  // Get EdgeTPU context
  std::shared_ptr<edgetpu::EdgeTpuContext> context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  std::unique_ptr<tflite::Interpreter> interpreter = BuildEdgeTpuInterpreter(*model, context.get());

  // Push the data to each named input tensor
  auto input_tensors = interpreter->inputs();
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    auto input_name = interpreter->GetInputName(i);

    for (size_t j = 0; j < tensor_count; ++j) {
      if (0 == strcmp(input_name, tensor_name[i])) {
        memcpy(interpreter->typed_input_tensor<uint8_t*>(j), tensor_data[j], tensor_size_bytes[j]);
        break;
      }
    }
  }

  // Run the interpreter
  interpreter->Invoke();

  // Extract the data from each output tensor
  auto output_tensors = interpreter->outputs();
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    auto output_name = interpreter->GetOutputName(i);

    for (size_t j = 0; j < tensor_count; ++j) {
      if (0 == strcmp(output_name, tensor_name[i])) {
        // TODO: Assumes that the size of the output tensor data is known
        // statically, but we should really get it from the output?_shape
        memcpy(tensor_data[j], interpreter->typed_output_tensor<uint8_t*>(i), tensor_size_bytes[j]);
        break;
      }
    }
  }
}

