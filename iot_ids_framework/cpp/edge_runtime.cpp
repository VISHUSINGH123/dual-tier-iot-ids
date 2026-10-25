// FR-08 reference: ONNX Runtime C++ inference with no Python interpreter.
#include <array>
#include <iostream>
#include <onnxruntime_cxx_api.h>

int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "Usage: iot_ids_edge_runtime model.onnx\n"; return 2; }
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "iot_ids");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);  // NFR-03: constrain inference to one core.
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  Ort::Session session(env, argv[1], options);
  std::array<float, 14> features{};
  for (size_t i = 0; i < features.size() && std::cin; ++i) std::cin >> features[i];
  std::array<int64_t, 2> shape{1, static_cast<int64_t>(features.size())};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input = Ort::Value::CreateTensor<float>(memory, features.data(), features.size(), shape.data(), shape.size());
  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name = session.GetInputNameAllocated(0, allocator);
  auto output_name = session.GetOutputNameAllocated(1, allocator); // LightGBM ONNX probabilities output.
  const char* inputs[] = {input_name.get()};
  const char* outputs[] = {output_name.get()};
  auto result = session.Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs, 1);
  const auto info = result[0].GetTensorTypeAndShapeInfo();
  const float* probabilities = result[0].GetTensorData<float>();
  size_t winner = 0;
  for (size_t i = 1; i < info.GetElementCount(); ++i) if (probabilities[i] > probabilities[winner]) winner = i;
  std::cout << "class_index=" << winner << " confidence=" << probabilities[winner] << '\n';
}
