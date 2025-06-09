using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntimeGenAI;
using Microsoft.Extensions.Options;
using System.Text;
using System.Text.Json;
using LocalLlmRagApp.Data;
using Tokenizer = Microsoft.ML.OnnxRuntimeGenAI.Tokenizer;

namespace LocalLlmRagApp.Llm;

public interface ILlmService
{
    Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default);
}

public class OnnxLlmService : ILlmService
{
    private InferenceSession? _session;
    private string? _modelPath;
    private readonly IOptions<AppConfig> _config;
    private bool _initialized = false;
    private LocalLlmRagApp.Data.Tokenizer? _tokenizer;

    public OnnxLlmService(IOptions<AppConfig> config)
    {
        _config = config;
    }

    public void Initialize()
    {
        if (_initialized) return;
        _modelPath = _config.Value.LlmOnnxModelPath ?? "./models/llm-chat-model.onnx";
        _session = new InferenceSession(_modelPath);
        _tokenizer = new LocalLlmRagApp.Data.Tokenizer(_config);
        _initialized = true;
    }

    public Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default)
    {
        if (!_initialized) throw new InvalidOperationException("OnnxLlmService is not initialized. Call Initialize() first.");
        var inputIds = _tokenizer!.Encode(prompt);
        var inputTensor = new DenseTensor<long>(new[] { 1, inputIds.Length });
        for (int i = 0; i < inputIds.Length; i++) inputTensor[0, i] = inputIds[i];
        var attentionMask = new long[inputIds.Length];
        for (int i = 0; i < inputIds.Length; i++) attentionMask[i] = 1;
        var attentionMaskTensor = new DenseTensor<long>(attentionMask, new[] { 1, inputIds.Length });
        var inputs = new List<NamedOnnxValue> {
            NamedOnnxValue.CreateFromTensor("input_ids", inputTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
        };
        // past_key_values.* の入力があればゼロ埋めで渡す
        foreach (var meta in _session!.InputMetadata)
        {
            var name = meta.Key;
            if (name.StartsWith("past_key_values"))
            {
                var dims = meta.Value.Dimensions.ToArray();
                // -1を1に置き換え（バッチ1、長さ1で初期化）
                for (int i = 0; i < dims.Length; i++)
                    if (dims[i] < 1) dims[i] = 1;
                // float32でゼロ埋め
                var zeroTensor = new DenseTensor<float>(dims);
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, zeroTensor));
            }
        }
        using var results = _session.Run(inputs);
        var output = results.First();
        string response;
        try
        {
            if (output.AsTensor<long>() is Tensor<long> longTensor)
            {
                var tokenIds = longTensor.ToArray();
                response = _tokenizer.Decode(tokenIds);
            }
            else if (output.AsTensor<int>() is Tensor<int> intTensor)
            {
                var tokenIds = intTensor.ToArray().Select(x => (long)x).ToArray();
                response = _tokenizer.Decode(tokenIds);
            }
            else if (output.AsTensor<float>() is Tensor<float> floatTensor)
            {
                var logits = floatTensor.ToArray();
                long tokenId = Array.IndexOf(logits, logits.Max());
                response = _tokenizer.Decode(new long[] { tokenId });
            }
            else
            {
                response = "[ERROR: Unknown output type from ONNX model]";
            }
        }
        catch
        {
            response = "[ERROR: Failed to decode ONNX output]";
        }
        return Task.FromResult(response);
    }
}
