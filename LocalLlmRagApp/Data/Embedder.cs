using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Options;

namespace LocalLlmRagApp.Data;

public enum EmbeddingModelType
{
    IntfloatMultilingualE5Base, // 768次元
    IntfloatMultilingualE5Large // 1024次元
}

public class Embedder
{
    private InferenceSession? _session;
    private EmbedderTokenizer? _tokenizer;
    private readonly IOptions<AppConfig> _config;
    private bool _initialized = false;
    private const int MaxLength = 128; // モデルの最大長に合わせて調整

    public static int GetDimensions(EmbeddingModelType type) => type switch
    {
        EmbeddingModelType.IntfloatMultilingualE5Base => 768,
        EmbeddingModelType.IntfloatMultilingualE5Large => 1024,
        _ => throw new NotSupportedException()
    };

    public EmbeddingModelType ModelType { get; }
    public int EmbeddingDimensions => GetDimensions(ModelType);

    public Embedder(IOptions<AppConfig> config, EmbeddingModelType modelType = EmbeddingModelType.IntfloatMultilingualE5Base)
    {
        _config = config;
        ModelType = modelType;
    }

    public void Initialize()
    {
        if (_initialized) return;
        var modelPath = _config.Value.EmbeddingOnnxModelPath;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new InvalidOperationException("AppConfig.EmbeddingOnnxModelPath is required.");
        _session = new InferenceSession(modelPath);
        _tokenizer = new EmbedderTokenizer(_config);
        _initialized = true;
    }

    public float[] Embed(string text)
    {
        if (!_initialized) throw new InvalidOperationException("Embedder is not initialized. Call Initialize() first.");
        var inputIds = _tokenizer!.Encode(text);
        // パディングまたは切り詰め
        var padded = new long[MaxLength];
        var attentionMask = new long[MaxLength];
        int len = Math.Min(inputIds.Length, MaxLength);
        Array.Copy(inputIds, padded, len);
        for (int i = 0; i < len; i++) attentionMask[i] = 1;
        var inputTensor = new DenseTensor<long>(padded, new[] { 1, MaxLength });
        var maskTensor = new DenseTensor<long>(attentionMask, new[] { 1, MaxLength });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", maskTensor)
        };
        using var results = _session!.Run(inputs);
        var tensor = results.First().AsTensor<float>();
        var shape = tensor.Dimensions.ToArray();
        if (shape.Length == 3 && shape[0] == 1) // [1, N, 1024]
        {
            int n = shape[1];
            int d = shape[2];
            var arr = tensor.ToArray();
            var pooled = new float[d];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < d; j++)
                    pooled[j] += arr[i * d + j];
            for (int j = 0; j < d; j++) pooled[j] /= n;
            return pooled;
        }
        else if (shape.Length == 2 && shape[0] > 1) // [N, 1024]
        {
            int n = shape[0];
            int d = shape[1];
            var arr = tensor.ToArray();
            var pooled = new float[d];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < d; j++)
                    pooled[j] += arr[i * d + j];
            for (int j = 0; j < d; j++) pooled[j] /= n;
            return pooled;
        }
        else if (shape.Length == 2 && shape[0] == 1) // [1, 1024]
        {
            return tensor.ToArray();
        }
        else if (shape.Length == 1 && shape[0] == 1024) // [1024]
        {
            return tensor.ToArray();
        }
        else
        {
            throw new InvalidOperationException($"Unexpected embedding shape: [{string.Join(",", shape)}]");
        }
    }
}
