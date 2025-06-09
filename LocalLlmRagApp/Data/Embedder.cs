using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Options;

namespace LocalLlmRagApp.Data;

public class Embedder
{
    private InferenceSession? _session;
    private Tokenizer? _tokenizer;
    private readonly IOptions<AppConfig> _config;
    private bool _initialized = false;
    private const int MaxLength = 128; // ���f���̍ő咷�ɍ��킹�Ē���

    public Embedder(IOptions<AppConfig> config)
    {
        _config = config;
    }

    public void Initialize()
    {
        if (_initialized) return;
        var modelPath = _config.Value.EmbeddingOnnxModelPath;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new InvalidOperationException("AppConfig.EmbeddingOnnxModelPath is required.");
        _session = new InferenceSession(modelPath);
        _tokenizer = new Tokenizer(_config);
        _initialized = true;
    }

    public float[] Embed(string text)
    {
        if (!_initialized) throw new InvalidOperationException("Embedder is not initialized. Call Initialize() first.");
        var inputIds = _tokenizer!.Encode(text);
        // �p�f�B���O�܂��͐؂�l��
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
        // ONNX�o��shape��[1, 1024]���̏ꍇ��1�����z��֕ϊ�
        var embedding = results.First().AsTensor<float>().ToArray();
        return embedding;
    }
}
