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

    public Embedder(IOptions<AppConfig> config)
    {
        _config = config;
    }

    public void Initialize()
    {
        if (_initialized) return;
        var modelPath = _config.Value.EmbeddingOnnxModelPath;
        _session = new InferenceSession(modelPath);
        _tokenizer = new Tokenizer();
        _initialized = true;
    }

    public float[] Embed(string text)
    {
        if (!_initialized) throw new InvalidOperationException("Embedder is not initialized. Call Initialize() first.");
        var inputIds = _tokenizer!.Encode(text);
        var inputTensor = new DenseTensor<long>(new[] { 1, inputIds.Length });
        for (int i = 0; i < inputIds.Length; i++) inputTensor[0, i] = inputIds[i];
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", inputTensor) };
        using var results = _session!.Run(inputs);
        var embedding = results.First().AsEnumerable<float>().ToArray();
        return embedding;
    }
}
