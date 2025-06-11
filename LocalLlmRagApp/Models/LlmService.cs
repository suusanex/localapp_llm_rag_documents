using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Options;
using LocalLlmRagApp.Data;

namespace LocalLlmRagApp.Llm;

public interface ILlmService
{
    Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default);
}

public class OnnxLlmService : ILlmService
{
    private readonly IOptions<AppConfig> _config;
    private readonly IVectorDb _vectorDb;
    private readonly ILogger<OnnxLlmService> _logger;
    private InferenceSession? _session;
    private LocalLlmRagApp.Data.Tokenizer? _tokenizer;
    private bool _initialized = false;

    public OnnxLlmService(IOptions<AppConfig> config, IVectorDb vectorDb, ILogger<OnnxLlmService> logger)
    {
        _config = config;
        _vectorDb = vectorDb;
        _logger = logger;
    }

    public void Initialize()
    {
        if (_initialized) return;

        var modelPath = _config.Value.LlmOnnxModelPath;
        if (string.IsNullOrEmpty(modelPath))
            throw new InvalidOperationException("Model path is not configured");

        _session = new InferenceSession(modelPath);
        _tokenizer = new LocalLlmRagApp.Data.Tokenizer(_config);
        _initialized = true;
    }

    public async Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default)
    {
        if (!_initialized)
            throw new InvalidOperationException("Service is not initialized");

        try
        {
            // RAG: プロンプトの拡張
            var queryVector = GetEmbedding(prompt);
            var similarChunks = await _vectorDb.SearchAsync(queryVector, topK: 3);
            var context = string.Join("\n", similarChunks.Select(x => x.text));

            // プロンプトの構築
            var format = $@"<|system|>You are a helpful AI assistant. Use this context to answer: {context}<|end|>
<|user|>{prompt}<|end|>
<|assistant|>";

            // トークン化
            var inputIds = _tokenizer!.Encode(format);
            var container = new List<NamedOnnxValue>();

            {
                // テンソルの作成
                var shape = new[] { 1, inputIds.Length };
                var inputTensor = new DenseTensor<long>(inputIds, shape);
                container.Add(NamedOnnxValue.CreateFromTensor("input_ids", inputTensor));

                // attention mask
                var attentionMask = new long[inputIds.Length];
                Array.Fill(attentionMask, 1);
                var attentionTensor = new DenseTensor<long>(attentionMask, shape);
                container.Add(NamedOnnxValue.CreateFromTensor("attention_mask", attentionTensor));

                // 推論の実行
                using var results = _session!.Run(container);
                var output = results.First();

                // 結果の処理
                var outputTensor = output.AsTensor<long>();
                var response = _tokenizer.Decode(outputTensor.ToArray());

                return response;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Chat error");
            return $"Error: {ex.Message}";
        }
    }

    private float[] GetEmbedding(string text)
    {
        var embedder = new Embedder(_config);
        embedder.Initialize();
        return embedder.Embed(text);
    }
}
