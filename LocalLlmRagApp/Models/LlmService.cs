using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Options;
using LocalLlmRagApp.Data;
using Microsoft.ML.OnnxRuntimeGenAI;
using System.IO;

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
    private Microsoft.ML.OnnxRuntimeGenAI.Tokenizer? _llmTokenizer;
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
        var tokenizerPath = _config.Value.TokenizerModelPath;
        if (string.IsNullOrEmpty(modelPath))
            throw new InvalidOperationException("Model path is not configured");
        if (string.IsNullOrEmpty(tokenizerPath))
            throw new InvalidOperationException("Tokenizer path is not configured");

        _session = new InferenceSession(modelPath);
        using var tokenizerStream = File.OpenRead(tokenizerPath);
        _llmTokenizer = Microsoft.ML.OnnxRuntimeGenAI.Tokenizer.CreateStream(tokenizerStream);
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
            var context = string.join("\n", similarChunks.Select(x => x.text));

            // プロンプトの構築
            var format = $"<|system|>You are a helpful AI assistant. Use this context to answer: {context}<|end|>\n<|user|>{prompt}<|end|>\n<|assistant|>";

            // LLM用トークナイザーでエンコード
            var encodeResult = _llmTokenizer!.Encode(format);
            var inputIds = encodeResult.Ids;
            var container = new List<NamedOnnxValue>();

            // テンソルの作成
            var shape = new[] { 1, inputIds.Count };
            var inputTensor = new DenseTensor<long>(inputIds.ToArray(), shape);
            container.Add(NamedOnnxValue.CreateFromTensor("input_ids", inputTensor));

            // attention mask
            var attentionMask = Enumerable.Repeat(1L, inputIds.Count).ToArray();
            var attentionTensor = new DenseTensor<long>(attentionMask, shape);
            container.Add(NamedOnnxValue.CreateFromTensor("attention_mask", attentionTensor));

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
                    container.Add(NamedOnnxValue.CreateFromTensor(name, zeroTensor));
                }
            }

            // 推論の実行
            using var results = _session!.Run(container);
            var output = results.First();

            string response;
            if (output.ElementType == TensorElementType.Int64)
            {
                var outputTensor = output.AsTensor<long>();
                response = _llmTokenizer.Decode(outputTensor.ToArray());
            }
            else if (output.ElementType == TensorElementType.Float)
            {
                // logitsの場合: 最大値インデックスをトークンIDとみなしてデコード
                var logits = output.AsTensor<float>().ToArray();
                long tokenId = Array.IndexOf(logits, logits.Max());
                response = _llmTokenizer.Decode(new long[] { tokenId });
            }
            else
            {
                response = $"[ERROR: Unknown output type: {output.ElementType}]";
            }

            return response;
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
