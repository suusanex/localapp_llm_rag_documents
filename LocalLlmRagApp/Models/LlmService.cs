using LocalLlmRagApp.Data;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntimeGenAI;
using Microsoft.ML.Tokenizers;
using System.IO;
using System.Runtime.CompilerServices;
using Tokenizer = Microsoft.ML.OnnxRuntimeGenAI.Tokenizer;

namespace LocalLlmRagApp.Llm;

public interface ILlmService
{
    Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default);
}

public class OnnxLlmService(IOptions<AppConfig> _config, IVectorDb _vectorDb, ILogger<OnnxLlmService> _logger, IConfiguration _iconfig) : ILlmService
{
    private Microsoft.ML.OnnxRuntimeGenAI.Tokenizer? _llmTokenizer;
    private bool _initialized = false;
    private Model _model;


    public void Initialize()
    {
        if (_initialized) return;

        var modelPath = _config.Value.LlmOnnxModelDirPath;
        var tokenizerPath = _config.Value.TokenizerModelPath;
        if (string.IsNullOrEmpty(modelPath))
            throw new InvalidOperationException("Model path is not configured");
        if (string.IsNullOrEmpty(tokenizerPath))
            throw new InvalidOperationException("Tokenizer path is not configured");

        _model = new Model(modelPath);
        _llmTokenizer = new Tokenizer(_model);

        var connectionString = _iconfig.GetSection("ConnectionStrings")["DefaultConnection"];
        if (string.IsNullOrEmpty( connectionString))
            throw new InvalidOperationException("connectionString is not configured");

        _vectorDb.InitializeForReadOnlyAsync(connectionString);

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
            var similarChunks = await _vectorDb.SearchAsync(queryVector, topK: 10);
            var context = string.Join("\n", similarChunks.Select(x => x.text));

            // プロンプトの構築
            var format = $"<|system|>You are a helpful AI assistant. Use this context to answer: {context}<|end|>\n<|user|>{prompt}<|end|>\n<|assistant|>";

            StringBuilder buf = new();
            await foreach (var messagePart in InferStreaming(format, cancellationToken))
            {
                buf.Append(messagePart);
            }

            return buf.ToString();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Chat error");
            return $"Error: {ex.Message}";
        }
    }

    public async IAsyncEnumerable<string> InferStreaming(string prompt, [EnumeratorCancellation] CancellationToken ct = default)
    {
        if (_model == null || _llmTokenizer == null)
        {
            throw new InvalidOperationException("Model is not ready");
        }

        var generatorParams = new GeneratorParams(_model);

        var sequences = _llmTokenizer.Encode(prompt);

        generatorParams.SetSearchOption("max_length", 2048);
        generatorParams.TryGraphCaptureWithMaxBatchSize(1);



        using var tokenizerStream = _llmTokenizer.CreateStream();
        using var generator = new Generator(_model, generatorParams);
        generator.AppendTokenSequences(sequences);
        StringBuilder stringBuilder = new();
        while (!generator.IsDone())
        {
            string part;
            await Task.Delay(10).ConfigureAwait(false);
            generator.GenerateNextToken();
            part = tokenizerStream.Decode(generator.GetSequence(0)[^1]);
            stringBuilder.Append(part);
            if (stringBuilder.ToString().Contains("<|end|>")
                || stringBuilder.ToString().Contains("<|user|>")
                || stringBuilder.ToString().Contains("<|system|>"))
            {
                break;
            }

            yield return part;
        }
    }

    private float[] GetEmbedding(string text)
    {
        var embedder = new Embedder(_config);
        embedder.Initialize();
        return embedder.Embed(text);
    }
}
