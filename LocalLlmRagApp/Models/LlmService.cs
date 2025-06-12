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
            // 1. ベクトルDBから最大100件取得
            var queryVector = GetEmbedding(prompt);
            var similarChunks = await _vectorDb.SearchAsync(queryVector, topK: 100);
            var chunkTexts = similarChunks.Select(x => x.text).ToList();
            var selectedChunks = new List<string>();

            // 2. 10件ずつLLMで関連性の高い2件を抽出
            for (int i = 0; i < chunkTexts.Count; i += 10)
            {
                var group = chunkTexts.Skip(i).Take(10).ToList();
                if (group.Count == 0) break;
                // プロンプトを工夫してLLMに渡す
                var selectionPrompt = BuildSelectionPrompt(prompt, group);
                var llmResult = await GetLlmResultAsync(selectionPrompt, cancellationToken);
                var selected = ParseSelectedChunksFromLlmResult(llmResult, group);
                selectedChunks.AddRange(selected);
            }

            // 3. 20件をcontextとして従来のプロンプトで最終回答
            var context = string.Join("\n", selectedChunks);
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

    private string BuildSelectionPrompt(string question, List<string> group)
    {
        var sb = new StringBuilder();
        sb.AppendLine("次の10個のテキストの中から、質問に最も関連性が高いものを2つ選んでください。選んだテキストの番号のみをカンマ区切りで出力してください。\n");
        sb.AppendLine($"質問: {question}\n");
        for (int i = 0; i < group.Count; i++)
        {
            sb.AppendLine($"[{i}] {group[i].Replace("\n", " ")}");
        }
        sb.AppendLine("\n出力例: 3,7");
        return sb.ToString();
    }

    private async Task<string> GetLlmResultAsync(string prompt, CancellationToken cancellationToken)
    {
        StringBuilder buf = new();
        await foreach (var messagePart in InferStreaming(prompt, cancellationToken))
        {
            buf.Append(messagePart);
        }
        return buf.ToString();
    }

    private List<string> ParseSelectedChunksFromLlmResult(string llmResult, List<string> group)
    {
        var result = new List<string>();
        var line = llmResult.Split('\n').FirstOrDefault(l => l.Trim().Any(char.IsDigit));
        if (line == null) return result;
        var nums = line.Split(',').Select(s => s.Trim()).Where(s => int.TryParse(s, out _)).Select(int.Parse);
        foreach (var idx in nums)
        {
            if (idx >= 0 && idx < group.Count)
                result.Add(group[idx]);
        }
        return result;
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
