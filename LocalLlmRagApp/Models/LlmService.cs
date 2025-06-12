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
    // モデル・プロンプト関連パラメータ
    private readonly int _contextLength = 4096; // Phi-3-mini-4k-instruct-onnx
    private readonly int _maxResponseTokens = 2048;
    private readonly int _maxLength = 4096;
    // ベクトルDB関連パラメータ
    private int _vectorDbTopK = 90;
    private int _selectGroupSize = 10;
    private int _selectPerGroup = 2;
    private int _finalContextChunks = 18;

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
        if (string.IsNullOrEmpty(connectionString))
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
            // 1. ベクトルDBから最大_topK件取得
            var queryVector = GetEmbedding(prompt);
            var similarChunks = await _vectorDb.SearchAsync(queryVector, topK: _vectorDbTopK);
            var chunkTexts = similarChunks.Select(x => x.text).ToList();
            var selectedChunks = new List<string>();

            // 2. _selectGroupSize件ずつLLMで関連性の高い_selectPerGroup件を抽出
            for (int i = 0; i < chunkTexts.Count; i += _selectGroupSize)
            {
                var group = chunkTexts.Skip(i).Take(_selectGroupSize).ToList();
                if (group.Count == 0) break;
                var selectionPrompt = BuildSelectionPrompt(prompt, group);
                var llmResult = await GetLlmResultAsync(selectionPrompt, cancellationToken);
                var selected = ParseSelectedChunksFromLlmResult(llmResult, group);
                selectedChunks.AddRange(selected);
            }

            // 3. context_length - maxResponseTokens 未満のトークン数になるようcontextを調整
            var contextChunks = new List<string>();
            int contextTokenLimit = _contextLength - _maxResponseTokens;
            int currentTokens = 0;
            foreach (var chunk in selectedChunks)
            {
                var sequences = _llmTokenizer.Encode(chunk);
                int tokenCount = sequences.NumSequences > 0 ? sequences[0].Length : 0;
                if (currentTokens + tokenCount > contextTokenLimit) break;
                contextChunks.Add(chunk);
                currentTokens += tokenCount;
            }
            var context = string.Join("\n", contextChunks);
            var format = $"<|system|>You are a helpful AI assistant. Use this context to answer: {context}<|end|>\n<|user|>{prompt}<|end|>\n<|assistant|>";

            // format全体のトークン数が contextLength-maxResponseTokens 未満かチェック
            var sequencesFormat = _llmTokenizer.Encode(format);
            int totalTokens = sequencesFormat.NumSequences > 0 ? sequencesFormat[0].Length : 0;
            if (totalTokens > contextTokenLimit)
            {
                // contextをさらに削る
                while (contextChunks.Count > 0)
                {
                    sequencesFormat = _llmTokenizer.Encode(format);
                    totalTokens = sequencesFormat.NumSequences > 0 ? sequencesFormat[0].Length : 0;
                    if (totalTokens <= contextTokenLimit) break;
                    contextChunks.RemoveAt(contextChunks.Count - 1);
                    context = string.Join("\n", contextChunks);
                    format = $"<|system|>You are a helpful AI assistant. Use this context to answer: {context}<|end|>\n<|user|>{prompt}<|end|>\n<|assistant|>";
                }
            }

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
        sb.AppendLine("次の10個のテキストの中から、質問に最も関連性が高いものを2つ選んでください。選んだテキストの番号のみを1行でカンマ区切りで出力してください。番号以外は一切出力しないでください。\n");
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
        // 正規表現で [3, 9] や 3,9 などを抽出
        var regex = new System.Text.RegularExpressions.Regex(@"\[?\s*(\d+)\s*,\s*(\d+)\s*\]?");
        var match = regex.Match(llmResult);
        if (match.Success)
        {
            for (int i = 1; i <= 2; i++)
            {
                if (int.TryParse(match.Groups[i].Value, out int idx) && idx >= 0 && idx < group.Count)
                    result.Add(group[idx]);
            }
            return result;
        }
        // それ以外のパターンも考慮
        var line = llmResult.Split('\n').FirstOrDefault(l => l.Trim().Any(char.IsDigit));
        if (line != null)
        {
            var nums = line.Split(',').Select(s => s.Trim()).Where(s => int.TryParse(s, out _)).Select(int.Parse);
            foreach (var idx in nums)
            {
                if (idx >= 0 && idx < group.Count)
                    result.Add(group[idx]);
            }
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

        generatorParams.SetSearchOption("max_length", _maxLength);
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
