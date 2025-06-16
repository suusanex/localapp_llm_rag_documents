using System;
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
            Console.WriteLine("[1/6] ベクトルDB検索開始...");
            // 1. ベクトルDBから最大_topK/2件取得（ベクトル検索）
            var queryVector = GetEmbedding(prompt);
            var similarChunks = await _vectorDb.SearchAsync(queryVector, topK: _vectorDbTopK / 2);
            var chunkTexts = similarChunks.Select(x => x.text).ToList();

            Console.WriteLine("[2/6] キーワード抽出・キーワード検索開始...");
            // 1b. キーワード検索で最大_topK/2件取得
            var keywords = await ExtractKeywordsAsync(prompt, cancellationToken);
            Console.WriteLine($"質問文から抽出した重要キーワード:{string.Join(",", keywords)}");
            var keywordChunks = await KeywordSearchAsync(keywords, _vectorDbTopK / 2, cancellationToken);

            Console.WriteLine("[3/6] ベクトル・キーワード検索結果合成・重複除去...");
            // 1c. ベクトル検索とキーワード検索の結果を合成（重複除去）
            chunkTexts.AddRange(keywordChunks);
            chunkTexts = chunkTexts.Distinct().ToList();
            var selectedChunks = new List<string>();

            Console.WriteLine($"[4/6] LLM選択処理開始...（{chunkTexts.Count}件を{_selectGroupSize}件ずつ処理）");
            // 2. _selectGroupSize件ずつLLMで関連性の高い_selectPerGroup件を抽出
            int groupCount = (chunkTexts.Count + _selectGroupSize - 1) / _selectGroupSize;
            for (int i = 0; i < chunkTexts.Count; i += _selectGroupSize)
            {
                int groupIndex = (i / _selectGroupSize) + 1;
                var group = chunkTexts.Skip(i).Take(_selectGroupSize).ToList();
                if (group.Count == 0) break;
                Console.WriteLine($"  LLM選択 {groupIndex}/{groupCount} ...");
                var selectionPrompt = BuildSelectionPrompt(prompt, group);
                var llmResult = await GetLlmResultAsync(selectionPrompt, cancellationToken);
                var selected = ParseSelectedChunksFromLlmResult(llmResult, group);
                selectedChunks.AddRange(selected);
            }

            Console.WriteLine("[5/6] コンテキストトークン長調整...");
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
            var format =
                "<|system|>あなたは親切なアシスタントです。以下のコンテキスト情報を基に、ユーザーの質問に正確かつ簡潔に回答してください。<|end|>\n" +
                "<|context|>\n" + context + "\n<|end|>\n" +
                "<|user|>\n" + prompt + "\n<|end|>\n" +
                "<|assistant|>";

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
                    format =
                        "<|system|>あなたは親切なアシスタントです。以下のコンテキスト情報を基に、ユーザーの質問に正確かつ簡潔に回答してください。<|end|>\n" +
                        "<|context|>\n" + context + "\n<|end|>\n" +
                        "<|user|>\n" + prompt + "\n<|end|>\n" +
                        "<|assistant|>";
                }
            }

            Console.WriteLine("[6/6] LLM最終推論開始...");
            StringBuilder buf = new();
            await foreach (var messagePart in InferStreaming(format, cancellationToken))
            {
                buf.Append(messagePart);
            }

            Console.WriteLine("[完了] LLM応答生成完了");
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
        // Markdown制御文字をバックスラッシュでエスケープする関数
        string EscapeMarkdown(string text)
        {
            var chars = new[] {'#', '*', '_', '`', '[', ']', '(', ')', '!', '>', '|', '~', '-'};
            foreach (var c in chars)
            {
                text = text.Replace(c.ToString(), "\\" + c);
            }
            return text;
        }

        var sb = new StringBuilder();
        sb.AppendLine("<|system|>あなたは質問に関連性の高いドキュメントを見つけ出すアシスタントです。ユーザープロンプトに含まれる1つの質問と、複数のドキュメントを読み取り、質問に関連性の高いドキュメントを2つ選定してください。\nユーザープロンプトはMarkdownで構成され、Markdownの見出しでインデックス数値が付いています。Markdownの本文内に、『エスケープされたMarkdown制御文字』が含まれています。エスケープされた文字は、Markdownの解釈としては無視し、ただの文字列として理解する必要があります。\n選定した2つのドキュメントの、インデックス数値を、数値のカンマ区切り表記で応答に返してください。その応答を返したら、処理を終了してください。<|end|>");
        sb.AppendLine("<|user|>");
        sb.AppendLine("# 質問\n");
        sb.AppendLine(EscapeMarkdown(question));
        sb.AppendLine("\n# 複数のドキュメント\n");
        for (int i = 0; i < group.Count; i++)
        {
            sb.AppendLine($"## {i}\n");
            sb.AppendLine(EscapeMarkdown(group[i]));
            sb.AppendLine();
        }
        sb.AppendLine("<|end|>");
        sb.AppendLine("<|assistant|>");
        return sb.ToString();
    }

    private async Task<string> GetLlmResultAsync(string prompt, CancellationToken cancellationToken)
    {
        const int maxRetry = 3;
        for (int retry = 0; retry < maxRetry; retry++)
        {
            StringBuilder buf = new();
            await foreach (var messagePart in InferStreamingForSelection(prompt, cancellationToken))
            {
                buf.Append(messagePart);
            }
            var regex = new System.Text.RegularExpressions.Regex(@"\b\d+\s*,\s*\d+\b");
            foreach (var line in buf.ToString().Split('\n'))
            {
                var trimmed = line.Trim();
                if (regex.IsMatch(trimmed))
                    return regex.Match(trimmed).Value;
            }
            foreach (var line in buf.ToString().Split('\n'))
            {
                var nums = line.Split(',').Select(s => s.Trim()).Where(s => int.TryParse(s, out _)).ToList();
                if (nums.Count >= 2)
                    return string.Join(",", nums.Take(2));
            }
        }
        return string.Empty;
    }

    private async IAsyncEnumerable<string> InferStreamingForSelection(string prompt, [EnumeratorCancellation] CancellationToken ct = default)
    {
        if (_model == null || _llmTokenizer == null)
            throw new InvalidOperationException("Model is not ready");

        var generatorParams = new GeneratorParams(_model);
        var sequences = _llmTokenizer.Encode(prompt);
        int promptTokens = sequences.NumSequences > 0 ? sequences[0].Length : 0;
        int responseTokens = 128; // さらに余裕を持たせる
        int maxLength = promptTokens + responseTokens;
        generatorParams.SetSearchOption("max_length", maxLength);
        generatorParams.SetSearchOption("min_length", 1);
        generatorParams.SetSearchOption("temperature", 0.5f); // 柔軟性を高める
        generatorParams.SetSearchOption("top_p", 0.95f); // 柔軟性を高める

        using var tokenizerStream = _llmTokenizer.CreateStream();
        using var generator = new Generator(_model, generatorParams);
        generator.AppendTokenSequences(sequences);
        while (!generator.IsDone())
        {
            string part;
            await Task.Delay(10).ConfigureAwait(false);
            generator.GenerateNextToken();
            part = tokenizerStream.Decode(generator.GetSequence(0)[^1]);
            yield return part;
        }
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
        // TryGraphCaptureWithMaxBatchSize(1); は非推奨のため削除

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

    // 質問文からキーワードを抽出する（LLM利用、プロンプト簡素化）
    private async Task<List<string>> ExtractKeywordsAsync(string question, CancellationToken cancellationToken)
    {
        var prompt = "次の文から重要なキーワード（名詞や固有名詞など）をカンマ区切りで1行だけ出力してください。説明や余計な語句は不要です。\n\n質問文: " + question + "\n\n出力例: XXXXXXXXXXX, チェック, ログ";
        var result = await GetLlmResultAsyncForKeywords(prompt, cancellationToken);
        return result.Split(',').Select(x => x.Trim()).Where(x => !string.IsNullOrEmpty(x)).ToList();
    }

    // キーワード抽出用に短文・低温度で推論
    private async Task<string> GetLlmResultAsyncForKeywords(string prompt, CancellationToken cancellationToken)
    {
        StringBuilder buf = new();
        await foreach (var messagePart in InferStreamingForKeywords(prompt, cancellationToken))
        {
            buf.Append(messagePart);
        }
        return buf.ToString();
    }

    // InferStreamingのパラメータを短文・低温度に（max_length=プロンプト長+応答分, temperature緩和）
    private async IAsyncEnumerable<string> InferStreamingForKeywords(string prompt, [EnumeratorCancellation] CancellationToken ct = default)
    {
        if (_model == null || _llmTokenizer == null)
            throw new InvalidOperationException("Model is not ready");

        var generatorParams = new GeneratorParams(_model);
        var sequences = _llmTokenizer.Encode(prompt);
        int promptTokens = sequences.NumSequences > 0 ? sequences[0].Length : 0;
        int responseTokens = 32; // 応答用トークン数
        int maxLength = promptTokens + responseTokens;
        generatorParams.SetSearchOption("max_length", maxLength);
        generatorParams.SetSearchOption("min_length", 1);
        generatorParams.SetSearchOption("temperature", 0.3f);
        generatorParams.SetSearchOption("top_p", 0.9f);
        // TryGraphCaptureWithMaxBatchSize(1); は非推奨のため削除

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
            if (stringBuilder.ToString().Contains("\n") || stringBuilder.Length > 128)
            {
                break;
            }
            yield return part;
        }
    }

    // PgvectorDbのSQLキーワード検索を利用するよう修正
    private async Task<List<string>> KeywordSearchAsync(List<string> keywords, int topK, CancellationToken cancellationToken)
    {
        var results = await _vectorDb.SearchByKeywordsAsync(keywords, topK);
        return results.Select(x => x.text).ToList();
    }
}
