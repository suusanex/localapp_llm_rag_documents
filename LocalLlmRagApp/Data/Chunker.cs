using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using LocalLlmRagApp.Llm;
using System.Text.RegularExpressions;

namespace LocalLlmRagApp.Data;

public class Chunker
{
    private readonly IEmbedder _embedder;
    private readonly ILlmService _llmService;
    private readonly int _embeddingTokenLimit;

    public Chunker(IEmbedder embedder, ILlmService llmService)
    {
        _embedder = embedder;
        _llmService = llmService;
        _embeddingTokenLimit = (int)(_embedder.MaxTokenLength - 10);
    }

    public IEnumerable<string> Chunk(string text)
    {
        var lines = text.Split(Environment.NewLine);
        string? lastH1 = null;
        string? lastH2 = null;
        string? lastH3 = null;
        StringBuilder? currentChunk = null;
        string? currentHeaderType = null;
        List<string> currentHeadings = new();
        foreach (var line in lines)
        {
            var trimmed = line.TrimStart();
            if (trimmed.StartsWith("# "))
            {
                if (currentChunk != null && IsValidChunk(currentChunk))
                {
                    foreach (var chunk in SummarizeAndValidateChunk(currentChunk.ToString().Trim(), currentHeadings).GetAwaiter().GetResult())
                        yield return chunk;
                }
                lastH1 = line;
                lastH2 = null;
                lastH3 = null;
                currentChunk = new StringBuilder();
                currentChunk.AppendLine(line);
                currentHeaderType = "h1";
                currentHeadings = new List<string> { lastH1 };
            }
            else if (trimmed.StartsWith("## "))
            {
                if (currentChunk != null && IsValidChunk(currentChunk))
                {
                    foreach (var chunk in SummarizeAndValidateChunk(currentChunk.ToString().Trim(), currentHeadings).GetAwaiter().GetResult())
                        yield return chunk;
                }
                lastH2 = line;
                lastH3 = null;
                currentChunk = new StringBuilder();
                if (lastH1 != null) currentChunk.AppendLine(lastH1);
                currentChunk.AppendLine(line);
                currentHeaderType = "h2";
                currentHeadings = new List<string>();
                if (lastH1 != null) currentHeadings.Add(lastH1);
                currentHeadings.Add(lastH2);
            }
            else if (trimmed.StartsWith("### "))
            {
                if (currentChunk != null && IsValidChunk(currentChunk))
                {
                    foreach (var chunk in SummarizeAndValidateChunk(currentChunk.ToString().Trim(), currentHeadings).GetAwaiter().GetResult())
                        yield return chunk;
                }
                lastH3 = line;
                currentChunk = new StringBuilder();
                if (lastH1 != null) currentChunk.AppendLine(lastH1);
                if (lastH2 != null) currentChunk.AppendLine(lastH2);
                currentChunk.AppendLine(line);
                currentHeaderType = "h3";
                currentHeadings = new List<string>();
                if (lastH1 != null) currentHeadings.Add(lastH1);
                if (lastH2 != null) currentHeadings.Add(lastH2);
                currentHeadings.Add(lastH3);
            }
            else
            {
                if (currentChunk != null)
                    currentChunk.AppendLine(line);
            }
        }
        if (currentChunk != null && IsValidChunk(currentChunk))
        {
            foreach (var chunk in SummarizeAndValidateChunk(currentChunk.ToString().Trim(), currentHeadings).GetAwaiter().GetResult())
                yield return chunk;
        }
    }

    private bool IsValidChunk(StringBuilder chunk)
    {
        var lines = chunk.ToString().Split(Environment.NewLine);
        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (!string.IsNullOrEmpty(trimmed) && !trimmed.StartsWith("#"))
                return true;
        }
        return false;
    }

    // 要約とバリデーションを行い、トークン上限を超える場合は分割して返す
    private async Task<List<string>> SummarizeAndValidateChunk(string chunk, List<string> headings)
    {
        var chunkLines = chunk.Split(Environment.NewLine);
        int headingCount = 0;
        var headingBuilder = new StringBuilder();
        foreach (var h in headings)
        {
            headingBuilder.AppendLine(h);
            headingCount++;
        }
        var bodyBuilder = new StringBuilder();
        for (int i = headingCount; i < chunkLines.Length; i++)
        {
            bodyBuilder.AppendLine(chunkLines[i]);
        }
        var body = bodyBuilder.ToString().Trim();
        var headingText = headingBuilder.ToString();

        string headingsConcat = string.Join("\n", headings);
        int headingsTokenCount = _embedder.GetTokenCount(headingsConcat);
        var summaryTokenLimit = _embeddingTokenLimit - headingsTokenCount;

        string summaryPrompt = "<|system|>あなたはテキストの要約を適切に行うエージェントです。ユーザープロンプトの内容を、名詞などの重要単語や内容が失われないように要約する必要があります。要約のみを出力し、そこで出力を終了してください。<|end|><|user|>{BODY}<|end|><|assistant|># 要約\n";

        int promptTokenCount = _llmService.GetTokenCount(summaryPrompt.Replace("{BODY}", ""));
        int summaryChatTokenLimit = summaryTokenLimit + promptTokenCount;
        if (summaryChatTokenLimit < 1) summaryChatTokenLimit = 1;

        int contextLength = _llmService.GetContextLength();
        var bodyLines = body.Split('\n');
        var summaries = new List<string>();
        int start = 0;
        while (start < bodyLines.Length)
        {
            var partBuilder = new StringBuilder();
            int partTokenCount = 0;
            int end = start;
            for (; end < bodyLines.Length; end++)
            {
                var testBuilder = new StringBuilder();
                for (int i = start; i <= end; i++)
                    testBuilder.AppendLine(bodyLines[i]);
                string testBody = testBuilder.ToString().Trim();
                string testPrompt = summaryPrompt.Replace("{BODY}", testBody);
                int testPromptTokenCount = _llmService.GetTokenCount(testPrompt);
                int testMaxLength = summaryTokenLimit + testPromptTokenCount;
                if (testMaxLength > contextLength)
                    break;
                partBuilder = testBuilder;
                partTokenCount = testPromptTokenCount;
            }
            if (partBuilder.Length == 0 && start < bodyLines.Length)
            {
                partBuilder.AppendLine(bodyLines[start]);
                end = start + 1;
                partTokenCount = _llmService.GetTokenCount(summaryPrompt.Replace("{BODY}", bodyLines[start]));
            }
            string partBody = partBuilder.ToString().Trim();
            string partPrompt = summaryPrompt.Replace("{BODY}", partBody);
            int partMaxLength = summaryTokenLimit + partTokenCount;
            if (partMaxLength < 1) partMaxLength = 1;
            string partSummary = await _llmService.ChatAsyncDirect(partPrompt, [("max_length", partMaxLength)], CancellationToken.None);
            var lines = partSummary.Split('\n');
            var filteredLines = new List<string>();
            var regexH1 = new Regex("^#\\s+");
            foreach (var line in lines)
            {
                if (regexH1.IsMatch(line))
                    break;
                filteredLines.Add(line);
            }
            string filteredSummary = string.Join("\n", filteredLines).Trim();
            summaries.Add(filteredSummary);
            start = end + 1;
        }
        // ここから分割処理
        var resultChunks = new List<string>();
        var summaryText = string.Join("\n", summaries).Trim();
        // headings+summaryTextをトークン上限で分割
        int maxToken = _embedder.MaxTokenLength;
        int summaryStart = 0;
        while (summaryStart < summaryText.Length)
        {
            // まず大まかに分割
            int approxLength = summaryText.Length - summaryStart;
            string candidate;
            do
            {
                candidate = summaryText.Substring(summaryStart, approxLength);
                string candidateChunk = headingText + candidate;
                int tokenCount = _embedder.GetTokenCount(candidateChunk);
                if (tokenCount <= maxToken)
                {
                    resultChunks.Add(candidateChunk);
                    summaryStart += approxLength;
                    break;
                }
                approxLength--;
            } while (approxLength > 0);
            if (approxLength == 0)
            {
                // 1文字も入らない場合は強制的に1文字進める
                summaryStart++;
            }
        }
        return resultChunks;
    }
}
