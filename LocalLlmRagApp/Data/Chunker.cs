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
                    yield return SummarizeAndValidateChunk(currentChunk.ToString().Trim(), currentHeadings).GetAwaiter().GetResult();
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
                    yield return SummarizeAndValidateChunk(currentChunk.ToString().Trim(), currentHeadings).GetAwaiter().GetResult();
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
                    yield return SummarizeAndValidateChunk(currentChunk.ToString().Trim(), currentHeadings).GetAwaiter().GetResult();
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
            yield return SummarizeAndValidateChunk(currentChunk.ToString().Trim(), currentHeadings).GetAwaiter().GetResult();
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

    // チャンク本文を要約し、見出しはC#側で付加
    private async Task<string> SummarizeAndValidateChunk(string chunk, List<string> headings)
    {
        var chunkLines = chunk.Split(Environment.NewLine);
        int headingCount = 0;
        var headingBuilder = new StringBuilder();
        foreach (var h in headings)
        {
            headingBuilder.AppendLine(h);
            headingCount++;
        }
        // 本文部分
        var bodyBuilder = new StringBuilder();
        for (int i = headingCount; i < chunkLines.Length; i++)
        {
            bodyBuilder.AppendLine(chunkLines[i]);
        }
        var body = bodyBuilder.ToString().Trim();
        var headingText = headingBuilder.ToString();

        // headingsのトークン数をEmbedderで計算
        string headingsConcat = string.Join("\n", headings);
        int headingsTokenCount = _embedder.GetTokenCount(headingsConcat);
        // summaryChatTokenLimitの計算
        var summaryTokenLimit = _embeddingTokenLimit - headingsTokenCount;

        // LLMで要約（見出し保持指示は削除）
        string summaryPrompt = "<|system|>あなたはテキストの要約を適切に行うエージェントです。ユーザープロンプトの内容を、名詞などの重要単語や内容が失われないように要約する必要があります。要約は" + summaryTokenLimit + "トークン以内で作成し、要約のみを出力し、そこで出力を終了してください。<|end|><|user|>{BODY}<|end|><|assistant|># 要約\n";

        // summaryPromptのトークン数をILlmService経由で取得
        int promptTokenCount = _llmService.GetTokenCount(summaryPrompt.Replace("{BODY}", ""));
        int summaryChatTokenLimit = summaryTokenLimit + promptTokenCount;
        if (summaryChatTokenLimit < 1) summaryChatTokenLimit = 1;

        // LLMモデルの最大長を取得
        int contextLength = _llmService.GetContextLength();
        // bodyが長すぎてmax_lengthがcontextLengthを超える場合は分割
        var bodyLines = body.Split('\n');
        var summaries = new List<string>();
        int start = 0;
        while (start < bodyLines.Length)
        {
            // 1チャンク分のbodyを決定
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
                // 1行すら入らない場合は強制的に1行だけ処理
                partBuilder.AppendLine(bodyLines[start]);
                end = start + 1;
                partTokenCount = _llmService.GetTokenCount(summaryPrompt.Replace("{BODY}", bodyLines[start]));
            }
            string partBody = partBuilder.ToString().Trim();
            string partPrompt = summaryPrompt.Replace("{BODY}", partBody);
            int partMaxLength = summaryTokenLimit + partTokenCount;
            if (partMaxLength < 1) partMaxLength = 1;
            string partSummary = await _llmService.ChatAsyncDirect(partPrompt, [("max_length", partMaxLength)], CancellationToken.None);
            // Markdown見出し1が出現したらそれ以降を無視（^#\s+のみ）
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
        string resultChunk = headingText + string.Join("\n", summaries).Trim();
        // トークン数チェックと切り詰め処理
        int resultTokenCount = _embedder.GetTokenCount(resultChunk);
        if (resultTokenCount > _embedder.MaxTokenLength)
        {
            // 文字単位で切り詰めてトークン数が上限以内になるまで調整
            int maxToken = _embedder.MaxTokenLength;
            string truncated = resultChunk;
            // 粗く末尾を削る（高速化のため）
            int approxLength = (int)((double)resultChunk.Length * maxToken / resultTokenCount);
            if (approxLength < resultChunk.Length)
                truncated = resultChunk.Substring(0, approxLength);
            // 1文字ずつ減らして調整
            while (_embedder.GetTokenCount(truncated) > maxToken && truncated.Length > 0)
            {
                truncated = truncated.Substring(0, truncated.Length - 1);
            }
            resultChunk = truncated;
        }
        return resultChunk;
    }
}
