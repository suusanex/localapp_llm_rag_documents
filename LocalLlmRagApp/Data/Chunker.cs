using System.Collections.Generic;
using System.Text;

namespace LocalLlmRagApp.Data;

public class Chunker
{
    private readonly IEmbedder _embedder;

    public Chunker(IEmbedder embedder)
    {
        _embedder = embedder;
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
                    foreach (var chunk in SplitWithHeadings(currentChunk.ToString().Trim(), currentHeadings))
                        yield return chunk;
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
                    foreach (var chunk in SplitWithHeadings(currentChunk.ToString().Trim(), currentHeadings))
                        yield return chunk;
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
                    foreach (var chunk in SplitWithHeadings(currentChunk.ToString().Trim(), currentHeadings))
                        yield return chunk;
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
            foreach (var chunk in SplitWithHeadings(currentChunk.ToString().Trim(), currentHeadings))
                yield return chunk;
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

    // 本文がEmbedder.MaxLengthトークン数を超える場合、見出しを付加して分割
    private IEnumerable<string> SplitWithHeadings(string chunk, List<string> headings)
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
        // 先頭・末尾の空行を除去
        var body = string.Join("\n", bodyBuilder.ToString().Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries));
        var headingText = headingBuilder.ToString();
        // 1チャンクとしてトークン数が収まる場合
        string fullChunk = headingText + body;
        if (_embedder.GetTokenCount(fullChunk) <= _embedder.MaxTokenLength)
        {
            yield return chunk;
            yield break;
        }
        // 本文を分割
        int maxBodyTokens = _embedder.MaxTokenLength - _embedder.GetTokenCount(headingText);
        if (maxBodyTokens <= 0)
            yield break; // 見出しだけでオーバー
        // 1文ずつ追加してトークン数が超えたら分割
        var sentences = body.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
        var partBuilder = new StringBuilder();
        foreach (var sentence in sentences)
        {
            // 追加前にトークン数を判定
            string candidate = headingText + (partBuilder.Length > 0 ? partBuilder.ToString() + "\n" : "") + sentence;
            if (_embedder.GetTokenCount(candidate) > _embedder.MaxTokenLength)
            {
                // 追加前のpartBuilderを出力
                if (partBuilder.Length > 0)
                {
                    yield return (headingText + partBuilder.ToString()).Trim();
                    partBuilder.Clear();
                }
                // 1文だけで超える場合はその文単体で出力
                if (_embedder.GetTokenCount(headingText + sentence) > _embedder.MaxTokenLength)
                {
                    yield return (headingText + sentence).Trim();
                }
                else
                {
                    partBuilder.Append(sentence);
                }
            }
            else
            {
                if (partBuilder.Length > 0)
                    partBuilder.Append('\n');
                partBuilder.Append(sentence);
            }
        }
        if (partBuilder.Length > 0)
        {
            yield return (headingText + partBuilder.ToString()).Trim();
        }
    }
}
