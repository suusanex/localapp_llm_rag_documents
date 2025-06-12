using System.Collections.Generic;
using System.Text;

namespace LocalLlmRagApp.Data;

public class Chunker
{
    public int MaxChunkLength { get; set; } = 800;

    public IEnumerable<string> Chunk(string text)
    {
        var lines = text.Split('\n');
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
        var lines = chunk.ToString().Split('\n');
        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (!string.IsNullOrEmpty(trimmed) && !trimmed.StartsWith("#"))
                return true;
        }
        return false;
    }

    // 本文がMaxChunkLengthを超える場合、見出しを付加して分割
    private IEnumerable<string> SplitWithHeadings(string chunk, List<string> headings)
    {
        // 見出し部分を抽出
        var chunkLines = chunk.Split('\n');
        int headingCount = 0;
        var headingBuilder = new StringBuilder();
        foreach (var h in headings)
        {
            headingBuilder.AppendLine(h);
            headingCount++;
        }
        int headingLength = headingBuilder.Length;
        // 本文部分
        var bodyBuilder = new StringBuilder();
        for (int i = headingCount; i < chunkLines.Length; i++)
        {
            bodyBuilder.AppendLine(chunkLines[i]);
        }
        var body = bodyBuilder.ToString().Trim();
        if (headingLength + body.Length <= MaxChunkLength)
        {
            yield return chunk;
            yield break;
        }
        // 本文を分割
        int maxBodyLen = MaxChunkLength - headingLength;
        int pos = 0;
        while (pos < body.Length)
        {
            int len = Math.Min(maxBodyLen, body.Length - pos);
            var part = body.Substring(pos, len);
            var sb = new StringBuilder();
            sb.Append(headingBuilder.ToString());
            sb.Append(part);
            yield return sb.ToString().Trim();
            pos += len;
        }
    }
}
