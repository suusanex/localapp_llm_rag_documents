using System.Collections.Generic;
using System.Text;

namespace LocalLlmRagApp.Data;

public class Chunker
{
    public IEnumerable<string> Chunk(string text)
    {
        var lines = text.Split('\n');
        string? lastH1 = null;
        string? lastH2 = null;
        StringBuilder? currentChunk = null;
        string? currentHeaderType = null;
        foreach (var line in lines)
        {
            var trimmed = line.TrimStart();
            if (trimmed.StartsWith("# "))
            {
                // flush previous chunk
                if (currentChunk != null && IsValidChunk(currentChunk))
                    foreach (var chunk in SplitByLength(currentChunk.ToString().Trim(), 800))
                        yield return chunk;
                lastH1 = line;
                lastH2 = null;
                currentChunk = new StringBuilder();
                currentChunk.AppendLine(line);
                currentHeaderType = "h1";
            }
            else if (trimmed.StartsWith("## "))
            {
                if (currentChunk != null && IsValidChunk(currentChunk))
                    foreach (var chunk in SplitByLength(currentChunk.ToString().Trim(), 800))
                        yield return chunk;
                lastH2 = line;
                currentChunk = new StringBuilder();
                if (lastH1 != null) currentChunk.AppendLine(lastH1);
                currentChunk.AppendLine(line);
                currentHeaderType = "h2";
            }
            else if (trimmed.StartsWith("### "))
            {
                if (currentChunk != null && IsValidChunk(currentChunk))
                    foreach (var chunk in SplitByLength(currentChunk.ToString().Trim(), 800))
                        yield return chunk;
                currentChunk = new StringBuilder();
                if (lastH1 != null) currentChunk.AppendLine(lastH1);
                if (lastH2 != null) currentChunk.AppendLine(lastH2);
                currentChunk.AppendLine(line);
                currentHeaderType = "h3";
            }
            else
            {
                if (currentChunk != null)
                    currentChunk.AppendLine(line);
            }
        }
        if (currentChunk != null && IsValidChunk(currentChunk))
            foreach (var chunk in SplitByLength(currentChunk.ToString().Trim(), 800))
                yield return chunk;
    }

    private bool IsValidChunk(StringBuilder chunk)
    {
        // �{���i���o���ȊO�j��1�s�ȏ�܂܂�Ă��邩����
        var lines = chunk.ToString().Split('\n');
        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (!string.IsNullOrEmpty(trimmed) && !trimmed.StartsWith("#"))
                return true;
        }
        return false;
    }

    private IEnumerable<string> SplitByLength(string text, int maxLength)
    {
        if (string.IsNullOrEmpty(text))
            yield break;
        for (int i = 0; i < text.Length; i += maxLength)
        {
            yield return text.Substring(i, System.Math.Min(maxLength, text.Length - i));
        }
    }
}
