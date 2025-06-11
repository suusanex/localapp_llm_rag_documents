using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace LocalLlmRagApp.Data;

public class Chunker
{
    public IEnumerable<string> Chunk(string text)
    {
        // Markdownの見出しレベル3ごとに分割し、直前の#と##を付加
        var lines = text.Split('\n');
        string? lastH1 = null;
        string? lastH2 = null;
        StringBuilder currentChunk = null;
        bool inH3 = false;
        foreach (var line in lines)
        {
            var trimmed = line.TrimStart();
            if (trimmed.StartsWith("# "))
            {
                lastH1 = line;
                lastH2 = null;
            }
            else if (trimmed.StartsWith("## "))
            {
                lastH2 = line;
            }
            else if (trimmed.StartsWith("### "))
            {
                if (currentChunk != null)
                {
                    yield return currentChunk.ToString().Trim();
                }
                currentChunk = new StringBuilder();
                if (lastH1 != null)
                {
                    currentChunk.AppendLine(lastH1);
                }
                if (lastH2 != null)
                {
                    currentChunk.AppendLine(lastH2);
                }
                currentChunk.AppendLine(line);
                inH3 = true;
                continue;
            }
            if (inH3 && currentChunk != null)
            {
                currentChunk.AppendLine(line);
            }
        }
        if (currentChunk != null && currentChunk.Length > 0)
        {
            yield return currentChunk.ToString().Trim();
        }
    }
}
