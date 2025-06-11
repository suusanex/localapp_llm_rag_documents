using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace LocalLlmRagApp.Data;

public class Chunker
{
    public IEnumerable<string> Chunk(string text)
    {
        // Markdown�̌��o�����x��1�`3�i#, ##, ###�j���ƂɃ`�����N����
        var lines = text.Split('\n');
        StringBuilder chunkBuilder = new StringBuilder();
        bool firstSection = true;
        var headingRegex = new Regex(@"^\s*#{1,3} ");
        foreach (var line in lines)
        {
            if (headingRegex.IsMatch(line))
            {
                if (!firstSection && chunkBuilder.Length > 0)
                {
                    yield return chunkBuilder.ToString().Trim();
                    chunkBuilder.Clear();
                }
                firstSection = false;
            }
            chunkBuilder.AppendLine(line);
        }
        if (chunkBuilder.Length > 0)
        {
            yield return chunkBuilder.ToString().Trim();
        }
    }
}
