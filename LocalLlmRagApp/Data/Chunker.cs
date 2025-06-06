using System.Collections.Generic;

namespace LocalLlmRagApp.Data;

public class Chunker
{
    public IEnumerable<string> Chunk(string text, int maxLength = 512)
    {
        // �V���v���ȃ`�����N�����imaxLength���Ƃɕ����j
        for (int i = 0; i < text.Length; i += maxLength)
        {
            yield return text.Substring(i, Math.Min(maxLength, text.Length - i));
        }
    }
}
