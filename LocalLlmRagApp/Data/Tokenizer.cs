using Microsoft.ML.Tokenizers;
using System.Linq;
using System.IO;

namespace LocalLlmRagApp.Data;

// --- TokenizerのE5用実装（Microsoft.ML.Tokenizers利用） ---
public class Tokenizer
{
    private readonly SentencePieceTokenizer _tokenizer;

    public Tokenizer(string modelPath = "./models/sentencepiece.model")
    {
        // E5用tokenizer.jsonをローカルからロード
        using var stream = File.OpenRead(modelPath);
        _tokenizer = SentencePieceTokenizer.Create(stream);
    }

    public long[] Encode(string text)
    {
        var encoding = _tokenizer.EncodeToIds(text);
        return encoding.Select(id => (long)id).ToArray();
    }
}
