using Microsoft.ML.Tokenizers;
using System.Linq;
using System.IO;

namespace LocalLlmRagApp.Data;

// --- Tokenizer��E5�p�����iMicrosoft.ML.Tokenizers���p�j ---
public class Tokenizer
{
    private readonly SentencePieceTokenizer _tokenizer;

    public Tokenizer(string modelPath = "./models/sentencepiece.model")
    {
        // E5�ptokenizer.json�����[�J�����烍�[�h
        using var stream = File.OpenRead(modelPath);
        _tokenizer = SentencePieceTokenizer.Create(stream);
    }

    public long[] Encode(string text)
    {
        var encoding = _tokenizer.EncodeToIds(text);
        return encoding.Select(id => (long)id).ToArray();
    }
}
