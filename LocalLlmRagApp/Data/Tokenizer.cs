using Microsoft.ML.Tokenizers;
using System.Linq;
using System.IO;
using Microsoft.Extensions.Options;

namespace LocalLlmRagApp.Data;

// --- Tokenizer��E5�p�����iMicrosoft.ML.Tokenizers���p�j ---
public class Tokenizer
{
    private readonly SentencePieceTokenizer _tokenizer;

    public Tokenizer(IOptions<AppConfig> config)
    {
        var modelPath = config.Value.TokenizerModelPath;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new InvalidOperationException("TokenizerModelPath is required in AppConfig.");
        using var stream = File.OpenRead(modelPath);
        _tokenizer = SentencePieceTokenizer.Create(stream);
    }

    public long[] Encode(string text)
    {
        var encoding = _tokenizer.EncodeToIds(text);
        return encoding.Select(id => (long)id).ToArray();
    }
}
