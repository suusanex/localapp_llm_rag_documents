using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Options;

namespace LocalLlmRagApp.Data;

public class Embedder
{
    private readonly InferenceSession _session;
    private readonly Tokenizer _tokenizer;

    public Embedder(IOptions<AppConfig> config)
    {
        // ONNXモデルファイルパスを設定ファイルから取得
        var modelPath = config.Value.EmbeddingOnnxModelPath;
        _session = new InferenceSession(modelPath);
        _tokenizer = new Tokenizer(); // Tokenizerは後述の簡易実装または外部ライブラリ利用
    }

    public float[] Embed(string text)
    {
        // Tokenize
        var inputIds = _tokenizer.Encode(text);
        var inputTensor = new DenseTensor<long>(new[] { 1, inputIds.Length });
        for (int i = 0; i < inputIds.Length; i++) inputTensor[0, i] = inputIds[i];
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", inputTensor) };
        // 推論
        using var results = _session.Run(inputs);
        var embedding = results.First().AsEnumerable<float>().ToArray();
        return embedding;
    }
}
