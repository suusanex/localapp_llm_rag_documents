using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntimeGenAI;
using Microsoft.Extensions.Options;
using System.Text;
using System.Text.Json;
using LocalLlmRagApp.Data;
using Tokenizer = Microsoft.ML.OnnxRuntimeGenAI.Tokenizer;

namespace LocalLlmRagApp.Llm;

public interface ILlmService
{
    Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default);
}

public class OnnxLlmService : ILlmService
{
    private readonly InferenceSession _session;
    private readonly string _modelPath;

    public OnnxLlmService(IOptions<AppConfig> config)
    {
        // appsettings.json等でONNXモデルパスを指定（例: "./models/llm-chat-model.onnx"）
        _modelPath = config.Value.LlmOnnxModelPath ?? "./models/llm-chat-model.onnx";
        _session = new InferenceSession(_modelPath);
    }

    public Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default)
    {
        // 入力をトークナイズ（ここではUTF-8バイト列をint化する簡易例。実際はモデルに合わせて要修正）
        var inputIds = Encoding.UTF8.GetBytes(prompt).Select(b => (long)b).ToArray();
        var inputTensor = new DenseTensor<long>(new[] { 1, inputIds.Length });
        for (int i = 0; i < inputIds.Length; i++) inputTensor[0, i] = inputIds[i];
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", inputTensor) };
        // 推論
        using var results = _session.Run(inputs);
        // 出力（ここでは最初の出力テンソルをUTF-8文字列に変換する簡易例。モデル仕様に合わせて要修正）
        var outputTensor = results.First().AsEnumerable<long>().ToArray();
        var response = Encoding.UTF8.GetString(outputTensor.Select(x => (byte)x).ToArray());
        return Task.FromResult(response);
    }
}

public class Phi3MiniOnnxLlmService : ILlmService
{
    private readonly Model _model;
    private readonly Microsoft.ML.OnnxRuntimeGenAI.Tokenizer _tokenizer;
    private readonly Embedder _embedder;
    private readonly IVectorDb _vectorDb;

    public Phi3MiniOnnxLlmService(IOptions<AppConfig> config, Embedder embedder, IVectorDb vectorDb)
    {
        // Phi-3-mini-4k-instructのONNXモデルパスをAppConfigから取得
        var modelPath = config.Value.LlmOnnxModelPath ?? "./models/phi-3-mini-4k-instruct.onnx";
        _model = new Model(modelPath);
        _tokenizer = new Tokenizer(_model);
        _embedder = embedder;
        _vectorDb = vectorDb;
    }

    public async Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default)
    {
        // 1. 入力文をベクトル化
        var queryVec = _embedder.Embed(prompt);
        // 2. ベクトルDBから類似チャンク検索
        var similarChunks = _vectorDb.Search(queryVec, topK: 3).Select(x => x.text).ToArray();
        // 3. RAGプロンプト生成（Phi-3のプロンプト形式に合わせる）
        var ragPrompt = $"<|system|>You are a helpful assistant. Use the following context to answer the user's question.<|end|>\n<|context|>\n{string.Join("\n---\n", similarChunks)}\n<|end|>\n<|user|>{prompt}<|end|>\n<|assistant|>";
        // 4. トークナイズ
        var inputTokens = _tokenizer.Encode(ragPrompt);
        // 5. Generator生成
        var generatorParams = new GeneratorParams(_model);
        generatorParams.SetSearchOption("max_length", 2048);
        generatorParams.TryGraphCaptureWithMaxBatchSize(1);
        using var generator = new Generator(_model, generatorParams);
        generator.AppendTokenSequences(inputTokens);
        var sb = new StringBuilder();
        while (!generator.IsDone())
        {
            await Task.Delay(TimeSpan.FromMilliseconds(10), cancellationToken).ConfigureAwait(false);
            generator.GenerateNextToken();
            var part = _tokenizer.Decode(generator.GetSequence(0).ToArray());
            sb.Append(part);
            if (sb.ToString().Contains("<|end|>")
                || sb.ToString().Contains("<|user|>")
                || sb.ToString().Contains("<|system|>"))
            {
                break;
            }
        }
        return sb.ToString();
    }
}

public class DummyLlmService : ILlmService
{
    public Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default)
    {
        // ダミー応答
        return Task.FromResult($"Echo: {prompt}");
    }
}
