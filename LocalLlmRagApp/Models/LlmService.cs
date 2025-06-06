using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Options;
using System.Text;
using System.Text.Json;

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

public class DummyLlmService : ILlmService
{
    public Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default)
    {
        // ダミー応答
        return Task.FromResult($"Echo: {prompt}");
    }
}
