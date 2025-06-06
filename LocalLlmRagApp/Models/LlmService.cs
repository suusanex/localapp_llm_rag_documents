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
        // appsettings.json����ONNX���f���p�X���w��i��: "./models/llm-chat-model.onnx"�j
        _modelPath = config.Value.LlmOnnxModelPath ?? "./models/llm-chat-model.onnx";
        _session = new InferenceSession(_modelPath);
    }

    public Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default)
    {
        // ���͂��g�[�N�i�C�Y�i�����ł�UTF-8�o�C�g���int������Ȉ՗�B���ۂ̓��f���ɍ��킹�ėv�C���j
        var inputIds = Encoding.UTF8.GetBytes(prompt).Select(b => (long)b).ToArray();
        var inputTensor = new DenseTensor<long>(new[] { 1, inputIds.Length });
        for (int i = 0; i < inputIds.Length; i++) inputTensor[0, i] = inputIds[i];
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", inputTensor) };
        // ���_
        using var results = _session.Run(inputs);
        // �o�́i�����ł͍ŏ��̏o�̓e���\����UTF-8������ɕϊ�����Ȉ՗�B���f���d�l�ɍ��킹�ėv�C���j
        var outputTensor = results.First().AsEnumerable<long>().ToArray();
        var response = Encoding.UTF8.GetString(outputTensor.Select(x => (byte)x).ToArray());
        return Task.FromResult(response);
    }
}

public class DummyLlmService : ILlmService
{
    public Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default)
    {
        // �_�~�[����
        return Task.FromResult($"Echo: {prompt}");
    }
}
