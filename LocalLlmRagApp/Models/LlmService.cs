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
    private InferenceSession? _session;
    private string? _modelPath;
    private readonly IOptions<AppConfig> _config;
    private bool _initialized = false;

    public OnnxLlmService(IOptions<AppConfig> config)
    {
        _config = config;
    }

    public void Initialize()
    {
        if (_initialized) return;
        _modelPath = _config.Value.LlmOnnxModelPath ?? "./models/llm-chat-model.onnx";
        _session = new InferenceSession(_modelPath);
        _initialized = true;
    }

    public Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default)
    {
        if (!_initialized) throw new InvalidOperationException("OnnxLlmService is not initialized. Call Initialize() first.");
        var inputIds = Encoding.UTF8.GetBytes(prompt).Select(b => (long)b).ToArray();
        var inputTensor = new DenseTensor<long>(new[] { 1, inputIds.Length });
        for (int i = 0; i < inputIds.Length; i++) inputTensor[0, i] = inputIds[i];
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", inputTensor) };
        using var results = _session!.Run(inputs);
        var outputTensor = results.First().AsEnumerable<long>().ToArray();
        var response = Encoding.UTF8.GetString(outputTensor.Select(x => (byte)x).ToArray());
        return Task.FromResult(response);
    }
}

public class Phi3MiniOnnxLlmService : ILlmService
{
    private Model? _model;
    private Microsoft.ML.OnnxRuntimeGenAI.Tokenizer? _tokenizer;
    private readonly IOptions<AppConfig> _config;
    private readonly Embedder _embedder;
    private readonly IVectorDb _vectorDb;
    private bool _initialized = false;

    public Phi3MiniOnnxLlmService(IOptions<AppConfig> config, Embedder embedder, IVectorDb vectorDb)
    {
        _config = config;
        _embedder = embedder;
        _vectorDb = vectorDb;
    }

    public void Initialize()
    {
        if (_initialized) return;
        var modelPath = _config.Value.LlmOnnxModelPath ?? "./models/phi-3-mini-4k-instruct.onnx";
        _model = new Model(modelPath);
        _tokenizer = new Microsoft.ML.OnnxRuntimeGenAI.Tokenizer(_model);
        _initialized = true;
    }

    public async Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default)
    {
        if (!_initialized) throw new InvalidOperationException("Phi3MiniOnnxLlmService is not initialized. Call Initialize() first.");
        var queryVec = _embedder.Embed(prompt);
        var similarChunks = _vectorDb.Search(queryVec, topK: 3).Select(x => x.text).ToArray();
        var ragPrompt = $"<|system|>You are a helpful assistant. Use the following context to answer the user's question.<|end|>\n<|context|>\n{string.Join("\n---\n", similarChunks)}\n<|end|>\n<|user|>{prompt}<|end|>\n<|assistant|>";
        var inputTokens = _tokenizer!.Encode(ragPrompt);
        var generatorParams = new GeneratorParams(_model!);
        generatorParams.SetSearchOption("max_length", 2048);
        generatorParams.TryGraphCaptureWithMaxBatchSize(1);
        using var generator = new Generator(_model!, generatorParams);
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
