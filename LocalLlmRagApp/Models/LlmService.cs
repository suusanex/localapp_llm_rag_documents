// LLMチャットCUIの雛形クラスを追加
namespace LocalLlmRagApp.Llm;

public interface ILlmService
{
    Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default);
}

public class DummyLlmService : ILlmService
{
    public Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default)
    {
        // ダミー応答
        return Task.FromResult($"Echo: {prompt}");
    }
}
