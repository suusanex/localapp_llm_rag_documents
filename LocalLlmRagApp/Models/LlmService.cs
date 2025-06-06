// LLM�`���b�gCUI�̐��`�N���X��ǉ�
namespace LocalLlmRagApp.Llm;

public interface ILlmService
{
    Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default);
}

public class DummyLlmService : ILlmService
{
    public Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default)
    {
        // �_�~�[����
        return Task.FromResult($"Echo: {prompt}");
    }
}
