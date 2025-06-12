namespace LocalLlmRagApp.Data;

public interface IEmbedder
{
    int MaxTokenLength { get; }
    int GetTokenCount(string text);
}
