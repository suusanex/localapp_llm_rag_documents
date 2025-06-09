namespace LocalLlmRagApp.Data;

public interface IVectorDb
{
    Task AddAsync(string id, float[] vector, string text);
    Task<(string text, float score)[]> SearchAsync(float[] queryVector, int topK = 5);
}
