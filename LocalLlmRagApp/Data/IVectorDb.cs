namespace LocalLlmRagApp.Data;

public interface IVectorDb
{
    Task AddAsync(string id, float[] vector, string text);
    Task<(string text, float score)[]> SearchAsync(float[] queryVector, int topK = 5);
    Task<(string text, float score)[]> SearchByKeywordsAsync(IEnumerable<string> keywords, int topK = 5);
    Task InitializeForWriteAsync(string connectionString, int vectorDimensions, string tableName = "embeddings", bool recreateTable = false);

    void InitializeForReadOnlyAsync(string connectionString, string tableName = "embeddings");
}
