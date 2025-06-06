namespace LocalLlmRagApp.Data;

public interface IVectorDb
{
    void Add(string id, float[] vector, string text);
    (string text, float score)[] Search(float[] queryVector, int topK = 5);
}
