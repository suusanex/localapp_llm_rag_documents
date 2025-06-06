using System.Collections.Generic;

namespace LocalLlmRagApp.Data;

public class InMemoryVectorDb : IVectorDb
{
    private readonly List<(string id, float[] vector, string text)> _data = new();

    public void Add(string id, float[] vector, string text)
    {
        _data.Add((id, vector, text));
    }

    public (string text, float score)[] Search(float[] queryVector, int topK = 5)
    {
        // �R�T�C���ގ��x�Ō����i�_�~�[�����j
        return _data.Select(d => (d.text, score: 1.0f)).Take(topK).ToArray();
    }
}
