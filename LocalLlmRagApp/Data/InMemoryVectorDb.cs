using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace LocalLlmRagApp.Data;

public class InMemoryVectorDb : IVectorDb
{
    private readonly List<(string id, float[] vector, string text)> _data = new();

    public Task AddAsync(string id, float[] vector, string text)
    {
        _data.Add((id, vector, text));
        return Task.CompletedTask;
    }

    public Task<(string text, float score)[]> SearchAsync(float[] queryVector, int topK = 5)
    {
        // コサイン類似度で検索（ダミー実装）
        var result = _data.Select(d => (d.text, score: 1.0f)).Take(topK).ToArray();
        return Task.FromResult(result);
    }
}
