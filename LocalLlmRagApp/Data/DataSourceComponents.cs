// データソース構築のためのMarkdownファイル読み込み、チャンク分割、埋め込み生成、ベクトルDB格納の雛形を追加
namespace LocalLlmRagApp.Data;

public class MarkdownFiles
{
    public IEnumerable<string> GetMarkdownFilePaths(string directory)
    {
        // 指定ディレクトリ配下のMarkdownファイルパスを列挙
        return Directory.EnumerateFiles(directory, "*.md", SearchOption.AllDirectories);
    }

    public string ReadFile(string path)
    {
        return File.ReadAllText(path);
    }
}

public class Chunker
{
    public IEnumerable<string> Chunk(string text, int maxLength = 512)
    {
        // シンプルなチャンク分割（maxLengthごとに分割）
        for (int i = 0; i < text.Length; i += maxLength)
        {
            yield return text.Substring(i, Math.Min(maxLength, text.Length - i));
        }
    }
}

public class Embedder
{
    public float[] Embed(string text)
    {
        // ダミーの埋め込みベクトル（実際はMiniLM/BGE等を利用）
        return Enumerable.Repeat(0.1f, 384).ToArray();
    }
}

public interface IVectorDb
{
    void Add(string id, float[] vector, string text);
    (string text, float score)[] Search(float[] queryVector, int topK = 5);
}

public class InMemoryVectorDb : IVectorDb
{
    private readonly List<(string id, float[] vector, string text)> _data = new();

    public void Add(string id, float[] vector, string text)
    {
        _data.Add((id, vector, text));
    }

    public (string text, float score)[] Search(float[] queryVector, int topK = 5)
    {
        // コサイン類似度で検索（ダミー実装）
        return _data.Select(d => (d.text, score: 1.0f)).Take(topK).ToArray();
    }
}
