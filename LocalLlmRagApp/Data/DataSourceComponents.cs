using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.HuggingFace;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Options;
using Microsoft.ML.Tokenizers;
using Pgvector;
using Npgsql;

namespace LocalLlmRagApp.Data;

public class MarkdownFiles
{
    public IEnumerable<string> GetMarkdownFilePaths(string directory)
    {
        // �w��f�B���N�g���z����Markdown�t�@�C���p�X���
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
        // �V���v���ȃ`�����N�����imaxLength���Ƃɕ����j
        for (int i = 0; i < text.Length; i += maxLength)
        {
            yield return text.Substring(i, Math.Min(maxLength, text.Length - i));
        }
    }
}

public class Embedder
{
    private readonly InferenceSession _session;
    private readonly Tokenizer _tokenizer;

    public Embedder(IOptions<AppConfig> config)
    {
        // ONNX���f���t�@�C���p�X��ݒ�t�@�C������擾
        var modelPath = config.Value.EmbeddingOnnxModelPath;
        _session = new InferenceSession(modelPath);
        _tokenizer = new Tokenizer(); // Tokenizer�͌�q�̊ȈՎ����܂��͊O�����C�u�������p
    }

    public float[] Embed(string text)
    {
        // Tokenize
        var inputIds = _tokenizer.Encode(text);
        var inputTensor = new DenseTensor<long>(new[] { 1, inputIds.Length });
        for (int i = 0; i < inputIds.Length; i++) inputTensor[0, i] = inputIds[i];
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_ids", inputTensor) };
        // ���_
        using var results = _session.Run(inputs);
        var embedding = results.First().AsEnumerable<float>().ToArray();
        return embedding;
    }
}

// --- Tokenizer��E5�p�����iMicrosoft.ML.Tokenizers���p�j ---
public class Tokenizer
{
    private readonly Microsoft.ML.Tokenizers.Tokenizer _tokenizer;

    public Tokenizer(string tokenizerJsonPath = "./models/tokenizer.json")
    {
        // E5�ptokenizer.json�����[�J�����烍�[�h
        using var stream = File.OpenRead(tokenizerJsonPath);
        _tokenizer = Microsoft.ML.Tokenizers.Tokenizer.FromJson(stream);
    }

    public long[] Encode(string text)
    {
        var encoding = _tokenizer.Encode(text);
        return encoding.Ids.Select(id => (long)id).ToArray();
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
        // �R�T�C���ގ��x�Ō����i�_�~�[�����j
        return _data.Select(d => (d.text, score: 1.0f)).Take(topK).ToArray();
    }
}

public class PgvectorDb : IVectorDb
{
    private readonly string _connectionString;
    private readonly string _tableName;
    private readonly int _vectorDimensions;

    public PgvectorDb(string connectionString, string tableName = "embeddings", int vectorDimensions = 768)
    {
        _connectionString = connectionString;
        _tableName = tableName;
        _vectorDimensions = vectorDimensions;
        EnsureTable();
    }

    private void EnsureTable()
    {
        using var conn = new NpgsqlConnection(_connectionString);
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = $@"
            CREATE TABLE IF NOT EXISTS {_tableName} (
                id TEXT PRIMARY KEY,
                embedding VECTOR({_vectorDimensions}),
                text TEXT
            );";
        cmd.ExecuteNonQuery();
    }

    public void Add(string id, float[] vector, string text)
    {
        using var conn = new NpgsqlConnection(_connectionString);
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = $"INSERT INTO {_tableName} (id, embedding, text) VALUES (@id, @embedding, @text) ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, text = EXCLUDED.text;";
        cmd.Parameters.AddWithValue("@id", id);
        cmd.Parameters.AddWithValue("@embedding", new Vector(vector));
        cmd.Parameters.AddWithValue("@text", text);
        cmd.ExecuteNonQuery();
    }

    public (string text, float score)[] Search(float[] queryVector, int topK = 5)
    {
        using var conn = new NpgsqlConnection(_connectionString);
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = $@"
            SELECT text, (embedding <#> @query) AS score
            FROM {_tableName}
            ORDER BY score ASC
            LIMIT @topK;";
        cmd.Parameters.AddWithValue("@query", new Vector(queryVector));
        cmd.Parameters.AddWithValue("@topK", topK);
        using var reader = cmd.ExecuteReader();
        var results = new List<(string text, float score)>();
        while (reader.Read())
        {
            var text = reader.GetString(0);
            var score = reader.GetFloat(1);
            results.Add((text, score));
        }
        return results.ToArray();
    }
}
