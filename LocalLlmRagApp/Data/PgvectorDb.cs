using Pgvector;
using Npgsql;
using System.Collections.Generic;

namespace LocalLlmRagApp.Data;

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
