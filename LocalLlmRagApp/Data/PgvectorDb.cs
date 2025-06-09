using Pgvector;
using Npgsql;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace LocalLlmRagApp.Data;

public class PgvectorDb : IVectorDb
{
    private readonly NpgsqlDataSource _dataSource;
    private readonly string _tableName;
    private readonly int _vectorDimensions;

    public PgvectorDb(string connectionString, string tableName = "embeddings", int vectorDimensions = 1024)
    {
        var dataSourceBuilder = new NpgsqlDataSourceBuilder(connectionString);
        dataSourceBuilder.UseVector();
        _dataSource = dataSourceBuilder.Build();
        _tableName = tableName;
        _vectorDimensions = vectorDimensions;
    }

    public async Task InitializeAsync()
    {
        await EnsureTableAsync();
    }

    private async Task EnsureTableAsync()
    {
        await using var conn = await _dataSource.OpenConnectionAsync();
        await using var cmd = conn.CreateCommand();
        cmd.CommandText = $@"
            CREATE TABLE IF NOT EXISTS {_tableName} (
                id TEXT PRIMARY KEY,
                embedding VECTOR({_vectorDimensions}),
                text TEXT
            );";
        await cmd.ExecuteNonQueryAsync();
    }

    public async Task AddAsync(string id, float[] vector, string text)
    {
        await using var conn = await _dataSource.OpenConnectionAsync();
        await using var cmd = conn.CreateCommand();
        cmd.CommandText = $"INSERT INTO {_tableName} (id, embedding, text) VALUES (@id, @embedding, @text) ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, text = EXCLUDED.text;";

        cmd.Parameters.AddWithValue("@id", id);
        cmd.Parameters.AddWithValue("@embedding", new Vector(vector));
        cmd.Parameters.AddWithValue("@text", text);

        await cmd.ExecuteNonQueryAsync();
    }

    public async Task<(string text, float score)[]> SearchAsync(float[] queryVector, int topK = 5)
    {
        await using var conn = await _dataSource.OpenConnectionAsync();
        await using var cmd = conn.CreateCommand();
        cmd.CommandText = $@"
            SELECT text, (embedding <#> @query) AS score
            FROM {_tableName}
            ORDER BY score ASC
            LIMIT @topK;";
        cmd.Parameters.AddWithValue("@query", new Vector(queryVector));
        cmd.Parameters.AddWithValue("@topK", topK);
        await using var reader = await cmd.ExecuteReaderAsync();
        var results = new List<(string text, float score)>();
        while (await reader.ReadAsync())
        {
            var text = reader.GetString(0);
            var score = reader.GetFloat(1);
            results.Add((text, score));
        }
        return results.ToArray();
    }
}
