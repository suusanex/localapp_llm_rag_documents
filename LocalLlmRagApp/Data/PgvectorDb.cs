using Pgvector;
using Npgsql;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace LocalLlmRagApp.Data;

public class PgvectorDb : IVectorDb
{
    private NpgsqlDataSource _dataSource;
    private string _tableName;
    private int _vectorDimensions;
    private bool _recreateTable;

    public void InitializeForReadOnlyAsync(string connectionString, string tableName = "embeddings")
    {
        var dataSourceBuilder = new NpgsqlDataSourceBuilder(connectionString);
        dataSourceBuilder.UseVector();
        _dataSource = dataSourceBuilder.Build();
        _tableName = tableName;

    }
    
    public async Task InitializeForWriteAsync(string connectionString, int vectorDimensions, string tableName = "embeddings", bool recreateTable = false)
    {
        var dataSourceBuilder = new NpgsqlDataSourceBuilder(connectionString);
        dataSourceBuilder.UseVector();
        _dataSource = dataSourceBuilder.Build();
        _tableName = tableName;
        _vectorDimensions = vectorDimensions;
        _recreateTable = recreateTable;
        await EnsureTableAsync();
    }

    private async Task EnsureTableAsync()
    {
        await using var conn = await _dataSource.OpenConnectionAsync();
        await using var cmd = conn.CreateCommand();
        if (_recreateTable)
        {
            cmd.CommandText = $"DROP TABLE IF EXISTS {_tableName};";
            await cmd.ExecuteNonQueryAsync();
        }
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
            SELECT text, (embedding <=> @query) AS score
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

    public async Task<(string text, float score)[]> SearchByKeywordsAsync(IEnumerable<string> keywords, int topK = 5)
    {
        if (keywords == null || !keywords.Any())
            return Array.Empty<(string text, float score)>();
    
        await using var conn = await _dataSource.OpenConnectionAsync();
        await using var cmd = conn.CreateCommand();
        // 動的にOR条件を生成
        var conditions = keywords.Select((k, i) => $"text ILIKE @kw{i}").ToArray();
        cmd.CommandText = $@"
            SELECT text, 0.0 AS score
            FROM {_tableName}
            WHERE {string.Join(" OR ", conditions)}
            LIMIT @topK;";
        int idx = 0;
        foreach (var kw in keywords)
        {
            cmd.Parameters.AddWithValue($"@kw{idx}", $"%{kw}%");
            idx++;
        }
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
