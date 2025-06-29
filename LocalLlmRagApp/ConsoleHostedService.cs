using LocalLlmRagApp.Data;
using LocalLlmRagApp.Llm;

namespace LocalLlmRagApp;

public class ConsoleHostedService(ILogger<ConsoleHostedService> _Logger, IHostApplicationLifetime _AppLifetime,
    IOptions<AppConfig> _Options, ILlmService _llmService, IConfiguration _config,
    IVectorDb _VectorDb) : IHostedService
{
    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _AppLifetime.ApplicationStarted.Register(OnStarted);
        await Task.CompletedTask;
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        await Task.CompletedTask;
    }

    [STAThread]
    private void OnStarted()
    {
        try
        {
            StartedAsync().GetAwaiter().GetResult();
        }
        catch (Exception e)
        {
            Console.Error.WriteLine(e.Message);
            _Logger.LogError($"Unhandled Exception, {e}");

            Environment.ExitCode = -1;
        }
        finally
        {
            _Logger.LogInformation($"Exit, Code={Environment.ExitCode}");
            _AppLifetime.StopApplication();
        }

        async Task StartedAsync()
        {
            var args = Environment.GetCommandLineArgs();
            _Logger.LogInformation($"Start, {string.Join(" ", args)}");

            if (args.Length >= 3 && string.Equals(args[1], "/CreateDataSource", StringComparison.OrdinalIgnoreCase))
            {
                var folder = args[2];
                var markdown = new MarkdownFiles();
                var embedder = new Embedder(_Options);
                embedder.Initialize();
                // LLMサービスの初期化（OnnxLlmServiceの場合のみ）
                if (_llmService is OnnxLlmService onnx)
                {
                    onnx.Initialize();
                }
                // DIからChunkerを取得する場合は、下記のように修正
                // var chunker = serviceProvider.GetRequiredService<Chunker>();
                // ただし、EmbedderやILlmServiceの状態管理に注意
                // ここでは明示的にEmbedder/ILlmServiceを渡す形で修正
                var chunker = new Chunker(embedder, _llmService);
                // ConnectionStringをUserSecretsから取得
                var connectionString = _config.GetSection("ConnectionStrings")["DefaultConnection"];
                var vectorDimensions = Embedder.GetDimensions(EmbeddingModelType.IntfloatMultilingualE5Base);
                await _VectorDb.InitializeForWriteAsync(connectionString, vectorDimensions, recreateTable: true); // テーブルを削除して作り直すモードをハードコーディング
                foreach (var file in markdown.GetMarkdownFilePaths(folder))
                {
                    var text = markdown.ReadFile(file);
                    foreach (var chunk in chunker.Chunk(text))
                    {
                        var vec = embedder.Embed(chunk);
                        await _VectorDb.AddAsync(Guid.NewGuid().ToString(), vec, chunk);
                    }
                }
                Console.WriteLine($"RAGデータソース構築完了: {folder}");
            }
            else
            {
                // チャットモード
                if (_llmService is OnnxLlmService onnx)
                {
                    onnx.Initialize();
                }
                else if (_llmService is Embedder embedder)
                {
                    embedder.Initialize();
                }
                Console.WriteLine("Input your prompt (or 'exit'):");
                string? input;
                while ((input = Console.ReadLine()) != "exit")
                {
                    var response = await _llmService.ChatAsync(input ?? "");
                    Console.WriteLine($"LLM: {response}");
                }
            }
            await Task.CompletedTask;
        }
    }
}

