using LocalLlmRagApp.Data;
using LocalLlmRagApp.Llm;

namespace LocalLlmRagApp;

public class ConsoleHostedService(ILogger<ConsoleHostedService> _Logger, IHostApplicationLifetime _AppLifetime,
    IOptions<AppConfig> _Options, ILlmService _llmService, IConfiguration _config) : IHostedService
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
                var chunker = new Chunker();
                var embedder = new Embedder(_Options);
                embedder.Initialize();
                // ConnectionStringをUserSecretsから取得
                var connectionString = _config.GetSection("ConnectionStrings")["DefaultConnection"];
                var vectorDb = new PgvectorDb(connectionString);
                foreach (var file in markdown.GetMarkdownFilePaths(folder))
                {
                    var text = markdown.ReadFile(file);
                    foreach (var chunk in chunker.Chunk(text))
                    {
                        var vec = embedder.Embed(chunk);
                        vectorDb.Add(Guid.NewGuid().ToString(), vec, chunk);
                    }
                }
                Console.WriteLine($"RAGデータソース構築完了: {folder}");
            }
            else
            {
                // チャットモード
                if (_llmService is Phi3MiniOnnxLlmService phi3)
                {
                    phi3.Initialize();
                }
                else if (_llmService is OnnxLlmService onnx)
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

