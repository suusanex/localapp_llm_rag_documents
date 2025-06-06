using LocalLlmRagApp.Data;
using LocalLlmRagApp.Llm;

namespace LocalLlmRagApp;

public class ConsoleHostedService(ILogger<ConsoleHostedService> _Logger, IHostApplicationLifetime _AppLifetime,
    IOptions<AppConfig> _Options) : IHostedService
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
            _Logger.LogInformation($"Start, {string.Join(" ", args)}, {_Options.Value.Other1}");
            // RAGデータソース構築例
            var markdown = new MarkdownFiles();
            var chunker = new Chunker();
            var embedder = new Embedder(_Options);
            var vectorDb = new InMemoryVectorDb();
            foreach (var file in markdown.GetMarkdownFilePaths("./Data"))
            {
                var text = markdown.ReadFile(file);
                foreach (var chunk in chunker.Chunk(text))
                {
                    var vec = embedder.Embed(chunk);
                    vectorDb.Add(Guid.NewGuid().ToString(), vec, chunk);
                }
            }
            // LLMチャットCUI例
            var llm = new DummyLlmService();
            Console.WriteLine("Input your prompt (or 'exit'):");
            string? input;
            while ((input = Console.ReadLine()) != "exit")
            {
                var response = await llm.ChatAsync(input ?? "");
                Console.WriteLine($"LLM: {response}");
            }
            await Task.CompletedTask;
        }
    }
}

