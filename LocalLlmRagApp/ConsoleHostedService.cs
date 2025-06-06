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
            await Task.CompletedTask;
        }
    }
}

