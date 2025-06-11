global using Microsoft.Extensions.Configuration;
global using Microsoft.Extensions.DependencyInjection;
global using Microsoft.Extensions.Hosting;
global using Microsoft.Extensions.Logging;
global using Microsoft.Extensions.Options;
global using System;
global using System.Collections.Generic;
global using System.Linq;
global using System.Text;
global using System.Threading.Tasks;
global using System.Diagnostics;
using LocalLlmRagApp;
using LocalLlmRagApp.Data;
using LocalLlmRagApp.Llm;

_ = Host.CreateDefaultBuilder(args)
    .ConfigureLogging((context, logging) =>
    {
        logging.ClearProviders();
        logging.AddConfiguration(context.Configuration.GetSection("Logging"));
        logging.AddConsole();
        DebugLoggerAdd(logging);
    })
    .ConfigureAppConfiguration((context, config) =>
    {
        config.AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
            .AddUserSecrets<Program>(optional: true, reloadOnChange: true);
    })
    .ConfigureServices(ConfigureServices)
    .RunConsoleAsync();


void ConfigureServices(HostBuilderContext context, IServiceCollection services)
{
    services.AddHostedService<ConsoleHostedService>();
    services.Configure<AppConfig>(context.Configuration.GetSection(nameof(AppConfig)));
    // RAG・LLM関連サービスのDI登録
    services.AddSingleton<MarkdownFiles>();
    services.AddSingleton<Chunker>();
    services.AddSingleton<Embedder>();
    services.AddSingleton<IVectorDb, PgvectorDb>();
    services.AddSingleton<ILlmService, OnnxLlmService>();
    // IConfigurationをDIに追加（HostBuilderで自動登録されるが明示的に記載）
    services.AddSingleton<IConfiguration>(sp => context.Configuration);
}

[Conditional("DEBUG")]
static void DebugLoggerAdd(ILoggingBuilder loggingBuilder)
{
    loggingBuilder.AddDebug();
}
