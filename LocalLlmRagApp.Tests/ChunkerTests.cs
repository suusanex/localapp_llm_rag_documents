using NUnit.Framework;
using LocalLlmRagApp.Data;
using Microsoft.Extensions.Options;
using Moq;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Threading;

namespace LocalLlmRagApp.Tests;

public class ChunkerTests
{
    private class DummyEmbedder : IEmbedder
    {
        public int MaxTokenLength => 10;
        public int GetTokenCount(string text) => text.Length;
    }

    private class DummyLlmService : LocalLlmRagApp.Llm.ILlmService
    {
        public Task<string> ChatAsync(string prompt, CancellationToken cancellationToken = default) => Task.FromResult("");
        public Task<string> ChatAsyncDirect(string format, (string searchOption, double value)[] searchOptions, CancellationToken cancellationToken) => Task.FromResult(format.Replace("<|system|>", "").Replace("<|user|>", "").Replace("<|assistant|>", "").Replace("<|end|>", ""));
        public int GetTokenCount(string text) => text.Length;
        public int GetContextLength() => 1000;
    }

    [Test]
    public void Chunk_SplitsTextCorrectly()
    {
        var chunker = new Chunker(new DummyEmbedder(), new DummyLlmService());
        var text = new string('a', 25);
        var chunks = chunker.Chunk(text).ToList();
        Assert.That(chunks.Count, Is.GreaterThan(0));
        Assert.That(chunks.Select(x => x.Length).Sum(), Is.EqualTo(25));
    }
}
