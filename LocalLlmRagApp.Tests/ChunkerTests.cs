using NUnit.Framework;
using LocalLlmRagApp.Data;
using Microsoft.Extensions.Options;
using Moq;

namespace LocalLlmRagApp.Tests;

public class ChunkerTests
{
    private class DummyEmbedder : IEmbedder
    {
        public int MaxTokenLength => 10;
        public int GetTokenCount(string text) => text.Length;
    }

    [Test]
    public void Chunk_SplitsTextCorrectly()
    {
        var chunker = new Chunker(new DummyEmbedder());
        var text = new string('a', 25);
        var chunks = chunker.Chunk(text).ToList();
        Assert.That(chunks.Count, Is.EqualTo(3));
        Assert.That(chunks[0].Length, Is.EqualTo(10));
        Assert.That(chunks[1].Length, Is.EqualTo(10));
        Assert.That(chunks[2].Length, Is.EqualTo(5));
    }
}
