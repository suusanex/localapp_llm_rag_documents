using NUnit.Framework;
using LocalLlmRagApp.Data;

namespace LocalLlmRagApp.Tests;

public class ChunkerTests
{
    [Test]
    public void Chunk_SplitsTextCorrectly()
    {
        var chunker = new Chunker();
        var text = new string('a', 1200);
        var chunks = chunker.Chunk(text, 512).ToList();
        Assert.That(chunks.Count, Is.EqualTo(3));
        Assert.That(chunks[0].Length, Is.EqualTo(512));
        Assert.That(chunks[1].Length, Is.EqualTo(512));
        Assert.That(chunks[2].Length, Is.EqualTo(176));
    }
}
