// �f�[�^�\�[�X�\�z�̂��߂�Markdown�t�@�C���ǂݍ��݁A�`�����N�����A���ߍ��ݐ����A�x�N�g��DB�i�[�̐��`��ǉ�
namespace LocalLlmRagApp.Data;

public class MarkdownFiles
{
    public IEnumerable<string> GetMarkdownFilePaths(string directory)
    {
        // �w��f�B���N�g���z����Markdown�t�@�C���p�X���
        return Directory.EnumerateFiles(directory, "*.md", SearchOption.AllDirectories);
    }

    public string ReadFile(string path)
    {
        return File.ReadAllText(path);
    }
}

public class Chunker
{
    public IEnumerable<string> Chunk(string text, int maxLength = 512)
    {
        // �V���v���ȃ`�����N�����imaxLength���Ƃɕ����j
        for (int i = 0; i < text.Length; i += maxLength)
        {
            yield return text.Substring(i, Math.Min(maxLength, text.Length - i));
        }
    }
}

public class Embedder
{
    public float[] Embed(string text)
    {
        // �_�~�[�̖��ߍ��݃x�N�g���i���ۂ�MiniLM/BGE���𗘗p�j
        return Enumerable.Repeat(0.1f, 384).ToArray();
    }
}

public interface IVectorDb
{
    void Add(string id, float[] vector, string text);
    (string text, float score)[] Search(float[] queryVector, int topK = 5);
}

public class InMemoryVectorDb : IVectorDb
{
    private readonly List<(string id, float[] vector, string text)> _data = new();

    public void Add(string id, float[] vector, string text)
    {
        _data.Add((id, vector, text));
    }

    public (string text, float score)[] Search(float[] queryVector, int topK = 5)
    {
        // �R�T�C���ގ��x�Ō����i�_�~�[�����j
        return _data.Select(d => (d.text, score: 1.0f)).Take(topK).ToArray();
    }
}
