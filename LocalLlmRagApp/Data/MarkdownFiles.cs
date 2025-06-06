using System.Collections.Generic;
using System.IO;

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
