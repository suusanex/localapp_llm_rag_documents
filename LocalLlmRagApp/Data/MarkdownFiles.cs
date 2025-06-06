using System.Collections.Generic;
using System.IO;

namespace LocalLlmRagApp.Data;

public class MarkdownFiles
{
    public IEnumerable<string> GetMarkdownFilePaths(string directory)
    {
        // 指定ディレクトリ配下のMarkdownファイルパスを列挙
        return Directory.EnumerateFiles(directory, "*.md", SearchOption.AllDirectories);
    }

    public string ReadFile(string path)
    {
        return File.ReadAllText(path);
    }
}
