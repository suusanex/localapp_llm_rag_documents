using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LocalLlmRagApp;

public class AppConfig
{
    /// <summary>
    /// UserSecretから読み込む
    /// </summary>
    public string Secret1 { get; set; }

    /// <summary>
    /// 埋め込みモデル(ONNX)のパス（必須）
    /// </summary>
    public string EmbeddingOnnxModelPath { get; set; }

    /// <summary>
    /// LLM用ONNXモデルファイルのパス（必須）
    /// </summary>
    public string LlmOnnxModelPath { get; set; }

    /// <summary>
    /// トークナイザーモデルファイルのパス（必須）
    /// </summary>
    public string TokenizerModelPath { get; set; }

    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(EmbeddingOnnxModelPath))
            throw new InvalidOperationException("AppConfig.EmbeddingOnnxModelPath is required.");
        if (string.IsNullOrWhiteSpace(LlmOnnxModelPath))
            throw new InvalidOperationException("AppConfig.LlmOnnxModelPath is required.");
        if (string.IsNullOrWhiteSpace(TokenizerModelPath))
            throw new InvalidOperationException("AppConfig.TokenizerModelPath is required.");
    }
}
