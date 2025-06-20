using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using LocalLlmRagApp.Llm;
using System.Text.RegularExpressions;

namespace LocalLlmRagApp.Data;

public class Chunker
{
    private readonly IEmbedder _embedder;
    private readonly ILlmService _llmService;
    private readonly int _embeddingTokenLimit;

    public Chunker(IEmbedder embedder, ILlmService llmService)
    {
        _embedder = embedder;
        _llmService = llmService;
        _embeddingTokenLimit = (int)(_embedder.MaxTokenLength - 10);
    }

    public IEnumerable<string> Chunk(string text)
    {
        var lines = text.Split(Environment.NewLine);
        string? lastH1 = null;
        string? lastH2 = null;
        string? lastH3 = null;
        StringBuilder? currentChunk = null;
        string? currentHeaderType = null;
        List<string> currentHeadings = new();
        foreach (var line in lines)
        {
            var trimmed = line.TrimStart();
            if (trimmed.StartsWith("# "))
            {
                if (currentChunk != null && IsValidChunk(currentChunk))
                    yield return SummarizeAndValidateChunk(currentChunk.ToString().Trim(), currentHeadings).GetAwaiter().GetResult();
                lastH1 = line;
                lastH2 = null;
                lastH3 = null;
                currentChunk = new StringBuilder();
                currentChunk.AppendLine(line);
                currentHeaderType = "h1";
                currentHeadings = new List<string> { lastH1 };
            }
            else if (trimmed.StartsWith("## "))
            {
                if (currentChunk != null && IsValidChunk(currentChunk))
                    yield return SummarizeAndValidateChunk(currentChunk.ToString().Trim(), currentHeadings).GetAwaiter().GetResult();
                lastH2 = line;
                lastH3 = null;
                currentChunk = new StringBuilder();
                if (lastH1 != null) currentChunk.AppendLine(lastH1);
                currentChunk.AppendLine(line);
                currentHeaderType = "h2";
                currentHeadings = new List<string>();
                if (lastH1 != null) currentHeadings.Add(lastH1);
                currentHeadings.Add(lastH2);
            }
            else if (trimmed.StartsWith("### "))
            {
                if (currentChunk != null && IsValidChunk(currentChunk))
                    yield return SummarizeAndValidateChunk(currentChunk.ToString().Trim(), currentHeadings).GetAwaiter().GetResult();
                lastH3 = line;
                currentChunk = new StringBuilder();
                if (lastH1 != null) currentChunk.AppendLine(lastH1);
                if (lastH2 != null) currentChunk.AppendLine(lastH2);
                currentChunk.AppendLine(line);
                currentHeaderType = "h3";
                currentHeadings = new List<string>();
                if (lastH1 != null) currentHeadings.Add(lastH1);
                if (lastH2 != null) currentHeadings.Add(lastH2);
                currentHeadings.Add(lastH3);
            }
            else
            {
                if (currentChunk != null)
                    currentChunk.AppendLine(line);
            }
        }
        if (currentChunk != null && IsValidChunk(currentChunk))
            yield return SummarizeAndValidateChunk(currentChunk.ToString().Trim(), currentHeadings).GetAwaiter().GetResult();
    }

    private bool IsValidChunk(StringBuilder chunk)
    {
        var lines = chunk.ToString().Split(Environment.NewLine);
        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (!string.IsNullOrEmpty(trimmed) && !trimmed.StartsWith("#"))
                return true;
        }
        return false;
    }

    // �`�����N�{����v�񂵁A���o����C#���ŕt��
    private async Task<string> SummarizeAndValidateChunk(string chunk, List<string> headings)
    {
        var chunkLines = chunk.Split(Environment.NewLine);
        int headingCount = 0;
        var headingBuilder = new StringBuilder();
        foreach (var h in headings)
        {
            headingBuilder.AppendLine(h);
            headingCount++;
        }
        // �{������
        var bodyBuilder = new StringBuilder();
        for (int i = headingCount; i < chunkLines.Length; i++)
        {
            bodyBuilder.AppendLine(chunkLines[i]);
        }
        var body = bodyBuilder.ToString().Trim();
        var headingText = headingBuilder.ToString();

        // headings�̃g�[�N������Embedder�Ōv�Z
        string headingsConcat = string.Join("\n", headings);
        int headingsTokenCount = _embedder.GetTokenCount(headingsConcat);
        // summaryChatTokenLimit�̌v�Z
        var summaryTokenLimit = _embeddingTokenLimit - headingsTokenCount;

        // LLM�ŗv��i���o���ێ��w���͍폜�j
        string summaryPrompt = $"<|system|>���Ȃ��̓e�L�X�g�̗v���K�؂ɍs���G�[�W�F���g�ł��B���[�U�[�v�����v�g�̓��e���A�����Ȃǂ̏d�v�P�����e�������Ȃ��悤�ɗv�񂷂�K�v������܂��B{summaryTokenLimit}�g�[�N���ȓ��ŗv��݂̂��o�͂��A�����ŏo�͂��I�����Ă��������B<|end|><|user|>{{BODY}}<|end|><|assistant|># �v��\n";

        // summaryPrompt�̃g�[�N������ILlmService�o�R�Ŏ擾
        int promptTokenCount = _llmService.GetTokenCount(summaryPrompt.Replace("{{BODY}}", ""));
        int summaryChatTokenLimit = summaryTokenLimit + promptTokenCount;
        if (summaryChatTokenLimit < 1) summaryChatTokenLimit = 1;

        // LLM���f���̍ő咷���擾
        int contextLength = _llmService.GetContextLength();
        // body����������max_length��contextLength�𒴂���ꍇ�͕���
        var bodyLines = body.Split('\n');
        var summaries = new List<string>();
        int start = 0;
        while (start < bodyLines.Length)
        {
            // 1�`�����N����body������
            var partBuilder = new StringBuilder();
            int partTokenCount = 0;
            int end = start;
            for (; end < bodyLines.Length; end++)
            {
                var testBuilder = new StringBuilder();
                for (int i = start; i <= end; i++)
                    testBuilder.AppendLine(bodyLines[i]);
                string testBody = testBuilder.ToString().Trim();
                string testPrompt = summaryPrompt.Replace("{{BODY}}", testBody);
                int testPromptTokenCount = _llmService.GetTokenCount(testPrompt);
                int testMaxLength = summaryTokenLimit + testPromptTokenCount;
                if (testMaxLength > contextLength)
                    break;
                partBuilder = testBuilder;
                partTokenCount = testPromptTokenCount;
            }
            if (partBuilder.Length == 0 && start < bodyLines.Length)
            {
                // 1�s�������Ȃ��ꍇ�͋����I��1�s��������
                partBuilder.AppendLine(bodyLines[start]);
                end = start + 1;
                partTokenCount = _llmService.GetTokenCount(summaryPrompt.Replace("{{BODY}}", bodyLines[start]));
            }
            string partBody = partBuilder.ToString().Trim();
            string partPrompt = summaryPrompt.Replace("{{BODY}}", partBody);
            int partMaxLength = summaryTokenLimit + partTokenCount;
            if (partMaxLength < 1) partMaxLength = 1;
            string partSummary = await _llmService.ChatAsyncDirect(partPrompt, [("max_length", partMaxLength)], CancellationToken.None);
            // Markdown���o��1���o�������炻��ȍ~�𖳎��i^#\s+�̂݁j
            var lines = partSummary.Split('\n');
            var filteredLines = new List<string>();
            var regexH1 = new Regex("^#\\s+");
            foreach (var line in lines)
            {
                if (regexH1.IsMatch(line))
                    break;
                filteredLines.Add(line);
            }
            string filteredSummary = string.Join("\n", filteredLines).Trim();
            summaries.Add(filteredSummary);
            start = end + 1;
        }
        string resultChunk = headingText + string.Join("\n", summaries).Trim();
        // �g�[�N�����`�F�b�N
        if (_embedder.GetTokenCount(resultChunk) > _embedder.MaxTokenLength)
        {
            throw new System.Exception($"�`�����N���ő�g�[�N����({_embedder.MaxTokenLength})�𒴂��܂����B���o��: {headingText.Trim()}\n�v���g�[�N����: {_embedder.GetTokenCount(resultChunk)}");
        }
        return resultChunk;
    }
}
