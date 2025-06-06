# localapp_llm_rag_documents �d�l��

## �T�v

�{�\�t�g�E�F�A�́AWindows�N���C�A���gOS��œ��삷�郍�[�J�������^��RAG�iRetrieval-Augmented Generation�j�V�X�e���ł��B  
���C# (.NET 9) ��p���ĊJ�����ASphinx�v���W�F�N�g��Markdown�h�L�������g�Q��RAG�p�f�[�^�\�[�X�Ƃ��ăx�N�g��DB�����A  
���[�J��LLM�ƘA�g�����`���b�g�^CUI��񋟂��܂��B

---

## �@�\�v��

### 1. RAG�f�[�^�\�[�X����

- Sphinx�v���W�F�N�g�z���̕���Markdown�t�@�C������͂Ƃ��Ď󂯕t����
- Markdown�t�@�C����K�؂Ƀ`�����N�����i�Z�N�V�����E�i���E�g�[�N�������ŕ����j
- �e�`�����N�����[�J���œ���\�Ȍy�ʃ��f���i��: MiniLM, BGE���j�{Sewmantic Kernel�Ńx�N�g����
- �x�N�g��DB�i���[�J���t�@�C��DB�A��: SQLite, Faiss, Qdrant���j�֕ۑ�
- �x�N�g��DB��Windows�N���C�A���gOS�̃��[�J���X�g���[�W��ɔz�u

### 2. LLM�`���b�gCUI

- �R���\�[���A�v���iCUI�j�Ƃ��ē���
- ���[�U�[����̓��͂��󂯕t���ALLM�ƃe�L�X�g�`���b�g
- ���͕��ɑ΂��A�x�N�g��DB����ގ��`�����N���������ALLM�ւ̃v�����v�g�ɕt���iRAG����j
- LLM�̓��[�J���œ���\�Ȍy�ʃ��f�����g�p�i��: llama.cpp, GGML, ONNX Runtime���j
- .NET GenericHost�ɂ��A�v���P�[�V�������C�t�T�C�N���Ǘ�

---

## ��@�\�v��

- ���ׂĂ̏����̓��[�J��PC��Ŋ����i�O��API��N���E�h�T�[�r�X�͗��p���Ȃ��j
- ���C#�Ŏ����B��������g���ꍇ��C++��D��
- Windows 10/11 �N���C�A���gOS�œ���
- �R�}���h���C��������ݒ�t�@�C���iappsettings.json, UserSecrets���j�ɂ��_��Ȑݒ�

---

## ��ȏ����V�[�P���X

### 1. RAG�f�[�^�\�[�X����

```plantuml
@startuml

actor User 
User -> ConsoleApp : �f�[�^�\�[�X�����R�}���h(Sphinx�v���W�F�N�g�̃t�H���_�p�X)
ConsoleApp -> DataSourceBuilder : �f�[�^�\�[�X�����R�}���h 
DataSourceBuilder -> MarkdownFiles : Markdown�Ǎ� 
DataSourceBuilder -> Chunker : �`�����N���� 
Chunker -> Embedder : �x�N�g���� 
Embedder -> VectorDB : �x�N�g���ۑ� 
@enduml
```

### 2. LLM�`���b�gCUI

```plantuml
@startuml

actor User 
User -> ConsoleApp : �`���b�g���� 
ConsoleApp -> VectorDB : �ގ��`�����N���� 
ConsoleApp -> LLM : �v�����v�g�{�������ʂŐ��_ 
LLM -> ConsoleApp : ���� 
ConsoleApp -> User : �����\��
@enduml
```



## ��ȋZ�p�X�^�b�N

- .NET 9 / C#
- Microsoft.Extensions.Hosting (GenericHost)
- Sewmantic Kernel
- ���[�J��LLM�i��: llama.cpp, GGML, ONNX Runtime���j
- �x�N�g��DB�i��: SQLite, Faiss, Qdrant���j
- PlantUML�i�h�L�������g�p�j

---

## �\�[�X�R�[�h�f�B���N�g���\��

/LocalLlmRagApp
 |-- Program.cs
 |-- ConsoleHostedService.cs
 |-- AppConfig.cs
 |-- appsettings.json
 |-- /Data
 |-- /Models
 |-- /VectorDb
 |-- README.md

---

## ����̊g����

- GUI�Ή�
- ����LLM���f���ؑ�
- �x�N�g��DB�̎�ޒǉ�
- �`�����N�����E�x�N�g�����p�����[�^�̃J�X�^�}�C�Y

---
