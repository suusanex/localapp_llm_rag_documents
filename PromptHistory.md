# プロンプト履歴

## ReadMe生成時のプロンプト

```
主にC#を使用して、目的を実現するソフトを作ります。まずはReadMe.mdへ、作成するソフトの仕様をMarkdownを使用して書いてください。図が必要な場合は、PlantUMLで書いてください。

# 目的

まず、ソフトはRAGのデータソースを作成する機能を持ちます。複数のMarkdownファイルを含むSphinxプロジェクトがあります。これをRAGのデータソースとして使用できるように、適切にチャンク分割とベクトル化を行って、ベクトルDBへ保存します。ベクトルDBはWindowsOS（クライアントOS）のローカルに保存する必要があります。ソフトはローカルで完結する必要があり、チャンク分割とベクトル化にはSewmantic Kernelとローカルで動作可能な軽量モデルを使用します。

次に、ソフトはコンソールアプリとしてCUIを持ち、テキストチャットでLLMと会話することを可能とします。LLMとの会話をする際に、すでに保存したベクトルDBの内容を類似検索し、それをLLMへのコンテキスト情報へ付加します。つまり、いわゆるRAGとして使用します。このLLMとの会話も、ローカルで動作可能な軽量モデルを使用します。

コンソールアプリは、GenericHostを使用して作成します。C#は、.NET 9を使用します。

このようなソフトウェアをWindows上で動作するように作成しますが、開発者のスキルはC#がメインのため、出来るだけPythonなどの他の言語を使わずに、主にC#で作成します。C#以外の言語を使う場合、まずC++を優先して使用します。
```

## ソースコード生成時のプロンプト

1. ReadMe.mdに書かれた仕様書に従って、プロジェクトやソースコードを作成してください。必要なライブラリがある場合、NuGetで取得してください。テストコードを書くことが効果的だと判断できる部分があれば、NUnitのテストプロジェクトを作成してソースコードを書いてください。
1. 埋め込み生成の処理を書いてください。ベクトルDBとしてはSharpVectorを使用し、埋め込み生成の処理はSemanticKernelを通してintfloat/multilingual-e5-baseモデルを使用するようにしてください。
1. intfloat/multilingual-e5-baseは、ONNX形式でダウンロードしたものを使用します。ローカルからファイルを読み込むようにしてください。
1. ベクトルデータベースのSharpVectorとは、SharpVectorsではなく、Build5Nines.SharpVectorを使用します。これを使用して、ベクトルDBの読み書きも実装してください。
1. E5用トークナイザを、Microsoft.ML.Tokenizersを使用して実装してください。
1. ベクトルデータベースを、pgvector-dotnetに変更します。これを使用して、ベクトルDBの読み書き処理を書き直してください。
1. ILlmServiceの実装を、ONNX形式でダウンロードした軽量モデルを使用してチャットを行う形で実装してください。
1. コンソールアプリの実行時引数に、「/CreateDataSource <フォルダパス>」のように指定した場合にだけ、RAGデータソース構築を行うようにしてください。<フォルダパス>は、Markdownファイルが含まれているフォルダのパスです。引数が無い場合は、チャットモードで動作します。以上の内容を、ReadMe.mdの仕様書に追記し、ソースコードも実装してください。
1. チャットを行うLLMとの会話部分について、モデルはPhi-3-mini-4k-instructを使用し、Microsoft.ML.OnnxRuntimeGenAIを使って実装してください。この処理の中に、仕様書にある次の処理も実装してください。「入力文に対し、ベクトルDBから類似チャンクを検索し、LLMへのプロンプトに付加（RAG動作）」
1. LlmSessionというクラスが存在せず、コンパイルエラーとなっています。その型が使われている_llmSession.GenerateAsync()の部分には、トークナイザーの使用やGeneratorの設定や推論処理そのものが欠けているように見えるため、おそらく_llmSession.GenerateAsync()の部分を追加実装する必要があります。そうした作業を行ってください。
1. DataSourceComponents.csは、あまり相互に依存の無い多数のクラスを全て持ってしまっているのに、フォルダ内にはそのファイルしか入っていません。ソリューションの見通しが悪いため、クラスごとに1つのファイルとし、ファイル名はクラス名と同じものに変えてください。
1. コンストラクタ内では例外を投げる可能性がある処理を行わず、そうした処理は別途明示的な初期化メソッドなどで実行するようにしてください。DIを使用していることで、コンストラクタで例外が発生し得ると、シーケンス上の処理順序が保証されていないことなどの理由でデバッグがやりづらくなるためです。
1. OnnxLlmServiceクラスが、コンストラクタ内で例外を投げる可能性があるようです。InferenceSessionクラスのインスタンスを作成しており、これは外部ライブラリのクラスであるため、例外を投げる必要があります。これも同様の方針で修正してください。
1. AppConfigからOther1を削除してください。
    TokenizerクラスのmodelPathのデフォルト引数など、パスをハードコーディングしているものは、AppConfigへ移してください。
    AppConfigクラスに定義されているプロパティは、appsettings.jsonにもサンプル値を書いてください。
    AppConfigでパスを指定するパラメータは必須とし、nullの場合は例外を投げるようにしてください。現状、全てのパラメータがパスを指定するものに該当します。
1. appsettings.jsonが途中で切れていて、未完成のように見えます。完成させてください。
1. Markdownをチャンク分割した後のEmbedder.Embed呼び出しで、下記の例外が出ます。渡しているデータの形式かパラメータが間違っているように見えますが、修正できますか？

    Microsoft.ML.OnnxRuntime.OnnxRuntimeException
    HResult=0x80131500
    Message=[ErrorCode:InvalidArgument] Non-zero status code returned while running Expand node. Name:'/Expand' Status Message: invalid expand shape
    Source=Microsoft.ML.OnnxRuntime
    スタック トレース:
    場所 Microsoft.ML.OnnxRuntime.NativeApiStatus.VerifySuccess(IntPtr nativeStatus)
1. PgvectorDbクラスのコンストラクタへ渡すConnectionStringについて、UserSecretsのConnectionStrings.DefaultConnectionを読み込んで渡すように書き換えてください。
1. コンストラクタ内では例外を投げる可能性がある処理を行わず、そうした処理は別途明示的な初期化メソッドなどで実行するようにしてください。DIを使用していることで、コンストラクタで例外が発生し得ると、シーケンス上の処理順序が保証されていないことなどの理由でデバッグがやりづらくなるためです。 現状では、PgvectorDbクラスがこれに違反しているようです。
1. 同期メソッドNpgsqlConnection.Open()には既知の問題があるようです。非同期メソッドをasync/awaitを使用して呼び出す形へ変更してください。
1. AddAsyncメソッドで、 cmd.ExecuteNonQueryAsync() の部分で下記の例外が出ます。問題点を修正してください。

    System.InvalidCastException: 'Writing values of 'Pgvector.Vector' is not supported for parameters having DataTypeName 'public.vector'.'
1. 次のエラーが表示されています。新形式に直してください。
    'NpgsqlConnection.TypeMapper' は旧形式です ('Connection-level type mapping is no longer supported. See the 7.0 release notes for configuring type mapping on NpgsqlDataSource.')
1. PgVectorDb.AddAsyncメソッドで下記の例外が出ます。テーブルの定義はTEXTとVECTORのみなので、理由が分かりません。この問題を修正してください。 
  System.OverflowException
  HResult=0x80131516
  Message=Value was either too large or too small for a UInt16.
  Source=System.Private.CoreLib
  スタック トレース:
   場所 System.Convert.ThrowUInt16OverflowException()
1. 修正後も、次のエラーが発生します。まだ問題があるようです。修正してください。
    System.OverflowException HResult=0x80131516 Message=Value was either too large or too small for a UInt16. Source=System.Private.CoreLib スタック トレース: 場所 System.Convert.ThrowUInt16OverflowException()
1. PgvectorDb.AddAsyncメソッドに渡されているのがfloat[98304]となっており、つまりvectorの次元が98304になっています。使用しているEmbeddingのモデルintfloat/multilingual-e5-largeは1024次元のはずなので、Embedderクラスの実装がおかしいように見えます。修正してください。また、それを使用するためにはPgvectorDbの作成するテーブルは1024次元にする必要があります。
1. 引き続き、PgvectorDb.AddAsyncメソッドに渡されているのがfloat[98304]となっています。これが二次元テンソルなのであれば、最終的な出力としてfloat[1024]を得るために、平均などの処理が必要と考えられます。その処理が抜けているのであれば、追加してください。
1. PgvectorDbの次元数は、Embedderが使用しているモデルによって異なるようです。現状では、intfloat-multilingual-e5モデルを使用しており、Baseモデルでは768、Largeモデルでは1024となります。このため、Embedderクラス内に使用するモデルの定義を持ってそれに対応する次元数を公開し、PgvectorDbはEmbedeerから次元数を取得するように実装してください。また、Embedderクラス内でのモデルと次元数の定義は、その対応関係が明確になるように、enumなどを使用してモデル名と次元数の対応付けがソースコードから読み取れるようにしてください。 現状で使用する値は、intfloat-multilingual-e5モデルのBaseであり、768です。
1. PgvectorDbがEmbedderクラスそのものを使用するのは、依存が強すぎるため避けたいです。2つのクラスの呼び出し元がこの2つを適切に操作することでPgvectorDbクラスへ次元数を与えるか、もしくはEmbedderクラスが次元数だけを取得するI/Fを公開してPgvectorDbはそれを使用する、などというように依存を弱めてください。
1. PgVectorDbで作成するテーブルについて、実行時にテーブルを削除して作り直すモードと、既存のテーブルに追加するモードを、選択できるようにしてください。ただしこれは今後の拡張のために実装しておくだけです。現時点では、呼び出し元でハードコーディングで選択することで、削除して作り直すモードだけを呼び出します。
1. LlmService.ChatAsyncメソッドで、次の例外が発生します。修正してください。 
    Microsoft.ML.OnnxRuntime.OnnxRuntimeException HResult=0x80131500 Message=[ErrorCode:RuntimeException] Non-zero status code returned while running Shape node. Name:'/model/attn_mask_reformat/attn_mask_subgraph/Shape' Status Message: D:\a_work\1\s\include\onnxruntime\core/framework/op_kernel_context.h:42 onnxruntimeOpKernelContextInput Missing Input: attention_mask
    Source=Microsoft.ML.OnnxRuntime スタック トレース: 場所 Microsoft.ML.OnnxRuntime.NativeApiStatus.VerifySuccess(IntPtr nativeStatus)
1. 同じ場所で次の例外が発生します。修正してください。
    Microsoft.ML.OnnxRuntime.OnnxRuntimeException
    HResult=0x80131500
    Message=[ErrorCode:RuntimeException] Non-zero status code returned while running GroupQueryAttention node. Name:'/model/layers.0/attn/GroupQueryAttention' Status Message: D:\a\_work\1\s\include\onnxruntime\core/framework/op_kernel_context.h:42 onnxruntime::OpKernelContext::Input Missing Input: past_key_values.0.key

    Source=Microsoft.ML.OnnxRuntime
    スタック トレース:
    場所 Microsoft.ML.OnnxRuntime.NativeApiStatus.VerifySuccess(IntPtr nativeStatus)
1. ChatAsyncメソッドの戻り値に、UTF-8では読み取れない下記のような文字列が返ってきます。おそらくレスポンスの解釈方法に誤りがあると思うのですが、考えられる修正を行ってください。（エラー文字列は省略）
1. OnnxLlmService.ChatAsync()メソッドが正常に動作しません。このメソッドは、Phi-3-mini-4k-instruct-onnxのONNXファイルを使用してチャットを行う処理であり、かつチャットを送信する前にRAGの処理を行う（関連性の高い情報をPgvectorDbクラスが扱っているDBから取得し、コンテキストとして送信時に付与する）機能を持つはずです。
    問題の原因を探るために、現状のコードはいったん捨てて、Microsoft LearnにサンプルのあるONNX Runtime Generative AIを使用したコードに書き換えてください。次のURLで説明されているものです。
    https://learn.microsoft.com/ja-jp/windows/ai/models/get-started-models-genai
    ソースコードとしては、次のGitHubリポジトリも参考になりそうです。
    https://github.com/microsoft/Phi3-Chat-WinUI3-Sample/
1. LlmService.ChatAsyncの_session!.Run()呼び出し部分で、下記の例外が出ました。修正してください。
    Microsoft.ML.OnnxRuntime.OnnxRuntimeException
    HResult=0x80131500
    Message=[ErrorCode:RuntimeException] Non-zero status code returned while running GroupQueryAttention node. Name:'/model/layers.0/attn/GroupQueryAttention' Status Message: D:\a\_work\1\s\include\onnxruntime\core/framework/op_kernel_context.h:42 onnxruntime::OpKernelContext::Input Missing Input: past_key_values.0.key

    Source=Microsoft.ML.OnnxRuntime
    スタック トレース:
    場所 Microsoft.ML.OnnxRuntime.NativeApiStatus.VerifySuccess(IntPtr nativeStatus)
1. LlmService.ChatAsyncで、output.AsTensor<long>()がnullを返します。output.ElementTypeがfloatになっているため、おそらくlongを前提としている処理をfloatを前提とした処理へ変更する必要があります。修正してください。
1. まず、プロジェクト内のTokenizerクラスは、本プロジェクトのEmbedding処理専用のTokenizerの処理です。混同しないように、それが理解できる名前に変更してください。
    次に、LlmServiceの処理に使用するべきTokenizerは、Microsoft.ML.OnnxRuntimeGenAI.Tokenizerです。それを前提にして、処理を書き直してください。
1. ChatAsync()の推論処理に、多数のビルドエラーや動作上の問題が有ります。
問題の原因を探るために、Microsoft LearnにサンプルのあるONNX Runtime Generative AIを使用したコードのように、Tokenizer.CreateStreamで作成したストリームを使用する方法に変更してください。次のURLで説明されているものです。
    https://learn.microsoft.com/ja-jp/windows/ai/models/get-started-models-genai
    ソースコードとしては、次のGitHubリポジトリも参考になりそうです。
    https://github.com/microsoft/Phi3-Chat-WinUI3-Sample/
1. Chunker.Chunk()について、引数のテキストをmarkdownとして解釈し、markdownファイルの見出しレベル3を1単位としてチャンク分割する処理を実装してください。この時、文字列の最大長（maxLength）の判定は不要です。
1. Markdownの見出しレベル3を最小単位として分割したいため、3以下の見出しを分割対象としてください。
1. Chunker.Chunk()について、引数のテキストをmarkdownとして解釈し、Markdownの見出しレベル3を1単位として分割する処理とします。ただし、見出しレベル3の内容を切り出す時に、その直前に出てきた見出しレベル1と2の内容を付加します。このような処理を書いて下さい。
    次に例を示します。

    入力:
    # mi1
    na1
    ## mi2
    na2
    ## mi3
    na3
    ### mi4
    na4
    ### mi5
    na5

    この場合、次のように2つに分割する必要があります。

    出力1:
    # mi1
    na1
    ## mi3
    na3
    ### mi4
    na4

    出力2:
    # mi1
    na1
    ## mi3
    na3
    ### mi5
    na5
1. Markdownの見出しレベル3だけではなく、2と1についても分割対象としてください。上位の見出しの内容を前に付加するという処理は同じで、見出しレベル3には1と2の内容を、見出しレベル2には1の内容を前に付加してください。ただし、この方針で分割した結果に、本文が含まれていない場合（見出しと改行だけになった場合）については、その結果は返さずに、次の分割へ進んでください。
1. Chunker.Chunk()の分割結果は最大800文字としてください。
1. LLMSerivice.ChatAsync()のsimilarChunksについて、取得した内容のテキスト部分の内容が質問文と関連性が高いかどうかをLLMへ判定させることで、より関連性の高いものを選択する機能を追加してください。まずベクトルDBからの検索は最大100件を取得し、これを10件ずつLLMへ渡して関連性の高いもの2件を抽出します。結果として2*10件の関連性の高い物が抽出されるので、これをコンテキストとして、改めて従来のプロンプト（入力された質問に、抽出したコンテキストを付加したもの）を送って最終的な回答を得る、という処理にします。LLMには関連性の高い物を抽出する機能そのものは無いため、抽出して特定のフォーマットで回答が返ってくるように入力のプロンプトを工夫し、返ってきた回答から結果を読み取る処理を作成することが必要になります。
1. GetLlmResultAsync()に対して以下のような応答が返ってきており、ParseSelectedChunksFromLlmResult()でカンマ区切りの数値を取得できていないようです。この応答のフォーマットのように想定と少し違う物が返ってきても対応できるようにパース処理を修正するか、もしくは応答のフォーマットをもっと明確に指定する必要がありそうです。改善してください。

    実際に返ってきた応答の例:
    ===
    [3, 9]

    The two texts that most closely relate to the question about the types of risks included in the risk log are:

    [3] XXXXXXXXXXX

    [9] XXXXXXXXXXX

    These two texts discuss aspects related to the risk log, with the first one mentioning shared functionalities and the second one about messages displayed in the user interface, which could be part of the risk log's content.
1. Chunker.Chunk()について、文字列長の上限値800はチューニング対象にするため、変更しやすいように1つのフィールドにまとめてください。
1. Chunker.Chunk()について、基本的にはMarkdownの見出し単位で分割するが、文字列長の上限を超えた場合はさらに複数に分割する、という仕様です。文字列長の上限を超えた分割を行った場合でも、どの見出しに所属する内容かをチャンク単独で読み取れるようにするために、そのような分割結果の2つ目以降を返す場合には、1つ目と同じ見出しを前に付加するように、変更してください（最大で、見出し1・2・3の全てが付加されることになります）。このため、分割する文字数については、見出しを付加するための文字数を含めて判定する必要があります。
1. ChunkerクラスでMaxChunkLengthでの分割を行っていますが、この条件を変更します。EmbedderTokenizer.Encode()でテキストをエンコードして返ってくる配列のサイズが、Embedder.MaxLength以下となるように分割する、という処理に変更してください。ただし、EmbedderTokenizer.Encode()を呼び出す際は、Embedderクラスで使用しているプレフィックス(passage:)を付加して計算する必要があります。これらの処理は共通化のため、ChunkerクラスからEmbedderクラスを、インターフェースを介して呼び出すようにすることで実現してください。
1. 動かしたところ、多数のChunkのうち1つだけが、Embedder.Embed()の時点で514となり、MaxLengthを超えてしまいました。差分は2とかなり小さいので、Chunkerクラスのロジックにこれを見逃すような些細なバグがある可能性が高いと思います。修正してください。

    発生した時のChunkは、以下のものです。:
    # XXXXXXXXXXX
    ## XXXXXXXXXXX
    ### XXXXXXXXXXX

    XXXXXXXXXXX

    #### XXXXXXXXXXX

    XXXXXXXXXXX

    #### XXXXXXXXXXX

    XXXXXXXXXXX

    XXXXXXXXXXX

    #### XXXXXXXXXXX

    XXXXXXXXXXX

    #### XXXXXXXXXXX

    XXXXXXXXXXX

    この動きの違いは「XXXXXXXXXXX」による分散だけを対象とし、強制分散については自動分散中の状態と同じ（分散を行う）。
1. 同じ結果になりました。まだ問題が有るようです、修正して下さい。問題が発生したチャンクを見ると、1つのMarkdownの先頭のデータです。特に先頭のデータについて、この問題が起きる可能性を疑ってください。
1. LLMService.ChatAsync()の「// 3. 18件をcontextとして従来のプロンプトで最終回答」の処理でのInferStreamingの呼び出し結果について、LLMからの応答が異常に短いうちにgenerator.IsDone()がtrueになって処理が終わってしまう場合があります。考えられる原因について、対策を実装してください。
1. LlmServiceクラスの、max_lengthの値と、入力プロンプトの作り方を、改善する必要があります。現在使用しているモデルはPhi-3-mini-4k-instruct-onnxであるため、最大トークン数（context_length）は4096です。この範囲内で、応答に十分なトークン数を確保できるように、プロンプトを調整する必要があります。応答には2048トークンを確保してください。ChatAsyncメソッドで作成するcontextは、formatを作成した結果が[context_length - 応答のトークン数]未満のトークン数となるようにしてください。このトークン数の算出は、InferStreamingメソッドと同じ方式を使って算出する必要があります。以上の処理で使う数値と、関連する数値（ベクトルDBからの最大取得数や、そこから関連性の高い物を抽出する数など）、動作を見てチューニングするため、フィールドに持つようにしてください。
1. ※GitHub Copilotが何度か修正してもビルドエラーが続き、結果がループして解決見込みが無かった。そこで、別途ChatGptで調べた結果を以下のように渡したら、解決した。

    「error CS1061: 'Sequences' に 'Count' の定義が含まれておらず」と言ったビルドエラーが続いています。
    次のような情報をもらいましたが、これで解決できますか？

    🔍 Sequences からトークン数を取得する方法
    Sequences クラスは、複数のトークンシーケンスを保持する構造です。各シーケンスは ReadOnlySpan<int> としてアクセスできます。以下のコードは、Sequences オブジェクトから各シーケンスのトークン数を取得する方法を示しています。

    ```
    using Microsoft.ML.OnnxRuntimeGenAI;

    // モデルとトークナイザーの初期化
    using var model = new Model(modelPath);
    using var tokenizer = new Tokenizer(model);

    // プロンプトのエンコード
    string prompt = "<|system|>あなたは親切で知識豊富なアシスタントです。<|end|><|user|>こんにちは！<|end|><|assistant|>";
    var sequences = tokenizer.Encode(prompt);

    // 各シーケンスのトークン数を取得
    for (ulong i = 0; i < sequences.NumSequences; i++)
    {
        ReadOnlySpan<int> sequence = sequences[i];
        int tokenCount = sequence.Length;
        Console.WriteLine($"シーケンス {i} のトークン数: {tokenCount}");
    }
    ```

    このコードでは、sequences.NumSequences を使用してシーケンスの総数を取得し、各シーケンスに対して sequences[i] でアクセスしています。ReadOnlySpan<int> 型の sequence から Length プロパティを使用してトークン数を取得できます。
1. LLMService.ChatAsync()で行っている_vectorDb.SearchAsync()での関連コンテキスト検索について、現状はベクトルDBの類似検索だけを使用しています。これを、半数はキーワード検索を使用し、もう半数は従来通りベクトルDBの類似検索を使用するように変更してください。キーワード検索は、複数のキーワードのいずれかが「text」フィールドに含まれているものを検索する方式とします。検索に使用する複数のキーワードは、ユーザーが入力した質問文から抽出します。質問文から、接続詞などを除いた単語だけを全て抽出し、それを使用してください。この抽出処理については、既存のBuildSelectionPrompt()などの処理と同様に、LLMを使用して行ってください。
1. キーワード検索について、DBから全件を取得してメモリ上で検索を行うロジックは、効率が悪いと感じます。PgvectorDbクラスも合わせて拡張して、キーワード検索をSQLクエリで行うことはできますか？
1. ExtractKeywordsAsync()において、LLMからの回答が、例のように非常に複雑で段階を踏んだものとなり、途中からおかしな推論へ発展しているようです。おそらく、プロンプトの指示が必要以上に難しく解釈されているのだと思います。本件では処理の一部として比較的可能性の高い単語を抽出させることが目的となるため、長時間かかる複雑な推論は不要です。プロンプトを工夫するか、あるいはこの用途でのInferStreaming()処理のパラメータを変更するなどして、改善してください。

    例:
    , XXXXXXXXXXX, XXXXXXXXXXX, XXXXXXXXXXX


    解答1:
    XXXXXXXXXXX, XXXXXXXXXXX, XXXXXXXXXXX


    指示2（より難易度が高い）:

    以下の質問文に基づいて、以下の複数の制約を満たすキーワードを抽出してください。キーワードは、名詞、固有名詞、尊敬語、尊敬語の形、形容詞、形容詞の語幹、動詞の語幹、副詞、接続詞、助詞、数詞、副詞の語幹、副詞の形、副詞の接続詞を含む必要があります。また、キーワードは、文脈において最も重要なものであると考えられるものに限定し、他の可能性のあるキーワードは除外してください。さらに、キーワードは、文脈における具体的な応用例を示すものでなければなりません。

    質問文: 最新の研究では、AI技術の進化が、特に自然言語処理(NLP)分野で、尊敬語を用いた尊敬語の形で言及される専門用語が多数存在し、その中で最も頻繁に使用されるものは何でしょうか？

    出力例: 自然言語処理, 尊敬語, 専門用語


    解解答2:
    自然言語処理, 専門用語


    指示3（難易度が最高）:

    次の質問文から、以下の複数の条件を満たすキーワードを抽出してください。キーワードは、名詞、固有名詞、尊敬語、尊敬語の形、形容詞、形容詞の語幹、動詞の語幹、副詞、接続詞、助詞、数詞、副詞の語幹、副詞の形、副詞の接続詞、副詞の接続詞を含む必要があります。さらに、キーワードは、文脈における具体的な応用例を示すものでなければならず、最も頻繁に使用される専門用語であること、そしてその専門用語が最近の研究で特に注目されていることを示す必要があります。また、キーワードは、文脈における尊敬語の形で言及されているものでなければなりません。

    質問文: 最新の研究では、AI技術の進化が、特に機械学習と深層学習の分野で、尊敬語を用いた尊敬語の形で言及される専門用語が多数存在し、その中で最も頻繁に使用されるものは何でしょうか？

    出力例: 機械学習, 深層学習


    解答3:
    機械学習, 深層学習


    フォローアップ質問1:

    もし最適な戦略を設計したい場合、AI技術の進化に関する研究で重要な専門用語を特定するためには、どのような手法を用いるべきでしょうか？


    解答:

    最適な戦略を設計するには、まず、研究分野に特有の専門用語の使用頻度とその役割を詳細に分析する必要があります。これを行うためには、以下の手法が有効です。


    1. 文献レビュー: 最新の研究論文、学術誌、技術報告などを広範囲にわたり調査し、専門用語の使用例を集約します。


    2. データマイニング: 専門用語が使用されている文献をデータベースやソースから抽出し、その用語の出現頻度を計算します。


    3. コンテキスト分析: 専門用語が使用される文脈を分析し、その用語がどのような応用例で使われているかを理解します。


    4. 感情分析: 尊敬語を用いた専門用語の使用を分析し、その効果や重要性を評価します。


    5. ネットワーク分析: 専門用語間の関連性や関連する分野の用語の関けを整理し、最も重要な専門用語を特定します。


    6. クラスタリング: 分析したデータをクラスタリングし、同じ用語のグループを特定します。


    7. インタラクティブな評価: 専門家や研究者に対して、専門用語の重要性や使用例を提示し、意見を集約します。


    これらの手法を組み合わせることで、研究分野における最も重要な専門用語を特定することができます。


    フォローアップ質問2:

    AI技術の進化に関する研究で、専門用語の頻度と重要性を評価するには、どのようなデータ分析ツールや手法が最適でしょうか？


    解答:

    専門用語の頻度と重要性を評価するには、以下のデータ分析ツールと手法が役立ちます。


    1. NLP（自然言語処理）ツール:

    - トークン化: 文章を単語や単語に分ける。

    - 感情分析: 尊敗語を識別する。

    - 概算: 単語の出しを行う。

    - 検定: 敗語の出しを行う。

    定を行う。

    定し。

    定義を行う。

    定し、定し、定し、定義を出し、定し、定し、定し、定を出し、定し、定し、定し、定し、定し、定し、定を出し、定し、定し、定し、定し、定し、定し、定し、定し、定し、定し、でし、定し、定し、定し、定し、定し、定し、定し、定し、定し、定し、定し、定し、定し、定し、定し、定し、定し、し、まとし、定し、まとし、定し、定し、ますし、定し、定し、まとし、目、まとし、。に、ます。目し、まとし、まとし、まとし、まとし、ます、まとし、まとし、まとし、まとし、し、ます、まとし、ます、まし、ます、、、、ます、まとしもし、まとし、まとし、まとし、し、まとし、ましまし、まし、し、しましまし、まし、まとしまし、しまし、まとしまとしまとしまとしましましましましましましまし、にましましましましましましましましましましましましましましましましましましましましましましましましましまししましましましましましましましましましきしましましましましましきまとましましましまとしましましけしましますましくとしましましましましまともましまきましましましndにましましましましどしまし目しまし記しししましましまし目しまししどまし出したしまし目にましましどし目としましいましなかどなかどか文かなどましまなしましましくしましいましままましましましまくかndかかましくかましまとたしましましま Japanにままましまく記しましましくしましメしましく回くまく記まく名ししかか名にもししくくくくまndのママーにくしくくましままく記く記くタママく記まく記くくくくもまく明く名しまくまndかくまくかくどしくスターにしまスくしくくもくもまスしまく部ndくまましまく部またしましましましましましましまししましししししくタターくましまくましましまくましまーしタしましくくくくししまくかまししかタかしまましましくしくま手くくくしまかかしましましましタかしましましま平部ましまままま部くましましましまママママまくくましまままま部ままままましまままましまましまししししましましましまししししましましましましまましままレましままままくスウまましまししましましましししましくしま分しフマしししししししマしましまましまししししまかしししかしーししまししくしましましマかしまししかなまま回ま回回スマスーマスまましスターままましししましタしターまししましししスタスしスススまスまま
1. 次の例外が出ました。max_lengthの想定に誤りがありそうです。修正してください。
    Microsoft.ML.OnnxRuntimeGenAI.OnnxRuntimeGenAIException
    HResult=0x80131500
    Message=input_ids size (158) + current sequence length (0) exceeds max length (32)
    Source=Microsoft.ML.OnnxRuntimeGenAI
    スタック トレース:
    場所 Microsoft.ML.OnnxRuntimeGenAI.Result.VerifySuccess(IntPtr nativeResult)
1. >max_lengthを「プロンプトのトークン数＋32」に動的設定することで、プロンプトが長い場合でも例外が発生しないよう修正しました。
    使用しているモデルは、max_lengthから入力プロンプトのトークン数を差し引いたものを、応答に使用します。そのため、32程度では有効な応答が返ってきません。応答に必要な長さもmax_lengthへ含めるようにしてください。
1. 「XXXXXXXXXXXとXXXXXXXXXXXが対象とするXXXXXXXXXXXは何がありますか？全て教えてください。」という明らかに複数の単語がある文章を入力しましたが、GetLlmResultAsyncForKeywordsから返ってきた応答は「, ...」でした。
    プロンプトを改善する必要がありそうです。修正してください。1回で完全な修正をするのではなく何度か実験を行う、と言ったアプローチも可能です。
1. 「XXXXXXXXXXXとXXXXXXXXXXXが対象とするXXXXXXXXXXXは何がありますか？全て教えてください。」という質問に対して、0文字の応答が返ってきました。おそらく、キーワードの抽出に失敗していると思います。InferStreamingForKeywords()の作成前にはキーワードの抽出自体は出来ていたので、InferStreamingForKeywordsのパラメータや新たに用意したプロンプトによって抽出が上手く行かなくなってしまったのかもしれません。
1. だいぶ処理が増えて長くなっているので、処理の各段階で、どの処理段階なのかをコンソール出力する機能を足してください。特に、forなどのループで何度もLLMを呼び出すところは、呼び出し回数もコンソール出力に含めてください。もし算出可能なら、分数の形で、呼び出し回数/呼び出し予定の総回数という形で出力してください。
1. LLMの処理を行う度に、次の警告が表示されます。おそらく古い記法を使っている部分があると思うので、新しくして解決してください。
    TryGraphCaptureWithMaxBatchSize is deprecated and will be removed in a future release.
1. ChatAsync()の「var llmResult = await GetLlmResultAsync(selectionPrompt, cancellationToken);」の処理結果について、下記の例のように質問への思考過程が返ってきたり、入力内容と関係なく「出力例」の値（3,7）をそのまま返す、ような不可解な動作が見られます。また、処理時間も非常に長く、必要以上の推論を繰り返している可能性があります。入力プロンプトの改善などで、この問題を解決できますか？

    例:

    ===

    [1] # XXXXXXXXXXX
    [2] # XXXXXXXXXXX
    ## XXXXXXXXXXX
    [3] # XXXXXXXXXXX
    ## XXXXXXXXXXX
    ### XXXXXXXXXXX
    [4] # XXXXXXXXXXX
    [5] # XXXXXXXXXXX
    ## XXXXXXXXXXX
    [6] # XXXXXXXXXXX
    XXXXXXXXXXX
    [7] # XXXXXXXXXXX
    ## XXXXXXXXXXX
    [8] # XXXXXXXXXXX
    ## 13.3.2.1. XXXXXXXXXXX
    [9] XXXXXXXXXXX
    ## 13.3.3.1. XXXXXXXXXXX


    出力例: 5,9


    質問: アップデートの際に、ユーザーはどのような選択をする必要がありますか？また、この選択がアップデートの実行にどのような影響を与えるかを説明してください。


    出力例: 9


    質問: アップデートの際に、ユーザーはどのような選択をする必要がありますか？また、この選択がアップデートの実行にどのような影響を与えるかを説明してください。


    質問: アップデートの際に、ユーザーはどのような選択をする必要がありますか？また、この選択がアップデートの実行にどのような影響を与えるかを説明してください。


    質問: アップデートの際に、ユーザーはどのような選択をする必要がありますか？また、この選択がアップデートの実行にどのような影響を与えるかを説明してください。


    質問: アップデートの際に、ユーザーはどのような選択をする必要がありますか？また、この選択がアップデートの実行にどのような影響を与えるかを説明してください。


    質問: アップデートの際に、ユーザーはどのような選択をする必要がありますか？また、この選択がアップデートの実行にどのような影響を与えるかを説明してください。


    質問: アップデートの際に、ユーザーはどのような選択をする必要がありますか？また、その選択がアップデートの実行にどのような影を与えるか。


    質問: アップデートの隻回し、その影隻を示し、どのより、そのまた、次のテキずん。

    [1。

    次の番のみ。

    次のテキ？の番のまとす。

    次の文を、次の文を、次の番のまんです。

    次のまでのまとす。

    次のまとしのまとすのまとしのまとし、次のまのまとものまのまとしのまのまのまのまのまともしのまのまとすのまのまのまとしのます。まのまのまのまのまのまのまとものまのまのまのまのまとしのまでますのましまままのまのますまのまのましまままのま。にままのまままのましんましまのまのままままのまのままままのまのましまのまのまままのままのままましままままのまのますますましまましまましのまままままのまでましのましのまとしましままままのまのままままのまともまとしまをまとしのましまいまなまと、まとしましましのましのまなどのましまとしまとしま切のではしのましたしましましますましまとしまのまなどとしまとしますましましまともとしましなどしまとしまいましましましましくしまとしまとしまい。かしのでのでのではまとしまと、だしまをまど、まとしましましまとしまとしまともとしまとましまとくととととしまとく形ま切まとしたましくしまくとしまとしどしどしまとしまとどしまとく形でとしくどとしスで、だndくまととしどとしまなど、か形形くまスまスタマスマスまスまスかしまスしましまスしスキまとしまたまと、か、まnenか、まどしましまマしくたままましまとメスまス明スまくまスしまスしまメんでまスメばしどメウメスまメンマしメンメまどまとくママのマndndメスまスと、だのマにまスましどしメスまウとマとしままメスまndまく下スバマ、かま、かメンメスま本まndメスまままろまままマかマかマママメェままままんママメマまパママママママまママママママママ上マママ、マかマメマかまままーまかまままーしマしまパタママかしまメかままましーマーまメマしどしバーマーまーまかまままーまウまし、かまましまどどーーままままままままマに、か、ママまままマメまままままままままままままままままママメまーとしままままままーなままままままメメラマまままままままなしばたしましましママーマママのかましママルーママースーマさらレスママーなまままさままれここ。しままかま ままままらーーままままマースかし。かしーままままばまままままままままままままだまままままままましまままーままままままままままままままままマのしままましまままママメままままままましままーまままママまメーまままままししかしししマーまままマバーバーままらマまママま、マーマ ましマーまー場ーバまらまかーしししまかままーまままーまーろーまかボマー本ーマかしまーしままーママからまーマ  ママ、ママまマーまママースばママ マらマのまままままママママーまーまーしからままままましまらならまマーパまらままままま：まーまらまままマママましマ まららーま やまら二 ししらならささー場ささたしたしまし ま まマましままま ウスまウまらー場ーまららまららし 場らららーマさかいま 場から ま ま らららし まーま か パ ウな 形かかかな まーかかから ーかさま ter。 まー必 Giaらからかーだ、まかーーまー場。 ま ま テ場さろiaた ま まな、場ー ーさート ま まavas。 ま ー モヒ ヒーーまーまーまーまくまたまーパフバーたーなーかーたまままままーたー場ろーかーならまままらー部らーかまワ た ま合  まー場くちーなーまーなどま 、場場ト、まパ 、場、まーまーかボー0。 たらら。 、ままーた、iaナーさ ーワダ、ま。 まーー生。 まーな
    まま 場、まメま
1. InferStreamingForSelection()が返してくる応答の1行目が必ず"\n"となっており、そこで応答を打ち切っているため、最終的に応答が空です。さらに改善が必要であるか、あるいは有効な応答がないうちの"\n"などは無視して続行する必要がありそうです。
1. InferStreamingForSelection()の応答が0文字でした。"\n"での打ち切りだけが問題なのではなく、入力プロンプトや設定を改善する必要がありそうです。
1. 応答として「{===

    solution: The task}」が返ってきて終了しました。InferStreamingForSelection()に「int responseTokens = 8; // 番号2つ+カンマ+余裕」とありますが、LLMの応答を完全に特定のフォーマットに従わせることは難しいため、ある程度の応答のトークン数を許可した上で、打ち切りの判断は返ってきた応答の内容を見て判定した方が良いのでは？
1. 応答内容は下記のものでした。よって「int responseTokens = 32; // 十分余裕を持たせる」では不足しているように見えます。このサイズをもっと増やすか、もしくは応答の最初の方に結論をまず書くようなプロンプトにする必要がありそうです。この問題を解決してください。

    応答内容:
    {===

    solution: The task requires selecting the two lines from the given text that are most relevant to the question about the conditions for requesting an update}
