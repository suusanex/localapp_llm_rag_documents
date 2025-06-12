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
