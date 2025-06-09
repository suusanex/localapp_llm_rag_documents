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

