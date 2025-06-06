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
    /// appsettings.jsonから読み込む
    /// </summary>
    public string Other1 { get; set; }

}
