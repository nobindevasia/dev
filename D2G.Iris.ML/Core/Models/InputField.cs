using System;
using System.Text.Json.Serialization;

namespace D2G.Iris.ML.Core.Models
{
    public class InputField
    {
        public string Name { get; set; }
        public bool IsEnabled { get; set; }
        public string DataType { get; set; }

        // If you want to pick up the "targetField" JSON object in the same array,
        // you can make this nullable and JsonIgnore when it's not present.
        [JsonPropertyName("targetField")]
        public string TargetField { get; set; }

        public Type GetCSharpType()
        {
            return DataType?.ToLower() switch
            {
                "float" => typeof(float),
                "double" => typeof(double),
                "int" => typeof(int),
                "long" => typeof(long),
                "bool" => typeof(bool),
                "string" => typeof(string),
                _ => typeof(float)  // Default to float
            };
        }

        public System.Data.DbType GetDbType()
        {
            return DataType?.ToLower() switch
            {
                "float" => System.Data.DbType.Single,
                "double" => System.Data.DbType.Double,
                "int" => System.Data.DbType.Int32,
                "long" => System.Data.DbType.Int64,
                "bool" => System.Data.DbType.Boolean,
                "string" => System.Data.DbType.String,
                _ => System.Data.DbType.Single  // Default to float
            };
        }
    }
}