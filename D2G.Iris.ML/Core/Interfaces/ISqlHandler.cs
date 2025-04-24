﻿using System.Collections.Generic;
using Microsoft.ML;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Enums;

namespace D2G.Iris.ML.Core.Interfaces
{
    public interface ISqlHandler
    {
        void Connect(DatabaseConfig dbConfig);
        string GetConnectionString();

        // Original method for backward compatibility
        void SaveProcessedDataToTable(
            string tableName,
            List<Dictionary<string, object>> processedData,
            string[] featureNames,
            string targetField,
            ModelType modelType);

        // New method working directly with IDataView
        void SaveDataViewToTable(
            MLContext mlContext,
            IDataView dataView,
            string tableName,
            string[] featureNames,
            string targetField,
            ModelType modelType);
    }
}