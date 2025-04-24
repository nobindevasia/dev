using System;
using System.Collections.Generic;
using System.Data;
using Microsoft.Data.SqlClient;
using System.Linq;
using Microsoft.ML;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.Data
{
    public class SqlHandler
    {
        private SqlConnectionStringBuilder _builder;
        private readonly string _tableName;

        public SqlHandler(string tableName)
        {
            _tableName = tableName;
        }

        public void Connect(DatabaseConfig dbConfig)
        {
            _builder = new SqlConnectionStringBuilder()
            {
                DataSource = dbConfig.Server,
                InitialCatalog = dbConfig.Database,
                IntegratedSecurity = true,
                Pooling = true,
                TrustServerCertificate = true,
                ConnectTimeout = 60
            };
        }

        public string GetConnectionString()
        {
            if (_builder == null)
                throw new InvalidOperationException("Database connection not initialized");
            return _builder.ConnectionString;
        }

        public void SaveModelOutput(IDataView predictions, string connectionString, string outputTableName)
        {
            // Implementation for saving model outputs to SQL would go here
            Console.WriteLine($"Saving prediction results to {outputTableName}...");
            // This would typically involve:
            // 1. Converting the IDataView to an enumerable
            // 2. Creating a SQL table with the correct schema if it doesn't exist
            // 3. Bulk inserting the data
        }
    }
}