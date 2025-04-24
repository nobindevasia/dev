using Microsoft.ML;
using System.Collections.Generic;
using D2G.Iris.ML.Core.Enums;

namespace D2G.Iris.ML.Core.Interfaces
{
    /// <summary>
    /// Defines a data loader that streams SQL data into an ML.NET IDataView,
    /// optionally previewing rows and tracking row count.
    /// </summary>
    public interface IDataLoader
    {
        /// <summary>
        /// Loads data from a SQL table into an IDataView.
        /// </summary>
        /// <param name="connectionString">The ADO.NET connection string.</param>
        /// <param name="tableName">The (optionally schema-qualified) table name.</param>
        /// <param name="featureColumns">Columns to use as features.</param>
        /// <param name="modelType">Type of ML task to influence label type.</param>
        /// <param name="targetColumn">The label column name.</param>
        /// <param name="whereSyntax">Optional WHERE clause (without "WHERE").</param>
        /// <param name="previewRowCount">Number of rows to preview in console output.</param>
        /// <returns>An IDataView representing the streamed data.</returns>
        IDataView LoadDataFromSql(
            string connectionString,
            string tableName,
            IEnumerable<string> featureColumns,
            ModelType modelType,
            string targetColumn,
            string whereSyntax = "",
            int previewRowCount = 5);

        /// <summary>
        /// Gets the number of rows loaded by the last call to LoadDataFromSql.
        /// </summary>
        /// <returns>The row count, or null if not tracked.</returns>
        long? GetLastLoadedRowCount();
    }
}
