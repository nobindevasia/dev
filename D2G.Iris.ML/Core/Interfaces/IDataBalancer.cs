using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.Core.Interfaces
{
    /// <summary>
    /// Defines a data balancing strategy that can be applied to an IDataView.
    /// </summary>
    public interface IDataBalancer
    {
        /// <summary>
        /// Balances the input dataset according to the specified configuration and target field.
        /// </summary>
        /// <param name="mlContext">The MLContext instance.</param>
        /// <param name="data">The IDataView to balance.</param>
        /// <param name="featureColumns">The names of feature columns to include in balancing logic.</param>
        /// <param name="config">Configuration parameters for data balancing.</param>
        /// <param name="targetField">The name of the label/target column.</param>
        /// <returns>A task that returns a new balanced IDataView.</returns>
        Task<IDataView> BalanceDataset(
            MLContext mlContext,
            IDataView data,
            string[] featureColumns,
            DataBalancingConfig config,
            string targetField
        );
    }
}
