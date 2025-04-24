using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.Core.Interfaces
{
    /// <summary>
    /// Defines a feature selection strategy that can be applied to an IDataView.
    /// </summary>
    public interface IFeatureSelector
    {
        /// <summary>
        /// Selects a subset of features from the input dataset according to the specified
        /// configuration and returns the transformed IDataView, the new feature set, and a report.
        /// </summary>
        /// <param name="mlContext">The MLContext instance.</param>
        /// <param name="data">The IDataView to perform feature selection on.</param>
        /// <param name="featureColumns">The initial set of feature column names.</param>
        /// <param name="modelType">The type of ML task (e.g., Regression, Classification).</param>
        /// <param name="targetField">The name of the label/target column.</param>
        /// <param name="config">Configuration parameters for feature engineering.</param>
        /// <returns>
        /// A task that returns a <see cref="FeatureSelectionResult"/> containing:
        ///  • TransformedData: the IDataView with selected features.
        ///  • FeatureNames: the names of the selected features.
        ///  • Report: a textual summary of the selection process.
        /// </returns>
        Task<IDataView> SelectFeatures(
            MLContext mlContext,
            IDataView data,
            string[] featureColumns,
            ModelType modelType,
            string targetField,
            FeatureEngineeringConfig config
        );
    }
}
