using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.DataBalancing;
using D2G.Iris.ML.FeatureEngineering;

namespace D2G.Iris.ML.Data
{
    /// <summary>
    /// Orchestrates data balancing, feature selection, and saving of a pre-loaded IDataView.
    /// </summary>
    public class DataProcessor 
    {
        /// <inheritdoc />
        public async Task<ProcessedData> ProcessData(
            MLContext mlContext,
            IDataView rawData,
            string[] enabledFields,
            ModelConfig config,
            ISqlHandler sqlHandler)
        {
            Console.WriteLine("\n=============== Processing Data ===============");

            IDataView transformedData = rawData;
            string[] finalFeatureNames = enabledFields;
            string selectionReport = string.Empty;

            // Original row count
            long originalCount = rawData.GetRowCount() ?? 0;
            long balancedCount = originalCount;

            // Determine execution order
            bool balancingFirst = config.DataBalancing.ExecutionOrder <= config.FeatureEngineering.ExecutionOrder;
            if (config.DataBalancing.Method != DataBalanceMethod.None &&
                config.FeatureEngineering.Method != FeatureSelectionMethod.None)
            {
                Console.WriteLine($"Processing order: {(balancingFirst ?
                    "Data Balancing then Feature Selection" :
                    "Feature Selection then Data Balancing")}");
            }

            // Apply balancing and feature selection
            if (balancingFirst)
            {
                // Data Balancing
                if (config.DataBalancing.Method != DataBalanceMethod.None)
                {
                    var balancer = new DataBalancerFactory()
                        .CreateBalancer(config.DataBalancing.Method);
                    transformedData = await balancer.BalanceDataset(
                        mlContext,
                        transformedData,
                        enabledFields,
                        config.DataBalancing,
                        config.TargetField);
                    balancedCount = transformedData.GetRowCount() ?? originalCount;
                    Console.WriteLine($"Data balanced. New count: {balancedCount}");
                }

                // Feature Selection
                if (config.FeatureEngineering.Method != FeatureSelectionMethod.None)
                {
                    var selector = new FeatureSelectorFactory(mlContext)
                        .CreateSelector(config.FeatureEngineering.Method);
                    var result = await selector.SelectFeatures(
                        mlContext,
                        transformedData,
                        enabledFields,
                        config.ModelType,
                        config.TargetField,
                        config.FeatureEngineering);
                    transformedData = result.transformedData;
                    finalFeatureNames = result.featureNames;
                    selectionReport = result.report;
                    Console.WriteLine(selectionReport);
                }
                else
                {
                    transformedData = mlContext.Transforms
                        .Concatenate("Features", enabledFields)
                        .Fit(transformedData)
                        .Transform(transformedData);
                    selectionReport = "Feature selection disabled.";
                }
            }
            else
            {
                // Feature Selection first
                if (config.FeatureEngineering.Method != FeatureSelectionMethod.None)
                {
                    var selector = new FeatureSelectorFactory(mlContext)
                        .CreateSelector(config.FeatureEngineering.Method);
                    var result = await selector.SelectFeatures(
                        mlContext,
                        rawData,
                        enabledFields,
                        config.ModelType,
                        config.TargetField,
                        config.FeatureEngineering);
                    transformedData = result.transformedData;
                    finalFeatureNames = result.featureNames;
                    selectionReport = result.report;
                    Console.WriteLine(selectionReport);
                }

                // Then Data Balancing
                if (config.DataBalancing.Method != DataBalanceMethod.None)
                {
                    var balancer = new DataBalancerFactory()
                        .CreateBalancer(config.DataBalancing.Method);
                    transformedData = await balancer.BalanceDataset(
                        mlContext,
                        transformedData,
                        finalFeatureNames,
                        config.DataBalancing,
                        config.TargetField);
                    balancedCount = transformedData.GetRowCount() ?? originalCount;
                    Console.WriteLine($"Data balanced. New count: {balancedCount}");
                }
            }

            // Save to SQL if configured
            if (!string.IsNullOrEmpty(config.Database.OutputTableName))
            {
                try
                {
                    // Ensure connection
                    sqlHandler.Connect(config.Database);
                    var connString = sqlHandler.GetConnectionString();

                    sqlHandler.SaveDataViewToTable(
                        mlContext,
                        transformedData,
                        config.Database.OutputTableName,
                        finalFeatureNames,
                        config.TargetField,
                        config.ModelType);

                    Console.WriteLine($"Processed data saved to: {config.Database.OutputTableName}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error saving processed data: {ex.Message}");
                }
            }

            return new ProcessedData
            {
                Data = transformedData,
                FeatureNames = finalFeatureNames,
                OriginalSampleCount = (int)originalCount,
                BalancedSampleCount = (int)balancedCount,
                FeatureSelectionReport = selectionReport,
                FeatureSelectionMethod = config.FeatureEngineering.Method,
                DataBalancingMethod = config.DataBalancing.Method,
                DataBalancingExecutionOrder = config.DataBalancing.ExecutionOrder,
                FeatureSelectionExecutionOrder = config.FeatureEngineering.ExecutionOrder
            };
        }
    }
}
