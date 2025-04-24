using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.DataBalancing;
using D2G.Iris.ML.FeatureEngineering;

namespace D2G.Iris.ML.Data
{
    public class DataProcessor
    {
        public async Task<ProcessedData> ProcessData(
            MLContext mlContext,
            IDataView rawData,
            string[] enabledFields,
            ModelConfig config
            )
        {
            Console.WriteLine("\n=============== Processing Data ===============");

            IDataView transformedData = rawData;
            string[] finalFeatureNames = enabledFields.Where(f => f != config.TargetField).ToArray();
            string selectionReport = string.Empty;

            // Original row count
            long originalCount = rawData.GetRowCount() ?? 0;
            long balancedCount = originalCount;

            // Create feature vector
            var featurePipeline = mlContext.Transforms
                .Concatenate("Features", finalFeatureNames);

            transformedData = featurePipeline.Fit(rawData).Transform(rawData);

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
                    var balancer = new SmoteDataBalancer();
                    transformedData = await balancer.BalanceDataset(
                        mlContext,
                        transformedData,
                        finalFeatureNames,
                        config.DataBalancing,
                        config.TargetField);
                    balancedCount = transformedData.GetRowCount() ?? originalCount;
                    Console.WriteLine($"Data balanced. New count: {balancedCount}");
                }

                // Feature Selection
                if (config.FeatureEngineering.Method != FeatureSelectionMethod.None)
                {
                    var selector = new CorrelationFeatureSelector(mlContext); 
                    var result = await selector.SelectFeatures(
                        mlContext,
                        transformedData,
                        finalFeatureNames,
                        config.ModelType,
                        config.TargetField,
                        config.FeatureEngineering);

                    transformedData = result.transformedData;
                    finalFeatureNames = result.selectedFeatures;
                    selectionReport = result.report;
                    Console.WriteLine(selectionReport);
                }
            }
            else
            {
                // Feature Selection first
                if (config.FeatureEngineering.Method != FeatureSelectionMethod.None)
                {
                    var selector = new CorrelationFeatureSelector(mlContext);
                    var result = await selector.SelectFeatures(
                        mlContext,
                        transformedData,
                        finalFeatureNames,
                        config.ModelType,
                        config.TargetField,
                        config.FeatureEngineering);

                    transformedData = result.transformedData;
                    finalFeatureNames = result.selectedFeatures;
                    selectionReport = result.report;
                    Console.WriteLine(selectionReport);
                }

                // Then Data Balancing
                if (config.DataBalancing.Method != DataBalanceMethod.None)
                {
                    var balancer = new SmoteDataBalancer();
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
            //if (!string.IsNullOrEmpty(config.Database.OutputTableName))
            //{
            //    try
            //    {
            //        sqlHandler.SaveDataViewToTable(
            //            mlContext,
            //            transformedData,
            //            config.Database.OutputTableName,
            //            finalFeatureNames,
            //            config.TargetField,
            //            config.ModelType);

            //        Console.WriteLine($"Processed data saved to: {config.Database.OutputTableName}");
            //    }
            //    catch (Exception ex)
            //    {
            //        Console.WriteLine($"Error saving processed data: {ex.Message}");
            //    }
            //}

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