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
            InputField[] enabledFields,
            ModelConfig config
            )
        {
            Console.WriteLine("\n=============== Processing Data ===============");

            IDataView transformedData = rawData;
            string[] finalFeatureNames = enabledFields.Where(f => f.Name != config.TargetField).Select(f => f.Name).ToArray();
            string selectionReport = string.Empty;

            // Original row count
            long originalCount = rawData.GetRowCount() ?? 0;
            long balancedCount = originalCount;

            // Handle case-insensitive enum parsing for methods
            EnsureEnumValuesAreParsed(config);

            // Create feature vector with appropriate data type awareness
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
                    var balancer = CreateDataBalancer(config.DataBalancing.Method);
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
                    var selector = CreateFeatureSelector(mlContext, config.FeatureEngineering.Method);
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
                    var selector = CreateFeatureSelector(mlContext, config.FeatureEngineering.Method);
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
                    var balancer = CreateDataBalancer(config.DataBalancing.Method);
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

        private void EnsureEnumValuesAreParsed(ModelConfig config)
        {
            // Handle case-insensitive enum parsing for data balancing method
            if (config.DataBalancing != null && config.DataBalancing.Method == 0)
            {
                if (Enum.TryParse<DataBalanceMethod>("Smote", true, out var balanceMethod))
                {
                    Console.WriteLine($"Parsed data balancing method from string: {balanceMethod}");
                    config.DataBalancing.Method = balanceMethod;
                }
            }

            // Handle case-insensitive enum parsing for feature selection method
            if (config.FeatureEngineering != null && config.FeatureEngineering.Method == 0)
            {
                if (Enum.TryParse<FeatureSelectionMethod>("Correlation", true, out var featureMethod))
                {
                    Console.WriteLine($"Parsed feature selection method from string: {featureMethod}");
                    config.FeatureEngineering.Method = featureMethod;
                }
            }
        }

        private dynamic CreateDataBalancer(DataBalanceMethod method)
        {
            return method switch
            {
                DataBalanceMethod.SMOTE => new SmoteDataBalancer(),
                DataBalanceMethod.ADASYN => new SmoteDataBalancer(), // Use SMOTE for now
                _ => new NoDataBalancer()
            };
        }

        private dynamic CreateFeatureSelector(MLContext mlContext, FeatureSelectionMethod method)
        {
            return method switch
            {
                FeatureSelectionMethod.Correlation => new CorrelationFeatureSelector(mlContext),
                FeatureSelectionMethod.Forward => new CorrelationFeatureSelector(mlContext), // Use Correlation for now
                FeatureSelectionMethod.PCA => new CorrelationFeatureSelector(mlContext), // Use Correlation for now
                _ => new NoFeatureSelector(mlContext)
            };
        }
    }
}