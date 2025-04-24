using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Training;

namespace D2G.Iris.ML.Training
{
    public class BinaryClassificationTrainer
    {
        private readonly MLContext _mlContext;
        private readonly TrainerFactory _trainerFactory;

        public BinaryClassificationTrainer(MLContext mlContext, TrainerFactory trainerFactory)
        {
            _mlContext = mlContext;
            _trainerFactory = trainerFactory;
        }

        public async Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            InputField[] inputFields,
            ModelConfig config,
            ProcessedData processedData)
        {
            Console.WriteLine($"\nStarting binary classification model training using {config.TrainingParameters.Algorithm}...");

            try
            {
                // Get feature column names
                string[] featureNames = processedData.FeatureNames;

                // Use the IDataView directly without converting to custom classes
                // This maintains the data types throughout the pipeline

                // Split data into training and test sets
                var splitData = mlContext.Data.TrainTestSplit(
                    dataView,
                    testFraction: config.TrainingParameters.TestFraction,
                    seed: 42);

                Console.WriteLine($"Training set size: {splitData.TrainSet.GetRowCount() ?? 0} rows");
                Console.WriteLine($"Test set size: {splitData.TestSet.GetRowCount() ?? 0} rows");

                // The label column should be casted to Boolean for binary classification
                // Only do this if the target field is not already a boolean
                IDataView trainData = splitData.TrainSet;
                IDataView testData = splitData.TestSet;

                var targetDataType = config.DataType?.ToLower() ?? "bool";
                if (targetDataType != "bool")
                {
                    // Create transformation to convert label to boolean
                    var labelPipeline = mlContext.Transforms.Conversion.MapValue(
                        outputColumnName: "Label",
                        inputColumnName: config.TargetField,
                        map: new[] { new KeyValuePair<long, bool>(0, false), new KeyValuePair<long, bool>(1, true) }
                    );

                    trainData = labelPipeline.Fit(splitData.TrainSet).Transform(splitData.TrainSet);
                    testData = labelPipeline.Fit(splitData.TestSet).Transform(splitData.TestSet);
                }

                // Get the trainer from factory
                var trainer = _trainerFactory.GetTrainer(
                    config.ModelType,
                    config.TrainingParameters);

                // Create training pipeline
                // Keep it simple, focusing on maintaining the types
                var pipeline = mlContext.Transforms
                    .NormalizeMinMax("Features")
                    .AppendCacheCheckpoint(mlContext)
                    .Append(trainer);

                // Train the model
                Console.WriteLine("Training model...");
                var model = await Task.Run(() => pipeline.Fit(trainData));

                // Evaluate the model
                Console.WriteLine("Evaluating model...");
                var predictions = model.Transform(testData);
                var metrics = mlContext.BinaryClassification.Evaluate(
                    predictions,
                    labelColumnName: "Label",
                    scoreColumnName: "Score",
                    predictedLabelColumnName: "PredictedLabel");

                // Print metrics
                PrintMetrics(metrics);

                // Save the model
                var modelPath = $"BinaryClassification_{config.TrainingParameters.Algorithm}_Model.zip";
                mlContext.Model.Save(model, dataView.Schema, modelPath);
                Console.WriteLine($"Model saved to: {modelPath}");

                // Create standardized metrics
                var standardizedMetrics = new StandardizedBinaryMetrics
                {
                    Accuracy = metrics.Accuracy,
                    AreaUnderRocCurve = metrics.AreaUnderRocCurve,
                    PositivePrecision = metrics.PositivePrecision,
                    PositiveRecall = metrics.PositiveRecall,
                    F1Score = metrics.F1Score,
                    AreaUnderPrecisionRecallCurve = metrics.AreaUnderPrecisionRecallCurve
                };

                Console.WriteLine(standardizedMetrics.CreateStandardizedMetricsMsg());

                return model;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during model training: {ex.Message}");
                throw;
            }
        }

        private void PrintMetrics(BinaryClassificationMetrics metrics)
        {
            Console.WriteLine();
            Console.WriteLine("Model Evaluation Metrics:");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F4}");
            Console.WriteLine($"Area Under ROC Curve: {metrics.AreaUnderRocCurve:F4}");
            Console.WriteLine($"Area Under PR Curve: {metrics.AreaUnderPrecisionRecallCurve:F4}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F4}");
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision:F4}");
            Console.WriteLine($"Negative Precision: {metrics.NegativePrecision:F4}");
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F4}");
            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F4}");
            Console.WriteLine();
            Console.WriteLine("Confusion Matrix:");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
    }
}