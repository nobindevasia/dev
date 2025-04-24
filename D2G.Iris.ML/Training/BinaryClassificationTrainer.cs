using System;
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
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            Console.WriteLine($"\nStarting binary classification model training using {config.TrainingParameters.Algorithm}...");

            try
            {
                // Split data
                var dataSplit = mlContext.Data.TrainTestSplit(
                    dataView,
                    testFraction: config.TrainingParameters.TestFraction,
                    seed: 42);

                // Get trainer
                var trainer = _trainerFactory.GetTrainer(
                    config.ModelType,
                    config.TrainingParameters);

                // Create training pipeline
                var pipeline = _mlContext.Transforms
    .CopyColumns("Label", config.TargetField)
    .Append(_mlContext.Transforms.NormalizeMinMax("Features"))  // Add normalization
    .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label"))  // Convert label to key
    .Append(trainer)
    .Append(_mlContext.Transforms.CopyColumns("Probability", "Score"));

                // Before training, verify data schema
                var previewSet = dataSplit.TrainSet.Preview();
                Console.WriteLine("Training Data Schema:");
                foreach (var col in previewSet.Schema)
                {
                    Console.WriteLine($"Column: {col.Name}, Type: {col.Type}");
                }

                // Train model
                var model = await Task.Run(() => pipeline.Fit(dataSplit.TrainSet));

                // Evaluate model
                Console.WriteLine("Evaluating model...");
                var predictions = model.Transform(dataSplit.TestSet);
                var metrics = mlContext.BinaryClassification.Evaluate(
                    predictions,
                    labelColumnName: "Label",
                    scoreColumnName: "Score",
                    predictedLabelColumnName: "PredictedLabel");

                // Print metrics
                PrintMetrics(metrics);

                // Save model
                var modelPath = $"BinaryClassification_{config.TrainingParameters.Algorithm}_Model.zip";
                mlContext.Model.Save(model, dataView.Schema, modelPath);
                Console.WriteLine($"Model saved to: {modelPath}");

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