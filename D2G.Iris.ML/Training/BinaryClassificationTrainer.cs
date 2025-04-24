using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Models;
using System.IO;
using System.Linq;

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
            // Load all training parameters directly from the JSON schema
            Console.WriteLine("\n=============== Training Using JSON Schema Configuration ===============");
            Console.WriteLine($"Using algorithm from schema: {config.TrainingParameters.Algorithm}");

            try
            {
                // Split data exactly as defined in JSON schema
                Console.WriteLine($"Using test fraction from schema: {config.TrainingParameters.TestFraction}");
                var splitData = mlContext.Data.TrainTestSplit(
                    dataView,
                    testFraction: config.TrainingParameters.TestFraction,
                    seed: 42);

                // Use the target field name directly from the JSON schema
                string targetField = config.TargetField;
                Console.WriteLine($"Using target field from schema: {targetField}");

                // Get algorithm parameters from JSON schema
                var algorithmParameters = config.TrainingParameters.AlgorithmParameters;
                Console.WriteLine("Using algorithm parameters from schema:");
                if (algorithmParameters != null)
                {
                    foreach (var param in algorithmParameters)
                    {
                        Console.WriteLine($"  • {param.Key}: {param.Value}");
                    }
                }

                // Get the trainer using the exact algorithm specified in JSON
                var trainer = _trainerFactory.GetTrainer(
                    config.ModelType,
                    config.TrainingParameters);

                // Create and use the pipeline
                var pipeline = mlContext.Transforms
                    .CopyColumns("Label", targetField)
                    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                    .AppendCacheCheckpoint(mlContext)
                    .Append(trainer);

                Console.WriteLine("Training model using pipeline from schema...");
                var model = await Task.Run(() => pipeline.Fit(splitData.TrainSet));

                // Evaluate the model
                Console.WriteLine("Evaluating model...");
                var predictions = model.Transform(splitData.TestSet);
                var metrics = mlContext.BinaryClassification.Evaluate(
                    predictions,
                    labelColumnName: "Label",
                    scoreColumnName: "Score",
                    predictedLabelColumnName: "PredictedLabel");

                // Print metrics
                PrintMetrics(metrics);

                // Save model using schema information
                var modelPath = Path.Combine(
                    AppDomain.CurrentDomain.BaseDirectory,
                    $"{config.Author}_{config.TrainingParameters.Algorithm}_Model.zip"
                );

                mlContext.Model.Save(model, splitData.TrainSet.Schema, modelPath);
                Console.WriteLine($"Model saved to: {modelPath}");

                // Save to output table if specified in schema
                if (!string.IsNullOrEmpty(config.Database.OutputTableName))
                {
                    Console.WriteLine($"Predictions would be saved to: {config.Database.OutputTableName}");
                    // Implementation would go here
                }

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
            Console.WriteLine($"F1 Score: {metrics.F1Score:F4}");
            Console.WriteLine($"Confusion Matrix:");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
    }
}