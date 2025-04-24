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
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            Console.WriteLine($"\nStarting binary classification model training using {config.TrainingParameters.Algorithm}...");

            try
            {
                // Step 1: Read the data with our required structure
                var featureVectors = mlContext.Data.CreateEnumerable<FeatureVector>(dataView, reuseRowObject: false).ToList();
                Console.WriteLine($"Loaded {featureVectors.Count} records for training");

                // Step 2: Find the feature vector size (using featureNames length or max vector size in data)
                int vectorSize = featureNames?.Length > 0
                    ? featureNames.Length
                    : featureVectors.Max(v => v.Features?.Length ?? 0);
                Console.WriteLine($"Using feature vector size: {vectorSize}");

                // Step 3: Create binary classification rows with fixed-length features
                var binaryRows = featureVectors.Select(row => new BinaryRow
                {
                    // Create a new array of exact size and copy/pad the features
                    Features = CreateFixedLengthArray(row.Features, vectorSize),
                    Label = row.Label > 0  // Convert numeric label to boolean
                }).ToList();

                // Step 4: Create schema that explicitly defines vector size
                var schema = SchemaDefinition.Create(typeof(BinaryRow));
                schema["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, vectorSize);

                // Step 5: Load data with fixed schema
                var typedData = mlContext.Data.LoadFromEnumerable(binaryRows, schema);

                // Step 6: Split data into training and test sets
                var splitData = mlContext.Data.TrainTestSplit(
                    typedData,
                    testFraction: config.TrainingParameters.TestFraction,
                    seed: 42);

                // Step 7: Get the right trainer from factory
                var trainer = _trainerFactory.GetTrainer(
                    config.ModelType,
                    config.TrainingParameters);

                // Step 8: Create simple training pipeline
                var pipeline = mlContext.Transforms
                    .NormalizeMinMax("Features")
                    .Append(trainer)
                    .Append(mlContext.Transforms.CopyColumns("Probability", "Score"));

                // Step 9: Train the model
                Console.WriteLine("Training model...");
                var model = await Task.Run(() => pipeline.Fit(splitData.TrainSet));

                // Step 10: Evaluate the model
                Console.WriteLine("Evaluating model...");
                var predictions = model.Transform(splitData.TestSet);
                var metrics = mlContext.BinaryClassification.Evaluate(
                    predictions,
                    labelColumnName: "Label",
                    scoreColumnName: "Score",
                    predictedLabelColumnName: "PredictedLabel");

                // Step 11: Print metrics
                PrintMetrics(metrics);

                // Step 12: Save the model
                var modelPath = $"BinaryClassification_{config.TrainingParameters.Algorithm}_Model.zip";
                mlContext.Model.Save(model, typedData.Schema, modelPath);
                Console.WriteLine($"Model saved to: {modelPath}");

                return model;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during model training: {ex.Message}");
                throw;
            }
        }

        // Helper method to create fixed-length feature arrays
        private float[] CreateFixedLengthArray(float[] source, int length)
        {
            // Create new array of the required length
            var result = new float[length];

            // If source exists, copy values (up to the minimum of source length and required length)
            if (source != null && source.Length > 0)
            {
                int copyLength = Math.Min(source.Length, length);
                Array.Copy(source, result, copyLength);
            }

            return result;
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

        // Schema classes for data
        private class BinaryRow
        {
            [VectorType]
            public float[] Features { get; set; }
            public bool Label { get; set; }
        }

        private class FeatureVector
        {
            [VectorType]
            public float[] Features { get; set; }
            public long Label { get; set; }
        }
    }
}