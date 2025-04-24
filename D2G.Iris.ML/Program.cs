using Microsoft.ML;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using D2G.Iris.ML.Configuration;
using D2G.Iris.ML.Data;
using D2G.Iris.ML.Training;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            try
            {
                Console.WriteLine("Starting ML.NET Pipeline...");

                // Load configuration
                string configPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "modelconfig.json");
                var configManager = new ConfigManager();
                var config = configManager.LoadConfiguration(configPath);

                // Setup SQL connection
                var sqlHandler = new SqlHandler(config.Database.TableName);
                sqlHandler.Connect(config.Database);

                // Get enabled fields for features (excluding the target field)
                var enabledFields = config.InputFields
                    .Where(f => f.IsEnabled && f.TargetField == null)
                    .ToArray();

                // Find target field definition from InputFields list
                var targetField = config.InputFields
                    .FirstOrDefault(f => f.TargetField != null);

                if (targetField == null)
                {
                    throw new ArgumentException("Target field not found in the configuration");
                }

                // Set the target field name from the configuration
                config.TargetField = targetField.TargetField;

                // Create ML.NET context with fixed seed for reproducibility
                var mlContext = new MLContext(seed: 42);

                Console.WriteLine("Loading data...");
                var dataLoader = new DatabaseDataLoader();
                var rawData = dataLoader.LoadDataFromSql(
                    sqlHandler.GetConnectionString(),
                    config.Database.TableName,
                    enabledFields,
                    config.ModelType,
                    config.TargetField,
                    targetField,
                    config.Database.WhereClause);

                // Process the data
                Console.WriteLine("Processing data...");
                var dataProcessor = new DataProcessor();
                var processedData = await dataProcessor.ProcessData(
                    mlContext,
                    rawData,
                    enabledFields,
                    config
                    );

                // Train the model
                Console.WriteLine("Training model...");
                var modelTrainer = config.ModelType switch
                {
                    ModelType.BinaryClassification => new BinaryClassificationTrainer(mlContext, new TrainerFactory(mlContext)),
                    //ModelType.MultiClassClassification => new MultiClassClassificationTrainer(mlContext, new TrainerFactory(mlContext)),
                    //ModelType.Regression => new RegressionTrainer(mlContext, new TrainerFactory(mlContext)),
                    _ => throw new ArgumentException($"Unsupported model type: {config.ModelType}")
                };

                var model = await modelTrainer.TrainModel(
                    mlContext,
                    processedData.Data,
                    enabledFields,
                    config,
                    processedData);

                // Save predictions to output table if specified
                if (!string.IsNullOrEmpty(config.Database.OutputTableName))
                {
                    var predictions = model.Transform(processedData.Data);
                    sqlHandler.SaveModelOutput(
                        predictions,
                        sqlHandler.GetConnectionString(),
                        config.Database.OutputTableName);
                }

                Console.WriteLine("Pipeline completed successfully.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in pipeline: {ex.Message}");
                Console.WriteLine($"Stack Trace: {ex.StackTrace}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner Exception: {ex.InnerException.Message}");
                }
                throw;
            }
        }

        private static string GetDefaultDataTypeForModelType(ModelType modelType)
        {
            return modelType switch
            {
                ModelType.BinaryClassification => "bool",
                ModelType.MultiClassClassification => "int",
                ModelType.Regression => "float",
                _ => "float"
            };
        }
    }
}