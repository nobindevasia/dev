﻿using Microsoft.ML;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Utils;
using D2G.Iris.ML.Configuration;
using D2G.Iris.ML.Data;
using D2G.Iris.ML.Training;

namespace D2G.Iris.ML
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            try
            {
                Console.WriteLine("Starting...");
                string configPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "modelconfig.json");

                var configManager = new ConfigManager();
                var config = configManager.LoadConfiguration(configPath);

                var sqlHandler = new SqlHandler(config.Database.TableName);
                sqlHandler.Connect(config.Database);

                var enabledFields = config.InputFields
                    .Where(f => f.IsEnabled)
                    .Select(f => f.Name)
                    .ToArray();

                // Create ML.NET context with a fixed random seed for reproducibility
                var mlContext = new MLContext(seed: 42);

                // Load data from SQL 
                var dataLoader = new DatabaseDataLoader();
                var rawData = dataLoader.LoadDataFromSql(
                    sqlHandler.GetConnectionString(),
                    config.Database.TableName,
                    enabledFields,
                    config.ModelType,
                    config.TargetField,
                    config.Database.WhereClause);

                // Process the data
                var dataProcessor = new DataProcessor();
                var processedData = await dataProcessor.ProcessData(
                    mlContext,
                    rawData,
                    enabledFields,
                    config,
                    sqlHandler);

                //Train the model
               var modelTrainerFactory = new ModelTrainerFactory(mlContext);
                var modelTrainer = modelTrainerFactory.CreateTrainer(config.ModelType);

                // The data is already prepared through the pipeline
                var model = await modelTrainer.TrainModel(
                    mlContext,
                    processedData.Data,
                    processedData.FeatureNames,
                    config,
                    processedData);

                Console.WriteLine("Processing completed successfully.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine($"Stack Trace: {ex.StackTrace}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner Exception: {ex.InnerException.Message}");
                }
                throw;
            }
        }
    }
}