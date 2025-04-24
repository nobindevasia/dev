using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core;
using D2G.Iris.ML.Core.Enums;

namespace D2G.Iris.ML.Configuration
{
    public class ConfigManager
    {
        public ModelConfig LoadConfiguration(string configPath)
        {
            try
            {
                if (!File.Exists(configPath))
                {
                    throw new FileNotFoundException($"Configuration file not found at: {configPath}");
                }

                var configText = File.ReadAllText(configPath);
                Console.WriteLine("Reading config file...");

                var options = new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true,
                    WriteIndented = true,
                    Converters = { new JsonStringEnumConverter() }
                };

                var config = JsonSerializer.Deserialize<Dictionary<string, ModelConfig>>(configText, options);
                var modelConfig = config["modelConfig"];

                ValidateConfiguration(modelConfig);

                // Handle case-insensitive enum parsing for data balancing method
                if (modelConfig.DataBalancing != null && modelConfig.DataBalancing.Method == 0)
                {
                    // Try to parse the method from string if it's not correctly deserialized
                    if (Enum.TryParse<DataBalanceMethod>(modelConfig.DataBalancing.Method.ToString(), true, out var method))
                    {
                        modelConfig.DataBalancing.Method = method;
                    }
                }

                return modelConfig;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading configuration: {ex.Message}");
                throw;
            }
        }

        public void ValidateConfiguration(ModelConfig config)
        {
            if (config.Database == null)
                throw new ArgumentNullException(nameof(config.Database), "Database configuration is missing");

            if (string.IsNullOrEmpty(config.Database.TableName))
                throw new ArgumentException("Table name is missing in database configuration");

            if (config.TrainingParameters == null)
                throw new ArgumentNullException(nameof(config.TrainingParameters), "Training parameters are missing");

            if (string.IsNullOrEmpty(config.TrainingParameters.Algorithm))
                throw new ArgumentException("Algorithm name is missing in training parameters");

            // Find the target field from InputFields
            var targetField = config.InputFields.Find(f => f.TargetField != null);
            if (targetField == null)
                throw new ArgumentException("Target field is missing in input fields");

            config.TargetField = targetField.TargetField;

            // Set data type for target field
            config.DataType = targetField.DataType;

            ValidateFeatureEngineeringConfig(config.FeatureEngineering);
            ValidateDataBalancingConfig(config.DataBalancing);
        }

        private void ValidateFeatureEngineeringConfig(FeatureEngineeringConfig config)
        {
            if (config == null) return;

            switch (config.Method)
            {
                case Core.Enums.FeatureSelectionMethod.PCA:
                    if (config.NumberOfComponents <= 0)
                        throw new ArgumentException("Number of components must be greater than 0 for PCA");
                    break;

                case Core.Enums.FeatureSelectionMethod.Forward:
                    if (config.MaxFeatures <= 0)
                        throw new ArgumentException("Max features must be greater than 0 for Forward Selection");
                    if (config.MinImprovement <= 0)
                        throw new ArgumentException("Minimum improvement must be greater than 0 for Forward Selection");
                    break;

                case Core.Enums.FeatureSelectionMethod.Correlation:
                    if (config.MulticollinearityThreshold <= 0 || config.MulticollinearityThreshold >= 1)
                        throw new ArgumentException("Multicollinearity threshold must be between 0 and 1 for Correlation Selection");
                    break;
            }
        }

        private void ValidateDataBalancingConfig(DataBalancingConfig config)
        {
            if (config == null) return;

            switch (config.Method)
            {
                case DataBalanceMethod.SMOTE:
                case DataBalanceMethod.ADASYN:
                    if (config.KNeighbors <= 0)
                        throw new ArgumentException($"K neighbors must be greater than 0 for {config.Method}");
                    if (config.UndersamplingRatio <= 0 || config.UndersamplingRatio > 1)
                        throw new ArgumentException($"Undersampling ratio must be between 0 and 1 for {config.Method}");
                    if (config.MinorityToMajorityRatio <= 0 || config.MinorityToMajorityRatio > 1)
                        throw new ArgumentException($"Minority to majority ratio must be between 0 and 1 for {config.Method}");
                    break;
            }
        }
    }
}