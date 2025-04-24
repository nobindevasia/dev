using System;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using MathNet.Numerics.Statistics;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.FeatureEngineering
{
    public class CorrelationFeatureSelector
    {
        private readonly MLContext _mlContext;
        private readonly StringBuilder _report;

        public CorrelationFeatureSelector(MLContext mlContext)
        {
            _mlContext = mlContext;
            _report = new StringBuilder();
        }

        public async Task<(IDataView transformedData, string[] selectedFeatures, string report)> SelectFeatures(
            MLContext mlContext,
            IDataView data,
            string[] candidateFeatures,
            ModelType modelType,
            string targetField,
            FeatureEngineeringConfig config)
        {
            _report.Clear();
            _report.AppendLine("\nCorrelation-based Feature Selection Results:");
            _report.AppendLine("----------------------------------------------");

            try
            {
                var preview = data.Preview();
                Console.WriteLine("Input Data Schema:");
                foreach (var col in preview.Schema)
                {
                    Console.WriteLine($"Column: {col.Name}, Type: {col.Type}");
                }
                // Create enumerable to calculate correlations
                var dataEnumerable = mlContext.Data.CreateEnumerable<FeatureVector>(
                    data, reuseRowObject: false).ToList();

                // Extract target values and features
                var targetValues = dataEnumerable.Select(x => (double)x.Label).ToArray();
                var featureMatrix = new List<double[]>();

                // Extract feature values for each candidate feature
                foreach (var feature in candidateFeatures)
                {
                    var featureValues = dataEnumerable
                        .Select(x => x.Features[Array.IndexOf(candidateFeatures, feature)])
                        .Select(x => (double)x)
                        .ToArray();
                    featureMatrix.Add(featureValues);
                }

                // Calculate correlations with target
                var targetCorrelations = new Dictionary<string, double>();
                for (int i = 0; i < candidateFeatures.Length; i++)
                {
                    var correlation = Math.Abs(Correlation.Pearson(featureMatrix[i], targetValues));
                    targetCorrelations[candidateFeatures[i]] = correlation;
                }

                // Calculate feature-feature correlations
                var correlationMatrix = new double[candidateFeatures.Length, candidateFeatures.Length];
                for (int i = 0; i < candidateFeatures.Length; i++)
                {
                    for (int j = 0; j < candidateFeatures.Length; j++)
                    {
                        correlationMatrix[i, j] = Math.Abs(Correlation.Pearson(featureMatrix[i], featureMatrix[j]));
                    }
                }

                // Select features based on correlation thresholds
                var selectedFeatures = new List<string>();
                var sortedFeatures = targetCorrelations
                    .OrderByDescending(x => x.Value)
                    .Select(x => x.Key)
                    .ToList();

                _report.AppendLine("\nFeatures Ranked by Target Correlation:");
                foreach (var feature in sortedFeatures)
                {
                    _report.AppendLine($"{feature,-40} | {targetCorrelations[feature]:F4}");
                }

                // Feature selection logic
                foreach (var feature in sortedFeatures)
                {
                    if (selectedFeatures.Count >= config.MaxFeatures)
                        break;

                    bool isHighlyCorrelated = false;
                    foreach (var selectedFeature in selectedFeatures)
                    {
                        var i1 = Array.IndexOf(candidateFeatures, feature);
                        var i2 = Array.IndexOf(candidateFeatures, selectedFeature);
                        if (correlationMatrix[i1, i2] > config.MulticollinearityThreshold)
                        {
                            isHighlyCorrelated = true;
                            break;
                        }
                    }

                    if (!isHighlyCorrelated)
                    {
                        selectedFeatures.Add(feature);
                    }
                }

                _report.AppendLine($"\nSelection Summary:");
                _report.AppendLine($"Original features: {candidateFeatures.Length}");
                _report.AppendLine($"Selected features: {selectedFeatures.Count}");
                _report.AppendLine($"Multicollinearity threshold: {config.MulticollinearityThreshold}");
                _report.AppendLine("\nSelected Features:");
                foreach (var feature in selectedFeatures)
                {
                    _report.AppendLine($"- {feature} (correlation with target: {targetCorrelations[feature]:F4})");
                }

                // Create pipeline that keeps original features and creates new concatenated features
                var pipeline = _mlContext.Transforms
        .CopyColumns("OriginalFeatures", "Features")
        .Append(_mlContext.Transforms.SelectColumns(selectedFeatures.ToArray()))
        .Append(_mlContext.Transforms.Concatenate("Features", selectedFeatures.ToArray()));
                Console.WriteLine($"Creating pipeline with selected features: {string.Join(", ", selectedFeatures)}");
                var transformedData = pipeline.Fit(data).Transform(data);

                return (transformedData, selectedFeatures.ToArray(), _report.ToString());
            }
            catch (Exception ex)
            {
                _report.AppendLine($"Error during correlation analysis: {ex.Message}");
                return (data, candidateFeatures, _report.ToString());
            }
        }

        private class FeatureVector
        {
            [VectorType]
            public float[] Features { get; set; }
            public long Label { get; set; }
        }
    }
}