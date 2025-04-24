using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.DataBalancing
{
    public class SmoteDataBalancer 
    {
        public async Task<IDataView> BalanceDataset(
            MLContext mlContext,
            IDataView data,
            string[] featureNames,
            DataBalancingConfig config,
            string targetField)
        {
            ValidateConfig(config);
            Console.WriteLine("=============== Balancing Dataset with SMOTE ===============");

            // Convert IDataView to enumerable for SMOTE processing
            var dataEnumerable = mlContext.Data.CreateEnumerable<FeatureVector>(
                data, reuseRowObject: false);

            var minorityClass = new List<float[]>();
            var majorityClass = new List<float[]>();

            // Separate minority and majority classes
            foreach (var row in dataEnumerable)
            {
                if (row.Label == 1) // Instead of if(row.Label)
                    minorityClass.Add(row.Features);
                else
                    majorityClass.Add(row.Features);
            }

            // Ensure minority class is the smaller one
            if (minorityClass.Count > majorityClass.Count)
            {
                var temp = minorityClass;
                minorityClass = majorityClass;
                majorityClass = temp;
            }

            // Undersample majority class
            var random = new Random(42);
            int undersampledMajorityCount = (int)(majorityClass.Count * config.UndersamplingRatio);
            var undersampledMajority = ShuffleMajorityClass(majorityClass, undersampledMajorityCount, random);

            // Generate synthetic samples
            int targetMinorityCount = (int)(undersampledMajorityCount * config.MinorityToMajorityRatio);
            int syntheticCount = Math.Max(0, targetMinorityCount - minorityClass.Count);

            var syntheticSamples = await GenerateSyntheticSamples(
                minorityClass,
                syntheticCount,
                config.KNeighbors,
                random);

            // Combine all samples
            var balancedFeatures = new List<FeatureVector>();
            balancedFeatures.AddRange(undersampledMajority.Select(f => new FeatureVector { Features = f, Label = 0 }));
            balancedFeatures.AddRange(minorityClass.Select(f => new FeatureVector { Features = f, Label = 1 }));
            balancedFeatures.AddRange(syntheticSamples.Select(f => new FeatureVector { Features = f, Label = 1 }));

            // Convert back to IDataView
            return mlContext.Data.LoadFromEnumerable(balancedFeatures);
        }

        private void ValidateConfig(DataBalancingConfig config)
        {
            if (config.UndersamplingRatio <= 0 || config.UndersamplingRatio > 1)
                throw new ArgumentException("Undersampling ratio must be between 0 and 1");

            if (config.MinorityToMajorityRatio <= 0 || config.MinorityToMajorityRatio > 1)
                throw new ArgumentException("Minority to majority ratio must be between 0 and 1");

            if (config.KNeighbors < 1)
                throw new ArgumentException("K should be greater than 0");
        }

        private List<float[]> ShuffleMajorityClass(List<float[]> samples, int targetCount, Random random)
        {
            var indices = Enumerable.Range(0, samples.Count).ToList();
            int n = indices.Count;

            while (n > 1)
            {
                n--;
                int k = random.Next(n + 1);
                int temp = indices[k];
                indices[k] = indices[n];
                indices[n] = temp;
            }

            return indices.Take(targetCount)
                         .Select(i => samples[i])
                         .ToList();
        }

        private async Task<List<float[]>> GenerateSyntheticSamples(
            List<float[]> minoritySamples,
            int syntheticCount,
            int k,
            Random random)
        {
            if (syntheticCount <= 0) return new List<float[]>();

            var synthetic = new List<float[]>();
            var samplesPerInstance = (int)Math.Ceiling((double)syntheticCount / minoritySamples.Count);

            await Task.Run(() =>
            {
                for (int i = 0; i < minoritySamples.Count && synthetic.Count < syntheticCount; i++)
                {
                    var neighbors = FindKNearestNeighbors(minoritySamples, minoritySamples[i], i, k);

                    for (int j = 0; j < samplesPerInstance && synthetic.Count < syntheticCount; j++)
                    {
                        var neighborIdx = random.Next(neighbors.Length);
                        var syntheticSample = InterpolateFeatures(
                            minoritySamples[i],
                            minoritySamples[neighbors[neighborIdx]],
                            random);
                        synthetic.Add(syntheticSample);
                    }
                }
            });

            return synthetic;
        }

        private int[] FindKNearestNeighbors(List<float[]> samples, float[] target, int excludeIndex, int k)
        {
            var distances = new List<(int index, float distance)>();

            for (int i = 0; i < samples.Count; i++)
            {
                if (i == excludeIndex) continue;
                distances.Add((i, EuclideanDistance(samples[i], target)));
            }

            return distances.OrderBy(x => x.distance)
                          .Take(k)
                          .Select(x => x.index)
                          .ToArray();
        }

        private float EuclideanDistance(float[] a, float[] b)
        {
            float sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                float diff = a[i] - b[i];
                sum += diff * diff;
            }
            return MathF.Sqrt(sum);
        }

        private float[] InterpolateFeatures(float[] a, float[] b, Random random)
        {
            float ratio = (float)random.NextDouble();
            var result = new float[a.Length];

            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] + ratio * (b[i] - a[i]);
            }

            return result;
        }

        private class FeatureVector
        {
            [VectorType]
            public float[] Features { get; set; }
            public long Label { get; set; }
        }
    }
}