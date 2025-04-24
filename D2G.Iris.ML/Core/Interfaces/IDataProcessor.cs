using Microsoft.ML;
using System.Threading.Tasks;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.Core.Interfaces
{
    public interface IDataProcessor
    {
        Task<ProcessedData> ProcessData(
            MLContext mlContext,
            IDataView rawData,
            string[] enabledFields,
            ModelConfig config,
            ISqlHandler sqlHandler);
    }
}