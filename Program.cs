
using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace AgePrediction
{
    class PersonData
    {
        [LoadColumn(0)] public string? Name;
        [LoadColumn(1)] public string? Place;
        [LoadColumn(2)] public string? Gender;
        [LoadColumn(3)] public string? DateOfBirth;
        [LoadColumn(4)] public float Height;
        [LoadColumn(5)] public float Weight;
        [LoadColumn(7)] public float Age;
    }

    class AgePredictionResult
    {
        [ColumnName("Score")]
        public float PredictedAge;
    }

    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            var dataPath = "new.csv"; // Make sure this file is in the same folder as the executable
            var dataView = mlContext.Data.LoadFromTextFile<PersonData>(dataPath, separatorChar: ',', hasHeader: true);

            var trainTestData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding("Place")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Gender"))
                .Append(mlContext.Transforms.Text.FeaturizeText("DateOfBirth"))
                .Append(mlContext.Transforms.Concatenate("Features", "Place", "Gender", "DateOfBirth", "Height", "Weight"))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.Transforms.CopyColumns("Label", "Age"));

            var estimator = dataProcessPipeline.Append(mlContext.Regression.Trainers.LbfgsPoissonRegression());

            var model = estimator.Fit(trainTestData.TrainSet);

            var predictions = model.Transform(trainTestData.TestSet);
            var metrics = mlContext.Regression.Evaluate(predictions);

            Console.WriteLine($"Mean Squared Error: {metrics.MeanSquaredError}");
            Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");

            var predictionEngine = mlContext.Model.CreatePredictionEngine<PersonData, AgePredictionResult>(model);

            var sampleData = new PersonData
            {
                Name = "David Kim",
                Place = "South Korea",
                DateOfBirth = "11/16/86",
                Gender = "Male",
                Height = 175,
                Weight = 70
            };

            var prediction = predictionEngine.Predict(sampleData);
            Console.WriteLine($"Predicted Age for {sampleData.Name}: {prediction.PredictedAge}");
        }
    }
}
