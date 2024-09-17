using System.Diagnostics;
using System.Runtime.InteropServices;
using CommandLine;
using DarknetDotnet;

namespace DarknetImageValidator
{
	internal class Program
	{
		static void Main(string[] args)
		{
			var parsed = Parser.Default.ParseArguments<Options>(args);

			parsed.WithParsed<Options>(options =>
			{
				// Read the .data file to find

				Dictionary<string, string> dataValues = File.ReadAllLines(options.DataFile)
					.Select(q => q.Split('=', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
					.Where(q => q.Length == 2)
					.ToDictionary(q => q[0], q => q[1])
				;

				var testFile = dataValues["valid"];
				var namesFile = dataValues["names"];

				var validator = new Validator(options.ConfigFile, options.WeightsFile, namesFile, 0);


				var lines = File.ReadAllLines(testFile);
				var stats = new Stats(validator.ClassNames.Keys);

				int index = 0;
				foreach (var imgFile in lines)
				{
					var txtFile = Path.ChangeExtension(imgFile, ".txt");

					var newStats = validator.ValidateImage(".", imgFile, txtFile, 0.5f, 0.5f);
					//validator.PrintStats(newStats);


					foreach (var classId in newStats.ClassIds)
					{
						int fn = newStats.TruthsPerClass[classId] - newStats.TruePositivePerClass[classId] ;
						if (fn > 0)
						{
							Console.WriteLine($"{fn} FN for file '{imgFile}'");
						}
					}

					stats.Combine(newStats);
					index++;
					Console.Write($"Processed {index} of {lines.Length}\r");

				}

				validator.PrintStats(stats);

			});


		}
	}
}
