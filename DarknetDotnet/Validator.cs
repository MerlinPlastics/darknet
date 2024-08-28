using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DarknetDotnet
{
	public class Validator : YoloDetector
	{
		public Validator(string configFile, string weightsFile, int gpu, Dictionary<int, string> classNames)
			: base(configFile, weightsFile, gpu, classNames)
		{
		}

		public Validator(string configFile, string weightsFile, string namesFile, int gpu)
		: base(configFile, weightsFile, gpu, new Dictionary<int, string>())
		{
			if (!Path.Exists(namesFile))
				throw new FileNotFoundException($"Unable to find .names file {namesFile}");

			var lines = File.ReadAllLines(namesFile);
			if (lines.Length == 0)
				throw new InvalidDataException("Names file is empty");

			var classNames = File.ReadAllLines(namesFile)
				.Select(q => q.Split('#', StringSplitOptions.TrimEntries)[0])
				.Where(q => !string.IsNullOrWhiteSpace(q))
				.Select((value, index) => new { index, value })
				.ToDictionary(q => q.index, q => q.value);

			if (classNames.Count == 0)
				throw new InvalidDataException(
					"No usable names in the names file.  There must be one entry per line.  Use # for the first character of a line to comment the entire line, which are ignored.");

			base.ClassNames = classNames;

		}



	}

	class TruthBox
	{

	}
}
