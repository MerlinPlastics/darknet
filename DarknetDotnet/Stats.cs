using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace DarknetDotnet
{
	public class Stats
	{
		public Stats(IEnumerable<int> classIds)
		{
			ClassIds = new List<int>(classIds);
			DetectionsPerClass = new Dictionary<int, int>();
			TruthsPerClass = new Dictionary<int, int>();
			TruePositivePerClass = new Dictionary<int, int>();
			FalsePositivePerClass = new Dictionary<int, int>();
			TotalIOUPerClass = new Dictionary<int, double>();

			foreach (var classId in classIds)
			{
				DetectionsPerClass.Add(classId, 0);
				TruthsPerClass.Add(classId, 0);
				TruePositivePerClass.Add(classId, 0);
				FalsePositivePerClass.Add(classId, 0);
				TotalIOUPerClass.Add(classId, 0);
			}
		}

		public int TotalDetections { get; set; }
		public List<int> ClassIds { get; set; }

		public Dictionary<int, int> DetectionsPerClass { get; private set; }
		public Dictionary<int, int> TruthsPerClass { get; private set; }
		public Dictionary<int, int> TruePositivePerClass { get; private set; }
		public Dictionary<int, int> FalsePositivePerClass { get; private set; }

		public Dictionary<int, double> TotalIOUPerClass { get; private set; }

		public Stats Combine(Stats value)
		{
			if (value == null)
				throw new ArgumentNullException(nameof(value));

			// Assme everythign is valid...

			// Then match and add or copy
			if (ClassIds == null) ClassIds = value.ClassIds;

			TotalDetections += value.TotalDetections;

			CombineDictionary(DetectionsPerClass, value.DetectionsPerClass);
			CombineDictionary(TruthsPerClass, value.TruthsPerClass);
			CombineDictionary(TruePositivePerClass, value.TruePositivePerClass);
			CombineDictionary(FalsePositivePerClass, value.FalsePositivePerClass);
			CombineDictionary(TotalIOUPerClass, value.TotalIOUPerClass);

			return this;
		}


		private void CombineDictionary<T, U>(Dictionary<T, U> target, Dictionary<T, U> value)
			where T : notnull
			where U : notnull, INumber<U>
		{
			foreach (var key in value.Keys)
			{
				//	if (!target.ContainsKey(key))
				//		target[key] = value[key];
				//	else
				target[key] += value[key];
			}
		}
	}
}
