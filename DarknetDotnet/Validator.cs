using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace DarknetDotnet
{
	public class Validator //: YoloDetector
	{
		private YoloDetector detector;
		private Dictionary<int, string> classNames;

		public Dictionary<int, string> ClassNames
		{
			get { return classNames; }
		}

		public Validator(string configFile, string weightsFile, int gpu, Dictionary<int, string> classNames)
		{
			detector = new YoloDetector(configFile, weightsFile, gpu, classNames);
			this.classNames = classNames;
		}

		public Validator(string configFile, string weightsFile, string namesFile, int gpu)
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

			detector = new YoloDetector(configFile, weightsFile, gpu, classNames);
			this.classNames = classNames;

		}

		public Stats ValidateImage(string basePath, string imageFile, string truthsFile, float confidenceThresh = 0.5f, float iouThresh = 0.25f)
		{
			var stats = new Stats(classNames.Keys);

			var mat = Cv2.ImRead(imageFile, ImreadModes.Color);


			var truths = File.ReadAllLines(truthsFile)
				.Select((l, index) => TruthBox.Parse(l, index))
				.Where(t => t != null)
				.Select(t => new TruthBox() // Have to adjust TruthBox to match the detections output
				{
					IndexId = t.IndexId,
					ClassId = t.ClassId,
					X = mat.Width * (t.X - t.Width / 2),
					Y = mat.Height * (t.Y - t.Height / 2),
					Width = mat.Width * t.Width,
					Height = mat.Height * t.Height
				})
				.Cast<TruthBox>().ToList() ?? new List<TruthBox>();


			// Total number of truths for this class
			//foreach (var classId in classNames.Keys)
			foreach (var truth in truths)
				stats.TruthsPerClass[truth.ClassId]++;


			var detections = detector.Detect(mat, 0.005f);   // detector.cpp line 668

			// For each detection box
			int detectionIndex = 0;
			foreach (var detection in detections)
			{
				if (detection.Confidence <= 0) continue;    // Should never happen
				stats.DetectionsPerClass[detection.ObjectTypeId]++;
				stats.TotalDetections++;

				// Match detection with all truths of the same class, and find the highest IOU 
				var bestMatches = truths.Where(t => t!.ClassId == detection.ObjectTypeId)
					.Select(truth => new { truth, iou = truth!.Rectangle.IOU(detection.Rectangle) })
					.OrderByDescending(t => t.iou)      // Highest IOU at top
					.Where(t => t.iou > iouThresh)      // Only consider those of sufficent IOU
														//.ToList()
					;

				var bestMatch = bestMatches?.FirstOrDefault();

				// remove this truth from future considerations
				if (bestMatch != null /*&& bestMatch.iou > iouThresh*/)
					truths.Remove(bestMatch.truth);

				// If we exceed the confidence threshold, increment TP or FP counters as appropriate
				if (detection.Confidence > confidenceThresh)
				{

					// We found a match from the remaining available truths
					if (bestMatch != null)
					{
						stats.TruePositivePerClass[detection.ObjectTypeId]++;
						stats.TotalIOUPerClass[detection.ObjectTypeId] += bestMatch.iou;
					}
					else
					{
						// Detected something that matched no (available) truths.  This is a false-positive
						stats.FalsePositivePerClass[detection.ObjectTypeId]++;
					}
				}

			}

			return stats;

		}

		public void ValidateImages(string basePath, List<string> imageFiles, List<string> truthsFiles,
			float confidenceThresh = 0.5f, float iouThresh = 0.25f)
		{
			var sources = imageFiles.Zip(imageFiles, truthsFiles);

			HashSet<Stats> allStats = new HashSet<Stats>();
			Stats? aggregateStats = null;
			foreach (var source in sources)
			{
				var stats = this.ValidateImage(basePath, source.First, source.Second, confidenceThresh, iouThresh);
				allStats.Add(stats);
			}
		}

		public void PrintStats(Stats stats)
		{
			Console.WriteLine("  Id Name             AvgPrecision     TP     FN     FP     TN Accuracy ErrorRate Precision Recall Specificity FalsePosRate");
			Console.WriteLine("  -- ----             ------------ ------ ------ ------ ------ -------- --------- --------- ------ ----------- ------------");

			foreach (var classId in stats.ClassIds)
			{
				var line = BuildStatsLine(classId, classNames[classId], stats.DetectionsPerClass[classId], 0
					, stats.TruthsPerClass[classId], stats.TruePositivePerClass[classId], stats.FalsePositivePerClass[classId]);
				Console.WriteLine(line);
			}
		}

		string BuildStatsLine(int classId, string className
			, int allDetections, double averagePrecision
			, int truths, int tp, int fp
			)
		{
			int fn = truths - tp;
			int tn = allDetections - tp - fn - fp;
			double accuracy = allDetections == 0 ? 0 : (tp + tn) / (1.0 * allDetections);   // Dumb math tricks to convert to double
			double errorRate = allDetections == 0 ? 0 : (fp + fn) / (1.0 * allDetections);
			double precision = ((tp + fp) == 0) ? 0 : tp / (1.0 * tp + fp);
			double recall = ((tp + fn) == 0) ? 0 : tp / (1.0 * tp + fn);
			double specificity = ((tn + fp) == 0) ? 0 : tn / (1.0 * tn + fp);
			double falsePositiveRate = ((tn + fp) == 0) ? 0 : fp / (1.0 * tn + fp);

			string line = $"  {classId,2} {className,16} {averagePrecision,-12:N4} " +
						  $"{tp,6} {fn,6} {fp,6} {tn,6} " +
						  $"{accuracy,8:N4} {errorRate,9:N4} " +
						  $"{precision,9:N4} {recall,6:N4} " +
						  $"{specificity,-11:N4} {falsePositiveRate,12:N4}"
				;
			return line;
		}
	}




}
