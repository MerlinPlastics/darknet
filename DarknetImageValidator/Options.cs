using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;

namespace DarknetImageValidator
{
	internal class Options
	{
		[Option('c', "config", HelpText = "The input .cfg file")]
		public string ConfigFile { get; set; }

		[Option('w', "weights", HelpText = "The input .weights file")]
		public string WeightsFile { get; set; }

		[Option('d', "data", HelpText = "The input .data file")]
		public string DataFile { get; set; }

		//[Option('b', "base", HelpText = "The base path to work from for the ")]
	}
}
