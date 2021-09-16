using System.Collections.Generic;
using System.Xml.Serialization;

namespace EasyNN {
    [XmlRoot]
    public class TrainingData {
        public List<TrainingEpoch> TrainingEpochs = new List<TrainingEpoch>(0);

        public TrainingData()
        {

        }
    }
}