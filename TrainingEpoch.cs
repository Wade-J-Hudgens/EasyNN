using System.Xml.Serialization;

namespace EasyNN {
    public class TrainingEpoch {
        [XmlArray]
        [XmlArrayItem("e")]
        public float[] ExpectedValue;
        [XmlArray]
        [XmlArrayItem("i")]
        public float[] Input;

        public TrainingEpoch(int InputLayerSize, int OutputLayerSize) {
            ExpectedValue = new float[OutputLayerSize];
            Input = new float[InputLayerSize];
        }

        public TrainingEpoch()
        {
            ExpectedValue = new float[0];
            Input = new float[0];
        }
    }
}