async function run() {
  const csvUrl = "./Iris.csv";

  const trainingData = tf.data.csv(csvUrl, {
    columnConfigs: {
      Species: {
        isLabel: true,
      },
    },
  });
  //trainigData is a promise
  const numberOfFeatures = (await trainingData.columnNames()).length - 1; //pause code till trainingData is resolved
  const numberOfSamples = 150;
  const processedData = trainingData
    .map(({ xs, ys }) => {
      const labels = [
        ys.Species == "Iris-setosa" ? 1 : 0,
        ys.Species == "Iris-virginica" ? 1 : 0,
        ys.Species == "Iris-versicolor" ? 1 : 0,
      ];
      return { xs: Object.values(xs), ys: Object.values(labels) };
    })
    .batch(10);//batchsize

  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [numberOfFeatures],
      activation: "sigmoid",
      units: 5,
    })
  );
  model.add(
    tf.layers.dense({
      activation: "softmax",
      units: 3,
    })
  );
  model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.adam(0.06),
  });
  console.log(model.summary());

  await model.fitDataset(processedData, {
    epochs: 100,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch: ${epoch}\tLoss: ${logs.loss}`);
      },
    },
  }); //pause this execution thread till the promise based function model.fitDataset is resolved
  const classNames = ["setosa", "virginica", "versicolor"];
//  const testVal = tf.tensor2d([4.4, 2.9, 1.4, 0.2], [1, 4]);//setosa
  const testVal = tf.tensor2d([6.4, 3.2, 4.5, 1.5], [1, 4]);//versicolor
//  const testVal = tf.tensor2d([5.8, 2.7, 5.1, 1.9], [1, 4]);//virginica
  const prediction=model.predict(testVal)
  console.log(`Prediction: ${prediction}`)
  console.log(`Shape: ${prediction.shape}`)
  const pIndex=await tf.argMax(prediction,axis=1).data()
  console.log(pIndex)
  alert(classNames[pIndex])
}
result=run()
