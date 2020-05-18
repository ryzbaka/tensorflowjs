async function doTraining(model) {
  const history = await model.fit(xs, ys, {
    epochs: 500,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch: ${epoch}\tLoss:${logs.loss}`);
      },
    },
  });
} //actual training is asynchronous as it takes an indeterminate
//amount of time.

const xs = tf.tensor2d([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], [6, 1]);
const ys = tf.tensor2d([-3.0, -1.0, 0.0, 3.0, 5.0, 7.0], [6, 1]);
//target function is f(x)=2x-1
console.log(xs);
console.log(ys);

const model = tf.sequential();

model.add(
  tf.layers.dense({
    units: 1,
    inputShape: [1], //trainx.shape[1:]
  })
);

model.compile({
  loss: "meanSquaredError",
  optimizer: "sgd",
});

console.log(model.summary());

doTraining(model).then(() => {
  alert(model.predict(tf.tensor2d([10, 2], [2, 1])));
  //[values_per_batch,...],[batches,batch_size]
});
