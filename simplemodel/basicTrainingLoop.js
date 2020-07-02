const model=tf.sequential()
model.add(tf.layers.dense({
    units:1,
    inputShape:[1] //one is the shape of the input value the first value of a tensor shape(6 in our case) indicates number of elements
    //which is not important to the network
}))
model.compile({
    loss:'meanSquaredError',
    optimizer:'sgd'
})
console.log(model.summary())

const xs=tf.tensor2d([-1.0,0.0,1.0,2.0,3.0,4.0],[6,1]) //values and shape of 2d tensor

const ys=tf.tensor2d([-3.0,-1.0,2.0,3.0,5.0,7.0],[6,1]) //values and shape of 2d tensor
//we do this becase we don't have numpy in javascript :(

//The training process has to be asynchronous as it can take a lot of time

async function trainingLoop(model){
    const history = await model.fit(xs,ys,{ //fitting the model is also an synchronous process so we'll
                                            //assign the value to history only when that process is complete
                                            //using await
        epochs:500,
        callbacks:{
            onEpochEnd:(epoch,logs)=>{console.log(`Epoch: ${epoch}\t Loss: ${logs.loss}`)}
            //onEpochEnd is a callback that runs the function assigned to it at the end of every epoch
        }
    })
}
trainingLoop(model).then(()=>{
    alert('model trained')//send an alert that our model has been trained
    const testValue=tf.tensor2d([10],[1,1])
    console.log(`models prediction is ${model.predict(testValue)}`)
})

/* EXPLAINATION

the simplest way of defining a sequential model in tensorflow.js is as follows

const model=tf.sequential()// a sequential model kinda like model = Sequential() in python

//adding layers
model.add(tf.layers.dense({
    units:1,
    inputShape:[1]
}))// this is the same as model.add(Dense(1,input_shape=[1])) we're adding a single layer with one neuron

model.compile({
    loss:'meanSquaredError',
    optimizer:'sgd'
})//this is the same as model.compile(loss="mse",optimizer="sgd") in python

console.log(model.summary())//this is the same as writing print(model.summary())
_________________________________________________________________
layer_utils.js:152 Layer (type)                 Output shape              Param #   
layer_utils.js:64 =================================================================
layer_utils.js:152 dense_Dense1 (Dense)         [null,1]                  2         
layer_utils.js:74 =================================================================
layer_utils.js:83 Total params: 2
layer_utils.js:84 Trainable params: 2
layer_utils.js:85 Non-trainable params: 0
layer_utils.js:86 _________________________________________________________________

//DATA
const xs=tf.tensor2d([-1.0,0.0,1.0,2.0,3.0,4.0],[6,1]) //values and shape of 2d tensor
it could also have been 3 rows and two columns
const xs=tf.tensor2d([-1.0,0.0,1.0,2.0,3.0,4.0],[3,2]) //values and shape of 2d tensor
but we're gonna stick with the first one

these are the values that our xs have to be mapped to using the neural network
const ys=tf.tensor2d([-3.0,-1.0,2.0,3.0,5.0,7.0],[6,1]) //values and shape of 2d tensor

rest of the code is explained above!

*/
