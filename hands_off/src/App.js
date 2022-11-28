import React, { useEffect, useRef, useState } from 'react';
import '@tensorflow/tfjs-backend-cpu';
import { initNotifications, notify } from '@mycv/f8-notification';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import {Howl} from 'howler';
import soundURL from './assets/hey_sound.mp3'
import './App.css';

var sound = new Howl({
  src: [soundURL]
});

const NOT_TOUCH_LABEL = 'not_touch';
const TOUCHED_LABEL = 'touched';
const TRAINING_TIMES = 50;
const TOUCHED_CONFIDENCE = 0.8;

function App() {
  const video = useRef();
  const mobilenetModule = useRef();
  const classifier = useRef();
  const canPlaySound = useRef(true);
  const [touched, setTouched] = useState(false);


  const init = async () => {
    console.log('init...')
    await setupCamera();
    console.log('set up camera process');

    mobilenetModule.current = await mobilenet.load();

    classifier.current = knnClassifier.create();

    console.log('set up done');
    console.log('DO NOT TOUCH ON YOUR FACE AND PRESS TRAIN 1')
    initNotifications({ cooldown: 3000 });
  }

  const setupCamera = () => {
    return new Promise((resolve, reject) => {
      navigator.getUserMedia = navigator.getUserMedia ||
      navigator.webkitGetUserMedia ||
      navigator.mozGetUserMedia ||
      navigator.msGetUserMedia;

      if (navigator.getUserMedia) {
        navigator.getUserMedia(
          { video: true },
          stream => {
            video.current.srcObject = stream;
            video.current.addEventListener('loadeddata', resolve)
          },
          error => reject(error)
        );
      } else {
        reject();
      }
    });
  }

  const train = async label => {
    console.log(`[${label}] training...`)
    for (let i=0; i < TRAINING_TIMES; i++){
      console.log(`progress ${parseInt((i+1) / TRAINING_TIMES * 100)}%`)
      await training(label);
    }
  }

  /**
   * Step 1: Train with not touching face (50 times each)
   * Step 2: Train with touching face (50 times each)
   * Step 3: Use real time camera to predict situation
   * --> If match --> warning
   * @param {*} label 
   * @returns 
   */
  const training = label => {
    return new Promise( async resolve => {
      const embedding = mobilenetModule.current.infer(
        video.current,
        true
      );
      classifier.current.addExample(embedding, label);
      await sleep(100);
      resolve();
    });
  }

  const run = async () => {
    const embedding = mobilenetModule.current.infer(
      video.current,
      true
    );
    const result = await classifier.current.predictClass(embedding);
    
    // console.log('Label: ', result.label);
    // console.log('Confidence: ', result.confidences);

      if (
        result.label === TOUCHED_LABEL &&
        result.confidences[result.label] > TOUCHED_CONFIDENCE
        ) {
          console.log('Touched');
          if (canPlaySound.current) {
            canPlaySound.current = false;
            sound.play()
          }
          sound.play()
          notify('Hands off!', { body: "You've just touched your face!"})
          setTouched(true);
        } else {
          console.log('Not touched')
          setTouched(false);
        }

    await sleep(200);

    run();
  }

  const sleep = (ms = 0) => {
    return new Promise(resolve => setTimeout(resolve, ms))
  }


  useEffect(() => {
    init();

    sound.on('end', function(){
      canPlaySound.current = true;
    });
    // clean up
    return () => {

    }
  }, []);


  return (
    <div className={`main ${touched ? 'touched': ''}`}>
      <video
      ref={video}
      className='video'
      autoPlay
      />
      <div className='control'>
        <button className='btn' onClick={()=> train(NOT_TOUCH_LABEL)}>Train 1</button>
        <button className='btn' onClick={()=> train(TOUCHED_LABEL)}>Train 2</button>
        <button className='btn' onClick={()=> run()}>Run</button>
      </div>
    </div>
  );
} 

export default App;
