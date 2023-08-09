import React from 'react';
import * as ort from 'onnxruntime-web';


const W = 46;
const H = 46;
const MODEL_NAME = 'short40_default';

const angleArr = Float64Array.from([0]);
const angle = new ort.Tensor('float64', angleArr, [1]);


type OriginalConfig = {
  name: string,
  image: string,
  CH_ALL: number,
  image_pad: number,
  batch_size: number,
  betas: number[],
  damage_samples_in_batch: 3,
  fire_rate: number,
  hidden_size: number,
  lr: number,
  lr_gamma: number,
  n_epoch: number,
  pool_size: number,
  steps_max: number,
  steps_min: number,
}


function App() {
  const sessionRef = React.useRef<ort.InferenceSession>();
  const [state, setState] = React.useState<ort.Tensor>();
  const [vmin, setVmin] = React.useState<number>(0);

  const isMouseDownRef = React.useRef<boolean>(false);
  const mouseEventsRef = React.useRef<React.MouseEvent[]>([]);

  const configRef = React.useRef<OriginalConfig>();

  const canvasRatio = 0.8;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const offsetX = Math.floor((vw - vmin * canvasRatio)/2);
  const offsetY = Math.floor((vh - vmin * canvasRatio)/2);


  React.useEffect(() => {
    // Resize
    const onresize = () => {
      setVmin(Math.min(window.innerHeight, window.innerWidth));
    };
    onresize();
    window.addEventListener('resize', onresize);

    // Init Model
    const init = async () => {
      try {
        // Get config
        const response = await fetch(process.env.PUBLIC_URL + `/weights/${MODEL_NAME}.json`);
        const cfg = await response.json() as OriginalConfig;
        configRef.current = cfg;
        const CH = cfg.CH_ALL;
        console.log('Config:', cfg);

        // Create session
        const session = await ort.InferenceSession.create(process.env.PUBLIC_URL + `/weights/${MODEL_NAME}.onnx`);

        sessionRef.current = session;
        console.log(session.inputNames, session.outputNames);

        // Initial state
        const stateArr = new Float32Array(W*H*CH);
        for (let i=3; i<CH; i++)
          stateArr[Math.floor(H/2)*W*CH + Math.floor(W/2)*CH+i] = 1;
        const state0 = new ort.Tensor('float32', stateArr, [1, H, W, CH]);

        setState(state0);

      } catch (e) {
          document.write(`failed to inference ONNX model: ${e}.`);
      }
    }
    init();

    return () => {
      window.removeEventListener('resize', onresize);
    };
  }, []);


  React.useEffect(() => {
    if (!state || !sessionRef.current) return;
    const session = sessionRef.current;

    (async () => {
      await new Promise((resolve) => setTimeout(resolve, 10));
      let s = state;
      const CH = configRef.current!.CH_ALL;
      const t0 = Date.now()

      // Damage
      const es = mouseEventsRef.current;
      const dotSizeX = vmin * canvasRatio / W;
      const dotSizeY = vmin * canvasRatio / H;
      es.forEach(e => {
        const mx = (e.clientX-offsetX)/dotSizeX-0.5;
        const my = (e.clientY-offsetY)/dotSizeY-0.5;
        const array = state.data as Float32Array;
        const r = 2;

        for (let y=0; y<H; y++) {
          for (let x=0; x<W; x++) {
            if ((mx-x)*(mx-x) + (my-y)*(my-y) <= r*r) {
              for (let ch=0; ch<CH; ch++)
                array[y*W*CH+x*CH+ch] = 0;
            }
          }
        }
        s = new ort.Tensor('float32', array, [1, H, W, CH]);
      });
      mouseEventsRef.current = [];

      // Evolve
      const feeds = { 'x.1': s, 'angle': angle };
      const results = await session.run(feeds);
      const stateArr_ = results[89].data as Float32Array;

      setState(new ort.Tensor('float32', stateArr_, [1, H, W, CH]));
    })();
  }, [state]);


  const handleMouseDown = (e: React.MouseEvent) => {
    isMouseDownRef.current = true;
  }
  const handleMouseUp = (e: React.MouseEvent) => {
    isMouseDownRef.current = false;
  }
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isMouseDownRef.current) return;
    mouseEventsRef.current.push(e);
  }


  const _render = () => {
    if (!state) return undefined;
    const array = state.data as Float32Array;
    const CH = configRef.current!.CH_ALL;
    const dotSizeX = vmin * canvasRatio / W;
    const dotSizeY = vmin * canvasRatio / H;

    const res = [];
    for (let y=0; y<H; y++) {
      for (let x=0; x<W; x++) {
        let r = array[y*W*CH+x*CH+0];
        let g = array[y*W*CH+x*CH+1];
        let b = array[y*W*CH+x*CH+2];
        const a = array[y*W*CH+x*CH+3];
        r = Math.max(0, Math.min(1-a+r, 0.999)) * 255;
        g = Math.max(0, Math.min(1-a+g, 0.999)) * 255;
        b = Math.max(0, Math.min(1-a+b, 0.999)) * 255;
        r = Math.floor(r);
        g = Math.floor(g);
        b = Math.floor(b);

        let left = Math.floor(dotSizeX*x);
        let top = Math.floor(dotSizeY*y);
        let w = Math.floor(dotSizeX*(x+1)) - left;
        let h = Math.floor(dotSizeY*(y+1)) - top;
        left += offsetX;
        top += offsetY;

        res.push(
          <div key={y*W+x} style={{
            position: 'absolute',
            width: `${w}px`,
            height: `${h}px`,
            top: `${top}px`,
            left: `${left}px`,
            backgroundColor: `rgb(${r},${g},${b})`
          }}></div>
        )
      }
    }
    return res;
  }


  return (
    <div className="App" onMouseDown={handleMouseDown} onMouseUp={handleMouseUp} onMouseMove={handleMouseMove} style={{width: '100vw', height: '100vh'}}>
      {_render()}
    </div>
  );
}

export default App;
