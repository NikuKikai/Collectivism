import React from 'react';
import * as ort from 'onnxruntime-web';


const angleArr = Float64Array.from([0]);
const angle = new ort.Tensor('float64', angleArr, [1]);


function App() {
  const sessionRef = React.useRef<ort.InferenceSession>();
  const [state, setState] = React.useState<ort.Tensor>();
  const [vmin, setVmin] = React.useState<number>(0);

  const isMouseDownRef = React.useRef<boolean>(false);
  const mouseEventsRef = React.useRef<React.MouseEvent[]>([]);

  const canvasRatio = 0.8;
  const dotSize = vmin * canvasRatio / 32;
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
        // Create session
        const session = await ort.InferenceSession.create( process.env.PUBLIC_URL + '/weights/original_ch16_step48-64.pth.onnx');

        sessionRef.current = session;
        console.log(session.inputNames, session.outputNames);

        // Initial state
        const stateArr = new Float32Array(32 * 32 * 16);
        for (let i=3; i<16; i++)
          stateArr[15*32*16+15*16+i] = 1;
        const state0 = new ort.Tensor('float32', stateArr, [1, 32, 32, 16]);

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
      await new Promise((resolve) => setTimeout(resolve, 30));
      let s = state;

      // Damage
      const es = mouseEventsRef.current;
      es.forEach(e => {
        const mx = (e.clientX-offsetX)/dotSize-0.5;
        const my = (e.clientY-offsetY)/dotSize-0.5;
        const array = state.data as Float32Array;
        const r = 2;

        for (let y=0; y<32; y++) {
          for (let x=0; x<32; x++) {
            if ((mx-x)*(mx-x) + (my-y)*(my-y) <= r*r) {
              for (let ch=0; ch<16; ch++)
                array[y*32*16+x*16+ch] = 0;
            }
          }
        }
        s = new ort.Tensor('float32', array, [1, 32, 32, 16]);
      });
      mouseEventsRef.current = [];

      // Evolve
      const feeds = { 'x.1': s, 'angle': angle };
      const results = await session.run(feeds);
      const stateArr_ = results[89].data as Float32Array;

      setState(new ort.Tensor('float32', stateArr_, [1, 32, 32, 16]));
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

    const res = [];
    for (let y=0; y<32; y++) {
      for (let x=0; x<32; x++) {
        let r = array[y*32*16+x*16+0];
        let g = array[y*32*16+x*16+1];
        let b = array[y*32*16+x*16+2];
        const a = array[y*32*16+x*16+3];
        r = Math.max(0, Math.min(1-a+r, 0.999)) * 255;
        g = Math.max(0, Math.min(1-a+g, 0.999)) * 255;
        b = Math.max(0, Math.min(1-a+b, 0.999)) * 255;
        r = Math.floor(r);
        g = Math.floor(g);
        b = Math.floor(b);

        let left = Math.floor(dotSize*x);
        let top = Math.floor(dotSize*y);
        let w = Math.floor(dotSize*(x+1)) - left;
        let h = Math.floor(dotSize*(y+1)) - top;
        left += offsetX;
        top += offsetY;

        res.push(
          <div key={y*32+x} style={{
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
