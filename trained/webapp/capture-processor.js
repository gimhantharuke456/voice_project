'use strict';

// Runs on the audio render thread. Forwards raw mic samples to the main
// thread in native 128-sample render quanta (~2.7ms at 48kHz) — batching
// into larger hops happens in live.js, off the audio thread.
class CaptureProcessor extends AudioWorkletProcessor {
    process(inputs) {
        const input = inputs[0];
        if (input && input[0] && input[0].length > 0) {
            // Float32Array must be copied — the underlying buffer is reused
            // by the audio engine on the next render quantum.
            this.port.postMessage(input[0].slice(0));
        }
        return true;
    }
}

registerProcessor('capture-processor', CaptureProcessor);
