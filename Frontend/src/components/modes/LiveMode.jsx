import { useEffect } from 'react';
import { CameraFeed } from '@/components/CameraFeed';
import { TamilLetterDisplay } from '@/components/TamilLetterDisplay';
import { LetterGrid } from '@/components/LetterGrid';
import { useWebcam } from '@/hooks/useWebcam';
import { useSignRecognition } from '@/hooks/useSignRecognition';
import { useSpeech } from '@/hooks/useSpeech';

export const LiveMode = () => {
    const { videoRef, isStreaming, error, startCamera, stopCamera } = useWebcam();
    const {
        detectedLetter,
        startDetection,
        stopDetection,
        simulateDetection,
        isHandPresent
    } = useSignRecognition(videoRef);
    const { speak, isSpeaking } = useSpeech();

    // Start detection when camera starts
    useEffect(() => {
        if (isStreaming) {
            startDetection();
        } else {
            stopDetection();
        }
    }, [isStreaming, startDetection, stopDetection]);

    // Speak when letter is detected
    useEffect(() => {
        if (detectedLetter && !isSpeaking) {
            speak(detectedLetter.letter);
        }
    }, [detectedLetter, speak, isSpeaking]);

    const handleSpeak = () => {
        if (detectedLetter) {
            speak(detectedLetter.letter);
        }
    };

    const handleLetterClick = (letter) => {
        simulateDetection(letter.id);
    };

    const getCameraStatus = () => {
        if (!isStreaming) return 'idle';
        if (detectedLetter) return 'success';
        if (isHandPresent) return 'detecting';
        return 'idle';
    };

    return (
        <div className="container mx-auto px-4 py-6">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-2">
                        Live Sign to Speech
                    </h2>
                    <p className="text-muted-foreground">
                        Show a Tamil sign to your camera and hear it spoken aloud
                    </p>
                </div>

                <div className="grid lg:grid-cols-2 gap-6">
                    {/* Camera Section */}
                    <div className="space-y-4">
                        <CameraFeed
                            ref={videoRef}
                            isStreaming={isStreaming}
                            error={error}
                            onStart={startCamera}
                            onStop={stopCamera}
                            status={getCameraStatus()}
                        />

                        {/* Demo instruction removed */}
                    </div>

                    {/* Result Section */}
                    <div className="space-y-4">
                        {detectedLetter ? (
                            <TamilLetterDisplay
                                letter={detectedLetter}
                                onSpeak={handleSpeak}
                                isSpeaking={isSpeaking}
                                showDetails={true}
                            />
                        ) : (
                            <div className="flex flex-col items-center justify-center h-full min-h-[300px] glass-card p-8 text-center animate-in fade-in-50">
                                {isStreaming ? (
                                    <>
                                        {isHandPresent ? (
                                            <>
                                                <span className="text-4xl mb-4 animate-pulse">⏳</span>
                                                <h3 className="text-xl font-semibold mb-2">Processing...</h3>
                                                <p className="text-muted-foreground">
                                                    Analyzing your sign. Hold steady!
                                                </p>
                                            </>
                                        ) : (
                                            <>
                                                <span className="text-4xl mb-4">✋</span>
                                                <h3 className="text-xl font-semibold mb-2">No Hand Detected</h3>
                                                <p className="text-muted-foreground">
                                                    Bring your hand into the camera view to start translating.
                                                </p>
                                            </>
                                        )}
                                    </>
                                ) : (
                                    <>
                                        <span className="text-4xl mb-4">📷</span>
                                        <h3 className="text-xl font-semibold mb-2">Camera Off</h3>
                                        <p className="text-muted-foreground">
                                            Start the camera to begin sign recognition.
                                        </p>
                                    </>
                                )}
                            </div>
                        )}
                    </div>
                </div>

            </div>
        </div>
    );
};
