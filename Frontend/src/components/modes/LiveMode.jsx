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
    } = useSignRecognition();
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

                        {/* Demo instruction */}
                        {isStreaming && (
                            <div className="glass-card p-4 text-center">
                                <p className="text-sm text-muted-foreground">
                                    <strong>Demo Mode:</strong> Click a letter below to simulate detection
                                </p>
                            </div>
                        )}
                    </div>

                    {/* Result Section */}
                    <div className="space-y-4">
                        <TamilLetterDisplay
                            letter={detectedLetter}
                            onSpeak={handleSpeak}
                            isSpeaking={isSpeaking}
                            showDetails={true}
                        />
                    </div>
                </div>

                {/* Letter Grid for Demo */}
                <div className="mt-8">
                    <h3 className="text-lg font-semibold text-foreground mb-4 text-center">
                        Click a letter to simulate detection
                    </h3>
                    <LetterGrid
                        onLetterClick={handleLetterClick}
                        selectedLetterId={detectedLetter?.id}
                        showHints={true}
                    />
                </div>
            </div>
        </div>
    );
};
