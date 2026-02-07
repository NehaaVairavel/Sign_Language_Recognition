import { forwardRef } from 'react';
import { Camera, CameraOff, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export const CameraFeed = forwardRef(
    ({ isStreaming, error, onStart, onStop, status = 'idle', className }, ref) => {

        return (
            <div className={cn("relative", className)}>
                <div
                    className={cn(
                        "camera-container aspect-video bg-muted flex items-center justify-center",
                        status === 'detecting' && 'detecting',
                        status === 'success' && 'success',
                        status === 'error' && 'error'
                    )}
                >
                    {isStreaming ? (
                        <video
                            ref={ref}
                            autoPlay
                            playsInline
                            muted
                            className="w-full h-full object-cover rounded-2xl transform -scale-x-100"
                        />
                    ) : error ? (
                        <div className="flex flex-col items-center gap-4 p-8 text-center">
                            <div className="w-16 h-16 rounded-full bg-destructive/10 flex items-center justify-center">
                                <AlertCircle className="w-8 h-8 text-destructive" />
                            </div>
                            <div>
                                <p className="font-medium text-foreground mb-1">Camera Error</p>
                                <p className="text-sm text-muted-foreground max-w-xs">{error}</p>
                            </div>
                            <Button onClick={onStart} variant="outline" size="sm">
                                Try Again
                            </Button>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center gap-4 p-8 text-center">
                            <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center animate-pulse-scale">
                                <Camera className="w-10 h-10 text-primary" />
                            </div>
                            <div>
                                <p className="font-medium text-foreground mb-1">Camera Ready</p>
                                <p className="text-sm text-muted-foreground">
                                    Click to start your camera
                                </p>
                            </div>
                            <Button onClick={onStart} variant="hero" size="lg">
                                <Camera className="w-5 h-5" />
                                Start Camera
                            </Button>
                        </div>
                    )}
                </div>

                {/* Camera controls overlay */}
                {isStreaming && (
                    <div className="absolute bottom-4 left-1/2 -translate-x-1/2">
                        <Button
                            onClick={onStop}
                            variant="secondary"
                            size="sm"
                            className="backdrop-blur-sm bg-background/80"
                        >
                            <CameraOff className="w-4 h-4" />
                            Stop Camera
                        </Button>
                    </div>
                )}

                {/* Status indicator */}
                {isStreaming && status !== 'idle' && (
                    <div className="absolute top-4 left-4">
                        <div className={cn(
                            "flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium backdrop-blur-sm",
                            status === 'detecting' && "bg-primary/20 text-primary",
                            status === 'success' && "bg-success/20 text-success",
                            status === 'error' && "bg-destructive/20 text-destructive"
                        )}>
                            <span className={cn(
                                "w-2 h-2 rounded-full animate-pulse",
                                status === 'detecting' && "bg-primary",
                                status === 'success' && "bg-success",
                                status === 'error' && "bg-destructive"
                            )} />
                            {status === 'detecting' && 'Detecting...'}
                            {status === 'success' && 'Sign Detected!'}
                            {status === 'error' && 'No Sign Detected'}
                        </div>
                    </div>
                )}
            </div>
        );
    }
);

CameraFeed.displayName = 'CameraFeed';
