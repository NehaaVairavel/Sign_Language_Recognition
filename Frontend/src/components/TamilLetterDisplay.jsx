import { Volume2, VolumeX } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export const TamilLetterDisplay = ({
    letter,
    onSpeak,
    isSpeaking = false,
    showDetails = true,
    size = 'lg',
    className
}) => {
    if (!letter) {
        return (
            <div className={cn(
                "elevated-card p-8 flex flex-col items-center justify-center min-h-[200px]",
                className
            )}>
                <p className="text-muted-foreground text-center">
                    Show a sign to see the Tamil letter
                </p>
            </div>
        );
    }

    return (
        <div className={cn(
            "elevated-card p-6 sm:p-8 flex flex-col items-center animate-bounce-in",
            className
        )}>
            {/* Letter display */}
            <div className={cn(
                "font-tamil font-bold text-primary mb-4",
                size === 'sm' && "text-4xl sm:text-5xl",
                size === 'md' && "text-5xl sm:text-6xl",
                size === 'lg' && "tamil-display"
            )}>
                {letter.letter}
            </div>

            {/* Transliteration */}
            <p className={cn(
                "font-semibold text-foreground mb-2",
                size === 'sm' && "text-lg",
                size === 'md' && "text-xl",
                size === 'lg' && "text-2xl"
            )}>
                "{letter.transliteration}"
            </p>

            {/* Description */}
            {showDetails && (
                <p className="text-muted-foreground text-center text-sm sm:text-base mb-4 max-w-xs">
                    {letter.description}
                </p>
            )}

            {/* Speak button */}
            {onSpeak && (
                <Button
                    onClick={onSpeak}
                    variant={isSpeaking ? "secondary" : "default"}
                    size="lg"
                    className="mt-2"
                >
                    {isSpeaking ? (
                        <>
                            <VolumeX className="w-5 h-5" />
                            Speaking...
                        </>
                    ) : (
                        <>
                            <Volume2 className="w-5 h-5" />
                            Speak Letter
                        </>
                    )}
                </Button>
            )}
        </div>
    );
};
