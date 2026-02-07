import { getSignImage } from '@/lib/signImages';
import { cn } from '@/lib/utils';

export const SignAvatar = ({
    letter,
    size = 'md',
    className,
    showLabel = true
}) => {
    const signImage = getSignImage(letter.id);

    return (
        <div className={cn(
            "elevated-card p-4 flex flex-col items-center",
            className
        )}>
            {showLabel && (
                <p className="text-sm font-medium text-muted-foreground mb-3">
                    Sign Gesture for "{letter.transliteration}"
                </p>
            )}

            <div className={cn(
                "relative rounded-xl overflow-hidden bg-secondary/30 flex items-center justify-center",
                size === 'sm' && "w-32 h-32",
                size === 'md' && "w-48 h-48",
                size === 'lg' && "w-64 h-64"
            )}>
                {signImage ? (
                    <img
                        src={signImage}
                        alt={`Sign gesture for ${letter.letter} (${letter.transliteration})`}
                        className="w-full h-full object-contain p-2"
                    />
                ) : (
                    <div className="flex flex-col items-center justify-center text-muted-foreground p-4">
                        <div className="text-4xl mb-2">✋</div>
                        <p className="text-xs text-center">{letter.signHint}</p>
                    </div>
                )}
            </div>

            {showLabel && (
                <p className="text-xs text-muted-foreground mt-3 text-center max-w-[200px]">
                    {letter.signHint}
                </p>
            )}
        </div>
    );
};
