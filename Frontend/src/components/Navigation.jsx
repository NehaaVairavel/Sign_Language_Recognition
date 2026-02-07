import { Video, BookOpen, Target, Trophy } from 'lucide-react';
import { cn } from '@/lib/utils';

const modes = [
    { id: 'live', label: 'Live', icon: Video, description: 'Real-time recognition' },
    { id: 'learn', label: 'Learn', icon: BookOpen, description: 'Learn the signs' },
    { id: 'practice', label: 'Practice', icon: Target, description: 'Practice with feedback' },
    { id: 'quiz', label: 'Quiz', icon: Trophy, description: 'Test your knowledge' },
];

export const Navigation = ({ currentMode, onModeChange }) => {
    return (
        <nav className="w-full py-4">
            <div className="container mx-auto px-4">
                <div className="flex flex-wrap justify-center gap-2 sm:gap-3">
                    {modes.map((mode) => {
                        const Icon = mode.icon;
                        const isActive = currentMode === mode.id;

                        return (
                            <button
                                key={mode.id}
                                onClick={() => onModeChange(mode.id)}
                                className={cn(
                                    "mode-tab flex items-center gap-2",
                                    isActive && "active"
                                )}
                            >
                                <Icon className="w-4 h-4 sm:w-5 sm:h-5" />
                                <span className="hidden sm:inline">{mode.label}</span>
                                <span className="sm:hidden text-sm">{mode.label}</span>
                            </button>
                        );
                    })}
                </div>

                <p className="text-center text-sm text-muted-foreground mt-3">
                    {modes.find(m => m.id === currentMode)?.description}
                </p>
            </div>
        </nav>
    );
};
