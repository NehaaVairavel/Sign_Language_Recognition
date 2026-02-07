import { useState } from 'react';
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Header } from '@/components/Header';
import { Navigation } from '@/components/Navigation';
import { LiveMode } from '@/components/modes/LiveMode';
import { LearnMode } from '@/components/modes/LearnMode';
import { PracticeMode } from '@/components/modes/PracticeMode';
import { QuizMode } from '@/components/modes/QuizMode';

const queryClient = new QueryClient();

function AppContent() {
    const [currentMode, setCurrentMode] = useState('learn');

    const renderMode = () => {
        switch (currentMode) {
            case 'live':
                return <LiveMode />;
            case 'learn':
                return <LearnMode />;
            case 'practice':
                return <PracticeMode />;
            case 'quiz':
                return <QuizMode />;
            default:
                return <LearnMode />;
        }
    };

    return (
        <div className="min-h-screen bg-background">
            <Header />
            <Navigation
                currentMode={currentMode}
                onModeChange={setCurrentMode}
            />
            <main className="pb-12">
                {renderMode()}
            </main>
        </div>
    );
}

const App = () => (
    <QueryClientProvider client={queryClient}>
        <TooltipProvider>
            <Toaster />
            <Sonner />
            <AppContent />
        </TooltipProvider>
    </QueryClientProvider>
);

export default App;
