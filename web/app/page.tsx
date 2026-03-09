import { VideoSearch } from "@/components/video-search";

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto flex h-16 items-center px-4">
          <h1 className="text-xl font-semibold tracking-tight text-foreground">
            vidfetch
          </h1>
          <span className="ml-2 text-sm text-muted-foreground">
            Lightweight video retrieval
          </span>
        </div>
      </header>
      <main className="container mx-auto px-4 py-8">
        <section className="mb-10 text-center">
          <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Search your videos by what’s in them
          </h2>
          <p className="mx-auto mt-3 max-w-2xl text-lg text-muted-foreground">
            Find clips by object (person, fire, surfboard…) or by color. No GPU
            needed — <strong className="text-foreground">runs on CPU only</strong>, so it works on any machine.
          </p>
        </section>
        <VideoSearch />
      </main>
    </div>
  );
}
