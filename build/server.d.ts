export declare class TitanExpressServer {
    private app;
    private server;
    private model;
    private memoryVec;
    private port;
    private wss;
    constructor(port?: number);
    private setupMiddleware;
    private setupRoutes;
    private setupWebSocket;
    start(): Promise<void>;
    stop(): Promise<void>;
}
