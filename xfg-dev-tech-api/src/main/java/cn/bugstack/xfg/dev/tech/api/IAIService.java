package cn.bugstack.xfg.dev.tech.api;

import org.springframework.ai.chat.ChatResponse;
import reactor.core.publisher.Flux;

public interface IAIService {
    ChatResponse generate(String model,String message);

    Flux<ChatResponse> generateStream(String model,String message);

    Flux<ChatResponse> generateStreamRag(String model, String ragTag, String message);
}
