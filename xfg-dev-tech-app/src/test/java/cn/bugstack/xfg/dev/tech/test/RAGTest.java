package cn.bugstack.xfg.dev.tech.test;

import com.alibaba.fastjson.JSON;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.ai.chat.ChatResponse;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.SystemPromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.ollama.OllamaChatClient;
import org.springframework.ai.ollama.api.OllamaOptions;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.PgVectorStore;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

//public class Document {
//    private final String id;           // 每一个片段的唯一 ID (通常是 UUID)
//    private final String content;      // 这一段的内容
//    private final Map<String, Object> metadata; // 附加信息
//    // ... 还有对应的向量数据 (Embedding)
//}

@Slf4j
@RunWith(SpringRunner.class)
@SpringBootTest
public class RAGTest {

    @Resource
    private OllamaChatClient ollamaChatClient;
    @Resource
    private TokenTextSplitter tokenTextSplitter;
    @Resource
    private SimpleVectorStore simpleVectorStore;
    @Resource
    private PgVectorStore pgVectorStore;

    @Test
    public void upload() {
        TikaDocumentReader reader = new TikaDocumentReader("data/test.txt");

        List<Document> documents = reader.get();
//        for(Document doc:documents){
//            log.info(doc.getContent());
//        }
        //文本分割,将documents分割成为更小的chunk
        List<Document> documentSplitterList = tokenTextSplitter.apply(documents);

        //为每一条数据添加对应的元组信息
        documents.forEach(doc -> doc.getMetadata().put("knowledge", "知识库名称"));
        documentSplitterList.forEach(doc -> doc.getMetadata().put("knowledge", "知识库名称"));

        // pgVectorStore 会做三件事：
        //   a. 调用 Embedding 模型（你配置的 nomic-embed-text）将文字转为 768 维向量。
        //   b. 将原始文本、元数据和向量数据打包。
        //   c. 执行 SQL 插入到 PostgreSQL 的 vector_store 表中。
        pgVectorStore.accept(documentSplitterList);

        log.info("上传完成");
    }

    @Test
    public void chat(){
        String message = "耄耋,哪年出生";
        //角色设定(给AI的指令)
        String SYSTEM_PROMPT = """
                Use the information from the DOCUMENTS section to provide accurate answers but act as if you knew this information innately.
                If unsure, simply state that you don't know.
                Another thing you need to note is that your reply must be in Chinese!
                DOCUMENTS:
                    {documents}
                """;
        //构建搜索请求
        // .query(message): 将用户的问题转换成 768 维向量进行比对。
        // .withTopK(5): 取最相似的前 5 条数据。
        // .withFilterExpression(...): 过滤条件，只在“知识库名称”这个项目下搜索，防止搜到别的知识库。
        SearchRequest request = SearchRequest.query(message).withTopK(5).withFilterExpression("knowledge == '知识库名称'");
        // 在pg数据库当中根据余弦相似度计算
        List<Document> documents = pgVectorStore.similaritySearch(request);
        // 把搜出来的 Document 对象转成纯文本字符串。
        String documentsCollectors = documents.stream().map(Document::getContent).collect(Collectors.joining());
        //将搜索出来的内容填到PROMPT当中
        Message ragMessage = new SystemPromptTemplate(SYSTEM_PROMPT).createMessage(Map.of("documents", documentsCollectors));
        //组装上下文信息
        ArrayList<Message> messages = new ArrayList<>();
        messages.add(new UserMessage(message));
        messages.add(ragMessage);
        //发送给大模型进行推理
        ChatResponse chatResponse = ollamaChatClient.call(new Prompt(messages, OllamaOptions.create().withModel("deepseek-r1:1.5b")));

        log.info("测试结果:{}", JSON.toJSONString(chatResponse));
    }

}
