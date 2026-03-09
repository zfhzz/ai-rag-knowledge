package cn.bugstack.xfg.dev.tech.trigger.http;

import cn.bugstack.xfg.dev.tech.api.IRAGService;
import cn.bugstack.xfg.dev.tech.api.response.Response;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.transport.UsernamePasswordCredentialsProvider;
import org.redisson.api.RList;
import org.redisson.api.RedissonClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.ollama.OllamaChatClient;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.PgVectorStore;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.core.io.PathResource;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.List;

@Slf4j
@RestController()
@CrossOrigin("*")
@RequestMapping("/api/v1/rag/")
public class RAGController implements IRAGService {

    @Resource
    private OllamaChatClient ollamaChatClient;
    @Resource
    private TokenTextSplitter tokenTextSplitter;
    @Resource
    private SimpleVectorStore simpleVectorStore;
    @Resource
    private PgVectorStore pgVectorStore;
    @Resource
    private RedissonClient redissonClient;

    @RequestMapping(value = "query_rag_tag_list", method = RequestMethod.GET)
    @Override
    public Response<List<String>> queryRagTagList() {
        RList<String> elements = redissonClient.getList("ragTag");
        return Response.<List<String>>builder()
                .code("0000")
                .info("调用成功")
                .data(elements)
                .build();
    }
    @RequestMapping(value = "file/upload",method = RequestMethod.POST,headers = "content-type=multipart/form-data")
    @Override
    public Response<String> uploadFile(@RequestParam String ragTag,@RequestParam("file") List<MultipartFile> files) {
        log.info("开始上传知识库");
        log.info(ragTag);
        for(MultipartFile file:files){
            TikaDocumentReader documentReader = new TikaDocumentReader(file.getResource());
            List<Document> documents = documentReader.get();
            List<Document> documentsSplitterList = tokenTextSplitter.apply(documents);

            documents.forEach(doc -> doc.getMetadata().put("knowledge",ragTag));
            documentsSplitterList.forEach(doc -> doc.getMetadata().put("knowledge",ragTag));

            pgVectorStore.accept(documentsSplitterList);
            RList<String> elements = redissonClient.getList("ragTag");
            //如果redis当中不存在对应的知识库存储,则将该ragTag加入到其中
            if (!elements.contains(ragTag)){
                elements.add(ragTag);
            }
        }
        log.info("上传知识库完成 {}", ragTag);
        return Response.<String>builder().code("0000").info("调用成功").build();
    }

    @RequestMapping(value = "analyze_git_repository", method = RequestMethod.POST)
    @Override
    public Response<String> analyzeGitRepository(@RequestParam String repoUrl, @RequestParam String userName, @RequestParam String token) throws Exception {
        String localPath = "E:\\clone_path";
        String repoProjectName = extractProjectName(repoUrl);
        File directory = new File(localPath);

        // 1. 清理旧目录（如果存在）
        if (directory.exists()) {
            FileUtils.deleteDirectory(directory);
        }

        // 2. 使用 try-with-resources 自动管理 Git 资源关闭
        try (Git git = Git.cloneRepository()
                .setURI(repoUrl)
                .setDirectory(directory)
                .setCredentialsProvider(new UsernamePasswordCredentialsProvider(userName, token))
                .call()) {

            log.info("克隆成功，开始遍历项目: {}", repoProjectName);

            Files.walkFileTree(Paths.get(localPath), new SimpleFileVisitor<>() {
                @Override
                public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) {
                    // 【核心优化】跳过 .git 隐藏文件夹，不进入其内部
                    if (dir.getFileName().toString().equals(".git") || dir.getFileName().toString().equals(".idea")) {
                        return FileVisitResult.SKIP_SUBTREE;
                    }
                    return FileVisitResult.CONTINUE;
                }

                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                    String fileName = file.getFileName().toString();

                    // 【核心优化】只解析有意义的文本文件，跳过图片、压缩包、编译产物
                    if (isBinaryFile(fileName)) {
                        return FileVisitResult.CONTINUE;
                    }

                    log.info("{} 正在解析: {}", repoProjectName, fileName);
                    try {
                        TikaDocumentReader reader = new TikaDocumentReader(new PathResource(file));
                        List<Document> documents = reader.get();
                        if (documents.isEmpty()) return FileVisitResult.CONTINUE;

                        List<Document> documentSplitterList = tokenTextSplitter.apply(documents);

                        // 打标入库
                        documentSplitterList.forEach(doc -> doc.getMetadata().put("knowledge", repoProjectName));
                        pgVectorStore.accept(documentSplitterList);

                    } catch (Exception e) {
                        log.error("解析文件失败: {}，原因: {}", fileName, e.getMessage());
                    }
                    return FileVisitResult.CONTINUE;
                }
            });
        } // 执行到这里，git 会自动 close，释放文件锁

        // 3. 此时删除目录就不会报错了
        FileUtils.deleteDirectory(directory);

        // 4. 更新 Redis 标签列表
        RList<String> elements = redissonClient.getList("ragTag");
        if (!elements.contains(repoProjectName)) elements.add(repoProjectName);

        return Response.<String>builder().code("0000").info("调用成功").build();
    }

    // 简单的文件类型判定，防止 Tika 强行解析二进制导致内存或 CPU 飙升
    private boolean isBinaryFile(String fileName) {
        String name = fileName.toLowerCase();
        return name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg") ||
                name.endsWith(".gif") || name.endsWith(".idx") || name.endsWith(".pack") ||
                name.endsWith(".exe") || name.endsWith(".class") || name.endsWith(".jar");
    }

    //根据git地址解析知识库名称
    private String extractProjectName(String repoUrl) {
        String[] parts = repoUrl.split("/");
        String projectNameWithGit = parts[parts.length - 1];
        return projectNameWithGit.replace(".git", "");
    }
}
