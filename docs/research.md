# Research source

RecallMesh is inspired by **HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents**, by Jinchang Zhu, Jindong Li, Cheng Zhang, Jiahong Liu, and Menglin Yang (2026).

- [Paper and metadata](https://arxiv.org/abs/2604.16839)
- [Paper PDF](https://arxiv.org/pdf/2604.16839)
- DOI: [10.48550/arXiv.2604.16839](https://doi.org/10.48550/arXiv.2604.16839)

The paper proposes co-activation-based graph memory, associative retrieval, and a separate semantic store populated through reflective distillation. RecallMesh explores the associative graph direction with Hebbian reinforcement, decay, bounded retrieval, and a durable agent-facing service. It is an independent implementation, not the authors' official software or a complete reproduction: reflective semantic distillation is not implemented, lexical seeding is currently used, and our small evaluation does not reproduce the paper's full benchmark.

The local tests and answer-quality pilot are reported separately in [evaluation.md](evaluation.md). Paper results must not be presented as RecallMesh results. This citation does not relicense the paper or third-party datasets under the project's MIT license.

```bibtex
@article{zhu2026helamem,
  title={HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents},
  author={Zhu, Jinchang and Li, Jindong and Zhang, Cheng and Liu, Jiahong and Yang, Menglin},
  year={2026},
  eprint={2604.16839},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  doi={10.48550/arXiv.2604.16839}
}
```

## Related retrieval research

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2023. [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCoT)](https://aclanthology.org/2023.acl-long.557/). ACL 2023, pages 10014–10037. DOI: [10.18653/v1/2023.acl-long.557](https://doi.org/10.18653/v1/2023.acl-long.557).

IRCoT informs the related-work discussion of reasoning-guided retrieval. RecallMesh's bounded evidence planner is not a reproduction of IRCoT. See [methodology and engineering extensions](methodology.md) for joint fact selection, support checks, guardrails, and the distinction between existing co-activation edges and proposed typed relations.
