# Plano de melhoria de qualidade do GMFlow3D

## Escopo

Este plano foi produzido a partir da comparação entre:

- o código atual em `/home/fred/Projetos/Einstein/GMFlow`;
- o código de referência local em `/home/fred/Projetos/Einstein/GMFlow-dev`;
- o treinamento em `/media/fred/FRED5TB/work_dirs/gmflow3d_openbhb_k4_gpu5_20260801_101648`;
- o OpenBHB processado em `/media/fred/FRED5TB/Einstein/Open_BHB_processado`.

Nenhum código-fonte fora de `GMFlow/` deve ser alterado. O dataset e os logs externos são somente leitura.

## Diagnóstico

O treinamento de 100 mil iterações convergiu numericamente. A NLL e a norma dos gradientes permaneceram finitas no final do treino. O defeito quadrangular, portanto, não é explicado por divergência simples.

A assinatura espacial do defeito coincide com a grade de patches:

1. O modelo opera em Haar 3D com volumes `8 x 32 x 32 x 32`.
2. O patch size histórico é 2 no domínio wavelet.
3. O unpatchify produz blocos independentes de `2 x 2 x 2` nesse domínio.
4. A IDWT dobra cada dimensão, transformando a grade em blocos periódicos de 4 voxels no MRI reconstruído.
5. Os maiores saltos de intensidade medidos aparecem exatamente com período 4.

A transformada Haar não é a origem do erro. O teste DWT -> IDWT apresentou erro máximo de reconstrução de aproximadamente `4.77e-7`.

### Evidência quantitativa

| Medida | Resultado observado |
| --- | ---: |
| Razão de gradiente nas bordas de período 4, iter. 50k | aproximadamente 2,14 |
| Razão de gradiente nas bordas de período 4, iter. 100k | aproximadamente 1,34 a 1,40 |
| Razão equivalente em MRI real normalizado | aproximadamente 1,02 |
| Energia relativa média das 7 bandas de detalhe, gerado | 0,0569 |
| Energia relativa média das 7 bandas de detalhe, real | 0,0884 |
| Déficit relativo de detalhe do gerado | aproximadamente 36% |

Uma razão de borda igual a 1 significa que as fronteiras da grade não têm mais energia de gradiente que o interior dos patches. O valor observado confirma que o aspecto quadrangular está no volume gerado, e não apenas na visualização do TensorBoard.

### Causas principais

1. **Saída por patch sem mistura local:** cada token projeta diretamente seus próprios voxels no unpatchify.
2. **Entrada sem sobreposição:** tokens vizinhos não compartilham contexto local no patch embedding histórico.
3. **NLL sem restrição espacial explícita:** a likelihood pode melhorar enquanto descontinuidades periódicas permanecem.
4. **Variância global no treinador standalone histórico:** uma única escala não representa adequadamente LLL e as sete bandas de alta frequência.
5. **Amostragem histórica de primeira ordem:** Euler de primeira ordem aumenta o erro de discretização e o aspecto suavizado.
6. **Seleção por loss:** a NLL sozinha não mede periodicidade de patches nem energia anatômica de alta frequência.

## Modificações implementadas

### Arquitetura

Em `lib/models/architecture/gmflow3d.py`:

- `PatchEmbed3D` recebeu `overlap=True`, usando kernel `2 * patch_size - 1`, stride igual ao patch e padding que preserva a grade de tokens.
- Foi adicionado `LocalMixtureMeanRefiner3D`, um refinador residual Conv3d com kernel 3.
- A última convolução do refinador é inicializada em zero. Ativar o módulo começa como identidade.
- A correção local é somada a todas as médias gaussianas. As diferenças entre componentes da mistura são preservadas.
- A máscara de dropout da condição de idade agora é amostrada uma vez por amostra e reutilizada em todos os blocos.

As opções são aditivas. `overlap_patch_embed=False` e `local_refinement=False` preservam o caminho antigo.

### Função objetivo

Em `lib/models/losses/diffusion_loss.py`:

- foi adicionada L1 ponderada sobre a média efetiva da mistura;
- foi adicionada correspondência de gradientes 3D;
- quando `reconstruct_wavelet=True`, os gradientes são comparados após IDWT, no espaço de voxels;
- fronteiras do período de patch podem receber peso maior;
- os novos termos aparecem nos logs como `loss_gm_mean`, `loss_voxel_gradient` e `loss_auxiliary`.

A configuração inicial é:

```python
mixture_mean_weight = 0.05
voxel_gradient_weight = 0.25
boundary_period = 4
boundary_weight = 2.0
```

A loss compara o gradiente previsto com o gradiente real; ela não aplica suavização indiscriminada em bordas anatômicas.

### Mistura, frequência e amostragem

Na configuração principal e no treinador standalone:

- variância por subbanda ativada;
- pesos Haar `[1, 2, 2, 3, 2, 3, 3, 4]`;
- transição aleatória entre 0,05 e 1,0;
- 25 passos externos x 4 subpassos, total de 100 passos ODE;
- correção de segunda ordem ativada;
- Fourier features de idade mantidas;
- treino e inferência executados em `8 x 32 x 32 x 32`, com IDWT apenas no fim.

### Dados e observabilidade

- O caminho padrão agora aponta para `/media/fred/FRED5TB/Einstein/Open_BHB_processado`.
- O treinador standalone aceita cache `.pt` ou os volumes `.npy` diretamente.
- A faixa de idade é inferida dos exames encontrados.
- TensorBoard e standalone registram `patch_boundary_ratio`.
- O hook do TensorBoard registra média, desvio padrão e fração não finita, além das três vistas ortogonais.

## Compatibilidade de checkpoints

O patch embedding sobreposto altera o shape do kernel de entrada. A variância por subbanda também altera parte da cabeça GM. Por isso, o estado completo do otimizador antigo não deve ser retomado como se a arquitetura fosse idêntica.

O treinador standalone possui `--resume_weights_only`:

- carrega todos os tensores com nome e shape compatíveis;
- ignora explicitamente os tensores incompatíveis;
- reinicializa otimizador, scaler e contador;
- mantém o refinador como identidade no início.

Para maximizar o reaproveitamento do checkpoint antigo em uma primeira ablação, use `--no_overlap_patch_embed`. Para o resultado final, faça um treinamento novo com overlap ativado.

## Plano experimental

### Fase 1: validação funcional curta

Objetivo: confirmar loss finita, backward, IDWT, geração e TensorBoard no dataset solicitado.

```bash
/home/fred/anaconda3/envs/gmflow/bin/python tools/train_standalone.py \
  --data_root /media/fred/FRED5TB/Einstein/Open_BHB_processado/train/quasiraw_3d \
  --metadata /media/fred/FRED5TB/Einstein/Open_BHB_processado/train.tsv \
  --work_dir work_dirs/quality_smoke \
  --volume_size 16 \
  --num_heads 2 \
  --head_dim 32 \
  --num_layers 2 \
  --batch_size 1 \
  --grad_accum 1 \
  --num_workers 0 \
  --total_iters 2 \
  --warmup_iters 1 \
  --log_interval 1 \
  --save_interval 2 \
  --sample_interval 2 \
  --sample_timesteps 3 \
  --sample_substeps 2 \
  --sample_order 2
```

Esse comando valida o pipeline, não a qualidade final.

### Fase 2: ablações controladas

Usar a mesma semente e o mesmo ruído para todas as idades.

| Experimento | Inicialização | Mudanças | Objetivo |
| --- | --- | --- | --- |
| A | checkpoint 100k | apenas sampler ordem 1 versus 2 | separar erro de integração de erro aprendido |
| B | checkpoint 100k | refiner + losses + variância por banda, sem overlap | reaproveitar o patch stem antigo |
| C | checkpoint 100k | configuração completa com overlap | avaliar warm start com novo stem |
| D | aleatória | configuração completa | referência final sem incompatibilidade de checkpoint |

Comando sugerido para B:

```bash
/home/fred/anaconda3/envs/gmflow/bin/python tools/train_standalone.py \
  --data_root /media/fred/FRED5TB/Einstein/Open_BHB_processado/train/quasiraw_3d \
  --metadata /media/fred/FRED5TB/Einstein/Open_BHB_processado/train.tsv \
  --work_dir work_dirs/gmflow3d_quality_warmstart \
  --resume /media/fred/FRED5TB/work_dirs/gmflow3d_openbhb_k4_gpu5_20260801_101648/checkpoints/latest.pt \
  --resume_weights_only \
  --no_overlap_patch_embed \
  --batch_size 2 \
  --grad_accum 4
```

Em GPU B200, ajustar `--batch_size` para 64 após confirmar memória e throughput.

### Fase 3: treinamento final

Treinar do zero com a configuração completa. No caminho OpenMMLab:

```bash
BATCH_SIZE=2 LEARNING_RATE=1e-4 ./train.sh
```

Para continuar um checkpoint compatível com a mesma arquitetura:

```bash
./train.sh --resume checkpoints/gmflow3d_openbhb_k4/latest.pth
```

Um checkpoint anterior às mudanças estruturais deve usar o carregamento parcial do standalone, não o resume completo do runner.

### Fase 4: ajuste fino

Ajustar somente depois de medir cada ablação:

1. Se `patch_boundary_ratio > 1.15`, aumentar `voxel_gradient_weight` de 0,25 para 0,4.
2. Se detalhes anatômicos forem suavizados, reduzir `boundary_weight` antes de reduzir a loss global de gradiente.
3. Se as bandas de detalhe continuarem fracas, elevar os pesos das bandas HHH/HHL/HLH/LHH em passos de 25%.
4. Se a NLL ficar instável, limitar log-std por subbanda e reduzir LR da cabeça GM.
5. Se ordem 2 não melhorar visualmente, comparar 50 x 2 com 25 x 4 mantendo 100 subpassos.

## Critérios de aprovação

Não selecionar checkpoints apenas pela loss de treino.

### Integridade

- nenhuma amostra com NaN ou Inf;
- round trip Haar com erro máximo menor que `1e-6`;
- mesma semente produz exatamente o mesmo volume;
- shape final `B x 1 x 64 x 64 x 64`.

### Ausência de grade

- média de `patch_boundary_ratio` menor que 1,10;
- nenhum eixo maior que 1,15;
- inspeção axial, coronal e sagital sem periodicidade de 4 voxels.

### Frequência e anatomia

- razão detalhe/LLL entre 0,075 e 0,10 como faixa inicial;
- espectro radial sem picos na frequência da grade;
- bordas de ventrículos, córtex e substância branca preservadas;
- distribuição de volumes cerebrais compatível com o conjunto real.

### Condicionamento por idade

Com o mesmo ruído inicial:

- diferenças entre idades devem ser graduais, não saltos globais de intensidade;
- um regressor de idade independente deve responder monotonicamente;
- mudanças devem se concentrar em regiões plausíveis e preservar a identidade estrutural global;
- comparar idades mínima, quartis e máxima.

### Métricas recomendadas

- distância de features 3D usando encoder médico fixo;
- erro de um regressor de idade em volumes gerados;
- distribuição de GM/WM/CSF após segmentação;
- Dice/topologia de estruturas segmentáveis, quando houver pseudo-rótulos;
- espectro radial e energia por subbanda Haar;
- `patch_boundary_ratio`;
- diversidade entre sementes e consistência sob idade fixa.

## Decisão recomendada

1. Executar primeiro a ablação B por 5 a 10 mil iterações.
2. Comparar B com o checkpoint antigo usando exatamente o mesmo sampler de ordem 2.
3. Se a razão de borda cair sem perda de detalhe, iniciar D do zero.
4. Manter o checkpoint antigo somente como baseline; ele não deve ser considerado inicialização final obrigatória.
5. Avaliar a cada 100 iterações no smoke test. Em treino longo, aumentar o intervalo se a geração de 16 volumes dominar o tempo de treino.
