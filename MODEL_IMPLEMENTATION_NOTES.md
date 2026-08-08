# Wavelet-GMFlow-3D: Nota de Implementacao do Modelo

## Decisao arquitetural

A modificacao proposta para substituir o backbone global-attention DiT por um
backbone 3D mais escalavel nao precisa transformar o modelo em FlowLet puro.

A direcao recomendada e manter a formulacao probabilistica do GMFlow e usar
componentes inspirados no FlowLet apenas como vies indutivo espacial e
frequencial para MRI 3D.

Em outras palavras, o modelo alvo deve ser entendido como:

```text
Wavelet-GMFlow-3D
```

e nao como uma copia direta do FlowLet.

## O que permanece GMFlow

O modelo continua sendo GMFlow se preservarmos:

- cabeca de mistura gaussiana;
- predicao de `means`, `logstds` e `logweights`;
- `GMOutput3D`;
- `GMFlowNLLLoss3D`;
- algebra de transicao reversa em mistura gaussiana;
- `reverse_transition_3d`;
- sampler GM-ODE;
- sampler de segunda ordem `gm_2nd_order_3d`;
- classifier-free guidance probabilistico;
- treinamento via NLL/KL de mistura gaussiana.

Esses componentes definem a formulacao central do GMFlow.

## O que vem do FlowLet

As ideias inspiradas no FlowLet devem ser usadas para melhorar o vies
indutivo do modelo em dados volumetricos de MRI:

- treinamento em dominio wavelet 3D Haar;
- separacao entre anatomia grossa e bandas de alta frequencia;
- arquitetura 3D multiescala;
- backbone 3D U-Net, windowed attention, axial attention ou outro mecanismo
  local/hierarquico;
- possivel condicionamento por FiLM/AdaGN em multiplas escalas;
- perdas ou diagnosticos separados para LLL e bandas de detalhe.

Essas modificacoes tornam o modelo mais adequado para MRI 3D, mas nao removem
a formulacao probabilistica do GMFlow.

## Quando o modelo viraria FlowLet puro

O modelo deixaria de ser GMFlow e passaria a ser essencialmente FlowLet se
fossem feitas as seguintes trocas:

- remocao da cabeca de mistura gaussiana;
- remocao de `means`, `logstds` e `logweights`;
- substituicao do objetivo GMFlow NLL/KL por uma perda direta de flow matching
  como no FlowLet;
- uso integral do backbone FlowLet;
- uso completo de FiLM e spatial cross-attention como mecanismo principal de
  condicionamento;
- substituicao do sampler GM-ODE pelo solver usado na formulacao FlowLet.

Essa nao e a modificacao recomendada para uma refatoracao incremental deste
repositorio.

## Recomendacao de implementacao

A estrategia mais segura e incremental e adicionar uma opcao de backbone, por
exemplo:

```python
backbone_type = "windowed_dit3d"
```

ou, para uma refatoracao mais forte:

```python
backbone_type = "unet3d"
```

mantendo:

```python
diffusion.type = "GMFlow3D"
use_wavelet = True
denoising output = Gaussian mixture parameters
loss = GMFlowNLLLoss3D
sampler = GM-ODE order 2
```

## Interpretacao final

A arquitetura recomendada e um hibrido:

```text
GMFlow:
  - formulacao probabilistica
  - mistura gaussiana
  - NLL/KL
  - GM-ODE

FlowLet-inspired:
  - dominio Haar wavelet 3D
  - vies multiescala
  - backbone volumetrico escalavel
  - condicionamento mais forte por idade
```

Portanto, a modificacao 3 deve preservar a identidade do GMFlow, mas substituir
o backbone por uma versao mais apropriada para volumes 3D e detalhes
anatomicos de alta frequencia.
