# Erros encontrados e correções

Análise feita sobre o commit `c0dec6c` ("resolving dependencies").
Cada erro tem um teste em `tests/test_erros.py`. Todos os testes falharam
antes da correção, pelo motivo descrito, e passam depois dela.

## Como rodar os testes

```bash
uv run --no-project --with numpy --with scipy --with matplotlib --with pytest \
    python -m pytest tests -q
```

`tests/conftest.py` acrescenta `src/2D` e `src/3D` ao `sys.path` e usa o
backend `Agg` do matplotlib, para que nenhuma janela seja aberta.

---

## 1. Volume e energia em 3D calculados sobre a malha não deformada

- **Onde:** `src/3D/quadrature_triangle.py`, em `energy_3D` e `volume_3D`.
- **Problema:** as normais e as áreas vinham da malha deformada `m`, mas os
  vértices dos triângulos (`T(x)`) vinham de `mesh.points`, a malha original.
  Cada integral misturava as duas geometrias.
- **Evidência:** ao deformar uma esfera com `x = -0.5` (raio pela metade), o
  volume caía para 0,25 do original, quando o esperado é 0,125. A energia com
  `Bond = 2` também diferia da calculada diretamente sobre a malha deformada.
- **Impacto:** a restrição de volume e a função objetivo do SLSQP em 3D
  estavam erradas, então nenhum resultado 3D anterior é confiável.
- **Correção:** usar `m.points` e `m.connectivity`.
- **Testes:** `test_volume_3d_usa_malha_deformada`,
  `test_energy_3d_usa_malha_deformada`.

## 2. Restrições acumuladas entre chamadas (argumento padrão mutável)

- **Onde:** `find_droplet_optimization` (`src/2D/droplet_optimization.py`) e
  `find_droplet_optimization_3d` (`src/3D/optimization_3d.py`).
- **Problema:** o padrão `constraints: List = []` é a mesma lista em todas as
  chamadas, e a função faz `constraints.append(volume_constraint)`. Na segunda
  chamada o SLSQP recebia 2 restrições de volume; num laço sobre volumes
  diferentes, elas se contradizem.
- **Evidência:** duas chamadas seguidas passavam `[1, 2]` restrições ao otimizador.
- **Correção:** `constraints: list | None = None` e, no início da função,
  `constraints = list(constraints or [])`. A lista passada por quem chama
  também deixa de ser alterada.
- **Efeito colateral corrigido:** as anotações usavam `List` (nunca importado)
  e, no 3D, `Droplet` (que não existe no módulo). No Python 3.14 isso passa
  despercebido porque as anotações só são avaliadas quando alguém as lê, mas no
  3.12 os dois módulos davam `NameError` já no import. Agora as anotações usam
  `list | None` e o tipo de retorno do 3D é `Mesh`, que é o que a função
  realmente devolve.
- **Testes:** `test_2d_restricoes_nao_acumulam`, `test_3d_restricoes_nao_acumulam`.

## 3. Nomes indefinidos em `run/comparison.py`

- **Problema:** os ramos do `if` usavam `V` (linha 27) e `v_adim` (linha 29),
  mas a variável se chama `V_adim`. Para `2,17 < V_adim ≤ 10` o script dava
  `NameError`. Além disso, se nenhum ramo fosse executado, `A` ficava sem valor.
- **Correção:** trocar por `V_adim` e transformar o último `elif` em `else`, o
  mesmo comportamento para volumes válidos, mas garantindo que `A` sempre recebe valor.
- **Teste:** `test_comparison_sem_nomes_indefinidos`, que procura no código do
  script nomes usados e nunca definidos.

## 4. Modo `sessile`: índice errado no teste de segmento degenerado

- **Onde:** `find_droplets`, em `src/2D/droplet.py`.
- **Problema:** a interseção no modo `sessile` é procurada em θ (`u[2]`), mas o
  teste de "segmento degenerado" olhava sempre o raio (`u[0]`). Quando θ fica
  constante num segmento, a interpolação calculava `0/0` e o perfil recebia `NaN`.
- **Correção:** calcular `stop_idx` antes do teste e usá-lo no teste e na
  interpolação.
- **Teste:** `test_sessile_segmento_com_theta_constante`, com uma solução de
  EDO montada à mão e `np.errstate(all="raise")`.

## 5. Gota da otimização 2D criada com área `S=None`

- **Onde:** fim de `find_droplet_optimization`.
- **Problema:** a gota era criada com `S=None`, e `Droplet.dimensionalize()`
  falhava com `TypeError` em `None * a**2`.
- **Correção:** nova função `surface_droplet(points)`, que calcula a área da
  superfície de revolução do perfil poligonal. Cada segmento gera um tronco de
  cone de área `2π · r_médio · comprimento`, então a conta é exata para o
  polígono.
- **Teste:** `test_2d_otimizacao_calcula_area`. Com `x = 0` o perfil é a
  calota esférica inicial (base de raio 1, altura H), cuja área é π(1 + H²);
  depois `dimensionalize()` deve rodar sem erro.

## 6. `Gauss_Legendre` devolvia 0 sem avisar

- **Onde:** `src/2D/quadrature.py`.
- **Problema:** com `n_points_quad` diferente de 2 ou 3, `integrate` e
  `integrate_vector` devolviam `0.0` sem nenhum erro.
- **Correção:** um método `_rule` escolhe pontos e pesos e lança `ValueError`
  para valores não suportados. As duas funções passaram a usá-lo, o que também
  eliminou o código repetido dos dois ramos.
- **Teste:** `test_gauss_legendre_rejeita_n_invalido`.

## 7. `Arc.get_points_eq_space` com início igual ao fim

- **Onde:** `src/2D/objects.py`.
- **Problema:** quando o arco tinha tamanho angular 0 (início igual ao fim), o
  método chamava `super().plot(ax, ...)` com uma variável `ax` que não existe
  (`NameError`) e, mesmo que existisse, devolveria `None`.
- **Correção (premissa):** o arco que começa e termina no mesmo ponto passa a
  ser tratado como o círculo completo, começando em `x1` e seguindo o sentido
  de desenho. Se a intenção era outra (lançar erro, por exemplo), basta trocar
  essas duas linhas.
- **Teste:** `test_arco_com_inicio_igual_ao_fim`.

## 8. Arquivos `.pyc` versionados

- **Problema:** `src/3D/__pycache__/*.pyc` estava no repositório. A regra
  `__pycache__` no `.gitignore` não remove arquivos que já estavam versionados.
- **Correção:** `git rm -r --cached src/3D/__pycache__`. Os arquivos continuam
  no disco e passam a ser ignorados.

---

## Verificação de ponta a ponta

Depois das correções, as duas otimizações reais (SLSQP, sem nada simulado)
convergem:

| Caso | Iterações | Volume alvo | Volume obtido |
|---|---|---|---|
| 2D, 40 pontos, V = 5 | 51 | 5,0 | 5,0000000000002 |
| 3D, malha 4×8, bico quadrado, V = 5 | 26 | 5,0 | 4,99999999999956 |

## Unificação da versão 3D em malha triangular

`src/Quad` (malha de quadriláteros) foi removida; o que ela trazia de novo foi
levado para `src/3D`, que é agora a única versão 3D:

- `energy_3D` e `volume_3D` vetorizadas (sem laço por triângulo), com a regra
  escolhida por `QuadRule` (`strang3`, `strang7`, `strang10`).
- `find_droplet_optimization_3d` ganhou `fix_boundary` (prende a linha de
  contato), `QuadRule` e `verbose`.
- Testes em `tests/test_malha_triangular.py`: a versão vetorizada coincide com
  o laço original (rel. 1e-12) nas três regras; `fix_boundary` fixa os limites
  do anel do corte; regra desconhecida lança `ValueError`.

## Pendências (não corrigidas)

Não são erros confirmados; ficam como sugestões.

- **Linha de contato em 3D:** no 2D o ponto do bico é fixo (`x[0] = 0`); no 3D
  os vértices de `cut_boundary` ficam livres. Confirmar se é intencional.
- **Volume 3D com superfície aberta:** a fórmula pelo teorema da divergência
  só está certa porque o bico fica no plano `z = 0` e a integral usa a origem.
  Vale documentar essa premissa no código ou fechar a superfície.
- **Desempenho do 3D:** as integrais já são vetorizadas, mas o SLSQP ainda
  estima o gradiente por diferenças finitas (malha 8×16: ~40 s). O próximo
  passo é passar o gradiente analítico em `jac=`.
- **Organização:** imports com `*`, `sys.path` relativo ao diretório atual em
  `run/`, ausência de `requirements.txt`, pasta `imgs/` que não existe (usada
  em `pendant.py`).

## Mudanças na versão 3D (bico circular, parede, gradiente)

- `fix_boundary` foi removido. Agora os vértices variáveis ficam em
  `Mesh.free_ids` e só os vértices do bico são fixos: no bico circular o anel
  do corte é a própria borda do bico (equivalente a `x[0] = 0` do 2D); no
  bico quadrado o anel continua livre e um colar o liga ao quadrado.
- A calota inicial tem exatamente o volume da gota: `initial_mesh` corrige
  `h` por `brentq` sobre o volume da própria malha (antes a esfera era
  deslocada `R/4` para baixo e o volume inicial ficava maior que `V0`).
- Os vértices passam a se mover em relação ao centro real da esfera.
- `grad_energy_3D` e `grad_volume_3D` dão os gradientes exatos, passados ao
  SLSQP em `jac=` (malha 8×16: de ~10 s para ~0,05 s). Isso resolve a
  pendência "Desempenho do 3D" acima.
- `wall_constraint` / `wall=`: parede plana `x <= w` como restrição linear.

## Dobra do colar no bico quadrado (corrigido)

- **Sintoma:** na animação da gota no bico quadrado aparecia uma
  "singularidade" perto do bico.
- **Causa:** o anel do corte da calota ficava 0,05 abaixo do bico, ligado ao
  quadrado por um colar de uma camada de triângulos, e era livre. Cada vértice
  anda na reta que sai do centro da esfera (abaixo do bico), então ao ir para
  fora ele também sobe. No meio dos lados do quadrado o anel saía do quadrado e
  subia acima de z = 0, e 12 dos 48 triângulos do colar viravam (malha 12×24,
  V = 20). A energia contava a dobra como superfície e o volume pela divergência
  deixava de valer.
- **Correção:** não há mais colar. O topo da calota é esticado na horizontal
  (`Mesh.fit_cut_to_polygon`) para o anel do corte ficar exatamente sobre o
  quadrado, e o anel é fixo, como no bico circular e como `x[0] = 0` no 2D.
- **Teste:** `test_malha_final_sem_dobras` exige que todos os tetraedros
  formados com um ponto interno da gota tenham volume positivo e que nenhum
  vértice passe de z = 0. Com o código antigo, 24 triângulos falham (malha 8×16).
