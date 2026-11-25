# Roteiro do vídeo: a forma de uma gota

**Duração:** cerca de 10 minutos. **Narração:** Gabriel. **Público:** até o ensino médio.

A narração tem cerca de 960 palavras, o que dá uns 7 minutos falando com calma
(140 palavras por minuto); o plano para chegar a 10 minutos está em `transicoes.md`. Os tempos são aproximados. A coluna **Tela** indica a cena
de `cenas.py` (renderizada com `manim -qh cenas.py NomeDaCena`) ou uma filmagem.

Regras de linguagem: nada de derivada, gradiente ou multiplicador. Quando uma ideia
técnica aparecer, ela vem com uma comparação do dia a dia.

---

## 1. Abertura (0:00 – 0:45)

| Tela | Narração |
|---|---|
| Filmagem: torneira pingando, em câmera lenta se possível. | Repare numa torneira pingando. A gota não cai assim que aparece: ela vai crescendo, fica pendurada, estica… e, de repente, se solta. |
| `Abertura`: título "Qual é a forma de uma gota?" e uma gota pendurada crescendo. | Por que a gota tem exatamente essa forma, meio redonda e meio esticada? E por que ela cai num certo momento, e não antes? Neste vídeo a gente vai responder isso com uma ideia muito poderosa da matemática: a **otimização**, que é a arte de encontrar a melhor opção entre todas as possíveis. |

## 2. A pele da água (0:45 – 2:00)

| Tela | Narração |
|---|---|
| Filmagem ou imagem: inseto andando sobre a água, bolha de sabão. | A superfície da água se comporta como uma pele elástica bem fininha. É por isso que alguns insetos conseguem andar sobre a água, e que uma bolha de sabão fica redondinha. Essa "pele" tem um nome: **tensão superficial**. |
| `PeleDaAgua`: três formas com a mesma área (círculo, elipse, forma irregular) e o contorno de cada uma medido num contador. | Uma pele esticada guarda energia, como um elástico. Quanto maior a superfície, mais energia. Então a água "prefere" ter a menor superfície possível. Olha só: essas três figuras têm o mesmo tamanho por dentro, mas o círculo é o que tem a menor borda. No espaço, quem faz esse papel é a esfera. |
| Gota redonda flutuando (imagem de astronauta com gota no espaço, ou animação). | É por isso que, no espaço, onde quase não sentimos o peso, uma gota de água fica perfeitamente redonda. |

## 3. A gravidade puxa (2:00 – 3:00)

| Tela | Narração |
|---|---|
| `CaboDeGuerra`: a gota muda de forma, de achatada a muito esticada; à direita, duas barras: "energia da pele" e "energia da altura", e embaixo a soma. | Aqui na Terra tem mais alguém nessa disputa: a gravidade. Ela quer a água o mais baixo possível, e isso estica a gota para baixo. Só que esticar aumenta a superfície, e a pele reclama. |
| Mesmo cena: marcador no ponto onde a soma é a menor. | É um cabo de guerra. Se a gota fica muito redonda, a gravidade perde. Se fica muito esticada, a pele perde. A forma que a gente vê é o **ponto de equilíbrio**: aquela em que a soma das duas energias é a menor possível. |

## 4. Otimizar é achar o melhor (3:00 – 4:45)

| Tela | Narração |
|---|---|
| `DescendoOMorro`: uma curva (um "morro") e uma bolinha descendo em passos até o fundo do vale. | Achar o mínimo de alguma coisa é um problema de otimização. Imagine que você está num morro com muita neblina e quer chegar ao ponto mais baixo. Você não enxerga o vale, mas sente o chão sob os pés: dá um passo na direção que desce, depois outro, e outro, até que nenhum passo desça mais. |
| Mesma cena: aparece a frase "regra: a quantidade de água não muda". | Mas a gota tem uma regra a cumprir: a quantidade de água é fixa. Ela não pode simplesmente sumir para gastar menos energia. Em matemática isso se chama **restrição**. |
| `TrilhaRestrita`: um mapa com curvas de nível e uma trilha em forma de elipse; os passos do método vão chegando ao ponto mais baixo da trilha. | Com uma restrição, é como se a gente fosse obrigado a andar só sobre uma trilha. O método que usamos, chamado SLSQP, faz assim: a cada passo, ele monta um "mapa simplificado" do terreno perto de onde está, encontra o melhor ponto nesse mapa, respeitando a trilha, e anda até lá. No começo ele erra bastante, mas a cada passo o mapa fica mais parecido com o terreno de verdade, até que ele chega ao ponto mais baixo da trilha. |

## 5. O computador procura a gota (4:45 – 6:15)

| Tela | Narração |
|---|---|
| `GotaSeAjeitando` (parte 1): perfil da gota feito de 30 pontos ligados por retas; cada ponto anda sobre uma reta que sai do centro. | Para o computador, a gota é um desenho feito de pontos ligados por retas. Cada ponto pode andar para dentro ou para fora. O computador calcula a energia desse desenho e o volume de água dentro dele. |
| `GotaSeAjeitando` (parte 2): o perfil vai mudando a cada passo; contadores de energia e de volume ao lado. | Começamos com um pedaço de esfera com a quantidade certa de água. A cada passo, o método mexe em todos os pontos ao mesmo tempo para baixar a energia, sem mudar o volume. Repare: a energia vai caindo, e o volume fica parado em 20. Depois de umas trinta e poucas rodadas, a gota não muda mais. |
| `GotaSeAjeitando` (parte 3): a curva da física (EDO) aparece por cima dos pontos. | E aqui vem a melhor parte. Existe outro jeito, mais antigo, de calcular a forma da gota: usando as leis da física diretamente, com uma equação. Quando colocamos as duas respostas uma em cima da outra, elas coincidem. Dois caminhos completamente diferentes chegam à mesma gota. |

## 6. Crescendo até cair (6:15 – 7:30)

| Tela | Narração |
|---|---|
| `CrescendoAteCair` (parte 1): gotas de volume crescente, uma depois da outra, com o volume num contador. | Agora vamos colocar cada vez mais água. A gota cresce e estica. Até certo ponto, para cada quantidade de água existe uma forma de equilíbrio. |
| `CrescendoAteCair` (parte 2): a gota de volume máximo pisca. | Mas existe um limite. Para esse bico, quando o volume passa de um certo valor, não existe mais nenhuma forma em que a gota fique pendurada em equilíbrio. A pele não consegue mais segurar o peso. |
| `CrescendoAteCair` (parte 3): a forma estrangulada, com um "pescoço" fino, aparece ao lado. | Quando pedimos para o computador procurar a forma de menor energia perto desse limite, às vezes ele encontra esta aqui: uma gota com um pescoço bem fino, quase se soltando. Ela tem ainda menos energia que a gota pendurada. É exatamente o que acontece na torneira: a gota se solta. |

## 7. Em três dimensões (7:30 – 9:15)

| Tela | Narração |
|---|---|
| `Gota3D` (parte 1): a esfera inicial feita de triângulos, girando. | Até agora a gota era só um desenho de lado, porque ela é redonda em volta do bico. Mas e se o bico não for redondo? Aí precisamos da gota inteira, em três dimensões. O computador a monta com centenas de triangulinhos. |
| `Gota3D` (parte 2): a malha se deforma até a gota final enquanto a câmera gira. | O método é o mesmo: os cantos dos triângulos se mexem para baixar a energia, sempre com a mesma quantidade de água. São muito mais pontos, então fizemos uma conta extra que diz ao computador, de uma vez só, para que lado cada ponto deve ir. Isso deixou o cálculo muitas vezes mais rápido. |
| `Comparacao3D`: os pontos da gota 3D sobre a curva da equação. | Primeiro, um teste: com o bico redondo, a gota em 3D tem que dar a mesma resposta da equação. E dá, cada vez mais perto conforme usamos triângulos menores. |
| `QuadradoEParede` (parte 1): gota no bico quadrado, girando. | Agora, o que a equação antiga não consegue fazer. Num bico quadrado, a gota fica com cantinhos perto do bico, mas logo abaixo ela volta a ficar redonda. |
| `QuadradoEParede` (parte 2): gota no bico redondo encostado numa parede; a parede aparece como um plano. | E se o bico estiver encostado numa parede? A gota não pode atravessar a parede: isso é mais uma restrição, uma regra do tipo "não pode passar daqui". Ela se achata contra a parede, cresce para o outro lado e fica um pouco mais comprida. |

## 8. Para que serve (9:15 – 10:00)

| Tela | Narração |
|---|---|
| `Fechamento`: lista com ícones simples: medidor de tensão superficial, impressora jato de tinta, conta-gotas. | Isso não é só curiosidade. Medindo a forma de uma gota pendurada, laboratórios descobrem a tensão superficial de um líquido. Impressoras a jato de tinta precisam controlar o tamanho de cada gotinha. E o conta-gotas de um remédio funciona porque cada gota tem sempre o mesmo tamanho. |
| `Fechamento`: a gota final, e o título "A natureza também otimiza". | No fim das contas, a gota faz sozinha, num piscar de olhos, o que o computador leva vários passos para fazer: encontra a forma de menor energia que respeita as regras. A natureza também otimiza. |

---

## Observações para a gravação

- As filmagens (torneira, inseto, bolha) são opcionais: dá para trocar por imagens
  livres ou pular e usar só as cenas.
- Se o tempo apertar, a seção 4 pode perder a cena `DescendoOMorro` e ir direto à
  `TrilhaRestrita` (ganha uns 30 segundos).
- Os números falados (volume 20, "trinta e poucas rodadas", "muitas vezes mais
  rápido") vêm de `video/gerar_dados.py` e de `relatorio/gerar_figuras.py`. Vale
  conferir depois de rodar tudo de novo.
