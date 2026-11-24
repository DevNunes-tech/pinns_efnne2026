# PINNs para EDPs — EFNNE 2026

Projeto reprodutível para investigar Physics-Informed Neural Networks (PINNs) em:
1. Equação unidimensional do calor;
2. Equação de Schrödinger dependente do tempo para uma onda plana livre e periódica.

## Importante
Este repositório contém o código experimental inicial, não resultados científicos prontos.
Os resultados devem ser gerados executando os experimentos no ambiente local. Não use
valores de exemplo como se fossem resultados observados.

## Requisitos
- Python 3.10 ou superior recomendado
- PyTorch
- NumPy
- Matplotlib

## Instalação (Linux/macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No Windows, ative o ambiente com `.venv\\Scripts\\activate`.

## Executar
Na raiz do projeto:
```bash
python src/heat_pinn.py
python src/schrodinger_pinn.py
```

Os scripts gravam gráficos em `figures/` e métricas em `results/`.

## Experimentos e limitações
- Equação do calor: domínio espacial [0, 1], tempo [0, 0.5], difusividade alpha=1,
  condições de Dirichlet homogêneas e condição inicial u(x,0)=sin(pi*x).
- Schrödinger: equação adimensional de partícula livre em domínio periódico [−pi, pi],
  com uma onda plana como solução de referência analítica exata. A implementação usa
  parte real e imaginária da função de onda e impõe periodicidade para a função e sua
  primeira derivada espacial. Esse caso é compatível com o domínio, mas não representa
  um pacote localizado. O script gaussiano exploratório não é o experimento principal e
  mantém a incompatibilidade entre a referência livre e o domínio periódico.
- Os tempos de execução dependem do computador, versão do PyTorch e dispositivo.
- Os PINNs podem falhar em convergir; isso deve ser relatado, não ocultado.
- As métricas são calculadas em pontos de avaliação não usados como pontos de colocação.
- A semente padrão é 2026; defina `PINN_SEED` para executar uma semente diferente.
- Para Schrödinger, a integral de |psi|² registrada nas métricas é avaliada em t=0.
