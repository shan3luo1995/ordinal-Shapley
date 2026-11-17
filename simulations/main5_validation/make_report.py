#!/usr/bin/env python3
"""Generate the Chinese report source from the saved default 100-experiment run."""
from pathlib import Path
import json
import pandas as pd
PREAMBLE = r'''\documentclass[11pt,a4paper]{article}
\usepackage[left=24mm,right=24mm,top=22mm,bottom=20mm,includefoot]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx}
\usepackage{iftex}
\ifPDFTeX
  \usepackage{CJKutf8}
  % CJKutf8 supplies its own Unicode map for each Chinese subfont.
  \makeatletter
  \AddToHook{cmd/CJK@addcmap/before}{\pdfnobuiltintounicode\font@name}
  \makeatother
\else
  \usepackage[no-math]{fontspec}
  \IfFontExistsTF{Noto Serif CJK SC}{
    \setmainfont{Noto Serif CJK SC}
    \setsansfont{Noto Sans CJK SC}
  }{
    \setmainfont{FandolSong-Regular.otf}[BoldFont=FandolSong-Bold.otf]
    \setsansfont{FandolHei-Regular.otf}[BoldFont=FandolHei-Bold.otf]
  }
  \ifXeTeX
    \XeTeXlinebreaklocale "zh"
    \XeTeXlinebreakskip=0pt plus 0.5pt
  \fi
\fi
\pagestyle{plain}
\raggedbottom
\linespread{1.13}
\setlength{\parindent}{2em}
\setlength{\parskip}{3pt plus 1pt}
\setlength{\jot}{3pt}
\DeclareMathSizes{10.95}{10.95}{9}{7}
% Use a single stroke for Chinese headings when pdfLaTeX has no bold CJK font.
\newcommand{\headingweight}{\ifPDFTeX\mdseries\else\bfseries\fi}
\makeatletter
\renewcommand{\section}{\@startsection{section}{1}{0pt}
  {2.0ex plus .3ex minus .2ex}{.9ex}{\normalfont\large\headingweight}}
\makeatother

\newcommand{\indep}{\mathrel{\perp\!\!\!\perp}}
'''
BODY = r'''\begin{document}
\ifPDFTeX\begin{CJK*}{UTF8}{gbsn}\fi
\begin{center}
{\LARGE\headingweight 有序结局 Shapley 责任：100 次模拟核对}\\[5pt]
{\large 对 \texttt{main5.tex} 识别定理的数值验证}
\end{center}
\section*{设计}
随机种子为 \texttt{@@SEED@@}。独立生成 100 个满足报告假设的总体，
每个总体生成一条长度为 $50\,000$ 的随机试验数据流。
分别使用前 $2\,000$、$10\,000$ 和 $50\,000$ 个观测，
在每个样本量下四个暴露组各占四分之一。
每个样本量有 100 次独立实验，合计 300 个样本、1\,800 个责任估计值。
同一总体的三个样本相互嵌套。

类型顺序为 $T=(Y_{00},Y_{01},Y_{10},Y_{11})$，其中 $Y_{ab}\in\{0,1,2\}$。
核对 $\mathcal E_1$ 和 $\mathcal E_2$ 下两个暴露及背景的平均责任。
四条相邻增量均在 $[0,1]$ 内，且要求
\[
Y_{11}\indep Y_{00}\mid Y_{10},\qquad
Y_{11}\indep Y_{00}\mid Y_{01}.
\]

令 $B,D_1,D_2$ 相互独立，$B$ 取 $0,1,2$，并设
\[
\begin{aligned}
Y_{00}&=B,&Y_{01}&=\min(2,B+D_2),\\
Y_{10}&=\min(2,B+D_1),&Y_{11}&=\min(2,B+D_1+D_2).
\end{aligned}
\]
$B$ 的三个取值概率为 $0.1(1,1,1)+0.7V$，其中
$V\sim\mathrm{Dirichlet}(2,2,2)$。
给定独立抽取的 $\rho_1,\rho_2\sim\mathrm{Unif}(0.15,0.85)$，
$D_j\sim\mathrm{Bernoulli}(\rho_j)$。
该基础模型满足四条单步增量限制及两条 Markov 条件。

进一步改变两个中间潜在结果的联合取值：
将类型 $0001$ 和 $0111$ 的概率各增加同一个量，
并将类型 $0011$ 和 $0101$ 的概率各减少该量。
对类型 $1112$、$1222$ 与 $1122$、$1212$ 作同样调整。
两次调整量分别为对应可减最小概率的 $0.1$ 至 $0.8$ 倍，独立均匀抽取。
调整不改变任一加入顺序下三个潜在结果的联合分布，
因此保持两条 Markov 条件。每个生成总体均对全部 12 种单步增量相容类型赋予正概率。

\section*{总体层面的核对}
真实责任直接按完整类型分布加权计算。公式值只使用四个潜在结果的边际分布，
主分析固定使用经 $Y_{10}$ 的公式识别 $\theta=\mathrm{pr}(T=0112)$。
两条 Markov 条件通过枚举正概率条件层，计算条件联合概率与条件边际概率乘积的最大差核对。
\begin{center}
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{@{}lr@{}}
\toprule
核对项目 & 结果\\
\midrule
通过总体假设及识别公式核对 & $100/100$\\
两条 Markov 条件的最大数值残差 & $@@MARKOV@@$\\
$\theta$ 的公式值与真实值的最大绝对差 & $@@THETA@@$\\
六项责任公式值与真实值的最大绝对差 & $@@ORACLE@@$\\
另行检查的零分母边界例子 & $3/3$\\
\bottomrule
\end{tabular}
\end{center}
额外的三个确定性边界例子不计入 100 次随机实验，核对了
$\mathrm{pr}(Y_{10}=1)=0$ 以及某个病例等级发生概率为零时的公式处理。
在 100 个随机总体中，两个病例等级的最小概率为 @@MINCASE@@，
两种中间潜在结果取 $1$ 的最小概率为 @@MINMIDDLE@@。

\clearpage
\section*{有限样本结果}
估计时仅使用观测到的暴露组合和结局，以各组经验比例及经验均值代入识别公式。
不使用个体未观测到的潜在结果，也不对估计值截断或施加形状约束。
设第 $r$ 个总体中的真实责任为 $\mathcal R_{j,r}$，定义
\[
\overline e=\frac1{100}\sum_{r=1}^{100}
  (\widehat{\mathcal R}_{j,r}-\mathcal R_{j,r}),\qquad
\mathrm{RMSE}
 =\left\{\frac1{100}\sum_{r=1}^{100}
  (\widehat{\mathcal R}_{j,r}-\mathcal R_{j,r})^2\right\}^{1/2}.
\]
下表汇总不同总体和抽样数据共同形成的误差；
$\overline e$ 不是某一个固定总体下的 Monte Carlo 偏差。单位为等级分数。
\begin{center}
\small
\renewcommand{\arraystretch}{1.35}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}ccrrrrrr@{}}
\toprule
 & & \multicolumn{2}{c}{$n=2\,000$}
   & \multicolumn{2}{c}{$n=10\,000$}
   & \multicolumn{2}{c}{$n=50\,000$}\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}
病例 & 责任 & $\overline e$ & RMSE & $\overline e$ & RMSE & $\overline e$ & RMSE\\
\midrule
@@ROWS@@
\bottomrule
\end{tabular}
\end{center}
将六项责任合并计算，三个样本量下的综合 RMSE 分别为
@@POOLED@@。每项责任的 RMSE 均随样本量增大而下降。
平均误差的 Monte Carlo 标准误及逐次结果见配套 CSV 文件。
\begin{center}
\includegraphics[width=\linewidth]{rmse.pdf}
\end{center}
图中每条曲线对应一个病例等级和一项责任；每个点汇总 100 次独立实验。

全部 1\,800 个责任估计均有定义，未出现经验分母为零。
其中 4 项原始估计为负：$n=2\,000$ 时有 3 项，$n=10\,000$ 时有 1 项，
$n=50\,000$ 时为 0 项；均为一级病例的暴露责任，最小值为 $-0.0244$。
这是无约束代入估计的有限样本表现，不是总体识别公式失效。

\clearpage
\section*{真值与估计值}
下图显示 $n=10\,000$ 时每个总体的真实责任与观测数据估计值；
虚线为两者相等的位置。
\begin{center}
\includegraphics[width=\linewidth]{truth_vs_estimate.pdf}
\end{center}
\section*{完整联合分布与 Markov 条件}
对每个有效总体另构造一个具有不同中间联合取值的总体。
这 100 对总体的完整类型分布均不同，最小总变差距离为 @@TV@@；
四个边际分布及两个路径三元分布保持一致，
六项真实平均责任的最大差为 $@@ALTERR@@$。
这与“平均责任可识别、完整联合分布未必可识别”的结论一致。

另构造 100 个敏感性对照：保留四个潜在结果的边际分布和单步增量限制，
但改变 $\mathrm{pr}(T=0112)$，使 Markov 条件不再成立。
相同边际分布代入原公式后，其结果与新的真实责任出现偏差。
每个对照总体六项责任中最大绝对误差的平均为 @@BADMEAN@@，
所有对照中的最大绝对误差为 @@BADMAX@@。
这些对照不是主模拟中的有效总体，说明识别结论依赖额外的 Markov 条件。

本次结果支持报告公式在上述生成机制中的数值正确性，并展示了原始代入估计的抽样误差。
模拟核对不能替代定理证明，也未穷尽所有满足假设的联合分布。
全部代码、参数、类型概率、逐次估计和汇总结果均随报告保存。

\clearpage
\section*{补充：Markov 条件与可检验限制}
本次主模拟核对的是 \texttt{main5.tex} 中同时施加两条 Markov 条件的定理。
在两个中间等级概率均为正时，分别沿两条加入顺序计算可得
\[
\begin{aligned}
\theta_{10}
&=\frac{\{\mathrm{pr}(Y_{00}=0)-\mathrm{pr}(Y_{10}=0)\}(\xi_{11}-\xi_{10})}
        {\mathrm{pr}(Y_{10}=1)},\\[5pt]
\theta_{01}
&=\frac{\{\mathrm{pr}(Y_{00}=0)-\mathrm{pr}(Y_{01}=0)\}(\xi_{11}-\xi_{01})}
        {\mathrm{pr}(Y_{01}=1)}.
\end{aligned}
\]
两条条件同时成立时，必须有 $\theta=\theta_{10}=\theta_{01}$。
两式仅涉及可识别的边际概率，因此在假设 1、2 成立的前提下，
$\theta_{10}\ne\theta_{01}$ 可以否定假设 3。
有限样本中应检验两项估计的差，而不能因数值不完全相等就拒绝假设。
逐次的两个估计值保存在 \texttt{sample\_diagnostics.csv} 中。

例如，令类型 $0011$ 与 $1112$ 各占一半。
该总体满足单步增量限制，但其可观测边际给出
\[
\theta_{10}=\frac{(1/2-0)(1/2-0)}{1}=\frac14,
\qquad
\theta_{01}=\frac{(1/2-1/2)(1/2-0)}{1/2}=0.
\]
因此，没有满足两条 Markov 条件的潜在结果联合分布能够生成这些边际分布。
反之，两个表达式相等并不确认真实的潜在结果满足 Markov 条件。
前述 100 个敏感性对照与有效总体有相同边际分布，故仍满足该等式，
但它们的 Markov 条件均不成立。

对于当前 $p=2$、三个结局等级的责任识别，
可以仅选择并假设 $Y_{11}\indep Y_{00}\mid Y_{10}$ 成立。
此时 $\theta=\theta_{10}$ 已足够识别全部六项平均责任，
无需再要求另一条 Markov 条件。
也可以事先选择经 $Y_{01}$ 的条件，使用 $\theta_{01}$。
“选定一条并假设其成立”与“至少一条成立但不知道是哪条”不同；
后一种表述一般不保证点识别。
本次主模拟总体满足较强的双条件模型，并未对所有仅满足单条条件的总体作穷尽核对。
\ifPDFTeX\end{CJK*}\fi
\end{document}
'''

def scientific(x):
    mantissa, exponent = f"{x:.3e}".split("e")
    return mantissa + r"\times10^{" + str(int(exponent)) + "}"

def main():
    out = Path(__file__).resolve().parent/"results"
    summary = json.loads((out/"summary.json").read_text())
    metrics = pd.read_csv(out/"metrics.csv")
    if (summary["config"]["experiments"] != 100
            or summary["config"]["sample_sizes"] != [2000,10000,50000]
            or summary["config"]["seed"] != 20261117):
        raise ValueError("This report template describes the prespecified 100-experiment design and seed.")
    exact = summary["exact_checks"]
    rows = []
    for k in (1, 2):
        for j in (0, 1, 2):
            label = "背景" if j == 0 else "$X_"+str(j)+"$"
            vals = []
            for n in summary["config"]["sample_sizes"]:
                d = metrics[(metrics.n == n) & (metrics.grade == k) & (metrics.cause == j)].iloc[0]
                vals.extend([f"{d.mean_error:.4f}", f"{d.rmse:.4f}"])
            rows.append("$\\mathcal E_"+str(k)+"$ & "+label+" & "+" & ".join(vals)+r"\\")
        if k == 1:
            rows.append(r"\addlinespace[3pt]")
    replacements = {
        "SEED": str(summary["config"]["seed"]),
        "MARKOV": scientific(exact["max_markov_residual"]),
        "THETA": scientific(exact["max_theta_error"]),
        "ORACLE": scientific(exact["max_oracle_error"]),
        "MINCASE": f'{exact["min_case_probability"]:.4f}',
        "MINMIDDLE": f'{exact["min_middle_probability"]:.4f}',
        "ROWS": "\n".join(rows),
        "POOLED": "、".join("$"+f"{v:.4f}"+"$" for v in summary["finite_sample"]["pooled_rmse_by_n"].values()),
        "TV": f'{exact["alternative_min_tv"]:.4f}',
        "ALTERR": scientific(exact["alternative_max_responsibility_error"]),
        "BADMEAN": f'{exact["violation_mean_max_error"]:.4f}',
        "BADMAX": f'{exact["violation_max_error"]:.4f}',
    }
    body = BODY
    for key, value in replacements.items():
        body = body.replace("@@"+key+"@@", value)
    if "@@" in body:
        raise ValueError("Unfilled report marker")
    (out/"report.tex").write_text(PREAMBLE+body)
    print(out/"report.tex")

if __name__ == "__main__":
    main()

