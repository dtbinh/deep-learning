\ProvidesClass{amsdlaj}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[11pt,letterpaper]{article}


\def\FullBox{\hbox{\vrule width 8pt height 8pt depth 0pt}}

\def\qed{\ifmmode\qquad\FullBox\else{\unskip\nobreak\hfil
\penalty50\hskip1em\null\nobreak\hfil\FullBox
\parfillskip=0pt\finalhyphendemerits=0\endgraf}\fi}

\def\qedsketch{\ifmmode\Box\else{\unskip\nobreak\hfil
\penalty50\hskip1em\null\nobreak\hfil$\Box$
\parfillskip=0pt\finalhyphendemerits=0\endgraf}\fi}

\newenvironment{solution}
  {\begin{trivlist} \item {\bf Solution:~~} \sf}
  {\qed\end{trivlist}}


\newcommand{\relu}{ReLU}
