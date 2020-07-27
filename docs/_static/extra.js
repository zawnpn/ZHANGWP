//pangu
document.addEventListener('DOMContentLoaded', () => {
// listen to any DOM change and automatically perform spacing via MutationObserver()
  pangu.autoSpacingPage();
});
//mathjax
window.MathJax = {
    tex: {
      inlineMath: [ ["\\(","\\)"] ],
      displayMath: [ ["\\[","\\]"] ],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    }
  };
// // Re-render MathJax on document switch (instant loading, custom event)
// document.addEventListener("DOMContentSwitch", function() {
//   MathJax.typesetPromise();
// });