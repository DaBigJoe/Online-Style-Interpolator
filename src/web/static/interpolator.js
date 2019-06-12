document.addEventListener('DOMContentLoaded', function() {
  for (let i = 0; i < 5; i++) {
    let slider = document.getElementById("styleSlider" + i);
    let output = document.getElementById("styleValue" + i);
    output.innerHTML = slider.value;

    slider.oninput = function() {
      output.innerHTML = (this.value/100).toFixed(2);
    }
  }
}, false);