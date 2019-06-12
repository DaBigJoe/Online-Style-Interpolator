document.addEventListener('DOMContentLoaded', function() {
  let sliders = [];
  for (let i = 0; i < 5; i++) {
    let slider = document.getElementById("styleSlider" + i);
    let output = document.getElementById("styleValue" + i);
    output.innerHTML = slider.value;
    sliders.push(slider);

    slider.oninput = function() {
      // Update slider output value
      output.innerHTML = (this.value/100).toFixed(2);

      // Grab all interpolation values and call update method
      let interpolationValues = [];
      for (let j = 0; j < sliders.length; j++) {
        interpolationValues.push(sliders[j].value/100)
      }
      updateInterpolation(interpolationValues)
    }
  }
}, false);

function updateInterpolation(interpolation_values) {
  let values = {'values': interpolation_values};
  $.ajax({
    url: '/interpolate',
    type: 'POST',
    data: JSON.stringify(values),
    contentType: 'application/json'
    });
}
