document.addEventListener('DOMContentLoaded', function() {

  let passSliderValuesForInterpolation = function(sliders) {
    let interpolationValues = [];
    for (let j = 0; j < sliders.length; j++) {
      interpolationValues.push(sliders[j].value/100)
    }
    updateInterpolation(interpolationValues);
  };

  let sliders = [];
  for (let i = 0; i < 5; i++) {
    let slider = document.getElementById("styleSlider" + i);
    let output = document.getElementById("styleValue" + i);
    output.innerHTML = (slider.value/100).toFixed(2);
    sliders.push(slider);
    slider.oninput = function() {
      // Update slider output value
      output.innerHTML = (this.value/100).toFixed(2);

      // // Grab all interpolation values and call update method
      passSliderValuesForInterpolation(sliders)
    }
  }

  // let render_button = document.getElementById("render_button");
  // render_button.onclick = function() {
  //   // Grab all interpolation values and call update method
  //   passSliderValuesForInterpolation(sliders)
  // };

  let upload_button = document.getElementById("upload_button");
  $(upload_button).change(function () {
    console.log("File changed");

    // Form Data
    let formData = new FormData();

    if(upload_button.files && upload_button.files.length === 1){
      let file = upload_button.files[0];
      formData.set("file", file , file.name);
    }

    // let reader = new FileReader();
    // reader.onload = function(e) {
    //   $('#content_input').attr('src', e.target.result);
    // };
    // reader.readAsDataURL(upload_button.files[0]);

    let request = new XMLHttpRequest();
    request.onreadystatechange = function() {
      if (request.readyState === XMLHttpRequest.DONE) {
        $('#content_input').attr('src', 'data:image/jpeg;base64,' + request.responseText);
        passSliderValuesForInterpolation(sliders);
      }
    };
    request.open('POST', "/upload");
    request.send(formData);
  });

  passSliderValuesForInterpolation(sliders);
}, false);

function updateInterpolation(interpolation_values) {
  let values = {'values': interpolation_values};
  let output_image = document.getElementById("style_output");
  $.ajax({
    url: '/interpolate',
    type: 'POST',
    data: JSON.stringify(values),
    contentType: 'application/json',
    success: function (data) {
      if(data !== "")
        output_image.src = 'data:image/jpeg;base64,' + data;
    }
  });
}
