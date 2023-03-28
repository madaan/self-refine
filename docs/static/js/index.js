window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var current_cmd_idxs = {
        "gsm8k": 1,
        "gsm8khard": 1,
        "coloredobjects":1,
        "repeatcopy":1,
        "dateunderstanding":1
    }

    // examples
    $('select').on('click', function() {

        var sep_idx = this.value.indexOf('_');
        var domain_name = this.value.substring(0, sep_idx);
        var desired_cmd_idx = parseInt(this.value.substring(sep_idx + 1));
        var current_cmd_idx = current_cmd_idxs[domain_name];

        // hide current content
        var current_content = $('#content_' + domain_name + "_" + current_cmd_idx.toString());
        
        if (desired_cmd_idx == current_cmd_idx && current_content.is(":visible")) {
            current_content.hide();
            return;
        }
        current_content.hide();

        // show desired content
        var desired_content = $('#content_' + domain_name + "_" + desired_cmd_idx.toString());
        desired_content.show("slow");

        // set current to desired
        current_cmd_idxs[domain_name] = desired_cmd_idx;
    });



    // general function for xyzheader
    function toggle_options(header_id, options_id) {
        if ($(options_id).is(":visible")) {
            $(options_id).hide();
            // extract task name from header. e.g., #gsm8k_header -> gsm8k
            task_name = header_id.split("_")[0].substring(1);
            
            console.log("You have selected " + task_name + " as your task.");
            for (var i = 0; i <= 100; i++) {
                
                var content_id = "#content_" + task_name + "_" + i.toString();
                console.log(content_id);
                // check if content exists
                if ($(content_id).length == 0) {
                    break;
                }
                $(content_id).hide();
            }
            $(header_id).removeClass("is-active");
        } else {
            $(options_id).show("slow");
            $(header_id).addClass("is-active");
        }
    }

    $('#gsm8k_button').click(function() {
        toggle_options('#gsm8k_header', '#gsm8k_options');
    });
    $('#gsm8khard_button').click(function() {
        toggle_options('#gsm8khard_header', '#gsm8khard_options');
    }
    );
    $('#coloredobjects_button').click(function() {
        toggle_options('#coloredobjects_header', '#coloredobjects_options');
    }
    );
    $('#repeatcopy_button').click(function() {
        toggle_options('#repeatcopy_header', '#repeatcopy_options');
    }
    );
    $('#dateunderstanding_button').click(function() {
        toggle_options('#dateunderstanding_header', '#dateunderstanding_options');
    }
    );

    $('#gsm8khard_options').hide();
    $('#coloredobjects_options').hide();
    $('#repeatcopy_options').hide();
    $('#dateunderstanding_options').hide();
    

})




