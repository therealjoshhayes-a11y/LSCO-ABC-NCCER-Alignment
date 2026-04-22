$(document).ready(function(){
  $('.mobile-menu-toggle').click(function () {
    $(this).toggleClass('open');
    $('.block_n2_links.links_table').toggleClass('open');
    $('.block_n2_tools').toggleClass('open');
  });
  // Add keyboard shortcuts as per https://www.w3.org/WAI/ARIA/apg/patterns/accordion/
  $('.mobile-menu-toggle').keypress(function (event) {
    if(event.keyCode == 13 || event.keyCode == 32) { // enter or space 
      event.preventDefault();
      $(this).click();
    }
  });
  
  $(window).resize(function() {
    if ($(document).width() > 768) {
      $('.mobile-menu-toggle, .block_n2_links.links_table, .block_n2_tools').removeClass('open');
    }
  });
});