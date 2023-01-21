           
    $('.tab-button').eq(0).addClass('active');
    $('.tab-content').eq(0).addClass('show');

for(let i = 0; i < $('.tab-button').length; i++){
    $('.tab-button').eq(i).click(function(){
    탭열기(i)
});
}

/*$('.list').click(function(e){
    탭열기(e.target.dataset.id)
});*/


function 탭열기(i){
    $('.tab-button').removeClass('active');
    $('.tab-content').removeClass('show');
    $('.tab-button').eq(i).addClass('active');
    $('.tab-content').eq(i).addClass('show');  
}