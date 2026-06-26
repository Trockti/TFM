import { Component, OnInit } from '@angular/core';
import { MainService } from 'src/app/services/main.service';
import { DomSanitizer } from '@angular/platform-browser';
import { Word } from 'src/app/data/word';


@Component({
  selector: 'app-result',
  templateUrl: './result.component.html',
  styleUrls: ['./result.component.scss']
})
export class ResultComponent implements OnInit {

  text: string
  flag:string
  complexWords: Array<any>
  selected: boolean = false
  word: Word
  loading: boolean = true
  showResultText: boolean = true

  complexWordsString: Array<string>

  constructor(private mainService: MainService, private sanitizer: DomSanitizer) {
    this.complexWords = new Array()
    this.word = new Word()
    this.complexWordsString = new Array()
  }

  ngOnInit() {
    this.text = history.state.text
    this.flag = history.state.flag
    this.getComplexWords(this.text,this.flag)
  }

  async getComplexWords(text: string, flag: string) {
    console.log("Getting complex words for text: ", text, " with flag: ", flag);
    this.text = text
    this.complexWords = await this.mainService.api.getComplexWords(text, flag)
    console.log("Complex words found: ", this.complexWords);
    // La API ahora devuelve una lista simple de strings, no arrays
    for(let word of this.complexWords) {
      // Acceder a la palabra directamente porque ahora es un string
      this.replaceComplexWords(word, '', false)
    }
    this.addListener()
  }

  replaceComplexWords(word: string, synonym?: string, synonym2?: boolean) {
    let text = document.querySelector("#text");
    let regex = new RegExp(`\\b${word}\\b`, 'gi');
    if (synonym2==true){
      //console.log("TRUE!")
      text.innerHTML = text.innerHTML.replace(regex, `<div class=word>${word}<span class=wordtext>${synonym}</span></div>`)
    }
    else{
      //console.log("FALSE!")
      text.innerHTML = text.innerHTML.replace(regex, `<div class=word>${word}<span class=wordtext>${word}</span></div>`)
    }
    
    // this.sanitizer.bypassSecurityTrustHtml(text)    
  }

  addListener(){
    let words = document.querySelectorAll(".word");
    words.forEach(word => {
      word.addEventListener("click", () => {
        this.toggleDictionary(word.firstChild.textContent);
        if(this.isMobile()){
          this.showResultText = false
        }
      });
    });
  }

  async toggleDictionary(word: string) {
    //Obtener definicion y sinonimo
    if(! this.selected){
      this.selected = true
    }
    console.log(word)
    this.loading = true
    // Buscar la palabra compleja dentro de la lista
    // Modificar la palabra, sinonimos, definicion y pictograma que se muestra
    let complex_word = this.complexWords.find((element => element[4] == word))
    console.log(word, this.text)
    let result = await this.mainService.api.getDefinition(word, this.text)
    console.log(result)
    // this.word.synonyms = await this.mainService.api.getSynonyms(word, complex_word)
    // this.word.pictogram = await this.mainService.api.getPictogram(word)
    // if(! this.word.pictogram){
    //   this.word.pictogram = null
    // }
    this.word.synonyms = null
    this.word.pictogram = null
    this.word.word = word
    console.log(result)
    // this.word.definition = result['definition']
    // this.word.source = result['source']
    this.word.definition = result['definition']
    this.word.source = result['source']
    this.loading = false
  }

  goBack(){
    //Ocultar la información de la palabra y mostar el texto de nuevo
    this.showResultText = true
  }

  isMobile() {
    return !window.matchMedia("(min-width: 768px)").matches;
  }

  isHidden(elem) {
    var style = window.getComputedStyle(elem);
    return ((style.display === 'none') || (style.visibility === 'hidden'))
  }

  hideElement(elem) {
    elem.style.display = "none";
  }

  showElement(elem, type = "block") {
    elem.style.display = type;
  }

}