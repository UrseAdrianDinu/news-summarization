import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import {FormControl, FormGroup, FormsModule, ReactiveFormsModule, Validators} from "@angular/forms";
import {HttpClient, HttpClientModule, HttpHeaders} from "@angular/common/http";
import {map, Observable} from "rxjs";


@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, ReactiveFormsModule, HttpClientModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'server';
  inputText: string = '';
  apiUrl = "https://api-inference.huggingface.co/models/readerbench/RoSummary-medium";

  pgnUrl = "http://localhost:5000/api/predict"
  headers = new HttpHeaders({
    'Content-Type':'application/json',
    Authorization: "Bearer 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"
  });
  responseData: Array<any> | null = null;

  inputForm = new FormGroup({
    inputText: new FormControl('', {nonNullable:true, validators:[Validators.required]})
  });
  roGPTOutputText: string = 'ROGPT\'s output will appear here';
  modelOutputText: string = "Our model's output will appear here";

  constructor(protected http: HttpClient) {
  }
  summarize() {
    console.log(this.inputForm.getRawValue().inputText);
    var before = "\"Boema\" lui Puccini, pe timp de pandemie. Primul spectacol de operă live drive-in din Europa." + " " +
      "Englezilor le place să facă lucrurile altfel. Și încearcă să nu te dezamăgească niciodată. Primul spectacol de operă live drive-in din Europa a avut " +
      "loc în weekend la Londra. S-a jucat \"Boema\" lui Puccini, în parcarea de la Alexandra Palace.Sala de evenimente, faimoasă în oraş, e închisă încă de la începutul pandemiei." +
      "Așa că spectacolul a fost adaptat vremurilor. Spectatorii au stat în mașini și au fost serviţi cu îngheţată, " +
      "ciocolată şi băuturi.Şi-au reglat radiourile pe aceeaşi frecvenţă, ca să asculte operaAlții au preferat să stea cu " +
      "geamurile deschise şi să audă direct de pe scenă.La final, în loc de aplauze, artiștii au primit claxoane. " +
      "Tot în semn de admirație…Opera de la Bucureşti te aşteaptă şi ea din nou la spectacole începând din weekendul ăsta, " +
      "după mai bine de şase luni de pauză.Doar 130 din cele 915 bilete vor fi scoase la vânzare, pentru a respecta" +
      " normele impuse de autorităţi."
    var after = ""
    const result1 = this.http.post(this.apiUrl, JSON.stringify(before), { headers: this.headers, observe: 'response' });
    result1.subscribe(response => {
      this.responseData = response.body as Array<any>;
      if (this.responseData) {
        after = this.responseData[0].generated_text;
        while (after.length > before.length)
        {
          before = after;
          const result = this.http.post(this.apiUrl, JSON.stringify(before), { headers: this.headers, observe: 'response' });
          result.subscribe(response => {
            this.responseData = response.body as Array<any>;
            if (this.responseData) {
              after = this.responseData[0].generated_text;
            }
          }, error => {
            console.error(error);
          });
        }
      }
    }, error => {
      console.error(error);
    });
  }

  makeHttpRequest(before: string, apiUrl: string, headers: HttpHeaders): Observable<Array<any>> {
    return this.http.post(apiUrl, JSON.stringify(before), { headers, observe: 'response' })
      .pipe(
        map(response => response.body as Array<any>)
      );
  }

  pgnHttpRequest(text: string, apiUrl: string, headers: HttpHeaders) {
    const body = { text: text };
    return this.http.post(apiUrl, body, { headers, observe: 'response' })
      .subscribe(response=> {
          // Print the result string
          const responseBody = response.body;
          this.modelOutputText = responseBody as string
          this.modelOutputText = this.modelOutputText .charAt(0).toUpperCase() + this.modelOutputText .slice(1)
        },
        error => {
          // Handle errors if any
          console.error('Error:', error);
        });
  }

  performRecursiveRequests(before: string, apiUrl: string, headers: HttpHeaders): void {
    this.makeHttpRequest(before, apiUrl, headers).subscribe(
      (response: Array<any>) => {
        if (response && response.length > 0) {
          const after = response[0].generated_text;

          if (before !== after) {
            // Continue making requests recursively
            before = after;
            const summaryIndex = before.indexOf('Summary:');
            var summaryText = before.substring(summaryIndex + 'Summary:'.length).trim();
            console.log(summaryText)
            this.roGPTOutputText = summaryText;
            this.performRecursiveRequests(after, apiUrl, headers);
          } else {
            // The strings are equal, do something
            const summaryIndex = before.indexOf('Summary:');
            var summaryText = before.substring(summaryIndex + 'Summary:'.length).trim();
            this.roGPTOutputText = summaryText;
            console.log('Strings are equal:', before);
            console.log('Summary Text:', summaryText);
          }
        }
      },
      error => {
        console.error(error);
      }
    );
  }
  startRequests() {
    var pgnInputTextUser = this.inputForm.getRawValue().inputText
    var roGPTInputTextUser = pgnInputTextUser + " Summary:"
    this.pgnHttpRequest(pgnInputTextUser, this.pgnUrl, this.headers);
    this.performRecursiveRequests(roGPTInputTextUser, this.apiUrl, this.headers);

  }

}
