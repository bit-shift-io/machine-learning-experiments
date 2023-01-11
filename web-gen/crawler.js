import { screenshotWebsite } from './website.js'
import chunk from 'lodash/chunk.js'
import puppeteer from 'puppeteer';

/*
const Crawler = require('crawler');

const c = new Crawler({
    maxConnections: 10,
    // This will be called for each crawled page
    callback: (error, res, done) => {
        if (error) {
            console.log(error);
        } else {
            const $ = res.$;
            // $ is Cheerio by default
            //a lean implementation of core jQuery designed specifically for the server
            console.log($('title').text());
        }
        done();
    }
});

// Queue just one URL, with default callback
//c.queue('http://www.amazon.com');

// Queue a list of URLs
c.queue(['http://www.google.com/','http://www.yahoo.com']);
*/

const browser = await puppeteer.launch()
const r = await fetch('https://raw.githubusercontent.com/Kikobeats/top-sites/master/top-sites.json').then(r => r.json())
const c = chunk(r, r.length / 10)
const p = c.map(async (arr, idx) => {
    for (const w of arr) {
        console.log(`Processing: ${w.rootDomain}`)
        await screenshotWebsite(browser, 'http://' + w.rootDomain)
    }
})

await Promise.all(p)
await browser.close()
console.log('All Done')

