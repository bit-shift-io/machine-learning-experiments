import { screenshotWebsite } from './website.js'

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

const r = await fetch('https://raw.githubusercontent.com/Kikobeats/top-sites/master/top-sites.json').then(r => r.json())

for (const w of r) {
    console.log(`Processing: ${w.rootDomain}`)
    await screenshotWebsite('https://' + w.rootDomain)
}

console.log(r)

