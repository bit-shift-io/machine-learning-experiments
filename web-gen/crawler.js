/**
 * Crawl over website geenrating the dataset
 */

import chunk from 'lodash/chunk.js'
import puppeteer from 'puppeteer'
import fs from 'fs'
import path from 'path'

async function getPropertyValue(element, property) {
    return await (await element.getProperty(property)).jsonValue()
}

async function handleElement(page, element, dir) {
    fs.mkdirSync(dir, { recursive: true });

    const bounds = await element.boundingBox()
    const img_path = `${dir}/screenshot.jpg`
    const r = {
        img_path,
        offset_left: await getPropertyValue(element, 'offsetLeft'),
        offset_top: await getPropertyValue(element, 'offsetTop'),
        bounds,
        tag_name: await getPropertyValue(element, 'tagName'),
        // TODO: add css block to get layout info, padding, margins, border etc...
    }
    const r_children = []

    try {
        await element.screenshot({path: img_path})
    } catch (err) {
        try {
            fs.rmdirSync(dir)
        } catch (err) {
            console.warn(err)
        }

        return null
    }

    const children = await element.$$(':scope > *')
    //const children = await page.evaluateHandle(e => e.children, element)
    const p = children.map(async (c, idx) => {
        const rc = await handleElement(page, c, `${dir}/child_${idx}`)
        if (rc) {
            r_children.push(rc)
        }
    })

    await Promise.all(p)

    if (r_children.length > 0) {
        r.children = r_children
    } else {
        // for now, if we have non children, we are a leaf node.... we don't care about having a json file
        return r
    }

    const dataFile = `${dir}/data.json`
    const json = JSON.stringify(r, null, 4)
    fs.writeFileSync(dataFile, json)

    return r
}

export async function screenshotWebsite(browser, url) {
    console.log(`Starting processing: ${url}`)

    // create a dir from the website url
    let dir = 'data/' + url.replace('https://', '').replace('http://', '').replaceAll('.', '-')
    if (dir.endsWith('/')) {
        dir = dir.slice(0, -1)
    }
    /*
    const dataFile = `${dir}/data.json`
    if (fs.existsSync(dataFile)) {
        console.log(`Already processed: ${url}`)
        return
    }*/

    let page = null
    try {
        page = await browser.newPage()

        await page.goto(url)

        await page.waitForSelector('body')
        const element = await page.$('html')

        const results = await handleElement(page, element, `${dir}/html`)
        /*
        const json = JSON.stringify(results, null, 4)
        fs.writeFileSync(dataFile, json)
        */

        await page.close()
        console.log(`Done processing: ${url}`)
    } catch (err) {
        //console.warn(err)
        console.warn(`Failed processing: ${url}`)
        await page?.close()
    }
}


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



const TEST_SINGLE = true

if (TEST_SINGLE) {
    const browser = await puppeteer.launch()
    const url = 'https://www.wikipedia.org/'
    await screenshotWebsite(browser, url)
    await browser.close()
} else {
    const browser = await puppeteer.launch()
    const r = await fetch('https://raw.githubusercontent.com/Kikobeats/top-sites/master/top-sites.json').then(r => r.json())
    const c = chunk(r, r.length)// / 10)
    const p = c.map(async (arr, idx) => {
        for (const w of arr) {
            await screenshotWebsite(browser, 'http://' + w.rootDomain)
        }
    })

    await Promise.all(p)
    await browser.close()
}

console.log('All Done')

