/**
 * Crawl over website geenrating the dataset
 */

import chunk from 'lodash/chunk.js'
import pick from 'lodash/pick.js'
//import puppeteer from 'puppeteer'

import { chromium } from 'playwright' // Or 'chromium' or 'firefox'.

import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

async function getPropertyValue(element, property) {
    return await (await element.getProperty(property)).jsonValue()
}

/*
function computeBounds(boxModel) {
    // TODO: margins here are not working properly.... something is a bit off....
    const width = Math.floor(boxModel.margin[2].x - boxModel.margin[0].x)
    const height = Math.floor(boxModel.margin[2].y - boxModel.margin[0].y)
    return {
        ...boxModel.margin[0],
        width,
        height
    }
}*/

function makeRelativeToParent(childBounds, parentBounds) {
    return {
        ...childBounds,
        x: childBounds.x - parentBounds.x,
        y: childBounds.y - parentBounds.y
    } 
}

function computeBounds(boundingBox, styles) {
    if (!boundingBox) {
        return null
    }

    const b = Object.assign({}, boundingBox)
    b.y -= (parseFloat(styles.marginTop) || 0)
    b.x -= (parseFloat(styles.marginLeft) || 0)
    b.height += (parseFloat(styles.marginTop) || 0) + (parseFloat(styles.marginBottom) || 0)
    b.width += (parseFloat(styles.marginLeft) || 0) + (parseFloat(styles.marginRight) || 0)
    return b
}

async function handleElement(page, parent_element, parent_bounds, element, dir) {    
    const id = await getPropertyValue(element, 'id')

    const styles = await element.evaluate((element) => {
        return window.getComputedStyle(element)
    })

    // TODO: add other css properties we are interested in
    const pickedStyles = pick(styles, ['display', 'flex-direction', 'flex', 'marginTop', 'marginLeft', 'marginRight', 'marginBottom', 'paddingTop', 'paddingLeft', 'paddingRight', 'paddingBottom'])

    // this is relative to viewport apparently
    let boundingBox = computeBounds(await element.boundingBox(), styles)
    
    // not visible
    if (!boundingBox) {
        return null
    }

    // compute bounds relative to parent
    let boundsRelativeToParent = boundingBox
    if (parent_bounds) {
        boundsRelativeToParent = makeRelativeToParent(boundingBox, parent_bounds)
    }

    const img_path = `${dir}/screenshot.jpg`
    const r = {
        id,
        img_path,
        parent_size: { // need parent size to compute fractional scaling
            width: parent_bounds?.width || boundingBox.width,
            height: parent_bounds?.height || boundingBox.height,
        },
        bounds: boundsRelativeToParent,
        tag_name: await getPropertyValue(element, 'tagName'),
        css: {
            ...pickedStyles
        }
    }
    const r_children = []

    try {
        fs.mkdirSync(dir, { recursive: true })

        // Calculate your clip rect. For a single element, that's usually the same as it's bounding box.
        //const clipRelativeToViewport = await element.boundingBox();

        // Translate clip to be relative to the page.
        const clipRelativeToPage = {
            width: boundingBox.width,
            height: boundingBox.height,
            x: boundingBox.x + await page.evaluate(() => window.scrollX),
            y: boundingBox.y + await page.evaluate(() => window.scrollY),
        }

        // Take an area screenshot.
        const areaScreenshot = await page.screenshot({ path: img_path, clip: clipRelativeToPage, fullPage: true })

        //await element.screenshot({path: img_path, clip: boundingBox})
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

    /*
    const p = children.map(async (c, idx) => {
        const rc = await handleElement(page, element, boundingBox, c, `${dir}/child_${idx}`)
        if (rc) {
            r_children.push(rc)
        }
    })

    await Promise.all(p)
    */

    for (let i = 0; i < children.length; ++i) {
        const c = children[i]
        const rc = await handleElement(page, element, boundingBox, c, `${dir}/child_${i}`)
        if (rc) {
            r_children.push(rc)
        }
    }

    if (r_children.length > 0) {
        r.children = r_children
    } else {
        // for now, if we have non children, we are a leaf node.... we don't care about having a json file
        //return r
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
        const context = await browser.newContext()
        page = await context.newPage()

        await page.goto(url)

        await page.waitForSelector('body')
        const element = await page.$('html')

        const results = await handleElement(page, null, null, element, `${dir}/html`)
        /*
        const json = JSON.stringify(results, null, 4)
        fs.writeFileSync(dataFile, json)
        */

        await page.close()
        console.log(`Done processing: ${url}`)
    } catch (err) {
        //console.warn(err)
        console.warn(`Failed processing: ${url}:`, err)
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



const TEST_SINGLE = false

if (TEST_SINGLE) {
    const browser = await chromium.launch()
    //const url = 'https://www.wikipedia.org/'
    const url = `file://${__dirname}/test-web/test-1.html`
    await screenshotWebsite(browser, url)
    await browser.close()
} else {
    const browser = await chromium.launch()
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

