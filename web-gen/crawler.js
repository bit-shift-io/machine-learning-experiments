/**
 * Crawl over website geenrating the dataset
 */

import chunk from 'lodash/chunk.js'
import pick from 'lodash/pick.js'
//import puppeteer from 'puppeteer'
import sharp from 'sharp'
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
    // ignore margins as this will stuff up the AI learning
    /*
    b.y -= (parseFloat(styles.marginTop) || 0)
    b.x -= (parseFloat(styles.marginLeft) || 0)
    b.height += (parseFloat(styles.marginTop) || 0) + (parseFloat(styles.marginBottom) || 0)
    b.width += (parseFloat(styles.marginLeft) || 0) + (parseFloat(styles.marginRight) || 0)
    */
    return b
}

// TODO: if a container only has a single child with no margins, ignore it as it can be flattened with the child....
async function handleElement(page, parent_element, parent_bounds, element, dir) { 
    const MIN_SIZE = 10
    const MIN_ENTROPY = 0.2

    const id = await getPropertyValue(element, 'id')
    const tag_name = (await getPropertyValue(element, 'tagName')).toLowerCase()

    // we dont care about iframes or other tags
    const VALID_TAGS = ['div', 'body', 'a', 'ul', 'li']
    if (!VALID_TAGS.includes(tag_name)) {
        return null
    }

    const styles = await element.evaluate((element) => {
        return window.getComputedStyle(element)
    })

    // TODO: add other css properties we are interested in
    const pickedStyles = pick(styles, ['display', 'flex-direction', 'flex', 'marginTop', 'marginLeft', 'marginRight', 'marginBottom', 'paddingTop', 'paddingLeft', 'paddingRight', 'paddingBottom'])
    if (pickedStyles.display == 'none') {
        return null
    }

    // this is relative to viewport apparently
    let boundingBox = computeBounds(await element.boundingBox(), styles)
    
    // not visible
    if (!boundingBox) {
        return null
    }

    // too small to care about
    if (boundingBox.width < MIN_SIZE && boundingBox.height < MIN_SIZE) {
        return null
    }

    // compute bounds relative to parent
    let boundsRelativeToParent = boundingBox
    if (parent_bounds) {
        boundsRelativeToParent = makeRelativeToParent(boundingBox, parent_bounds)
    }

    // boil the layout down to row or column
    // TODO: handle grid
    let layout = 'column'
    if (pickedStyles.display == 'flex') {
        layout = pickedStyles['flex-direction'] || 'row'
    }


    const r_children = []
    const children = await element.$$(':scope > *')
    for (let i = 0; i < children?.length; ++i) {
        const c = children[i]
        const rc = await handleElement(page, element, boundingBox, c, `${dir}/child_${i}`)
        if (rc) {
            r_children.push(rc)
        }
    }

    // now we really only care about the height of the first child in the case of a column layout
    // and in the case of a row layout we only care about the width
    let firstChildSize = {width: boundsRelativeToParent.width, height: boundsRelativeToParent.height}
    let firstChild = r_children?.[0]
    if (firstChild) {
        if (layout == 'column') {
            firstChildSize.height = firstChild.bounds.height
        } else {
            firstChildSize.width = firstChild.bounds.width
        }
    }

    const img_path = `${dir}/screenshot.jpg`
    const img_path_200 = `${dir}/screenshot_200.jpg`
    const r = {
        id,
        img_path,
        img_path_200,

        layout,
        first_child_size: firstChildSize,

        parent_size: { // need parent size to compute fractional scaling
            width: parent_bounds?.width || boundingBox.width,
            height: parent_bounds?.height || boundingBox.height,
        },
        bounds: boundsRelativeToParent,
        tag_name,
        css: {
            ...pickedStyles
        }
    }
    
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

        const sImg = sharp(img_path)
        const stats = await sImg.stats()
        if (stats.entropy < MIN_ENTROPY) {
            //throw `low entropy for: ${img_path}`
            return null
        }
        
        await sImg.resize(200, 200, { fit: 'fill' })
            .toFile(img_path_200)

    } catch (err) {
        try {
            fs.rmSync(dir, { recursive: true, force: true })
        } catch (err) {
            console.warn(err)
        }

        return null
    }


    if (r_children.length > 0) {
        r.children = r_children
    }

    let filename = 'layout' // multiple children
    if (r_children.length == 0) {
        filename = 'leaf' // terminator (text or image)
    } else if (r_children.length == 1) {
        filename = 'container' // container of just a single child
    }

    const dataFile = `${dir}/${filename}.json`
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
    const dataFile = `${dir}/layout.json`
    if (fs.existsSync(dataFile)) {
        console.log(`Already processed: ${url}`)
        return
    }*/

    let page = null
    try {
        const context = await browser.newContext()
        page = await context.newPage()

        // might actually need to resize this AFTER we save the screnshot to help speed up the ML so it doesnt need to resize images...
        //page.setViewportSize({ width: 400, height: 400 })

        await page.goto(url)

        // https://github.com/microsoft/playwright/issues/662
        await page.waitForLoadState('networkidle')

        await page.waitForSelector('body')
        const element = await page.$('body')

        const results = await handleElement(page, null, null, element, `${dir}/body`)
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



const TEST_SINGLE = true

if (TEST_SINGLE) {
    const browser = await chromium.launch()
    //const url = 'https://www.wikipedia.org/'
    let url = `file://${__dirname}/test-web/test-1.html`
    url = `https://www.cambridge.org/`
    await screenshotWebsite(browser, url)
    await browser.close()
} else {
    const browser = await chromium.launch()
    const r = await fetch('https://raw.githubusercontent.com/Kikobeats/top-sites/master/top-sites.json').then(r => r.json())
    const c = chunk(r, r.length / 10)
    const p = c.map(async (arr, idx) => {
        for (const w of arr) {
            await screenshotWebsite(browser, 'http://' + w.rootDomain)
        }
    })

    await Promise.all(p)
    await browser.close()
}

console.log('All Done')

