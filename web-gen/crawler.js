/**
 * Crawl over website geenrating the dataset
 */

import chunk from 'lodash/chunk.js'
import pick from 'lodash/pick.js'
//import puppeteer from 'puppeteer'
import sharp from 'sharp'
import { chromium } from 'playwright' // Or 'chromium' or 'firefox'.
import fetch from 'node-fetch'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)


const TEST_SINGLE = false


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

    //const cls = await getPropertyValue(element, 'class') 
    const id = await getPropertyValue(element, 'id')
    const tag_name = (await getPropertyValue(element, 'tagName')).toLowerCase()

    /*
    if (dir == 'data/cloudflare-com/body/child_0/child_0/child_0/child_2/child_4/child_0/child_0/child_0/child_1/child_0/child_2/child_0') {
        debugger
    }*/

    // we dont care about iframes or other tags
    const VALID_TAGS = ['div', 'body', 'a', 'ul', 'li']
    if (!VALID_TAGS.includes(tag_name)) {
        return null
    }

    const styles = await element.evaluate((element) => {
        return window.getComputedStyle(element)
    })

    // TODO: add other css properties we are interested in
    const pickedStyles = pick(styles, ['display', 'flexDirection', 'flex', 'marginTop', 'marginLeft', 'marginRight', 'marginBottom', 'paddingTop', 'paddingLeft', 'paddingRight', 'paddingBottom'])
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
        layout = pickedStyles['flexDirection'] || 'row'
    }

    // we could handle this by getting the parent to check if its children contain 'inline-block'
    // then nchange its layout to a 'row'.... for now if we see inline block abort!
    if (pickedStyles.display == 'inline-block') {
        return null
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

        let sImg = sharp(img_path)
        sImg = await sImg.resize(200, 200, { fit: 'fill' })

        const stats = await sImg.stats()
        if (stats.entropy < MIN_ENTROPY) {
            //throw `low entropy for: ${img_path}`
            return null
        }
        
        await sImg.toFile(img_path_200)

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
    let dir = 'data/' + url.replace('https://', '').replace('http://', '').replace('www.', '').replaceAll('.', '-')
    if (dir.endsWith('/')) {
        dir = dir.slice(0, -1)
    }
    
    if (!TEST_SINGLE && fs.existsSync(dir)) {
        console.log(`Already processed: ${url}`)
        return
    }

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




if (TEST_SINGLE) {
    const browser = await chromium.launch()
    //const url = 'https://www.wikipedia.org/'
    let url = `file://${__dirname}/test-web/test-1.html`
    url = `https://www.cambridge.org/`
    url = `https://www.cloudflare.com/`
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

