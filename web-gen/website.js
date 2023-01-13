import puppeteer from 'puppeteer';
import fs from 'fs'

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
        // TODO: delete the dir?
        //console.warn(err)
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

    if (r_children.length) {
        r.children = r_children
    }
    return r
}

export async function screenshotWebsite(browser, url) {
    console.log(`Starting processing: ${url}`)

    // create a dir from the website url
    let dir = 'data/' + url.replace('https://', '').replace('http://', '').replaceAll('.', '-')
    if (dir.endsWith('/')) {
        dir = dir.slice(0, -1)
    }
    const dataFile = `${dir}/data.json`
    if (fs.existsSync(dataFile)) {
        console.log(`Already processed: ${url}`)
        return
    }

    let page = null
    try {
        page = await browser.newPage()

        await page.goto(url)

        await page.waitForSelector('body')
        const element = await page.$('html')

        const results = await handleElement(page, element, `${dir}/html`)
        const json = JSON.stringify(results, null, 4)
        fs.writeFileSync(dataFile, json)

        await page.close()
        console.log(`Done processing: ${url}`)
    } catch (err) {
        //console.warn(err)
        console.warn(`Failed processing: ${url}`)
        await page?.close()
    }
}
