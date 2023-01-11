import puppeteer from 'puppeteer';
import fs from 'fs'

async function handleElement(page, element, dir) {
    fs.mkdirSync(dir, { recursive: true });

    const img_path = `${dir}/screenshot.png`
    const r = {
        img_path
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

export async function screenshotWebsite(url) {
    const browser = await puppeteer.launch()
    const page = await browser.newPage()

    // create a dir from the website url
    const dir = 'data/' + url.replace('https://', '').replace('http://').replaceAll('.', '-')

    await page.goto(url)

    await page.waitForSelector('body')
    const element = await page.$('body')

    const results = await handleElement(page, element, `${dir}/body`)
    const json = JSON.stringify(results, null, 4)
    fs.writeFileSync(`${dir}/data.json`, json)

    await browser.close()
}


(async () => {
    const url = 'https://google.com'
    //await screenshotWebsite(url)
})();
