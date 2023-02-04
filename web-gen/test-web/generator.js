const MAX_DEPTH = 3
const RAND_STRING_MAX_LEN = 10

const images = [
    'https://www.incimages.com/uploaded_files/image/1920x1080/getty_481292845_77896.jpg',
    'https://statesman-homes.com.au/assets/Uploads/Packages.png',
    'https://i.guim.co.uk/img/media/03734ee186eba543fb3d0e35db2a90a14a5d79e3/0_173_5200_3120/master/5200.jpg?width=1200&height=900&quality=85&auto=format&fit=crop&s=9c30ed97ea8731f3e2a155467201afe3'
]

function rand_img() {
    return images[Math.floor(Math.random() * images.length)]
}

function rand_color() {
    return Math.floor(Math.random()*16777215).toString(16)
}

function rand_str(length) {
    let result = '';
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    const charactersLength = characters.length;
    let counter = 0;
    while (counter < length) {
      result += characters.charAt(Math.floor(Math.random() * charactersLength));
      counter += 1;
    }
    return result;
}

function create_flex_random(depth) {
    const d = Math.floor(Math.random() * 2)
    const dir = d == 0 ? 'row' : 'column'
    const flexEl = document.createElement('div')
    flexEl.style.cssText = `display: flex; flex-direction: ${dir}`

    if (depth >= MAX_DEPTH) {
        const pad = Math.floor(Math.random() * 10)
        const mar = Math.floor(Math.random() * 10)

        flexEl.style.cssText += `background-color: ${rand_color()}; padding: ${pad}; margin: ${mar}`
        const l = Math.floor(Math.random() * 5)
        if (l == 0) {
            flexEl.innerHTML = `<img src=\"${rand_img()}\" width=\"400px\" height=\"150px\">`
        } else {
            flexEl.innerHTML = rand_str(Math.floor(Math.random() * RAND_STRING_MAX_LEN))
        }
        return flexEl
    }

    const c = Math.floor(Math.random() * 10)
    for (let i = 0; i < c; ++i) {
        const childEl = create_flex_random(depth+1)
        flexEl.appendChild(childEl)
    }

    return flexEl
}

function generate_random() {
    const f = create_flex_random(0)
    document.body.appendChild(f)
}