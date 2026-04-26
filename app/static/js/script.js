// Climate Trend Analyzer - UI Enhancements

document.addEventListener("DOMContentLoaded", () => {

    // =====================================
    // Active Navbar Link
    // =====================================
    const currentPath = window.location.pathname;
    const links = document.querySelectorAll("nav a");

    links.forEach(link => {
        const href = link.getAttribute("href");

        if (href === currentPath) {
            link.classList.add("active-nav");
        }
    });

    // =====================================
    // Fade-in Animation on Scroll
    // =====================================
    const items = document.querySelectorAll(
        ".stat-card, .mini-card, .chart-card, .table-card"
    );

    const observer = new IntersectionObserver(entries => {

        entries.forEach(entry => {

            if (entry.isIntersecting) {
                entry.target.classList.add("show");
            }

        });

    }, {
        threshold: 0.15
    });

    items.forEach(item => {
        item.classList.add("hidden");
        observer.observe(item);
    });

    // =====================================
    // Scroll To Top Button
    // =====================================
    const btn = document.createElement("button");

    btn.innerHTML = "↑";
    btn.id = "scrollTopBtn";

    document.body.appendChild(btn);

    window.addEventListener("scroll", () => {

        if (window.scrollY > 300) {
            btn.style.display = "block";
        } else {
            btn.style.display = "none";
        }

    });

    btn.addEventListener("click", () => {

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });

    });

});