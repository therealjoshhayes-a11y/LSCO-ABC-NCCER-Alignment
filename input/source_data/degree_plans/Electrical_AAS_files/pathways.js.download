document.addEventListener('PathwaysJobMarketScriptLoaded', function (event) {
	let pathwaysWidget = document.getElementById('moderncampuspathwayspublic');
	let shadowRoot = pathwaysWidget.shadowRoot;
	let shadowRootObserver = new MutationObserver(function (mutationList, observer) {
		if (shadowRoot.querySelector('.ppw') !== null) {
			shadowRootObserver.disconnect();
			let link = document.createElement('link');
			link.rel = 'stylesheet';
			link.type = 'text/css';

			// Add styling for above and below
			if (pathwaysWidget.classList.contains('above') || pathwaysWidget.classList.contains('below')) {
				link.href = '/styles/pathways.css';
			// Add styling for left and right
			} else if (pathwaysWidget.classList.contains('left') || pathwaysWidget.classList.contains('right')) {
				link.href = '/styles/pathways_side.css';
			}

			shadowRoot.appendChild(link);
		}
	});
	shadowRootObserver.observe(shadowRoot, { attributes: false, childList: true, subtree: false});
	let style = document.createElement('style');

	style.textContent = `
		.loading-container {
			max-width: 100%;
		}
	`;

	shadowRoot.appendChild(style);
}, false);
