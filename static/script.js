console.log("Script loaded");

// Check if the startQuiz button exists
const startQuizButton = document.getElementById('startQuiz');
if (!startQuizButton) {
    console.error("Error: Start Quiz button not found.");
} else {
    console.log("Start Quiz button found:", startQuizButton);
}

// Fetch and parse the questionnaire document
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded, fetching questionnaire...");

    fetch('/static/questionnaire.txt')
        .then(response => {
            console.log("Fetch response status:", response.status);
            if (!response.ok) {
                throw new Error(`Failed to fetch questionnaire.txt: ${response.statusText}`);
            }
            return response.text();
        })
        .then(data => {
            console.log("Questionnaire fetched successfully, parsing...");
            const questions = parseQuestionnaire(data);
            console.log("Parsed questions:", questions);
            if (questions.length !== 30) {
                console.warn("Expected 30 questions, but parsed:", questions.length);
            }
            generateQuizForm(questions);
        })
        .catch(error => console.error('Error loading questionnaire:', error));
});

// Parse the questionnaire text into a structured format
function parseQuestionnaire(text) {
    const lines = text.split('\n').filter(line => line.trim() !== '');
    const questions = [];
    let currentQuestion = null;

    for (const line of lines) {
        if (line.startsWith('Learning Style Quiz')) continue;

        if (/^\d+\./.test(line.trim())) {
            if (currentQuestion) questions.push(currentQuestion);
            currentQuestion = {
                id: parseInt(line.match(/^\d+/)[0]),
                text: line.replace(/^\d+\.\s*/, '').trim(),
                options: {}
            };
        } else if (/^[abc]\)/.test(line.trim())) {
            const optionKey = line.trim()[0];
            const optionText = line.replace(/^[abc]\)\s*/, '').trim();
            currentQuestion.options[optionKey] = optionText;
        }
    }
    if (currentQuestion) questions.push(currentQuestion);
    return questions;
}

// Generate the quiz form dynamically
function generateQuizForm(questions) {
    const questionsContainer = document.getElementById('questionsContainer');
    if (!questionsContainer) {
        console.error("Error: questionsContainer not found.");
        return;
    }

    questions.forEach(question => {
        const questionDiv = document.createElement('div');
        questionDiv.className = 'question';

        const questionText = document.createElement('p');
        questionText.textContent = `${question.id}. ${question.text}`;
        questionDiv.appendChild(questionText);

        const labelA = document.createElement('label');
        labelA.innerHTML = `<input type="radio" name="q${question.id}" value="a" required> a) ${question.options.a}`;
        questionDiv.appendChild(labelA);
        questionDiv.appendChild(document.createElement('br'));

        const labelB = document.createElement('label');
        labelB.innerHTML = `<input type="radio" name="q${question.id}" value="b" required> b) ${question.options.b}`;
        questionDiv.appendChild(labelB);
        questionDiv.appendChild(document.createElement('br'));

        const labelC = document.createElement('label');
        labelC.innerHTML = `<input type="radio" name="q${question.id}" value="c" required> c) ${question.options.c}`;
        questionDiv.appendChild(labelC);

        questionsContainer.appendChild(questionDiv);
    });
    console.log("Quiz form generated successfully with", questions.length, "questions.");
}

// Handle Start Quiz button
if (startQuizButton) {
    startQuizButton.addEventListener('click', function() {
        console.log("Start Quiz clicked");
        const quizDiv = document.getElementById('quiz');
        if (!quizDiv) {
            console.error("Error: Quiz div not found.");
            return;
        }
        startQuizButton.style.display = 'none';
        quizDiv.style.display = 'block';
    });
}

// Handle form submission
const quizForm = document.getElementById('quizForm');
if (!quizForm) {
    console.error("Error: Quiz form not found.");
} else {
    console.log("Quiz form found:", quizForm);
    quizForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const responses = [];
        for (let i = 1; i <= 30; i++) {
            const answer = document.querySelector(`input[name="q${i}"]:checked`);
            if (!answer) {
                alert('Please answer all questions!');
                console.warn(`Question ${i} not answered.`);
                return;
            }
            responses.push(answer.value);
        }
        console.log("User responses:", responses);

        fetch('/submit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ responses })
        })
        .then(response => {
            console.log("Submit response status:", response.status);
            if (!response.ok) {
                throw new Error(`Failed to submit responses: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                alert(data.error);
                console.error("Backend error:", data.error);
                return;
            }
            console.log("Backend response:", data);
            document.getElementById('quiz').style.display = 'none';
            const resultDiv = document.getElementById('result');
            if (resultDiv) {
                resultDiv.style.display = 'block';
                document.getElementById('dominantStyle').textContent = `Dominant Style: ${data.dominant_style}`;
                document.getElementById('scores').textContent = `Scores - Visual: ${data.scores.Visual}, Auditory: ${data.scores.Auditory}, Read/Write: ${data.scores['Read/Write']}, Kinesthetic: ${data.scores.Kinesthetic}`;
            } else {
                console.error("Error: Result div not found.");
            }
        })
        .catch(error => console.error('Error submitting responses:', error));
    });
}