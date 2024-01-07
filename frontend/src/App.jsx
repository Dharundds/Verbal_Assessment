// import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Pages from "./Pages";

function App() {
  return (
    <div className="App">
      <Router>
        <Routes>
          <Route exact path="/" element={<Pages.Home />} />
          <Route exact path="/voiceroom" element={<Pages.VoiceRoom />} />
          <Route exact path="/test" element={<Pages.Test />} />
        </Routes>
      </Router>
    </div>
  );
}

export default App;
