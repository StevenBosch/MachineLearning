single = read.csv("ConvergenceSingle.csv", sep="\t")
team = read.csv("ConvergenceTeam.csv", sep="\t")

t.test(single$Not_grabbed, team$Not_grabbed)
t.test(single$Grabbed, team$Grabbed)

par(mfrow=c(1,2))

plot(single$Not_grabbed, ylim = c(0,3000), col="blue", pch=16, main = "Epochs before convergence (not grabbed)")
points(team$Not_grabbed, col="red", pch=17)
legend("bottomright", c("single", "team"), col=c("blue", "red"),pch=c(16, 17))

plot(single$Grabbed, col="blue", ylim = c(0,3000), pch=16, main = "Epochs before convergence (grabbed)")
points(team$Grabbed, col="red", pch=17)
legend("topright", c("single", "team"), col=c("blue", "red"),pch=c(16, 17))
