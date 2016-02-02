single = read.csv("ConvergenceSingle.csv", sep="\t")
team = read.csv("ConvergenceTeam.csv", sep="\t")

t.test(single$Not_grabbed, team$Not_grabbed)
t.test(single$Grabbed, team$Grabbed)

par(mfrow=c(1,2))

plot(single$Not_grabbed, col="blue", pch=16)
points(team$Not_grabbed, col="red", pch=17)
legend("bottomright", c("single", "team"), col=c("blue", "red"),pch=c(16, 17))

plot(single$Grabbed, col="blue", pch=16)
points(team$Grabbed, col="red", pch=17)
legend("bottomright", c("single", "team"), col=c("blue", "red"),pch=c(16, 17))