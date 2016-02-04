### ------- Convergence -------- ###

single = read.csv("Convergence_single.csv", sep="\t")
team = read.csv("Convergence_team.csv", sep="\t")

t.test(single$Not_grabbed, team$Not_grabbed)
t.test(single$Grabbed, team$Grabbed)
t.test(single$Grabbed + single$Not_grabbed, team$Grabbed + team$Not_grabbed)

#par(mfrow=c(1,2))

plot(single$Not_grabbed, ylim = c(0,3000), col=150, pch=16, main = "Epochs before convergence", ylab = "Epoch", xlab = "Run")
points(team$Not_grabbed, col=51, pch=17)
legend("bottomright", c("single", "team"), col=c(150, 51),pch=c(16, 17))

plot(single$Grabbed, col=150, ylim = c(0,3000), pch=16, main = "Epochs before convergence", ylab = "Epoch", xlab = "Run")
points(team$Grabbed, col=51, pch=17)
legend("topright", c("single", "team"), col=c(150, 51),pch=c(16, 17))


### ------- Path -------- ###

single = read.csv("Path_single.csv", sep="\t")
team = read.csv("Path_team.csv", sep="\t")

t.test(single$Not_grabbed, team$Not_grabbed)
t.test(single$Grabbed, team$Grabbed)
t.test(single$Grabbed + single$Not_grabbed, team$Grabbed + team$Not_grabbed)

#par(mfrow=c(1,2))

plot(single$Not_grabbed, ylim = c(0,60), col=150, pch=16, main = "Steps of final path", ylab = "Epoch", xlab = "Run")
points(team$Not_grabbed, col=51, pch=17)
legend("bottomright", c("single", "team"), col=c(150, 51),pch=c(16, 17))

plot(single$Grabbed, col=150, ylim = c(0,20), pch=16, main = "Steps of final path", ylab = "Epoch", xlab = "Run")
points(team$Grabbed, col=51, pch=17)
legend("topright", c("single", "team"), col=c(150, 51),pch=c(16, 17))
