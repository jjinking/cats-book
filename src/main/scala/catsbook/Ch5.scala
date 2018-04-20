package catsbook

object Ch5 {

  object Section5_4 {
    import scala.concurrent.{Await, Future}
    import scala.concurrent.ExecutionContext.Implicits.global
    import scala.concurrent.duration._
    import cats.data.EitherT
    import cats.instances.future._

    //type Response[A] = Future[Either[String, A]]
    type Response[A] = EitherT[Future, String, A]

    val powerLevels = Map(
      "Jazz"      -> 6,
      "Bumblebee" -> 8,
      "Hot Rod"   -> 10
    )

    def getPowerLevel(autobot: String): Response[Int] = {
      val powerOpt: Option[Int] = powerLevels.get(autobot)
      powerOpt match {
        case Some(p) => EitherT.right(Future(p)) 
        case _ => EitherT.left(Future(s"$autobot is unreachable"))
      }
    }

    def canSpecialMove(ally1: String, ally2: String): Response[Boolean] = for {
      p1 <- getPowerLevel(ally1)
      p2 <- getPowerLevel(ally2)
    } yield p1 + p2 > 15

    def tacticalReport(ally1: String, ally2: String): String = {
      val report = canSpecialMove(ally1, ally2).value.map {
        case Left(errorMsg) => errorMsg
        case Right(true) => "Special move is possible"
        case Right(false) => "Special move is not possible"
      }
      Await.result(report, 5.seconds)
    }
  }
}
