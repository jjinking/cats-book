package catsbook.unit

import org.scalatest._
import catsbook.Ch5._

class Ch5Spec extends FlatSpec with Matchers {

  it should "Section4 Check power levels" in {
    import Section5_4._

    assert(tacticalReport("Jazz", "Bumblebee") == "Special move is not possible")
    assert(tacticalReport("Bumblebee", "Hot Rod") == "Special move is possible")
    assert(tacticalReport("Jazz", "Ironhide") == "Ironhide is unreachable")
  }

}
