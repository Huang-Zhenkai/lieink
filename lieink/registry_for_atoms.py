from lieink.atoms import (
    SE3,
    SO3,
    Ad,
    Lie,
    Point3,
    Point4,
    RealScalar,
    Twist,
    Twist3,
    Vector3,
    Vector4,
    Wrench,
    ad,
    coAd,
    coad,
    se3,
    so3,
)

Lie.UPDATE_ALLOWED_OPERATORS(so3, "exp", RealScalar, SO3)
Lie.UPDATE_ALLOWED_OPERATORS(Twist3, "exp", RealScalar, SO3)
Lie.UPDATE_ALLOWED_OPERATORS(SO3, "log", None, {"Twist3": Twist3, "so3": so3})

Lie.UPDATE_ALLOWED_OPERATORS(
    se3, "exp", RealScalar, {"SE3": SE3, "Ad": Ad, "coAd": coAd}
)
Lie.UPDATE_ALLOWED_OPERATORS(
    Twist, "exp", RealScalar, {"SE3": SE3, "Ad": Ad, "coAd": coAd}
)
Lie.UPDATE_ALLOWED_OPERATORS(
    ad, "exp", RealScalar, {"SE3": SE3, "Ad": Ad, "coAd": coAd}
)
Lie.UPDATE_ALLOWED_OPERATORS(
    coad, "exp", RealScalar, {"SE3": SE3, "Ad": Ad, "coAd": coAd}
)
Lie.UPDATE_ALLOWED_OPERATORS(
    coAd, "exp", RealScalar, {"SE3": SE3, "Ad": Ad, "coAd": coAd}
)
Lie.UPDATE_ALLOWED_OPERATORS(
    SE3, "log", None, {"se3": se3, "Twist": Twist, "ad": ad, "coad": coad}
)
Lie.UPDATE_ALLOWED_OPERATORS(
    Ad, "log", None, {"se3": se3, "Twist": Twist, "ad": ad, "coad": coad}
)
Lie.UPDATE_ALLOWED_OPERATORS(
    coAd, "log", None, {"se3": se3, "Twist": Twist, "ad": ad, "coad": coad}
)

Lie.UPDATE_ALLOWED_OPERATORS(Point3, "+", Point3, Point3)
Lie.UPDATE_ALLOWED_OPERATORS(Point4, "+", Point4, Point4)
Lie.UPDATE_ALLOWED_OPERATORS(Vector3, "+", Vector3, Vector3)
Lie.UPDATE_ALLOWED_OPERATORS(Vector4, "+", Vector4, Vector4)
Lie.UPDATE_ALLOWED_OPERATORS(so3, "+", so3, so3)
Lie.UPDATE_ALLOWED_OPERATORS(Twist3, "+", Twist3, Twist3)
Lie.UPDATE_ALLOWED_OPERATORS(se3, "+", se3, se3)
Lie.UPDATE_ALLOWED_OPERATORS(Twist, "+", Twist, Twist)
Lie.UPDATE_ALLOWED_OPERATORS(Wrench, "+", Wrench, Wrench)
Lie.UPDATE_ALLOWED_OPERATORS(ad, "+", ad, ad)
Lie.UPDATE_ALLOWED_OPERATORS(coad, "+", coad, coad)

Lie.UPDATE_ALLOWED_OPERATORS(Point3, "*", RealScalar, Point3)
Lie.UPDATE_ALLOWED_OPERATORS(Point4, "*", RealScalar, Point4)
Lie.UPDATE_ALLOWED_OPERATORS(Vector3, "*", RealScalar, Vector3)
Lie.UPDATE_ALLOWED_OPERATORS(Vector4, "*", RealScalar, Vector4)
Lie.UPDATE_ALLOWED_OPERATORS(so3, "*", RealScalar, so3)
Lie.UPDATE_ALLOWED_OPERATORS(Twist3, "*", RealScalar, Twist3)
Lie.UPDATE_ALLOWED_OPERATORS(se3, "*", RealScalar, se3)
Lie.UPDATE_ALLOWED_OPERATORS(Twist, "*", RealScalar, Twist)
Lie.UPDATE_ALLOWED_OPERATORS(Wrench, "*", RealScalar, Wrench)
Lie.UPDATE_ALLOWED_OPERATORS(ad, "*", RealScalar, ad)
Lie.UPDATE_ALLOWED_OPERATORS(coad, "*", RealScalar, coad)

Lie.UPDATE_ALLOWED_OPERATORS(SO3, "*", Point3, Point3)
Lie.UPDATE_ALLOWED_OPERATORS(SO3, "*", Vector3, Vector3)
Lie.UPDATE_ALLOWED_OPERATORS(SO3, "*", Twist3, Twist3)
Lie.UPDATE_ALLOWED_OPERATORS(SO3, "*", SO3, SO3)
Lie.UPDATE_ALLOWED_OPERATORS(SE3, "*", Point4, Point4)
Lie.UPDATE_ALLOWED_OPERATORS(SE3, "*", Vector4, Vector4)
Lie.UPDATE_ALLOWED_OPERATORS(SE3, "*", SE3, SE3)
Lie.UPDATE_ALLOWED_OPERATORS(Ad, "*", Twist, Twist)
Lie.UPDATE_ALLOWED_OPERATORS(Ad, "*", Ad, Ad)
Lie.UPDATE_ALLOWED_OPERATORS(coAd, "*", Wrench, Wrench)
Lie.UPDATE_ALLOWED_OPERATORS(coAd, "*", coAd, coAd)

Lie.UPDATE_ALLOWED_OPERATORS(SO3, "@", so3, so3)
Lie.UPDATE_ALLOWED_OPERATORS(SE3, "@", se3, se3)
Lie.UPDATE_ALLOWED_OPERATORS(Ad, "@", ad, ad)
Lie.UPDATE_ALLOWED_OPERATORS(coAd, "@", coad, coad)
