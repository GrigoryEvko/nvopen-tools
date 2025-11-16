// Function: sub_21F81D0
// Address: 0x21f81d0
//
__int64 __fastcall sub_21F81D0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rax
  double v12; // xmm4_8
  double v13; // xmm5_8

  if ( !(unsigned __int8)sub_1C2F070(a2) )
    return 0;
  v11 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  *(_QWORD *)(a1 + 216) = v11;
  if ( !v11 )
    return 0;
  sub_21F6F00(a1, a2, a3, a4, a5, a6, v12, v13, a9, a10);
  return 0;
}
