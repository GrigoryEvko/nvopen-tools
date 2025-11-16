// Function: sub_1824B60
// Address: 0x1824b60
//
__int64 __fastcall sub_1824B60(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  double v10; // xmm4_8
  double v11; // xmm5_8

  if ( !(unsigned __int8)sub_3946E40(
                           *(_QWORD *)(a1 + 392),
                           (unsigned int)"dataflow",
                           8,
                           (unsigned int)"src",
                           3,
                           *(_QWORD *)(a1 + 392),
                           *(_QWORD *)(a2 + 176),
                           *(_QWORD *)(a2 + 184),
                           (__int64)"skip",
                           4) )
    sub_1821A70(a1, a2, a3, a4, a5, a6, v10, v11, a9, a10);
  return 0;
}
