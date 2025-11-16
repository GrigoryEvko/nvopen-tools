// Function: sub_34C9770
// Address: 0x34c9770
//
void __fastcall sub_34C9770(
        _QWORD *a1,
        __int64 a2,
        __m128i a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10)
{
  float v10; // xmm0_4

  v10 = sub_34C80D0(a1, a2, 0, 0, a3, a4, a5, a6, a7, a8, a9, a10);
  if ( v10 >= 0.0 )
    *(float *)(a2 + 116) = v10;
}
