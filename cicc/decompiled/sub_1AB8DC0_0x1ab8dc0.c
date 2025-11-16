// Function: sub_1AB8DC0
// Address: 0x1ab8dc0
//
void __fastcall sub_1AB8DC0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15)
{
  __int64 v16; // rdx
  __int64 v17; // rdx

  v16 = *(_QWORD *)(a2 + 80);
  if ( !v16 )
    BUG();
  v17 = *(_QWORD *)(v16 + 24);
  if ( v17 )
    v17 -= 24;
  sub_1AB7B70(a1, a2, v17, a3, a4, a5, a7, a8, a9, a10, a11, a12, a13, a14, a6, a15);
}
