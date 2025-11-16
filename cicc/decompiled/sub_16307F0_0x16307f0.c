// Function: sub_16307F0
// Address: 0x16307f0
//
__int64 __fastcall sub_16307F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 v14; // rdi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8

  v14 = *(_QWORD *)(a1 + 16);
  if ( (v14 & 4) == 0 )
    return sub_1623F10(a1, a2, a3, a4, a5);
  sub_16302D0((const __m128i *)(v14 & 0xFFFFFFFFFFFFFFF8LL), 0, a6, a7, a8, a9, a10, a11, a12, a13);
  return sub_1623F10(a1, 0, v16, v17, v18);
}
