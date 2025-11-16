// Function: sub_1BDC4B0
// Address: 0x1bdc4b0
//
__int64 __fastcall sub_1BDC4B0(
        __int64 a1,
        __int64 **a2,
        __int64 **a3,
        __int64 a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 **v12; // rbp
  __int64 **v14[3]; // [rsp-18h] [rbp-18h] BYREF

  if ( !a2 || !a3 )
    return 0;
  v14[2] = v12;
  v14[0] = a2;
  v14[1] = a3;
  return sub_1BDB410(a1, v14, 2u, a4, 0, 1, a5, a6, a7, a8, a9, a10, a11, a12);
}
