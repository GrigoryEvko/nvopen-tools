// Function: sub_1BE00E0
// Address: 0x1be00e0
//
__int64 __fastcall sub_1BE00E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 *a6,
        __m128i a7,
        __m128i a8,
        __m128i a9,
        __m128i a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 v14; // rbp
  int v17; // eax
  __int64 v19[2]; // [rsp-10h] [rbp-10h] BYREF

  if ( !a3 )
    return 0;
  v17 = *(unsigned __int8 *)(a3 + 16);
  if ( (unsigned __int8)v17 <= 0x17u )
    return 0;
  v19[1] = v14;
  v19[0] = a1;
  if ( (unsigned int)(v17 - 35) >= 0x12 )
    a2 = 0;
  return (unsigned int)sub_1BDDB00(
                         a2,
                         a3,
                         a4,
                         a5,
                         a6,
                         a7,
                         a8,
                         a9,
                         a10,
                         a11,
                         a12,
                         a13,
                         a14,
                         (__int64)a6,
                         (int)sub_1BDC7F0,
                         v19);
}
