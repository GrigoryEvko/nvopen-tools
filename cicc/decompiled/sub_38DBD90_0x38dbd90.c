// Function: sub_38DBD90
// Address: 0x38dbd90
//
__int64 __fastcall sub_38DBD90(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        const __m128i *a9,
        unsigned int a10)
{
  __int64 v10; // r10
  __m128i v12; // [rsp+0h] [rbp-30h] BYREF

  v10 = *(_QWORD *)(a2 + 8);
  if ( a9[1].m128i_i8[0] )
    v12 = _mm_loadu_si128(a9);
  sub_38C4140(a1, v10, a4, a5, a7, a8, a3, a6, &v12, a10);
  return a1;
}
