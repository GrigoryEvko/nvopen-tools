// Function: sub_1E6D510
// Address: 0x1e6d510
//
__int64 *__fastcall sub_1E6D510(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 i; // r13
  __int64 v9; // r14
  __int64 *v10; // rbx
  __m128i v11; // xmm0
  __int64 v15; // [rsp+20h] [rbp-60h]
  __m128i v16; // [rsp+30h] [rbp-50h] BYREF
  __int64 v17; // [rsp+40h] [rbp-40h]

  v15 = (a3 - 1) / 2;
  if ( a2 >= v15 )
  {
    v9 = a2;
  }
  else
  {
    for ( i = a2; ; i = v9 )
    {
      v9 = 2 * (i + 1);
      v10 = (__int64 *)(a1 + 16 * (i + 1));
      if ( (unsigned __int8)sub_1E6D280((__int64 *)&a7, *v10, *(_QWORD *)(a1 + 8 * (v9 - 1)), a4, a1 + 8 * (v9 - 1)) )
        v10 = (__int64 *)(a1 + 8 * --v9);
      *(_QWORD *)(a1 + 8 * i) = *v10;
      if ( v9 >= v15 )
        break;
    }
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v9 )
  {
    *(_QWORD *)(a1 + 8 * v9) = *(_QWORD *)(a1 + 8 * (2 * v9 + 1));
    v9 = 2 * v9 + 1;
  }
  v11 = _mm_loadu_si128((const __m128i *)&a7);
  v17 = a8;
  v16 = v11;
  return sub_1E6D460(a1, v9, a2, a4, (__int64)&v16);
}
