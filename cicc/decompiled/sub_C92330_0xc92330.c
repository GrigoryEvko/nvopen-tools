// Function: sub_C92330
// Address: 0xc92330
//
__int64 __fastcall sub_C92330(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 result; // rax
  __int64 v11; // rax
  __m128i v12; // xmm0
  __int64 v13; // rsi
  unsigned __int64 v14; // rdx
  __m128i si128; // xmm2
  __m128i v16; // [rsp+0h] [rbp-80h] BYREF
  __m128i v17; // [rsp+10h] [rbp-70h] BYREF
  __m128i v18; // [rsp+20h] [rbp-60h]
  __m128i v19; // [rsp+30h] [rbp-50h] BYREF
  __m128i v20[4]; // [rsp+40h] [rbp-40h] BYREF

  sub_C92270(&v17, a1, a2, a4, a5);
  result = v17.m128i_i64[1];
  if ( v17.m128i_i64[1] )
  {
    do
    {
      v17.m128i_i64[1] = result;
      v11 = *(unsigned int *)(a3 + 8);
      v12 = _mm_load_si128(&v17);
      if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        v16 = v12;
        sub_C8D5F0(a3, (const void *)(a3 + 16), v11 + 1, 0x10u, v8, v9);
        v11 = *(unsigned int *)(a3 + 8);
        v12 = _mm_load_si128(&v16);
      }
      v13 = v18.m128i_i64[0];
      *(__m128i *)(*(_QWORD *)a3 + 16 * v11) = v12;
      v14 = v18.m128i_u64[1];
      ++*(_DWORD *)(a3 + 8);
      sub_C92270(&v19, v13, v14, a4, a5);
      si128 = _mm_load_si128(v20);
      result = v19.m128i_i64[1];
      v17 = _mm_load_si128(&v19);
      v18 = si128;
    }
    while ( v19.m128i_i64[1] );
  }
  return result;
}
