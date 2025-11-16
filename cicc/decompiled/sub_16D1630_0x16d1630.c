// Function: sub_16D1630
// Address: 0x16d1630
//
_QWORD *__fastcall sub_16D1630(__int64 a1, unsigned __int64 a2, __int64 a3, const char *a4, __int64 a5)
{
  _QWORD *result; // rax
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rax
  __m128i v12; // kr00_16
  __m128i si128; // xmm2
  __m128i v14; // [rsp+0h] [rbp-70h] BYREF
  __m128i v15; // [rsp+10h] [rbp-60h]
  __m128i v16; // [rsp+20h] [rbp-50h] BYREF
  __m128i v17[4]; // [rsp+30h] [rbp-40h] BYREF

  for ( result = sub_16D1570(&v14, a1, a2, a4, a5); v14.m128i_i64[1]; v15 = si128 )
  {
    v11 = *(unsigned int *)(a3 + 8);
    if ( (unsigned int)v11 >= *(_DWORD *)(a3 + 12) )
    {
      sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v9, v10);
      v11 = *(unsigned int *)(a3 + 8);
    }
    v12 = v15;
    *(__m128i *)(*(_QWORD *)a3 + 16 * v11) = _mm_load_si128(&v14);
    ++*(_DWORD *)(a3 + 8);
    result = sub_16D1570(&v16, v12.m128i_i64[0], v12.m128i_u64[1], a4, a5);
    si128 = _mm_load_si128(v17);
    v14 = _mm_load_si128(&v16);
  }
  return result;
}
