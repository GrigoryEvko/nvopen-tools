// Function: sub_349DE60
// Address: 0x349de60
//
__int64 __fastcall sub_349DE60(__m128i *a1, __int64 a2)
{
  __m128i *v3; // rdi
  __m128i v4; // xmm0
  __m128i v5; // xmm1
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned int v8; // r13d
  __m128i *v9; // rdi
  __int64 result; // rax
  unsigned int v11; // r13d
  size_t v12; // rdx
  size_t v13; // rdx

  v3 = a1 + 5;
  v4 = _mm_loadu_si128((const __m128i *)a2);
  v5 = _mm_loadu_si128((const __m128i *)(a2 + 16));
  v3[-3].m128i_i64[0] = *(_QWORD *)(a2 + 32);
  v6 = *(_QWORD *)(a2 + 40);
  v3[-5] = v4;
  v3[-3].m128i_i64[1] = v6;
  v7 = *(_QWORD *)(a2 + 48);
  v3[-4] = v5;
  v3[-2].m128i_i64[0] = v7;
  v3[-2].m128i_i32[2] = *(_DWORD *)(a2 + 56);
  a1[4].m128i_i64[0] = (__int64)v3;
  a1[4].m128i_i64[1] = 0x800000000LL;
  v8 = *(_DWORD *)(a2 + 72);
  if ( v8 && &a1[4] != (__m128i *)(a2 + 64) )
  {
    v12 = 32LL * v8;
    if ( v8 <= 8
      || (sub_C8D5F0((__int64)a1[4].m128i_i64, v3, v8, 0x20u, (__int64)a1[4].m128i_i64, v8),
          v3 = (__m128i *)a1[4].m128i_i64[0],
          (v12 = 32LL * *(unsigned int *)(a2 + 72)) != 0) )
    {
      memcpy(v3, *(const void **)(a2 + 64), v12);
    }
    a1[4].m128i_i32[2] = v8;
  }
  v9 = a1 + 22;
  result = 0x800000000LL;
  a1[21].m128i_i64[0] = (__int64)a1[22].m128i_i64;
  a1[21].m128i_i64[1] = 0x800000000LL;
  v11 = *(_DWORD *)(a2 + 344);
  if ( v11 )
  {
    result = a2 + 336;
    if ( &a1[21] != (__m128i *)(a2 + 336) )
    {
      v13 = 4LL * v11;
      if ( v11 <= 8
        || (result = sub_C8D5F0((__int64)a1[21].m128i_i64, &a1[22], v11, 4u, (__int64)a1[21].m128i_i64, v11),
            v9 = (__m128i *)a1[21].m128i_i64[0],
            (v13 = 4LL * *(unsigned int *)(a2 + 344)) != 0) )
      {
        result = (__int64)memcpy(v9, *(const void **)(a2 + 336), v13);
      }
      a1[21].m128i_i32[2] = v11;
    }
  }
  return result;
}
