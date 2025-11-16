// Function: sub_2FEE7E0
// Address: 0x2fee7e0
//
const __m128i *__fastcall sub_2FEE7E0(__int64 a1, const void *a2, size_t a3)
{
  unsigned int v3; // eax
  unsigned int v4; // esi
  __int64 v5; // rdx
  __int64 v6; // rcx
  const __m128i *result; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __m128i v10; // xmm0
  __m128i v11; // xmm1

  v3 = sub_C55C90(a1 + 8, a2, a3);
  v4 = *(_DWORD *)(a1 + 32);
  v5 = 56 * (v3 + 1LL);
  v6 = 56LL * v4 - v5;
  result = (const __m128i *)(v5 + *(_QWORD *)(a1 + 24));
  v8 = 0x6DB6DB6DB6DB6DB7LL * (v6 >> 3);
  if ( v6 > 0 )
  {
    do
    {
      v9 = result[2].m128i_i64[1];
      v10 = _mm_loadu_si128(result);
      result = (const __m128i *)((char *)result + 56);
      v11 = _mm_loadu_si128((const __m128i *)((char *)result - 40));
      result[-5].m128i_i64[1] = v9;
      LOBYTE(v9) = result[-1].m128i_i8[8];
      result[-7] = v10;
      result[-6] = v11;
      result[-4].m128i_i8[0] = v9;
      --v8;
    }
    while ( v8 );
    v4 = *(_DWORD *)(a1 + 32);
  }
  *(_DWORD *)(a1 + 32) = v4 - 1;
  return result;
}
