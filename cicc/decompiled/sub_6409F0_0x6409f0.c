// Function: sub_6409F0
// Address: 0x6409f0
//
__int64 __fastcall sub_6409F0(const __m128i **a1)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 v4; // rdi
  __int64 v5; // rbx
  __m128i *v6; // rax
  __m128i *v7; // r15
  const __m128i *v8; // rdx
  __m128i *v9; // rcx
  __int64 result; // rax
  const __m128i *v11; // [rsp+8h] [rbp-38h]

  v2 = (__int64)a1[1];
  if ( v2 <= 1 )
  {
    v4 = 32;
    v3 = 2;
  }
  else
  {
    v3 = v2 + (v2 >> 1) + 1;
    v4 = 16 * v3;
  }
  v5 = (__int64)a1[2];
  v11 = *a1;
  v6 = (__m128i *)sub_823970(v4);
  v7 = v6;
  if ( v5 > 0 )
  {
    v8 = v11;
    v9 = &v6[v5];
    do
    {
      if ( v6 )
        *v6 = _mm_loadu_si128(v8);
      ++v6;
      ++v8;
    }
    while ( v9 != v6 );
  }
  result = sub_823A00(v11, 16 * v2);
  *a1 = v7;
  a1[1] = (const __m128i *)v3;
  return result;
}
