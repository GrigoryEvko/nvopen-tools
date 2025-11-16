// Function: sub_32184B0
// Address: 0x32184b0
//
void __fastcall sub_32184B0(__m128i *a1)
{
  __int64 v1; // rcx
  __int64 v2; // rdx
  unsigned int v3; // esi
  __m128i *v4; // rax
  __m128i v5; // xmm0
  __int64 v6; // rdx
  __m128i v7; // xmm2
  __m128i v8; // [rsp+0h] [rbp-20h] BYREF
  __int64 v9; // [rsp+10h] [rbp-10h]

  v1 = a1[1].m128i_i64[0];
  v2 = a1[-1].m128i_i64[1];
  v9 = v1;
  v3 = *(_DWORD *)(v1 + 16);
  v8 = _mm_loadu_si128(a1);
  if ( *(_DWORD *)(v2 + 16) > v3 )
  {
    v4 = (__m128i *)((char *)a1 - 24);
    do
    {
      v5 = _mm_loadu_si128(v4);
      v6 = v4[1].m128i_i64[0];
      a1 = v4;
      v4 = (__m128i *)((char *)v4 - 24);
      v4[3] = v5;
      v4[4].m128i_i64[0] = v6;
    }
    while ( *(_DWORD *)(v1 + 16) < *(_DWORD *)(v4[1].m128i_i64[0] + 16) );
  }
  v7 = _mm_loadu_si128(&v8);
  a1[1].m128i_i64[0] = v1;
  *a1 = v7;
}
