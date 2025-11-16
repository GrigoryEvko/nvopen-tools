// Function: sub_8E5140
// Address: 0x8e5140
//
void __fastcall sub_8E5140(
        __int64 a1,
        __int64 *a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  int v9; // r8d
  unsigned int v10; // edx
  __int64 v11; // rsi
  __m128i *v12; // rcx
  __int64 v13; // rax
  __m128i v14; // xmm1
  __int64 v15; // r8
  __int64 v16; // rdx
  __m128i v17; // xmm0
  __int64 v18; // rax
  __m128i *v19; // rsi
  __int64 v20; // rcx
  int v21; // eax
  __m128i v22; // xmm0
  __int64 v23; // rsi

  v9 = *(_DWORD *)(a1 + 8);
  v10 = v9 & a3;
  v11 = 32LL * v10;
  v12 = (__m128i *)(*(_QWORD *)a1 + v11);
  if ( *v12 != 0 || v12[1].m128i_i64[0] )
  {
    do
    {
      do
      {
        v10 = v9 & (v10 + 1);
        v13 = *(_QWORD *)a1 + 32LL * v10;
      }
      while ( *(_OWORD *)v13 != 0 );
    }
    while ( *(_QWORD *)(v13 + 16) );
    v14 = _mm_loadu_si128(v12);
    v15 = v12[1].m128i_i64[0];
    v16 = v12->m128i_i64[0];
    *(_QWORD *)(v13 + 16) = v15;
    *(__m128i *)v13 = v14;
    if ( v16 || *(_QWORD *)(v13 + 8) || v15 )
      *(_QWORD *)(v13 + 24) = v12[1].m128i_i64[1];
    v17 = _mm_loadu_si128((const __m128i *)&a7);
    v18 = a8;
    v12->m128i_i64[0] = 0;
    v12->m128i_i64[1] = 0;
    v12[1].m128i_i64[0] = 0;
    v19 = (__m128i *)(*(_QWORD *)a1 + v11);
    v20 = *a2;
    v19[1].m128i_i64[0] = v18;
    *v19 = v17;
    if ( v17.m128i_i64[0] || v19->m128i_i64[1] || v18 )
      v19[1].m128i_i64[1] = v20;
  }
  else
  {
    v22 = _mm_loadu_si128((const __m128i *)&a7);
    v23 = *a2;
    v12[1].m128i_i64[0] = a8;
    *v12 = v22;
    if ( v22.m128i_i64[0] || v12->m128i_i64[1] || v12[1].m128i_i64[0] )
      v12[1].m128i_i64[1] = v23;
  }
  v21 = *(_DWORD *)(a1 + 12) + 1;
  *(_DWORD *)(a1 + 12) = v21;
  if ( (unsigned int)(2 * v21) > *(_DWORD *)(a1 + 8) )
    sub_8E4FD0(a1);
}
