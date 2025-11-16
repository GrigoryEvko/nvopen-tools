// Function: sub_3060E80
// Address: 0x3060e80
//
void __fastcall sub_3060E80(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __m128i *v8; // rax
  __m128i *v9; // rdx
  __m128i *v10; // r12
  __int64 v11; // rcx
  unsigned __int64 v12; // r14
  __m128i *v13; // rcx
  __m128i v14; // xmm0
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // rsi
  __m128i *v18; // r15
  void (__fastcall *v19)(unsigned __int64, unsigned __int64, __int64); // rax
  int v20; // r15d
  unsigned __int64 v21[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = (__m128i *)sub_C8D7D0(a1, a1 + 16, a2, 0x20u, v21, a6);
  v9 = *(__m128i **)a1;
  v10 = v8;
  v11 = 2LL * *(unsigned int *)(a1 + 8);
  v12 = *(_QWORD *)a1 + v11 * 16;
  if ( *(_QWORD *)a1 != v12 )
  {
    v13 = &v8[v11];
    do
    {
      if ( v8 )
      {
        v8[1].m128i_i64[0] = 0;
        v14 = _mm_loadu_si128(v9);
        *v9 = _mm_loadu_si128(v8);
        *v8 = v14;
        v15 = v9[1].m128i_i64[0];
        v9[1].m128i_i64[0] = 0;
        v16 = v8[1].m128i_i64[1];
        v8[1].m128i_i64[0] = v15;
        v17 = v9[1].m128i_i64[1];
        v9[1].m128i_i64[1] = v16;
        v8[1].m128i_i64[1] = v17;
      }
      v8 += 2;
      v9 += 2;
    }
    while ( v8 != v13 );
    v18 = *(__m128i **)a1;
    v12 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v12 )
    {
      do
      {
        v19 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v12 - 16);
        v12 -= 32LL;
        if ( v19 )
          v19(v12, v12, 3);
      }
      while ( v18 != (__m128i *)v12 );
      v12 = *(_QWORD *)a1;
    }
  }
  v20 = v21[0];
  if ( v6 != v12 )
    _libc_free(v12);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v20;
}
