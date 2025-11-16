// Function: sub_313A0B0
// Address: 0x313a0b0
//
void __fastcall sub_313A0B0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __m128i *v7; // rax
  unsigned __int64 v8; // r14
  __m128i *v9; // rdx
  __m128i v10; // xmm0
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rcx
  __m128i *v14; // r15
  void (__fastcall *v15)(unsigned __int64, unsigned __int64, __int64); // rax
  int v16; // r15d
  unsigned __int64 v17[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = sub_C8D7D0(a1, a1 + 16, a2, 0x28u, v17, a6);
  v7 = *(__m128i **)a1;
  v8 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v8 )
  {
    v9 = (__m128i *)v6;
    do
    {
      if ( v9 )
      {
        v9[1].m128i_i64[0] = 0;
        v10 = _mm_loadu_si128(v7);
        *v7 = _mm_loadu_si128(v9);
        *v9 = v10;
        v11 = v7[1].m128i_i64[0];
        v7[1].m128i_i64[0] = 0;
        v12 = v9[1].m128i_i64[1];
        v9[1].m128i_i64[0] = v11;
        v13 = v7[1].m128i_i64[1];
        v7[1].m128i_i64[1] = v12;
        v9[1].m128i_i64[1] = v13;
        v9[2].m128i_i32[0] = v7[2].m128i_i32[0];
        v9[2].m128i_i8[4] = v7[2].m128i_i8[4];
      }
      v7 = (__m128i *)((char *)v7 + 40);
      v9 = (__m128i *)((char *)v9 + 40);
    }
    while ( (__m128i *)v8 != v7 );
    v14 = *(__m128i **)a1;
    v8 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v8 )
    {
      do
      {
        v15 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v8 - 24);
        v8 -= 40LL;
        if ( v15 )
          v15(v8, v8, 3);
      }
      while ( (__m128i *)v8 != v14 );
      v8 = *(_QWORD *)a1;
    }
  }
  v16 = v17[0];
  if ( a1 + 16 != v8 )
    _libc_free(v8);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 12) = v16;
}
