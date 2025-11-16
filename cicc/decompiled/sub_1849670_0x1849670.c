// Function: sub_1849670
// Address: 0x1849670
//
void __fastcall sub_1849670(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  __m128i *v7; // rax
  unsigned __int64 v8; // r14
  __m128i *v9; // rdx
  __m128i v10; // xmm0
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rcx
  __m128i v14; // xmm2
  __m128i v15; // xmm0
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rcx
  __m128i v19; // xmm3
  __m128i v20; // xmm0
  __int64 v21; // rcx
  __int64 v22; // rcx
  __m128i *v23; // r15
  void (__fastcall *v24)(unsigned __int64, unsigned __int64, __int64); // rax
  void (__fastcall *v25)(unsigned __int64, unsigned __int64, __int64); // rax
  void (__fastcall *v26)(unsigned __int64, unsigned __int64, __int64); // rax

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v2 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v3 = ((v2
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v2
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v4 = a2;
  v5 = (v3 | (v3 >> 16) | HIDWORD(v3)) + 1;
  if ( v5 >= a2 )
    v4 = v5;
  if ( v4 > 0xFFFFFFFF )
    v4 = 0xFFFFFFFFLL;
  v6 = malloc(104 * v4);
  if ( !v6 )
    sub_16BD1C0("Allocation failed", 1u);
  v7 = *(__m128i **)a1;
  v8 = *(_QWORD *)a1 + 104LL * *(unsigned int *)(a1 + 8);
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
        v14 = _mm_loadu_si128(v9 + 2);
        v9[3].m128i_i64[0] = 0;
        v9[1].m128i_i64[1] = v13;
        v15 = _mm_loadu_si128(v7 + 2);
        v7[2] = v14;
        v9[2] = v15;
        v16 = v7[3].m128i_i64[0];
        v7[3].m128i_i64[0] = 0;
        v17 = v9[3].m128i_i64[1];
        v9[3].m128i_i64[0] = v16;
        v18 = v7[3].m128i_i64[1];
        v7[3].m128i_i64[1] = v17;
        v19 = _mm_loadu_si128(v9 + 4);
        v9[5].m128i_i64[0] = 0;
        v9[3].m128i_i64[1] = v18;
        v20 = _mm_loadu_si128(v7 + 4);
        v7[4] = v19;
        v9[4] = v20;
        v21 = v7[5].m128i_i64[0];
        v7[5].m128i_i64[0] = 0;
        v9[5].m128i_i64[0] = v21;
        v22 = v7[5].m128i_i64[1];
        v7[5].m128i_i64[1] = v9[5].m128i_i64[1];
        v9[5].m128i_i64[1] = v22;
        v9[6].m128i_i32[0] = v7[6].m128i_i32[0];
        v9[6].m128i_i8[4] = v7[6].m128i_i8[4];
      }
      v7 = (__m128i *)((char *)v7 + 104);
      v9 = (__m128i *)((char *)v9 + 104);
    }
    while ( (__m128i *)v8 != v7 );
    v23 = *(__m128i **)a1;
    v8 = *(_QWORD *)a1 + 104LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v8 )
    {
      do
      {
        v24 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v8 - 24);
        v8 -= 104LL;
        if ( v24 )
          v24(v8 + 64, v8 + 64, 3);
        v25 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v8 + 48);
        if ( v25 )
          v25(v8 + 32, v8 + 32, 3);
        v26 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v8 + 16);
        if ( v26 )
          v26(v8, v8, 3);
      }
      while ( (__m128i *)v8 != v23 );
      v8 = *(_QWORD *)a1;
    }
  }
  if ( v8 != a1 + 16 )
    _libc_free(v8);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 12) = v4;
}
