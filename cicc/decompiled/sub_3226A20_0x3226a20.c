// Function: sub_3226A20
// Address: 0x3226a20
//
void __fastcall sub_3226A20(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // r12
  const __m128i *v9; // rax
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // r14
  __m128i *v12; // rcx
  __m128i *v13; // rdx
  __int64 v14; // rsi
  const __m128i *v15; // r15
  unsigned __int64 v16; // rdi
  int v17; // r15d
  unsigned __int64 v18[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x40u, v18, a6);
  v9 = *(const __m128i **)a1;
  v10 = (unsigned __int64)*(unsigned int *)(a1 + 8) << 6;
  v11 = *(_QWORD *)a1 + v10;
  if ( *(_QWORD *)a1 != v11 )
  {
    v12 = (__m128i *)(v8 + v10);
    v13 = (__m128i *)v8;
    do
    {
      if ( v13 )
      {
        *v13 = _mm_loadu_si128(v9);
        v13[1].m128i_i64[0] = v9[1].m128i_i64[0];
        v13[1].m128i_i32[2] = v9[1].m128i_i32[2];
        v13[2].m128i_i64[0] = v9[2].m128i_i64[0];
        v13[2].m128i_i64[1] = v9[2].m128i_i64[1];
        v13[3].m128i_i64[0] = v9[3].m128i_i64[0];
        v14 = v9[3].m128i_i64[1];
        v9[3].m128i_i64[0] = 0;
        v9[2].m128i_i64[1] = 0;
        v9[2].m128i_i64[0] = 0;
        v13[3].m128i_i64[1] = v14;
      }
      v13 += 4;
      v9 += 4;
    }
    while ( v13 != v12 );
    v15 = *(const __m128i **)a1;
    v11 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6);
    if ( *(_QWORD *)a1 != v11 )
    {
      do
      {
        v16 = *(_QWORD *)(v11 - 32);
        v11 -= 64LL;
        if ( v16 )
          j_j___libc_free_0(v16);
      }
      while ( v15 != (const __m128i *)v11 );
      v11 = *(_QWORD *)a1;
    }
  }
  v17 = v18[0];
  if ( v6 != v11 )
    _libc_free(v11);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v17;
}
