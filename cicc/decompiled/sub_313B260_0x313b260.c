// Function: sub_313B260
// Address: 0x313b260
//
void __fastcall sub_313B260(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // r12
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r14
  const __m128i *v11; // rdx
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  unsigned __int64 v14; // rsi
  __int32 v15; // ecx
  const __m128i *v16; // rcx
  unsigned __int64 v17; // r15
  unsigned __int64 v18; // rdi
  int v19; // r15d
  unsigned __int64 v20[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x38u, v20, a6);
  v9 = *(_QWORD *)a1;
  v10 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v10 )
  {
    v11 = (const __m128i *)(v9 + 24);
    v12 = 7 * (((0xDB6DB6DB6DB6DB7LL * ((v10 - v9 - 56) >> 3)) & 0x1FFFFFFFFFFFFFFFLL) + 1);
    v13 = v8;
    v14 = v8 + 8 * v12;
    do
    {
      if ( v13 )
      {
        *(_QWORD *)v13 = v11[-2].m128i_i64[1];
        *(_QWORD *)(v13 + 8) = v13 + 24;
        v16 = (const __m128i *)v11[-1].m128i_i64[0];
        if ( v11 == v16 )
        {
          *(__m128i *)(v13 + 24) = _mm_loadu_si128(v11);
        }
        else
        {
          *(_QWORD *)(v13 + 8) = v16;
          *(_QWORD *)(v13 + 24) = v11->m128i_i64[0];
        }
        *(_QWORD *)(v13 + 16) = v11[-1].m128i_i64[1];
        v15 = v11[1].m128i_i32[0];
        v11[-1].m128i_i64[0] = (__int64)v11;
        v11[-1].m128i_i64[1] = 0;
        v11->m128i_i8[0] = 0;
        *(_DWORD *)(v13 + 40) = v15;
        *(_DWORD *)(v13 + 44) = v11[1].m128i_i32[1];
        *(_DWORD *)(v13 + 48) = v11[1].m128i_i32[2];
        *(_DWORD *)(v13 + 52) = v11[1].m128i_i32[3];
      }
      v13 += 56;
      v11 = (const __m128i *)((char *)v11 + 56);
    }
    while ( v14 != v13 );
    v17 = *(_QWORD *)a1;
    v10 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v10 )
    {
      do
      {
        v10 -= 56LL;
        v18 = *(_QWORD *)(v10 + 8);
        if ( v18 != v10 + 24 )
          j_j___libc_free_0(v18);
      }
      while ( v10 != v17 );
      v10 = *(_QWORD *)a1;
    }
  }
  v19 = v20[0];
  if ( v6 != v10 )
    _libc_free(v10);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v19;
}
