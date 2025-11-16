// Function: sub_2DD9530
// Address: 0x2dd9530
//
__int64 __fastcall sub_2DD9530(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  const __m128i *v13; // r15
  __m128i *v14; // rdi
  unsigned __int64 v15; // r12
  __int64 v16; // r13
  __int64 v17; // rax
  char **v18; // rsi
  const __m128i *v19; // r15
  unsigned __int64 v20; // rdi
  int v21; // r15d
  __int64 v23; // [rsp+8h] [rbp-48h]
  unsigned __int64 v24[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x28u, v24, a6);
  v13 = *(const __m128i **)a1;
  v23 = v8;
  v14 = (__m128i *)v8;
  v15 = (unsigned __int64)v13 + 40 * *(unsigned int *)(a1 + 8);
  if ( v13 != (const __m128i *)v15 )
  {
    do
    {
      while ( 1 )
      {
        v16 = 40;
        if ( v14 )
        {
          v16 = (__int64)&v14[2].m128i_i64[1];
          *v14 = _mm_loadu_si128(v13);
          v17 = v13[1].m128i_i64[0];
          v14[1].m128i_i64[1] = (__int64)&v14[2].m128i_i64[1];
          v14[1].m128i_i64[0] = v17;
          v14[2].m128i_i32[0] = 0;
          v14[2].m128i_i32[1] = 0;
          if ( v13[2].m128i_i32[0] )
            break;
        }
        v13 = (const __m128i *)((char *)v13 + 40);
        v14 = (__m128i *)v16;
        if ( (const __m128i *)v15 == v13 )
          goto LABEL_7;
      }
      v18 = (char **)&v13[1].m128i_i64[1];
      v13 = (const __m128i *)((char *)v13 + 40);
      sub_2DD33A0((__int64)&v14[1].m128i_i64[1], v18, v9, v10, v11, v12);
      v14 = (__m128i *)((char *)v14 + 40);
    }
    while ( (const __m128i *)v15 != v13 );
LABEL_7:
    v19 = *(const __m128i **)a1;
    v15 = *(_QWORD *)a1 + 40LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v15 )
    {
      do
      {
        v15 -= 40LL;
        v20 = *(_QWORD *)(v15 + 24);
        if ( v20 != v15 + 40 )
          _libc_free(v20);
      }
      while ( (const __m128i *)v15 != v19 );
      v15 = *(_QWORD *)a1;
    }
  }
  v21 = v24[0];
  if ( v6 != v15 )
    _libc_free(v15);
  *(_DWORD *)(a1 + 12) = v21;
  *(_QWORD *)a1 = v23;
  return v23;
}
