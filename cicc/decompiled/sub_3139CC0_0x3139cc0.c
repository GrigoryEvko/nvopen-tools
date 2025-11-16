// Function: sub_3139CC0
// Address: 0x3139cc0
//
__int64 __fastcall sub_3139CC0(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *v6; // r12
  __int64 result; // rax
  __int64 v8; // r14
  __m128i v11; // xmm0
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  char **v16; // rsi
  __int64 v17; // rdi
  __m128i *v18; // r12
  __int64 v19; // rbx
  unsigned __int64 v20; // rdi

  v6 = *(__m128i **)a1;
  result = 11LL * *(unsigned int *)(a1 + 8);
  v8 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v8 )
  {
    do
    {
      while ( 1 )
      {
        if ( a2 )
        {
          a2[1].m128i_i64[0] = 0;
          v11 = _mm_loadu_si128(v6);
          *v6 = _mm_loadu_si128(a2);
          *a2 = v11;
          v12 = v6[1].m128i_i64[0];
          v6[1].m128i_i64[0] = 0;
          v13 = a2[1].m128i_i64[1];
          a2[1].m128i_i64[0] = v12;
          v14 = v6[1].m128i_i64[1];
          v6[1].m128i_i64[1] = v13;
          a2[1].m128i_i64[1] = v14;
          a2[2].m128i_i64[0] = v6[2].m128i_i64[0];
          a2[2].m128i_i64[1] = v6[2].m128i_i64[1];
          v15 = v6[3].m128i_i64[0];
          a2[4].m128i_i32[0] = 0;
          a2[3].m128i_i64[0] = v15;
          a2[3].m128i_i64[1] = (__int64)&a2[4].m128i_i64[1];
          a2[4].m128i_i32[1] = 2;
          if ( v6[4].m128i_i32[0] )
            break;
        }
        v6 = (__m128i *)((char *)v6 + 88);
        a2 = (__m128i *)((char *)a2 + 88);
        if ( (__m128i *)v8 == v6 )
          goto LABEL_7;
      }
      v16 = (char **)&v6[3].m128i_i64[1];
      v17 = (__int64)&a2[3].m128i_i64[1];
      v6 = (__m128i *)((char *)v6 + 88);
      a2 = (__m128i *)((char *)a2 + 88);
      sub_3120CF0(v17, v16, v13, a4, a5, a6);
    }
    while ( (__m128i *)v8 != v6 );
LABEL_7:
    v18 = *(__m128i **)a1;
    result = 11LL * *(unsigned int *)(a1 + 8);
    v19 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v19 )
    {
      do
      {
        v19 -= 88;
        v20 = *(_QWORD *)(v19 + 56);
        if ( v20 != v19 + 72 )
          _libc_free(v20);
        result = *(_QWORD *)(v19 + 16);
        if ( result )
          result = ((__int64 (__fastcall *)(__int64, __int64, __int64))result)(v19, v19, 3);
      }
      while ( (__m128i *)v19 != v18 );
    }
  }
  return result;
}
