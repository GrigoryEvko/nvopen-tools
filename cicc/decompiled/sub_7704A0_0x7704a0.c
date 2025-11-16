// Function: sub_7704A0
// Address: 0x7704a0
//
_QWORD *__fastcall sub_7704A0(__int64 a1)
{
  unsigned int v2; // r15d
  __m128i *v3; // r13
  unsigned int v4; // edi
  unsigned int v5; // eax
  int v6; // edx
  int v7; // esi
  int v8; // esi
  __int64 v9; // rax
  unsigned __int64 v10; // r14
  size_t v11; // rdx
  _QWORD *v12; // rcx
  int v13; // edi
  unsigned __int64 v14; // rdx
  char *v15; // rcx
  const __m128i *v16; // rsi
  __int64 m128i_i64; // r8
  __m128i *v18; // rax
  __m128i v19; // xmm0
  __int64 v21; // rax
  __int64 v22; // rax

  v2 = *(_DWORD *)(a1 + 8);
  v3 = *(__m128i **)a1;
  v4 = 32 * (v2 + 1);
  if ( v2 )
  {
    v5 = v2;
    v6 = 0;
    do
    {
      v7 = v6++;
      v5 &= v5 - 1;
    }
    while ( v5 );
    v8 = v7 + 2;
    v9 = v8;
    v10 = v8 - 1LL;
    if ( v8 > 10 )
    {
      v21 = sub_822B10(v4);
      v11 = v4;
      v12 = (_QWORD *)v21;
      goto LABEL_7;
    }
  }
  else
  {
    v10 = 0;
    v9 = 1;
  }
  v11 = v4;
  v12 = (_QWORD *)qword_4F08320[v9];
  if ( v12 )
  {
    qword_4F08320[v9] = *v12;
  }
  else
  {
    v22 = sub_823970(v4);
    v11 = v4;
    v12 = (_QWORD *)v22;
  }
LABEL_7:
  v13 = 2 * v2 + 1;
  v15 = (char *)memset(v12, 0, v11);
  if ( v2 != -1 )
  {
    v16 = v3;
    m128i_i64 = (__int64)v3[v2 + 1].m128i_i64;
    do
    {
      while ( 1 )
      {
        v14 = v16->m128i_i64[0];
        if ( v16->m128i_i64[0] )
          break;
        if ( (const __m128i *)m128i_i64 == ++v16 )
          goto LABEL_15;
      }
      for ( v14 >>= 3; ; LODWORD(v14) = v14 + 1 )
      {
        v14 = v13 & (unsigned int)v14;
        v18 = (__m128i *)&v15[16 * (unsigned int)v14];
        if ( !v18->m128i_i64[0] )
          break;
      }
      v19 = _mm_loadu_si128(v16++);
      *v18 = v19;
    }
    while ( (const __m128i *)m128i_i64 != v16 );
  }
LABEL_15:
  *(_QWORD *)a1 = v15;
  *(_DWORD *)(a1 + 8) = v13;
  if ( v10 > 0xA )
    return (_QWORD *)sub_822B90(v3, 16 * (v2 + 1), v14, v15);
  v3->m128i_i64[0] = qword_4F08320[v10];
  qword_4F08320[v10] = v3;
  return qword_4F08320;
}
