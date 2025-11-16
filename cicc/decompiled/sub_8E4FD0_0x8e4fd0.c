// Function: sub_8E4FD0
// Address: 0x8e4fd0
//
void __fastcall sub_8E4FD0(__int64 a1)
{
  __int64 v2; // r14
  unsigned int v3; // ebx
  _QWORD *v4; // rax
  unsigned __int64 v5; // rdx
  const __m128i *v6; // rcx
  __int64 v7; // r8
  __int64 *v8; // r9
  _QWORD *v9; // rsi
  __int64 v10; // rdi
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r10
  unsigned __int64 v13; // r10
  unsigned int i; // eax
  __int64 v15; // rax
  bool v16; // zf

  v2 = *(unsigned int *)(a1 + 8);
  v3 = 2 * v2 + 1;
  v4 = (_QWORD *)sub_823970(32LL * (unsigned int)(2 * v2 + 2));
  v9 = v4;
  if ( 2 * (_DWORD)v2 != -2 )
  {
    v5 = (unsigned __int64)&v4[4 * v3 + 4];
    do
    {
      if ( v4 )
      {
        *v4 = 0;
        v4[1] = 0;
        v4[2] = 0;
      }
      v4 += 4;
    }
    while ( (_QWORD *)v5 != v4 );
  }
  v10 = *(_QWORD *)a1;
  if ( (_DWORD)v2 != -1 )
  {
    v6 = *(const __m128i **)a1;
    v7 = v10 + 32 * v2 + 32;
    do
    {
      v11 = v6->m128i_i64[0];
      v5 = v6[1].m128i_u64[0];
      v12 = v6->m128i_u64[1];
      if ( v6->m128i_i64[0] || v12 || v5 )
      {
        v13 = v12 >> 3;
        v8 = (__int64 *)(v13 + 31 * ((v11 >> 3) + 527));
        for ( i = v3 & ((v5 >> 3) + 31 * (v13 + 31 * ((v11 >> 3) + 527))); ; i = v3 & (i + 1) )
        {
          v5 = (unsigned __int64)&v9[4 * i];
          if ( !*(_QWORD *)v5 && !*(_QWORD *)(v5 + 8) && !*(_QWORD *)(v5 + 16) )
            break;
        }
        *(__m128i *)v5 = _mm_loadu_si128(v6);
        v15 = v6[1].m128i_i64[0];
        v16 = *(_QWORD *)v5 == 0;
        *(_QWORD *)(v5 + 16) = v15;
        if ( !v16 || *(_QWORD *)(v5 + 8) || v15 )
          *(_QWORD *)(v5 + 24) = v6[1].m128i_i64[1];
      }
      v6 += 2;
    }
    while ( (const __m128i *)v7 != v6 );
  }
  *(_QWORD *)a1 = v9;
  *(_DWORD *)(a1 + 8) = v3;
  sub_823A00(v10, 32LL * (unsigned int)(v2 + 1), v5, (__int64)v6, v7, v8);
}
