// Function: sub_2AAA550
// Address: 0x2aaa550
//
__int64 __fastcall sub_2AAA550(__int64 ***a1, int *a2)
{
  __int64 **v2; // rdx
  __int64 *v3; // rcx
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rsi
  __int64 v9; // r9
  int v10; // edi
  char v11; // r10
  int v12; // r12d
  unsigned int i; // eax
  const __m128i *v14; // r11
  unsigned int v15; // eax
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  bool v18; // zf
  __int64 v19; // rax
  __int64 v20; // [rsp-8h] [rbp-8h]

  v2 = *a1;
  v3 = **a1;
  if ( *v3 )
    return 0;
  v6 = v2[1][5];
  v7 = *v2[2];
  v8 = *(unsigned int *)(v6 + 408);
  v9 = *(_QWORD *)(v6 + 392);
  if ( (_DWORD)v8 )
  {
    v10 = *a2;
    v11 = *((_BYTE *)a2 + 4);
    v12 = 1;
    for ( i = (v8 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)(v11 == 0) + 37 * v10 - 1)
                | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))) >> 31)
             ^ (484763065 * ((v11 == 0) + 37 * v10 - 1))); ; i = (v8 - 1) & v15 )
    {
      v14 = (const __m128i *)(v9 + ((unsigned __int64)i << 6));
      if ( v7 == v14->m128i_i64[0] && v10 == v14->m128i_i32[2] && v11 == v14->m128i_i8[12] )
        break;
      if ( v14->m128i_i64[0] == -4096 && v14->m128i_i32[2] == -1 && v14->m128i_i8[12] )
        goto LABEL_13;
      v15 = v12 + i;
      ++v12;
    }
  }
  else
  {
LABEL_13:
    v14 = (const __m128i *)(v9 + (v8 << 6));
  }
  v16 = _mm_loadu_si128(v14 + 2);
  v17 = _mm_loadu_si128(v14 + 3);
  v18 = v14[1].m128i_i32[0] == 6;
  v19 = v14[1].m128i_i64[1];
  *((__m128i *)&v20 - 5) = _mm_loadu_si128(v14 + 1);
  *((__m128i *)&v20 - 4) = v16;
  *((__m128i *)&v20 - 3) = v17;
  if ( !v18 )
    return 0;
  *v3 = v19;
  *v2[3] = *(__int64 *)((char *)&v20 - 60);
  return 1;
}
