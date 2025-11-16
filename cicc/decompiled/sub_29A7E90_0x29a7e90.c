// Function: sub_29A7E90
// Address: 0x29a7e90
//
__m128i *__fastcall sub_29A7E90(__int64 a1, __int64 a2, __m128i *a3, __int64 a4)
{
  __int64 v6; // rbx
  _BYTE *v7; // rdx
  __int64 v8; // rdi
  unsigned __int64 v9; // rsi
  __m128i *v10; // r8
  __int64 v11; // r9
  __int64 v12; // rcx
  __m128i *v13; // rax
  __m128i *v14; // rcx
  unsigned __int64 v15; // rdx
  const __m128i *v16; // r15
  unsigned int v17; // eax
  __int64 v19; // r9
  __int64 v20; // rax
  const __m128i *v21; // r15
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // r8
  __m128i *v24; // rax
  __int64 v25; // rdi
  __int64 v26; // r9
  __int64 v27; // rdi
  const void *v28; // rsi
  __int8 *v29; // r15
  _BYTE v30[96]; // [rsp+0h] [rbp-60h] BYREF

  if ( a1 )
  {
    v6 = a1;
    do
    {
      while ( 1 )
      {
        v7 = *(_BYTE **)(v6 + 24);
        if ( *v7 != 96 )
          goto LABEL_3;
        a3->m128i_i64[0] = (__int64)v7;
        if ( *(_DWORD *)(a4 + 16) )
        {
          sub_29A7A10((__int64)v30, a4, a3);
          if ( v30[32] )
          {
            v20 = *(unsigned int *)(a4 + 40);
            v21 = a3;
            v22 = *(_QWORD *)(a4 + 32);
            v23 = v20 + 1;
            if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 44) )
            {
              v27 = a4 + 32;
              v28 = (const void *)(a4 + 48);
              if ( v22 > (unsigned __int64)a3 || (unsigned __int64)a3 >= v22 + 32 * v20 )
              {
                v21 = a3;
                sub_C8D5F0(v27, v28, v23, 0x20u, v23, v19);
                v22 = *(_QWORD *)(a4 + 32);
                v20 = *(unsigned int *)(a4 + 40);
              }
              else
              {
                v29 = &a3->m128i_i8[-v22];
                sub_C8D5F0(v27, v28, v23, 0x20u, v23, v19);
                v22 = *(_QWORD *)(a4 + 32);
                v20 = *(unsigned int *)(a4 + 40);
                v21 = (const __m128i *)&v29[v22];
              }
            }
            v24 = (__m128i *)(v22 + 32 * v20);
            *v24 = _mm_loadu_si128(v21);
            v24[1] = _mm_loadu_si128(v21 + 1);
            ++*(_DWORD *)(a4 + 40);
          }
          goto LABEL_3;
        }
        v8 = *(unsigned int *)(a4 + 40);
        v9 = *(_QWORD *)(a4 + 32);
        v10 = (__m128i *)(v9 + 32 * v8);
        v11 = (32 * v8) >> 5;
        v12 = (32 * v8) >> 7;
        if ( !v12 )
          break;
        v13 = *(__m128i **)(a4 + 32);
        v14 = (__m128i *)(v9 + (v12 << 7));
        while ( v7 != (_BYTE *)v13->m128i_i64[0] )
        {
          if ( v7 == (_BYTE *)v13[2].m128i_i64[0] )
          {
            if ( v10 != &v13[2] )
              goto LABEL_3;
            goto LABEL_14;
          }
          if ( v7 == (_BYTE *)v13[4].m128i_i64[0] )
          {
            if ( v10 != &v13[4] )
              goto LABEL_3;
            goto LABEL_14;
          }
          if ( v7 == (_BYTE *)v13[6].m128i_i64[0] )
          {
            if ( v10 != &v13[6] )
              goto LABEL_3;
            goto LABEL_14;
          }
          v13 += 8;
          if ( v14 == v13 )
          {
            v11 = ((char *)v10 - (char *)v13) >> 5;
            goto LABEL_28;
          }
        }
LABEL_13:
        if ( v10 == v13 )
          goto LABEL_14;
LABEL_3:
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          return a3;
      }
      v13 = *(__m128i **)(a4 + 32);
LABEL_28:
      if ( v11 != 2 )
      {
        if ( v11 != 3 )
        {
          if ( v11 != 1 )
            goto LABEL_14;
          goto LABEL_31;
        }
        if ( v7 == (_BYTE *)v13->m128i_i64[0] )
          goto LABEL_13;
        v13 += 2;
      }
      if ( v7 == (_BYTE *)v13->m128i_i64[0] )
        goto LABEL_13;
      v13 += 2;
LABEL_31:
      if ( v7 == (_BYTE *)v13->m128i_i64[0] )
        goto LABEL_13;
LABEL_14:
      v15 = v8 + 1;
      v16 = a3;
      if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 44) )
      {
        v25 = a4 + 32;
        v26 = a4 + 48;
        if ( v9 > (unsigned __int64)a3 || v10 <= a3 )
        {
          sub_C8D5F0(v25, (const void *)(a4 + 48), v15, 0x20u, (__int64)v10, v26);
          v10 = (__m128i *)(*(_QWORD *)(a4 + 32) + 32LL * *(unsigned int *)(a4 + 40));
        }
        else
        {
          sub_C8D5F0(v25, (const void *)(a4 + 48), v15, 0x20u, (__int64)v10, v26);
          v16 = (__m128i *)((char *)a3 + *(_QWORD *)(a4 + 32) - v9);
          v10 = (__m128i *)(*(_QWORD *)(a4 + 32) + 32LL * *(unsigned int *)(a4 + 40));
        }
      }
      *v10 = _mm_loadu_si128(v16);
      v10[1] = _mm_loadu_si128(v16 + 1);
      v17 = *(_DWORD *)(a4 + 40) + 1;
      *(_DWORD *)(a4 + 40) = v17;
      if ( v17 <= 4 )
        goto LABEL_3;
      sub_29A7C60(a4);
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v6 );
  }
  return a3;
}
