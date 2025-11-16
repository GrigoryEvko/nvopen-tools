// Function: sub_3986C10
// Address: 0x3986c10
//
char *__fastcall sub_3986C10(
        const __m128i *src,
        const __m128i *a2,
        const __m128i *a3,
        const __m128i *a4,
        __m128i *a5,
        __int64 a6)
{
  const __m128i *v8; // r12
  __int64 v10; // rsi
  int v11; // eax
  __int64 v12; // r13
  int v13; // eax
  unsigned int v14; // r11d
  __int64 *v15; // rsi
  __int64 v16; // r15
  unsigned int v17; // esi
  __m128i v18; // xmm0
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rax
  int v22; // eax
  unsigned int v23; // edx
  __int64 v24; // r11
  int v25; // r14d
  int v26; // r15d
  __int64 *v27; // r14
  unsigned int v28; // eax
  __m128i v29; // xmm1
  size_t v30; // r13
  __int8 *v31; // r8
  int v33; // esi
  int v34; // r14d

  v8 = a3;
  if ( a3 != a4 && src != a2 )
  {
    while ( 1 )
    {
      v19 = v8->m128i_i64[0];
      v20 = src->m128i_i64[0];
      if ( !v8->m128i_i64[0] )
        break;
      v10 = *(_QWORD *)(*(_QWORD *)(a6 + 8) + 256LL);
      v11 = *(_DWORD *)(v10 + 104);
      if ( !v11 )
        goto LABEL_21;
      v12 = *(_QWORD *)(v10 + 88);
      v13 = v11 - 1;
      v14 = v13 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v15 = (__int64 *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( v19 != *v15 )
      {
        v33 = 1;
        while ( v16 != -8 )
        {
          v34 = v33 + 1;
          v14 = v13 & (v33 + v14);
          v15 = (__int64 *)(v12 + 16LL * v14);
          v16 = *v15;
          if ( v19 == *v15 )
            goto LABEL_6;
          v33 = v34;
        }
        if ( !v20 )
          goto LABEL_21;
        goto LABEL_14;
      }
LABEL_6:
      v17 = *((_DWORD *)v15 + 2);
      if ( !v20 )
        goto LABEL_7;
      v23 = v13 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v27 = (__int64 *)(v12 + 16LL * v23);
      v24 = *v27;
      if ( v20 != *v27 )
        goto LABEL_15;
LABEL_18:
      v28 = *((_DWORD *)v27 + 2);
      if ( v17 && (!v28 || v28 > v17) )
      {
LABEL_8:
        v18 = _mm_loadu_si128(v8);
        ++a5;
        ++v8;
        a5[-1] = v18;
        if ( src == a2 )
          goto LABEL_22;
        goto LABEL_9;
      }
LABEL_21:
      v29 = _mm_loadu_si128(src++);
      ++a5;
      a5[-1] = v29;
      if ( src == a2 )
        goto LABEL_22;
LABEL_9:
      if ( v8 == a4 )
        goto LABEL_22;
    }
    if ( !v20 )
      goto LABEL_21;
    v21 = *(_QWORD *)(*(_QWORD *)(a6 + 8) + 256LL);
    v12 = *(_QWORD *)(v21 + 88);
    v22 = *(_DWORD *)(v21 + 104);
    if ( !v22 )
      goto LABEL_21;
    v13 = v22 - 1;
LABEL_14:
    v23 = v13 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
    v24 = *(_QWORD *)(v12 + 16LL * v23);
    v17 = 0;
    if ( v20 == v24 )
      goto LABEL_21;
LABEL_15:
    v25 = 1;
    while ( v24 != -8 )
    {
      v26 = v25 + 1;
      v23 = v13 & (v25 + v23);
      v27 = (__int64 *)(v12 + 16LL * v23);
      v24 = *v27;
      if ( v20 == *v27 )
        goto LABEL_18;
      v25 = v26;
    }
LABEL_7:
    if ( v17 )
      goto LABEL_8;
    goto LABEL_21;
  }
LABEL_22:
  v30 = (char *)a2 - (char *)src;
  if ( a2 != src )
    a5 = (__m128i *)memmove(a5, src, v30);
  v31 = &a5->m128i_i8[v30];
  if ( a4 != v8 )
    v31 = (__int8 *)memmove(v31, v8, (char *)a4 - (char *)v8);
  return &v31[(char *)a4 - (char *)v8];
}
