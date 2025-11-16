// Function: sub_398BB50
// Address: 0x398bb50
//
__m128i *__fastcall sub_398BB50(__m128i *a1, __int64 a2)
{
  __m128i *v2; // r13
  __int64 v3; // r14
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rsi
  __m128i v9; // rax
  bool v10; // zf
  __int64 v11; // rax
  __int64 v12; // rdx
  char *v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdx
  char *v20; // rsi
  const char *v21; // rdx
  __m128i *v22; // rdi
  size_t v23; // rdx
  __m128i *v24; // rax
  __m128i *v25; // rdi
  __m128i *v26; // rax
  __int64 v27; // rdi
  size_t v28; // r8
  __int64 v29; // rsi
  __int8 v30; // al
  size_t v32; // r9
  __int64 v33; // rdi
  __int64 v34; // rsi
  size_t v35; // rdx
  __int64 v36; // rax
  __m128i v37; // xmm1
  __int64 v38; // [rsp+8h] [rbp-88h]
  char v39; // [rsp+17h] [rbp-79h]
  __int64 v40; // [rsp+18h] [rbp-78h]
  __m128i v41[2]; // [rsp+20h] [rbp-70h] BYREF
  __m128i *v42; // [rsp+40h] [rbp-50h] BYREF
  size_t n; // [rsp+48h] [rbp-48h]
  _QWORD src[8]; // [rsp+50h] [rbp-40h] BYREF

  v2 = 0;
  if ( !a1[282].m128i_i8[1] )
    return v2;
  v3 = *(_QWORD *)(a2 + 80);
  v2 = a1 + 307;
  v6 = v3;
  if ( *(_BYTE *)v3 == 15 )
  {
    v8 = *(_QWORD *)(a2 + 80);
    if ( *(_BYTE *)(v3 + 56) )
    {
      v8 = v3;
      goto LABEL_5;
    }
LABEL_8:
    v39 = 0;
    goto LABEL_9;
  }
  v7 = *(unsigned int *)(v3 + 8);
  v8 = *(_QWORD *)(v3 - 8 * v7);
  if ( !v8 )
    goto LABEL_8;
  if ( *(_BYTE *)(v8 + 56) )
  {
LABEL_5:
    v9.m128i_i64[0] = sub_161E970(*(_QWORD *)(v8 + 48));
    v10 = *(_BYTE *)v3 == 15;
    v41[0] = v9;
    if ( v10 )
    {
      v39 = 1;
      v8 = v3;
      goto LABEL_9;
    }
    v39 = 1;
    v7 = *(unsigned int *)(v3 + 8);
    goto LABEL_46;
  }
  v39 = 0;
LABEL_46:
  v8 = *(_QWORD *)(v3 - 8 * v7);
LABEL_9:
  v40 = sub_39A3100(a2, v8);
  v11 = *(unsigned int *)(v3 + 8);
  if ( *(_BYTE *)v3 == 15 )
  {
    v13 = *(char **)(v3 - 8 * v11);
    if ( !v13 )
    {
      v38 = 0;
      v16 = v3;
      goto LABEL_13;
    }
  }
  else
  {
    v12 = *(_QWORD *)(v3 - 8 * v11);
    if ( !v12 )
    {
      v38 = 0;
      v13 = (char *)byte_3F871B3;
      goto LABEL_36;
    }
    v13 = *(char **)(v12 - 8LL * *(unsigned int *)(v12 + 8));
    if ( !v13 )
    {
      v38 = 0;
LABEL_48:
      v36 = -v11;
      v6 = *(_QWORD *)(v3 + 8 * v36);
      if ( v6 )
      {
        v16 = *(_QWORD *)(v3 + 8 * v36);
        goto LABEL_13;
      }
LABEL_36:
      v21 = byte_3F871B3;
      v20 = (char *)byte_3F871B3;
      if ( a1[333].m128i_i64[1] )
        return v2;
      goto LABEL_37;
    }
  }
  v14 = sub_161E970((__int64)v13);
  v38 = v15;
  v13 = (char *)v14;
  v16 = v3;
  if ( *(_BYTE *)v3 != 15 )
  {
    v11 = *(unsigned int *)(v3 + 8);
    goto LABEL_48;
  }
LABEL_13:
  v17 = *(_QWORD *)(v16 + 8 * (1LL - *(unsigned int *)(v6 + 8)));
  if ( !v17 )
  {
    if ( a1[333].m128i_i64[1] )
      return v2;
LABEL_16:
    LOBYTE(src[0]) = 0;
    v22 = (__m128i *)a1[331].m128i_i64[0];
    v23 = 0;
    v42 = (__m128i *)src;
LABEL_17:
    a1[331].m128i_i64[1] = v23;
    v22->m128i_i8[v23] = 0;
    v24 = v42;
    goto LABEL_18;
  }
  v18 = sub_161E970(v17);
  v20 = (char *)v18;
  if ( a1[333].m128i_i64[1] )
    return v2;
  v21 = (const char *)(v18 + v19);
  if ( !v18 )
    goto LABEL_16;
LABEL_37:
  v42 = (__m128i *)src;
  sub_3984920((__int64 *)&v42, v20, (__int64)v21);
  v22 = (__m128i *)a1[331].m128i_i64[0];
  v24 = v22;
  if ( v42 == (__m128i *)src )
  {
    v23 = n;
    if ( n )
    {
      if ( n == 1 )
        v22->m128i_i8[0] = src[0];
      else
        memcpy(v22, src, n);
      v23 = n;
      v22 = (__m128i *)a1[331].m128i_i64[0];
    }
    goto LABEL_17;
  }
  v32 = n;
  v33 = src[0];
  if ( v24 == &a1[332] )
  {
    a1[331].m128i_i64[0] = (__int64)v42;
    a1[331].m128i_i64[1] = v32;
    a1[332].m128i_i64[0] = v33;
  }
  else
  {
    v34 = a1[332].m128i_i64[0];
    a1[331].m128i_i64[0] = (__int64)v42;
    a1[331].m128i_i64[1] = v32;
    a1[332].m128i_i64[0] = v33;
    if ( v24 )
    {
      v42 = v24;
      src[0] = v34;
      goto LABEL_18;
    }
  }
  v42 = (__m128i *)src;
  v24 = (__m128i *)src;
LABEL_18:
  n = 0;
  v24->m128i_i8[0] = 0;
  if ( v42 != (__m128i *)src )
    j_j___libc_free_0((unsigned __int64)v42);
  if ( !v13 )
  {
    v42 = (__m128i *)src;
    v25 = (__m128i *)a1[333].m128i_i64[0];
    v35 = 0;
    LOBYTE(src[0]) = 0;
LABEL_44:
    a1[333].m128i_i64[1] = v35;
    v25->m128i_i8[v35] = 0;
    v26 = v42;
    goto LABEL_25;
  }
  v42 = (__m128i *)src;
  sub_3984920((__int64 *)&v42, v13, (__int64)&v13[v38]);
  v25 = (__m128i *)a1[333].m128i_i64[0];
  v26 = v25;
  if ( v42 == (__m128i *)src )
  {
    v35 = n;
    if ( n )
    {
      if ( n == 1 )
        v25->m128i_i8[0] = src[0];
      else
        memcpy(v25, src, n);
      v35 = n;
      v25 = (__m128i *)a1[333].m128i_i64[0];
    }
    goto LABEL_44;
  }
  v27 = src[0];
  v28 = n;
  if ( v26 == &a1[334] )
  {
    a1[333].m128i_i64[0] = (__int64)v42;
    a1[333].m128i_i64[1] = v28;
    a1[334].m128i_i64[0] = v27;
    goto LABEL_52;
  }
  v29 = a1[334].m128i_i64[0];
  a1[333].m128i_i64[0] = (__int64)v42;
  a1[333].m128i_i64[1] = v28;
  a1[334].m128i_i64[0] = v27;
  if ( !v26 )
  {
LABEL_52:
    v42 = (__m128i *)src;
    v26 = (__m128i *)src;
    goto LABEL_25;
  }
  v42 = v26;
  src[0] = v29;
LABEL_25:
  n = 0;
  v26->m128i_i8[0] = 0;
  if ( v42 != (__m128i *)src )
    j_j___libc_free_0((unsigned __int64)v42);
  a1[335].m128i_i32[0] = 0;
  a1[335].m128i_i64[1] = v40;
  v30 = a1[337].m128i_i8[0];
  if ( v39 )
  {
    if ( v30 )
    {
      a1[336] = _mm_loadu_si128(v41);
    }
    else
    {
      v37 = _mm_loadu_si128(v41);
      a1[337].m128i_i8[0] = 1;
      a1[336] = v37;
    }
  }
  else if ( v30 )
  {
    a1[337].m128i_i8[0] = 0;
  }
  a1[337].m128i_i8[9] &= v40 != 0;
  a1[337].m128i_i8[10] |= v40 != 0;
  a1[337].m128i_i8[8] = v39;
  return v2;
}
