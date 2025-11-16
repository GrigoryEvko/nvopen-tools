// Function: sub_1852CC0
// Address: 0x1852cc0
//
__m128i *__fastcall sub_1852CC0(_QWORD *a1, const __m128i *a2)
{
  _QWORD *v2; // rbx
  const void *v3; // r12
  size_t v4; // r13
  int v5; // eax
  _QWORD *v6; // rax
  char v7; // dl
  size_t v8; // r15
  const void *v9; // r14
  _QWORD *v10; // r8
  int v11; // eax
  _BOOL4 v12; // r15d
  __m128i *v13; // rbx
  __int64 v15; // rax
  unsigned __int64 v16; // r12
  const void *v17; // rsi
  unsigned __int64 v18; // rbx
  const void *v19; // rdi
  int v20; // eax
  unsigned int v21; // r15d
  int v22; // eax
  _QWORD *v23; // [rsp+0h] [rbp-50h]
  _QWORD *v24; // [rsp+0h] [rbp-50h]
  _QWORD *v25; // [rsp+0h] [rbp-50h]
  _QWORD *v26; // [rsp+0h] [rbp-50h]
  _QWORD *v27; // [rsp+0h] [rbp-50h]
  _QWORD *v30; // [rsp+18h] [rbp-38h]

  v2 = (_QWORD *)a1[2];
  v30 = a1 + 1;
  if ( !v2 )
  {
    v2 = a1 + 1;
    goto LABEL_24;
  }
  v3 = (const void *)a2->m128i_i64[0];
  v4 = a2->m128i_u64[1];
  while ( 1 )
  {
    v8 = v2[5];
    v9 = (const void *)v2[4];
    if ( v4 > v8 )
      break;
    if ( v4 )
    {
      v5 = memcmp(v3, (const void *)v2[4], v4);
      if ( v5 )
        goto LABEL_12;
    }
    if ( v4 == v8 )
      goto LABEL_13;
LABEL_6:
    if ( v4 >= v8 )
      goto LABEL_13;
LABEL_7:
    v6 = (_QWORD *)v2[2];
    v7 = 1;
    if ( !v6 )
      goto LABEL_14;
LABEL_8:
    v2 = v6;
  }
  if ( !v8 )
    goto LABEL_13;
  v5 = memcmp(v3, (const void *)v2[4], v2[5]);
  if ( !v5 )
    goto LABEL_6;
LABEL_12:
  if ( v5 < 0 )
    goto LABEL_7;
LABEL_13:
  v6 = (_QWORD *)v2[3];
  v7 = 0;
  if ( v6 )
    goto LABEL_8;
LABEL_14:
  v10 = v2;
  if ( v7 )
  {
LABEL_24:
    if ( v2 == (_QWORD *)a1[3] )
    {
      v10 = v2;
      v12 = 1;
      if ( v30 == v2 )
      {
LABEL_22:
        v24 = v10;
        v13 = (__m128i *)sub_22077B0(48);
        v13[2] = _mm_loadu_si128(a2);
        sub_220F040(v12, v13, v24, v30);
        ++a1[5];
        return v13;
      }
LABEL_31:
      v16 = v10[5];
      v17 = (const void *)v10[4];
      v18 = a2->m128i_u64[1];
      v19 = (const void *)a2->m128i_i64[0];
      if ( v16 < v18 )
      {
        v12 = 0;
        if ( !v16 )
          goto LABEL_22;
        v27 = v10;
        v22 = memcmp(v19, v17, v10[5]);
        v10 = v27;
        v21 = v22;
        if ( v22 )
          goto LABEL_34;
      }
      else
      {
        if ( v18 )
        {
          v26 = v10;
          v20 = memcmp(v19, v17, a2->m128i_u64[1]);
          v10 = v26;
          v21 = v20;
          if ( v20 )
          {
LABEL_34:
            v12 = v21 >> 31;
            goto LABEL_22;
          }
        }
        v12 = 0;
        if ( v16 == v18 )
          goto LABEL_22;
      }
      v12 = v16 > v18;
      goto LABEL_22;
    }
    v15 = sub_220EF80(v2);
    v10 = v2;
    v8 = *(_QWORD *)(v15 + 40);
    v9 = *(const void **)(v15 + 32);
    v2 = (_QWORD *)v15;
    v4 = a2->m128i_u64[1];
    v3 = (const void *)a2->m128i_i64[0];
    if ( v8 <= v4 )
      goto LABEL_16;
LABEL_26:
    if ( !v4 )
      return (__m128i *)v2;
    v25 = v10;
    v11 = memcmp(v9, v3, v4);
    v10 = v25;
    if ( !v11 )
    {
LABEL_19:
      if ( v8 >= v4 )
        return (__m128i *)v2;
      goto LABEL_20;
    }
LABEL_28:
    if ( v11 >= 0 )
      return (__m128i *)v2;
LABEL_20:
    if ( !v10 )
      return 0;
    v12 = 1;
    if ( v30 == v10 )
      goto LABEL_22;
    goto LABEL_31;
  }
  if ( v8 > v4 )
    goto LABEL_26;
LABEL_16:
  if ( v8 )
  {
    v23 = v10;
    v11 = memcmp(v9, v3, v8);
    v10 = v23;
    if ( v11 )
      goto LABEL_28;
  }
  if ( v8 != v4 )
    goto LABEL_19;
  return (__m128i *)v2;
}
