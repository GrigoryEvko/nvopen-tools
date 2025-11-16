// Function: sub_C27720
// Address: 0xc27720
//
_QWORD *__fastcall sub_C27720(_QWORD *a1, const __m128i *a2)
{
  _QWORD *v2; // r12
  _QWORD *v3; // rbx
  size_t v4; // r15
  const void *v5; // r14
  size_t v6; // rdx
  int v7; // eax
  size_t v8; // r13
  const void *v9; // rdi
  size_t v10; // rbx
  const void *v11; // rsi
  size_t v12; // rdx
  int v13; // eax
  _QWORD *v17; // [rsp+18h] [rbp-48h]
  const __m128i *v18; // [rsp+28h] [rbp-38h] BYREF

  v2 = a1 + 1;
  v3 = (_QWORD *)a1[2];
  v17 = a1 + 1;
  if ( !v3 )
  {
    v2 = a1 + 1;
    goto LABEL_25;
  }
  v4 = a2->m128i_u64[1];
  v5 = (const void *)a2->m128i_i64[0];
  do
  {
    while ( 1 )
    {
      v8 = v3[5];
      v9 = (const void *)v3[4];
      if ( v4 < v8 )
        break;
      if ( v9 == v5 )
        goto LABEL_8;
      v6 = v3[5];
LABEL_5:
      if ( !v9 )
        goto LABEL_15;
      if ( !v5 )
        goto LABEL_10;
      v7 = memcmp(v9, v5, v6);
      if ( v7 )
      {
        if ( v7 >= 0 )
          goto LABEL_10;
        goto LABEL_15;
      }
LABEL_8:
      if ( v4 != v8 )
        goto LABEL_9;
LABEL_10:
      v2 = v3;
      v3 = (_QWORD *)v3[2];
      if ( !v3 )
        goto LABEL_16;
    }
    if ( v9 != v5 )
    {
      v6 = v4;
      goto LABEL_5;
    }
LABEL_9:
    if ( v4 <= v8 )
      goto LABEL_10;
LABEL_15:
    v3 = (_QWORD *)v3[3];
  }
  while ( v3 );
LABEL_16:
  if ( v17 == v2 )
    goto LABEL_25;
  v10 = v2[5];
  v11 = (const void *)v2[4];
  if ( v4 > v10 )
  {
    if ( v11 == v5 )
      goto LABEL_24;
    v12 = v2[5];
LABEL_20:
    if ( v5 )
    {
      if ( !v11 )
        return v2 + 6;
      v13 = memcmp(v5, v11, v12);
      if ( !v13 )
        goto LABEL_23;
      if ( v13 >= 0 )
        return v2 + 6;
    }
LABEL_25:
    v18 = a2;
    v2 = (_QWORD *)sub_C275A0(a1, v2, &v18);
    return v2 + 6;
  }
  if ( v11 != v5 )
  {
    v12 = v4;
    goto LABEL_20;
  }
LABEL_23:
  if ( v4 != v10 )
  {
LABEL_24:
    if ( v4 < v10 )
      goto LABEL_25;
  }
  return v2 + 6;
}
