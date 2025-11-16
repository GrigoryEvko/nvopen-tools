// Function: sub_C1BAB0
// Address: 0xc1bab0
//
_QWORD *__fastcall sub_C1BAB0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r13
  _QWORD *v3; // rbx
  size_t v4; // r14
  const void *v5; // r15
  size_t v6; // rdx
  int v7; // eax
  size_t v8; // r12
  const void *v9; // rdi
  size_t v10; // rbx
  const void *v11; // rsi
  size_t v12; // rdx
  int v13; // eax
  _QWORD *v15; // [rsp+8h] [rbp-38h]

  v2 = (_QWORD *)(a1 + 8);
  v3 = *(_QWORD **)(a1 + 16);
  v15 = (_QWORD *)(a1 + 8);
  if ( !v3 )
    return v15;
  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(const void **)a2;
  do
  {
    while ( 1 )
    {
      v8 = v3[5];
      v9 = (const void *)v3[4];
      if ( v4 < v8 )
        break;
      if ( v5 == v9 )
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
    if ( v5 != v9 )
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
  if ( v15 == v2 )
    return v2;
  v10 = v2[5];
  v11 = (const void *)v2[4];
  if ( v4 > v10 )
  {
    if ( v5 == v11 )
      goto LABEL_24;
    v12 = v2[5];
LABEL_20:
    if ( v5 )
    {
      if ( v11 )
      {
        v13 = memcmp(v5, v11, v12);
        if ( !v13 )
          goto LABEL_23;
        if ( v13 < 0 )
          return v15;
      }
      return v2;
    }
    return v15;
  }
  if ( v5 != v11 )
  {
    v12 = v4;
    goto LABEL_20;
  }
LABEL_23:
  if ( v4 != v10 )
  {
LABEL_24:
    if ( v4 < v10 )
      return v15;
  }
  return v2;
}
