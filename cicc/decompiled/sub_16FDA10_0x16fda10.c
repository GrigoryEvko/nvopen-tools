// Function: sub_16FDA10
// Address: 0x16fda10
//
_QWORD *__fastcall sub_16FDA10(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rbx
  const void *v3; // r15
  size_t v4; // r13
  _QWORD *v5; // r14
  int v6; // eax
  size_t v7; // r12
  const void *v8; // rdi
  size_t v9; // rbx
  const void *v10; // rsi
  int v11; // eax
  _QWORD *v13; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD **)(a1 + 16);
  v13 = (_QWORD *)(a1 + 8);
  if ( !v2 )
    return v13;
  v3 = *(const void **)a2;
  v4 = *(_QWORD *)(a2 + 8);
  v5 = (_QWORD *)(a1 + 8);
  do
  {
    while ( 1 )
    {
      v7 = v2[5];
      v8 = (const void *)v2[4];
      if ( v7 > v4 )
        break;
      if ( v7 )
      {
        v6 = memcmp(v8, v3, v2[5]);
        if ( v6 )
          goto LABEL_11;
      }
      if ( v7 == v4 )
        goto LABEL_12;
LABEL_6:
      if ( v7 >= v4 )
        goto LABEL_12;
LABEL_7:
      v2 = (_QWORD *)v2[3];
      if ( !v2 )
        goto LABEL_13;
    }
    if ( !v4 )
      goto LABEL_12;
    v6 = memcmp(v8, v3, v4);
    if ( !v6 )
      goto LABEL_6;
LABEL_11:
    if ( v6 < 0 )
      goto LABEL_7;
LABEL_12:
    v5 = v2;
    v2 = (_QWORD *)v2[2];
  }
  while ( v2 );
LABEL_13:
  if ( v13 == v5 )
    return v13;
  v9 = v5[5];
  v10 = (const void *)v5[4];
  if ( v4 > v9 )
  {
    if ( !v9 )
      return v5;
    v11 = memcmp(v3, v10, v5[5]);
    if ( v11 )
    {
LABEL_22:
      if ( v11 >= 0 )
        return v5;
      return v13;
    }
LABEL_18:
    if ( v4 >= v9 )
      return v5;
    return v13;
  }
  if ( v4 )
  {
    v11 = memcmp(v3, v10, v4);
    if ( v11 )
      goto LABEL_22;
  }
  if ( v4 != v9 )
    goto LABEL_18;
  return v5;
}
