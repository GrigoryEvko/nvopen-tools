// Function: sub_C1C800
// Address: 0xc1c800
//
_QWORD *__fastcall sub_C1C800(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  size_t v4; // r15
  const void *v5; // r14
  size_t v6; // rdx
  int v7; // eax
  _QWORD *v8; // rax
  char v9; // dl
  size_t v10; // r12
  const void *v11; // rsi
  const void *v12; // rdi
  size_t v13; // rdx
  int v14; // eax
  __int64 v16; // rax

  v3 = *(_QWORD **)(a1 + 16);
  if ( !v3 )
  {
    v3 = (_QWORD *)(a1 + 8);
    goto LABEL_28;
  }
  v4 = *(_QWORD *)(a2 + 8);
  v5 = *(const void **)a2;
  while ( 1 )
  {
    v10 = v3[5];
    v11 = (const void *)v3[4];
    if ( v10 >= v4 )
    {
      if ( v5 != v11 )
      {
        v6 = v4;
        goto LABEL_5;
      }
LABEL_8:
      if ( v10 == v4 )
        goto LABEL_10;
      goto LABEL_9;
    }
    if ( v5 != v11 )
      break;
LABEL_9:
    if ( v10 > v4 )
      goto LABEL_16;
LABEL_10:
    v8 = (_QWORD *)v3[3];
    v9 = 0;
    if ( !v8 )
      goto LABEL_17;
LABEL_11:
    v3 = v8;
  }
  v6 = v3[5];
LABEL_5:
  if ( !v5 )
    goto LABEL_16;
  if ( !v11 )
    goto LABEL_10;
  v7 = memcmp(v5, v11, v6);
  if ( !v7 )
    goto LABEL_8;
  if ( v7 >= 0 )
    goto LABEL_10;
LABEL_16:
  v8 = (_QWORD *)v3[2];
  v9 = 1;
  if ( v8 )
    goto LABEL_11;
LABEL_17:
  if ( !v9 )
  {
    v12 = (const void *)v3[4];
    if ( v10 <= v4 )
      goto LABEL_19;
LABEL_30:
    if ( v12 != v5 )
    {
      v13 = v4;
LABEL_21:
      if ( !v12 )
        return 0;
      if ( v5 )
      {
        v14 = memcmp(v12, v5, v13);
        if ( !v14 )
          goto LABEL_24;
        if ( v14 < 0 )
          return 0;
      }
      return v3;
    }
LABEL_25:
    if ( v10 < v4 )
      return 0;
    return v3;
  }
LABEL_28:
  if ( v3 != *(_QWORD **)(a1 + 24) )
  {
    v16 = sub_220EF80(v3);
    v4 = *(_QWORD *)(a2 + 8);
    v5 = *(const void **)a2;
    v10 = *(_QWORD *)(v16 + 40);
    v3 = (_QWORD *)v16;
    v12 = *(const void **)(v16 + 32);
    if ( v10 > v4 )
      goto LABEL_30;
LABEL_19:
    if ( v12 != v5 )
    {
      v13 = v10;
      goto LABEL_21;
    }
LABEL_24:
    if ( v10 == v4 )
      return v3;
    goto LABEL_25;
  }
  return 0;
}
