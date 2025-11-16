// Function: sub_26C4C60
// Address: 0x26c4c60
//
_QWORD *__fastcall sub_26C4C60(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r12
  __int64 v3; // rax
  size_t v4; // r14
  const void *v5; // r15
  size_t v6; // rdx
  int v7; // eax
  _QWORD *v8; // rax
  char v9; // dl
  __int64 v10; // r13
  size_t v11; // rbx
  const void *v12; // rsi
  const void *v13; // rdi
  const void *v14; // rsi
  size_t v15; // rdx
  int v16; // eax
  __int64 v18; // rax
  __int64 v19; // r13
  const void **v20; // [rsp+10h] [rbp-40h]
  __int64 v21; // [rsp+10h] [rbp-40h]

  v2 = *(_QWORD **)(a1 + 16);
  if ( !v2 )
  {
    v2 = (_QWORD *)(a1 + 8);
    goto LABEL_28;
  }
  v3 = *(_QWORD *)(a2 + 8);
  v4 = *(_QWORD *)(v3 + 8);
  v5 = *(const void **)v3;
  v20 = (const void **)v3;
  while ( 1 )
  {
    v10 = v2[5];
    v11 = *(_QWORD *)(v10 + 8);
    v12 = *(const void **)v10;
    if ( v11 >= v4 )
    {
      if ( v12 != v5 )
      {
        v6 = v4;
        goto LABEL_5;
      }
LABEL_8:
      if ( v11 == v4 )
        goto LABEL_10;
      goto LABEL_9;
    }
    if ( v12 != v5 )
      break;
LABEL_9:
    if ( v11 > v4 )
      goto LABEL_16;
LABEL_10:
    v8 = (_QWORD *)v2[3];
    v9 = 0;
    if ( !v8 )
      goto LABEL_17;
LABEL_11:
    v2 = v8;
  }
  v6 = *(_QWORD *)(v10 + 8);
LABEL_5:
  if ( !v5 )
    goto LABEL_16;
  if ( !v12 )
    goto LABEL_10;
  v7 = memcmp(v5, v12, v6);
  if ( !v7 )
    goto LABEL_8;
  if ( v7 >= 0 )
    goto LABEL_10;
LABEL_16:
  v8 = (_QWORD *)v2[2];
  v9 = 1;
  if ( v8 )
    goto LABEL_11;
LABEL_17:
  if ( !v9 )
  {
    v13 = *(const void **)v10;
    v14 = *v20;
    if ( v4 >= v11 )
      goto LABEL_19;
LABEL_30:
    if ( v14 != v13 )
    {
      v15 = v4;
LABEL_21:
      if ( !v13 )
        return 0;
      if ( v14 )
      {
        v16 = memcmp(v13, v14, v15);
        if ( !v16 )
          goto LABEL_24;
        if ( v16 < 0 )
          return 0;
      }
      return v2;
    }
LABEL_25:
    if ( v4 > v11 )
      return 0;
    return v2;
  }
LABEL_28:
  if ( v2 != *(_QWORD **)(a1 + 24) )
  {
    v18 = sub_220EF80((__int64)v2);
    v19 = *(_QWORD *)(v18 + 40);
    v2 = (_QWORD *)v18;
    v11 = *(_QWORD *)(v19 + 8);
    v13 = *(const void **)v19;
    v21 = *(_QWORD *)(a2 + 8);
    v4 = *(_QWORD *)(v21 + 8);
    v14 = *(const void **)v21;
    if ( v4 < v11 )
      goto LABEL_30;
LABEL_19:
    if ( v14 != v13 )
    {
      v15 = v11;
      goto LABEL_21;
    }
LABEL_24:
    if ( v4 == v11 )
      return v2;
    goto LABEL_25;
  }
  return 0;
}
