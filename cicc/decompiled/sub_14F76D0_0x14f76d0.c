// Function: sub_14F76D0
// Address: 0x14f76d0
//
_QWORD *__fastcall sub_14F76D0(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  _QWORD *v6; // rdi
  char *v7; // r10
  char *v8; // r8
  char *v9; // rsi
  char *v10; // rdx
  signed __int64 v11; // r9
  _QWORD *v12; // rcx
  _QWORD *v13; // rax
  _QWORD *v14; // rax
  char v15; // dl
  __int64 v17; // rax

  v4 = *(_QWORD **)(a1 + 16);
  if ( !v4 )
  {
    v4 = (_QWORD *)(a1 + 8);
    goto LABEL_24;
  }
  v5 = *(_QWORD **)(a2 + 8);
  v6 = *(_QWORD **)a2;
  v7 = (char *)v5 - *(_QWORD *)a2;
  while ( 1 )
  {
    v8 = (char *)v4[5];
    v9 = (char *)v4[4];
    v10 = v9;
    v11 = v8 - v9;
    v12 = (_QWORD *)((char *)v6 + v8 - v9);
    if ( (__int64)v7 <= v8 - v9 )
      v12 = v5;
    if ( v6 == v12 )
      break;
    v13 = v6;
    while ( *v13 >= *(_QWORD *)v10 )
    {
      if ( *v13 > *(_QWORD *)v10 )
        goto LABEL_13;
      ++v13;
      v10 += 8;
      if ( v12 == v13 )
        goto LABEL_12;
    }
LABEL_10:
    v14 = (_QWORD *)v4[2];
    v15 = 1;
    if ( !v14 )
      goto LABEL_14;
LABEL_11:
    v4 = v14;
  }
LABEL_12:
  if ( v8 != v10 )
    goto LABEL_10;
LABEL_13:
  v14 = (_QWORD *)v4[3];
  v15 = 0;
  if ( v14 )
    goto LABEL_11;
LABEL_14:
  if ( !v15 )
    goto LABEL_15;
LABEL_24:
  if ( *(_QWORD **)(a1 + 24) == v4 )
    return 0;
  v17 = sub_220EF80(v4);
  v5 = *(_QWORD **)(a2 + 8);
  v6 = *(_QWORD **)a2;
  v8 = *(char **)(v17 + 40);
  v9 = *(char **)(v17 + 32);
  v4 = (_QWORD *)v17;
  v7 = (char *)v5 - *(_QWORD *)a2;
  v11 = v8 - v9;
LABEL_15:
  if ( (__int64)v7 < v11 )
    v8 = &v7[(_QWORD)v9];
  if ( v8 != v9 )
  {
    while ( *(_QWORD *)v9 >= *v6 )
    {
      if ( *(_QWORD *)v9 > *v6 )
        return v4;
      v9 += 8;
      ++v6;
      if ( v8 == v9 )
        goto LABEL_26;
    }
    return 0;
  }
LABEL_26:
  if ( v5 != v6 )
    return 0;
  return v4;
}
