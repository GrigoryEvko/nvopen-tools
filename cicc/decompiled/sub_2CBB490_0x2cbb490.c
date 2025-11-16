// Function: sub_2CBB490
// Address: 0x2cbb490
//
_QWORD *__fastcall sub_2CBB490(__int64 a1, __int64 *a2)
{
  _QWORD *v2; // r14
  _QWORD *v3; // rax
  char v4; // dl
  __int64 v5; // r12
  const char *v6; // rax
  size_t v7; // rdx
  size_t v8; // rbx
  const char *v9; // r15
  const char *v10; // rax
  size_t v11; // rdx
  size_t v12; // r12
  bool v13; // cc
  size_t v14; // rdx
  int v15; // eax
  __int64 v16; // r15
  const char *v17; // rax
  size_t v18; // rdx
  size_t v19; // r13
  const char *v20; // r12
  const char *v21; // rax
  size_t v22; // rdx
  size_t v23; // rbx
  int v24; // eax

  v2 = *(_QWORD **)(a1 + 16);
  if ( !v2 )
  {
    v2 = (_QWORD *)(a1 + 8);
    goto LABEL_14;
  }
  while ( 1 )
  {
    v5 = *a2;
    v6 = sub_BD5D20(v2[4]);
    v8 = v7;
    v9 = v6;
    v10 = sub_BD5D20(v5);
    v12 = v11;
    v13 = v11 <= v8;
    v14 = v8;
    if ( v13 )
      v14 = v12;
    if ( v14 )
    {
      v15 = memcmp(v10, v9, v14);
      if ( v15 )
        break;
    }
    if ( v12 == v8 || v12 >= v8 )
    {
      v3 = (_QWORD *)v2[3];
      v4 = 0;
      goto LABEL_12;
    }
LABEL_3:
    v3 = (_QWORD *)v2[2];
    v4 = 1;
    if ( !v3 )
      goto LABEL_13;
LABEL_4:
    v2 = v3;
  }
  if ( v15 < 0 )
    goto LABEL_3;
  v3 = (_QWORD *)v2[3];
  v4 = 0;
LABEL_12:
  if ( v3 )
    goto LABEL_4;
LABEL_13:
  if ( !v4 )
  {
LABEL_16:
    v16 = v2[4];
    v17 = sub_BD5D20(*a2);
    v19 = v18;
    v20 = v17;
    v21 = sub_BD5D20(v16);
    v23 = v22;
    if ( v19 <= v22 )
      v22 = v19;
    if ( v22 && (v24 = memcmp(v21, v20, v22)) != 0 )
    {
      if ( v24 < 0 )
        return 0;
    }
    else if ( v19 != v23 && v19 > v23 )
    {
      return 0;
    }
    return v2;
  }
LABEL_14:
  if ( *(_QWORD **)(a1 + 24) != v2 )
  {
    v2 = (_QWORD *)sub_220EF80((__int64)v2);
    goto LABEL_16;
  }
  return 0;
}
