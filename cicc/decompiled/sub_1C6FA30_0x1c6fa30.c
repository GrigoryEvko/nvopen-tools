// Function: sub_1C6FA30
// Address: 0x1c6fa30
//
_QWORD *__fastcall sub_1C6FA30(__int64 a1, __int64 *a2)
{
  _QWORD *v2; // r14
  int v3; // eax
  _QWORD *v4; // rax
  char v5; // dl
  __int64 v6; // rbx
  const char *v7; // r15
  size_t v8; // rdx
  size_t v9; // r12
  size_t v10; // rdx
  const char *v11; // rdi
  size_t v12; // rbx
  __int64 v13; // r15
  const char *v14; // r12
  size_t v15; // rdx
  size_t v16; // r13
  size_t v17; // rdx
  const char *v18; // rdi
  size_t v19; // rbx
  int v20; // eax

  v2 = *(_QWORD **)(a1 + 16);
  if ( !v2 )
  {
    v2 = (_QWORD *)(a1 + 8);
    goto LABEL_15;
  }
  while ( 1 )
  {
    v6 = *a2;
    v7 = sub_1649960(v2[4]);
    v9 = v8;
    v11 = sub_1649960(v6);
    v12 = v10;
    if ( v9 < v10 )
      break;
    if ( v10 )
    {
      v3 = memcmp(v11, v7, v10);
      if ( v3 )
        goto LABEL_12;
    }
    if ( v9 == v12 )
      goto LABEL_13;
LABEL_6:
    if ( v9 <= v12 )
      goto LABEL_13;
LABEL_7:
    v4 = (_QWORD *)v2[2];
    v5 = 1;
    if ( !v4 )
      goto LABEL_14;
LABEL_8:
    v2 = v4;
  }
  if ( !v9 )
    goto LABEL_13;
  v3 = memcmp(v11, v7, v9);
  if ( !v3 )
    goto LABEL_6;
LABEL_12:
  if ( v3 < 0 )
    goto LABEL_7;
LABEL_13:
  v4 = (_QWORD *)v2[3];
  v5 = 0;
  if ( v4 )
    goto LABEL_8;
LABEL_14:
  if ( !v5 )
  {
LABEL_17:
    v13 = v2[4];
    v14 = sub_1649960(*a2);
    v16 = v15;
    v18 = sub_1649960(v13);
    v19 = v17;
    if ( v16 < v17 )
    {
      if ( !v16 )
        return v2;
      v20 = memcmp(v18, v14, v16);
      if ( !v20 )
      {
LABEL_21:
        if ( v16 > v19 )
          return 0;
        return v2;
      }
    }
    else if ( !v17 || (v20 = memcmp(v18, v14, v17)) == 0 )
    {
      if ( v16 == v19 )
        return v2;
      goto LABEL_21;
    }
    if ( v20 < 0 )
      return 0;
    return v2;
  }
LABEL_15:
  if ( *(_QWORD **)(a1 + 24) != v2 )
  {
    v2 = (_QWORD *)sub_220EF80(v2);
    goto LABEL_17;
  }
  return 0;
}
