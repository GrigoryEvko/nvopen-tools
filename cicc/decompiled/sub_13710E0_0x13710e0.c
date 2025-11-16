// Function: sub_13710E0
// Address: 0x13710e0
//
bool __fastcall sub_13710E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int *a4,
        unsigned int *a5,
        unsigned __int64 a6)
{
  unsigned __int64 v7; // r12
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 *v11; // rax
  __int64 *v12; // rdx
  unsigned int v13; // ecx
  __int64 v14; // rax
  _DWORD *v15; // rdi
  bool result; // al
  __int64 v17; // rdx
  __int64 *v18; // r15
  __int64 v19; // rax
  _DWORD *v20; // rdi
  __int64 v21; // rax
  _DWORD *v22; // rdi
  bool v23; // al
  __int64 v24; // rax
  bool v25; // al
  unsigned int v26; // [rsp+0h] [rbp-50h]
  __int64 v27; // [rsp+0h] [rbp-50h]
  unsigned int v28; // [rsp+8h] [rbp-48h]
  unsigned int v29; // [rsp+8h] [rbp-48h]
  unsigned int v30[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v7 = 1;
  v9 = *(_QWORD *)(a1 + 64);
  if ( a6 )
    v7 = a6;
  v10 = v9 + 24LL * *a5;
  v11 = *(__int64 **)(v10 + 8);
  if ( v11 && *((_BYTE *)v11 + 8) )
  {
    do
    {
      v12 = v11;
      v11 = (__int64 *)*v11;
    }
    while ( v11 && *((_BYTE *)v11 + 8) );
    v13 = *(_DWORD *)v12[12];
  }
  else
  {
    v13 = *(_DWORD *)v10;
  }
  v30[0] = v13;
  if ( a3 )
  {
    v14 = *(unsigned int *)(a3 + 12);
    v15 = *(_DWORD **)(a3 + 96);
    if ( (unsigned int)v14 > 1 )
    {
      v26 = v13;
      if ( sub_1369030(v15, &v15[v14], v30) )
        goto LABEL_12;
      v13 = v26;
    }
    else if ( v13 == *v15 )
    {
LABEL_12:
      sub_1370BE0(a2, v30, v7, 2u);
      return 1;
    }
    v17 = v9 + 24LL * v13;
    v18 = *(__int64 **)(v17 + 8);
    if ( !v18 )
    {
LABEL_17:
      sub_1370BE0(a2, v30, v7, 1u);
      return 1;
    }
  }
  else
  {
    v17 = v9 + 24LL * v13;
    v18 = *(__int64 **)(v17 + 8);
    if ( !v18 )
    {
      result = 0;
      if ( v13 >= *a4 )
        goto LABEL_26;
      return result;
    }
  }
  v19 = *((unsigned int *)v18 + 3);
  v20 = (_DWORD *)v18[12];
  if ( (unsigned int)v19 > 1 )
  {
    v28 = v13;
    v27 = v17;
    v23 = sub_1369030(v20, &v20[v19], (_DWORD *)v17);
    v13 = v28;
    if ( !v23 )
      goto LABEL_21;
    v17 = v27;
  }
  else if ( *(_DWORD *)v17 != *v20 )
  {
    goto LABEL_21;
  }
  v18 = (__int64 *)*v18;
  if ( !v18
    || (v24 = *((unsigned int *)v18 + 3), (unsigned int)v24 <= 1)
    || (v29 = v13, v25 = sub_1369030((_DWORD *)v18[12], (_DWORD *)(v18[12] + 4 * v24), (_DWORD *)v17), v13 = v29, !v25) )
  {
LABEL_21:
    if ( v18 != (__int64 *)a3 )
      goto LABEL_17;
    goto LABEL_22;
  }
  v18 = (__int64 *)*v18;
  if ( v18 != (__int64 *)a3 )
    goto LABEL_17;
LABEL_22:
  if ( v13 >= *a4 )
  {
LABEL_26:
    sub_1370BE0(a2, v30, v7, 0);
    return 1;
  }
  if ( !v18 )
    return 0;
  v21 = *((unsigned int *)v18 + 3);
  v22 = (_DWORD *)v18[12];
  if ( (unsigned int)v21 > 1 )
  {
    result = sub_1369030(v22, &v22[v21], a4);
    if ( result )
      goto LABEL_26;
  }
  else
  {
    result = 0;
    if ( *a4 == *v22 )
      goto LABEL_26;
  }
  return result;
}
