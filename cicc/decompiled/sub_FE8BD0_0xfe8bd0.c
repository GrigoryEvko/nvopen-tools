// Function: sub_FE8BD0
// Address: 0xfe8bd0
//
bool __fastcall sub_FE8BD0(__int64 a1, __int64 a2, __int64 a3, unsigned int *a4, unsigned int *a5, __int64 a6)
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
  bool v17; // al
  __int64 v18; // rdx
  __int64 *v19; // r15
  __int64 v20; // rax
  _DWORD *v21; // rdi
  __int64 v22; // rax
  _DWORD *v23; // rdi
  bool v24; // al
  __int64 v25; // rax
  bool v26; // al
  unsigned int v27; // [rsp+0h] [rbp-50h]
  __int64 v28; // [rsp+0h] [rbp-50h]
  unsigned int v29; // [rsp+8h] [rbp-48h]
  unsigned int v30; // [rsp+8h] [rbp-48h]
  unsigned int v31[13]; // [rsp+1Ch] [rbp-34h] BYREF

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
  v31[0] = v13;
  if ( a3 )
  {
    v14 = *(unsigned int *)(a3 + 12);
    v15 = *(_DWORD **)(a3 + 96);
    if ( (unsigned int)v14 > 1 )
    {
      v27 = v13;
      v17 = sub_FDC990(v15, &v15[v14], v31);
      a5 = v31;
      if ( v17 )
        goto LABEL_12;
      v13 = v27;
    }
    else if ( v13 == *v15 )
    {
LABEL_12:
      sub_FE8630(a2, v31, v7, 2u, (__int64)v31, a6);
      return 1;
    }
    v18 = v9 + 24LL * v13;
    v19 = *(__int64 **)(v18 + 8);
    if ( !v19 )
    {
LABEL_17:
      sub_FE8630(a2, v31, v7, 1u, (__int64)a5, a6);
      return 1;
    }
  }
  else
  {
    v18 = v9 + 24LL * v13;
    v19 = *(__int64 **)(v18 + 8);
    if ( !v19 )
    {
      result = 0;
      if ( v13 >= *a4 )
        goto LABEL_26;
      return result;
    }
  }
  v20 = *((unsigned int *)v19 + 3);
  v21 = (_DWORD *)v19[12];
  if ( (unsigned int)v20 > 1 )
  {
    v29 = v13;
    v28 = v18;
    v24 = sub_FDC990(v21, &v21[v20], (_DWORD *)v18);
    v13 = v29;
    if ( !v24 )
      goto LABEL_21;
    v18 = v28;
  }
  else if ( *(_DWORD *)v18 != *v21 )
  {
    goto LABEL_21;
  }
  v19 = (__int64 *)*v19;
  if ( !v19
    || (v25 = *((unsigned int *)v19 + 3), (unsigned int)v25 <= 1)
    || (v30 = v13, v26 = sub_FDC990((_DWORD *)v19[12], (_DWORD *)(v19[12] + 4 * v25), (_DWORD *)v18), v13 = v30, !v26) )
  {
LABEL_21:
    if ( v19 != (__int64 *)a3 )
      goto LABEL_17;
    goto LABEL_22;
  }
  v19 = (__int64 *)*v19;
  if ( v19 != (__int64 *)a3 )
    goto LABEL_17;
LABEL_22:
  if ( v13 >= *a4 )
  {
LABEL_26:
    sub_FE8630(a2, v31, v7, 0, (__int64)a5, a6);
    return 1;
  }
  if ( !v19 )
    return 0;
  v22 = *((unsigned int *)v19 + 3);
  v23 = (_DWORD *)v19[12];
  if ( (unsigned int)v22 > 1 )
  {
    result = sub_FDC990(v23, &v23[v22], a4);
    if ( result )
      goto LABEL_26;
  }
  else
  {
    result = 0;
    if ( *a4 == *v23 )
      goto LABEL_26;
  }
  return result;
}
