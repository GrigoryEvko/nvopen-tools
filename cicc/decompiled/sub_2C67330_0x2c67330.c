// Function: sub_2C67330
// Address: 0x2c67330
//
__int64 __fastcall sub_2C67330(_QWORD *a1, __int64 a2, __int64 a3)
{
  _DWORD *v6; // rbx
  _DWORD *v7; // rcx
  __int64 v8; // r9
  __int64 v9; // r15
  _DWORD *v10; // r8
  _DWORD *v11; // rsi
  _DWORD *v12; // rdx
  _DWORD *v13; // rdi
  _DWORD *v14; // rax
  __int64 result; // rax
  __int64 v16; // rdx
  _DWORD *v17; // rax
  _DWORD *v18; // rcx
  __int64 v19; // rsi
  _DWORD *v20; // rax
  __int64 v21; // rax
  _DWORD *v22; // r8
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rcx
  _DWORD *v26; // rax
  __int64 v27; // r12
  _DWORD *v28; // rdi
  _DWORD *v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rdx
  _DWORD *v32; // rcx
  _DWORD *v33; // r8
  bool v34; // cc
  _DWORD *v35; // rdx
  _DWORD *v36; // [rsp+8h] [rbp-38h]
  _DWORD *v37; // [rsp+8h] [rbp-38h]

  if ( (_QWORD *)a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_2C671E0((__int64)a1, a3);
    v27 = a1[4];
    v28 = *(_DWORD **)a3;
    v29 = *(_DWORD **)(v27 + 32);
    v30 = 4LL * *(unsigned int *)(a3 + 8);
    v31 = 4LL * *(unsigned int *)(v27 + 40);
    v32 = &v29[(unsigned __int64)v30 / 4];
    v33 = &v29[(unsigned __int64)v31 / 4];
    v34 = v30 < v31;
    v35 = v28;
    if ( !v34 )
      v32 = v33;
    if ( v29 == v32 )
    {
LABEL_53:
      if ( v35 == &v28[(unsigned __int64)v30 / 4] )
        return sub_2C671E0((__int64)a1, a3);
    }
    else
    {
      while ( *v29 >= *v35 )
      {
        if ( *v29 > *v35 )
          return sub_2C671E0((__int64)a1, a3);
        ++v29;
        ++v35;
        if ( v32 == v29 )
          goto LABEL_53;
      }
    }
    return 0;
  }
  v6 = *(_DWORD **)a3;
  v7 = *(_DWORD **)(a2 + 32);
  v8 = 4LL * *(unsigned int *)(a2 + 40);
  v9 = 4LL * *(unsigned int *)(a3 + 8);
  v10 = (_DWORD *)(*(_QWORD *)a3 + v9);
  v11 = (_DWORD *)(*(_QWORD *)a3 + v8);
  v12 = v7;
  if ( v8 >= v9 )
    v11 = v10;
  v13 = &v7[(unsigned __int64)v8 / 4];
  if ( v6 != v11 )
  {
    v14 = v6;
    while ( *v14 >= *v12 )
    {
      if ( *v14 > *v12 )
        goto LABEL_23;
      ++v14;
      ++v12;
      if ( v11 == v14 )
        goto LABEL_22;
    }
    goto LABEL_9;
  }
LABEL_22:
  if ( v13 != v12 )
  {
LABEL_9:
    result = a2;
    if ( a1[3] == a2 )
      return result;
    v36 = v10;
    v16 = sub_220EF80(a2);
    v17 = *(_DWORD **)(v16 + 32);
    v18 = &v17[(unsigned __int64)v9 / 4];
    v19 = 4LL * *(unsigned int *)(v16 + 40);
    if ( v9 >= v19 )
      v18 = &v17[(unsigned __int64)v19 / 4];
    if ( v17 != v18 )
    {
      while ( *v17 >= *v6 )
      {
        if ( *v17 > *v6 )
          return sub_2C671E0((__int64)a1, a3);
        ++v17;
        ++v6;
        if ( v18 == v17 )
          goto LABEL_51;
      }
      goto LABEL_17;
    }
LABEL_51:
    if ( v36 != v6 )
    {
LABEL_17:
      result = 0;
      if ( *(_QWORD *)(v16 + 24) )
        return a2;
      return result;
    }
    return sub_2C671E0((__int64)a1, a3);
  }
LABEL_23:
  if ( v8 > v9 )
    v13 = &v7[(unsigned __int64)v9 / 4];
  v20 = v6;
  if ( v7 == v13 )
  {
LABEL_49:
    if ( v10 == v20 )
      return a2;
  }
  else
  {
    while ( *v7 >= *v20 )
    {
      if ( *v7 > *v20 )
        return a2;
      ++v7;
      ++v20;
      if ( v13 == v7 )
        goto LABEL_49;
    }
  }
  if ( a1[4] == a2 )
    return 0;
  v37 = v10;
  v21 = sub_220EEE0(a2);
  v22 = v37;
  v23 = *(_QWORD *)(v21 + 32);
  v24 = v21;
  v25 = 4LL * *(unsigned int *)(v21 + 40);
  if ( v25 < v9 )
    v22 = &v6[(unsigned __int64)v25 / 4];
  v26 = *(_DWORD **)(v21 + 32);
  if ( v6 == v22 )
  {
LABEL_55:
    if ( v26 == (_DWORD *)(v23 + v25) )
      return sub_2C671E0((__int64)a1, a3);
  }
  else
  {
    while ( *v6 >= *v26 )
    {
      if ( *v6 > *v26 )
        return sub_2C671E0((__int64)a1, a3);
      ++v6;
      ++v26;
      if ( v22 == v6 )
        goto LABEL_55;
    }
  }
  result = 0;
  if ( *(_QWORD *)(a2 + 24) )
    return v24;
  return result;
}
