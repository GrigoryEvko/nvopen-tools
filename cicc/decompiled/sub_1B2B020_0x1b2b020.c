// Function: sub_1B2B020
// Address: 0x1b2b020
//
char __fastcall sub_1B2B020(__int64 *a1, __int64 a2, __int64 a3)
{
  bool v3; // cc
  char result; // al
  int v5; // eax
  int v6; // eax
  unsigned int v7; // eax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int64 v11; // r14
  unsigned __int64 v12; // r15
  unsigned __int64 v13; // rbx
  __int64 v14; // rax
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rcx
  _QWORD *v17; // r13
  __int64 v18; // r14
  _QWORD *v19; // r12
  __int64 v20; // r12
  _QWORD *v21; // rax
  _QWORD *v22; // rdi
  __int64 v23; // rbx
  _QWORD *v24; // rax
  _QWORD *v25; // rdi
  _QWORD *v26; // rax
  __int64 v27; // [rsp-48h] [rbp-48h]
  __int64 v28; // [rsp-48h] [rbp-48h]
  __int64 v29; // [rsp-40h] [rbp-40h]

  if ( a2 == a3 )
    return 0;
  v3 = *(_DWORD *)a2 < *(_DWORD *)a3;
  if ( *(_DWORD *)a2 != *(_DWORD *)a3 )
    return v3;
  v5 = *(_DWORD *)(a3 + 4);
  v3 = *(_DWORD *)(a2 + 4) < v5;
  if ( *(_DWORD *)(a2 + 4) != v5 )
    return v3;
  v6 = *(_DWORD *)(a2 + 8);
  if ( v6 != 2 )
  {
    if ( v6 != 1 || *(_DWORD *)(a3 + 8) != 1 )
    {
LABEL_9:
      v7 = *(_DWORD *)(a3 + 8);
      result = 1;
      if ( *(_DWORD *)(a2 + 8) >= v7 )
      {
        if ( *(_DWORD *)(a2 + 8) != v7 )
          return 0;
        v8 = *(_QWORD *)(a3 + 16);
        if ( *(_QWORD *)(a2 + 16) >= v8 && (*(_QWORD *)(a2 + 16) != v8 || *(_QWORD *)(a2 + 24) >= *(_QWORD *)(a3 + 24)) )
          return 0;
      }
      return result;
    }
    v17 = *(_QWORD **)(a2 + 16);
    v18 = *a1;
    v19 = *(_QWORD **)(a3 + 16);
    if ( v17 )
    {
      if ( v19 )
      {
        if ( *((_BYTE *)v17 + 16) == 17 )
          goto LABEL_33;
LABEL_62:
        if ( *((_BYTE *)v19 + 16) == 17 )
          v17 = 0;
        return sub_1B29A30(v18, (__int64)v17, (__int64)v19);
      }
      if ( !*(_QWORD *)(a3 + 24) )
        goto LABEL_51;
      goto LABEL_59;
    }
    if ( *(_QWORD *)(a2 + 24) )
    {
      if ( v19 )
        goto LABEL_38;
      if ( !*(_QWORD *)(a3 + 24) )
      {
        v19 = *(_QWORD **)(*(_QWORD *)(a3 + 32) + 48LL);
        goto LABEL_68;
      }
    }
    else
    {
      v17 = *(_QWORD **)(*(_QWORD *)(a2 + 32) + 48LL);
      if ( v19 )
      {
LABEL_52:
        if ( v17 )
        {
          if ( *((_BYTE *)v17 + 16) == 17 )
          {
            if ( v19 )
            {
LABEL_33:
              if ( *((_BYTE *)v19 + 16) != 17 )
                v19 = 0;
            }
            return sub_1B29A30(v18, (__int64)v17, (__int64)v19);
          }
          if ( v19 )
            goto LABEL_62;
LABEL_60:
          v19 = sub_1648700(*(_QWORD *)(a3 + 24));
          return sub_1B29A30(v18, (__int64)v17, (__int64)v19);
        }
LABEL_68:
        v17 = 0;
        if ( !v19 )
          goto LABEL_69;
LABEL_38:
        if ( *((_BYTE *)v19 + 16) != 17 )
          v17 = sub_1648700(*(_QWORD *)(a2 + 24));
        return sub_1B29A30(v18, (__int64)v17, (__int64)v19);
      }
      if ( !*(_QWORD *)(a3 + 24) )
      {
LABEL_51:
        v19 = *(_QWORD **)(*(_QWORD *)(a3 + 32) + 48LL);
        goto LABEL_52;
      }
      if ( v17 )
      {
LABEL_59:
        if ( *((_BYTE *)v17 + 16) == 17 )
        {
          v19 = 0;
          return sub_1B29A30(v18, (__int64)v17, (__int64)v19);
        }
        goto LABEL_60;
      }
    }
LABEL_69:
    v29 = a3;
    v26 = sub_1648700(*(_QWORD *)(a2 + 24));
    a3 = v29;
    v17 = v26;
    goto LABEL_60;
  }
  if ( *(_DWORD *)(a3 + 8) != 2 )
    goto LABEL_9;
  v9 = *(_QWORD *)(a2 + 16);
  if ( v9 || (v23 = *(_QWORD *)(a2 + 24)) == 0 )
  {
    v10 = *(_QWORD *)(a2 + 32);
    v11 = *(_QWORD *)(v10 + 48);
    v12 = *(_QWORD *)(v10 + 56);
  }
  else
  {
    v28 = a3;
    v24 = sub_1648700(*(_QWORD *)(a2 + 24));
    a3 = v28;
    v12 = v24[5];
    if ( (*((_BYTE *)v24 + 23) & 0x40) != 0 )
      v25 = (_QWORD *)*(v24 - 1);
    else
      v25 = &v24[-3 * (*((_DWORD *)v24 + 5) & 0xFFFFFFF)];
    v11 = v25[3 * *((unsigned int *)v24 + 14) + 1 + -1431655765 * (unsigned int)((v23 - (__int64)v25) >> 3)];
  }
  v13 = *(_QWORD *)(a3 + 16);
  if ( v13 || (v20 = *(_QWORD *)(a3 + 24)) == 0 )
  {
    v14 = *(_QWORD *)(a3 + 32);
    v15 = *(_QWORD *)(v14 + 48);
    v16 = *(_QWORD *)(v14 + 56);
  }
  else
  {
    v27 = a3;
    v21 = sub_1648700(*(_QWORD *)(a3 + 24));
    a3 = v27;
    v16 = v21[5];
    if ( (*((_BYTE *)v21 + 23) & 0x40) != 0 )
      v22 = (_QWORD *)*(v21 - 1);
    else
      v22 = &v21[-3 * (*((_DWORD *)v21 + 5) & 0xFFFFFFF)];
    v15 = v22[3 * *((unsigned int *)v21 + 14) + 1 + -1431655765 * (unsigned int)((v20 - (__int64)v22) >> 3)];
  }
  result = 1;
  if ( v11 >= v15 )
  {
    if ( v11 == v15 )
    {
      if ( v12 < v16 )
        return result;
    }
    else
    {
      result = 0;
      if ( v11 > v15 )
        return result;
    }
    result = 0;
    if ( v12 <= v16 )
    {
      result = 1;
      if ( v13 <= v9 )
      {
        result = 0;
        if ( v13 == v9 )
          return *(_QWORD *)(a2 + 24) < *(_QWORD *)(a3 + 24);
      }
    }
  }
  return result;
}
