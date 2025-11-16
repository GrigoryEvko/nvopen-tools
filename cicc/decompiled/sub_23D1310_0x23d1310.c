// Function: sub_23D1310
// Address: 0x23d1310
//
bool __fastcall sub_23D1310(__int64 a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  _BYTE *v5; // rax
  _BYTE *v6; // r13
  _BYTE *v7; // rax
  const void ***v8; // r14
  _BYTE *v9; // rbx
  _BYTE *v10; // rax
  unsigned __int8 *v11; // r13
  __int64 v12; // rdx
  unsigned int v13; // r14d
  __int64 v14; // rax
  __int64 v15; // rdx
  _BYTE *v16; // r15
  unsigned int v17; // r14d
  __int64 v18; // rax
  __int64 v19; // rdx
  _BYTE *v20; // rax
  unsigned __int8 *v21; // rax

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v5 != 57 )
    goto LABEL_4;
  v15 = *((_QWORD *)v5 - 8);
  if ( !v15 )
    goto LABEL_4;
  v8 = (const void ***)(a1 + 8);
  **(_QWORD **)a1 = v15;
  if ( !sub_10080A0((const void ***)(a1 + 8), *((_QWORD *)v5 - 4)) )
    goto LABEL_4;
  v6 = (_BYTE *)*((_QWORD *)a3 - 4);
  if ( *v6 != 57 )
    return 0;
  v7 = (_BYTE *)*((_QWORD *)v6 - 8);
  if ( *v7 != 55 || *((_QWORD *)v7 - 8) != **(_QWORD **)(a1 + 16) )
    goto LABEL_9;
  v16 = (_BYTE *)*((_QWORD *)v7 - 4);
  if ( !v16 )
    BUG();
  if ( *v16 == 17 )
    goto LABEL_26;
  v19 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v16 + 1) + 8LL) - 17;
  if ( (unsigned int)v19 > 1 || *v16 > 0x15u )
  {
LABEL_9:
    **(_QWORD **)a1 = v7;
    if ( !sub_10080A0(v8, *((_QWORD *)v6 - 4)) )
      return 0;
    v9 = (_BYTE *)*((_QWORD *)a3 - 8);
    if ( *v9 != 57 )
      return 0;
    v10 = (_BYTE *)*((_QWORD *)v9 - 8);
    if ( *v10 != 55 || *((_QWORD *)v10 - 8) != **(_QWORD **)(a1 + 16) )
      return 0;
    v11 = (unsigned __int8 *)*((_QWORD *)v10 - 4);
    if ( !v11 )
      BUG();
    v12 = *v11;
    if ( (_BYTE)v12 != 17 )
    {
      if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v11 + 1) + 8LL) - 17 > 1 )
        return 0;
      if ( (unsigned __int8)v12 > 0x15u )
        return 0;
      v21 = sub_AD7630((__int64)v11, 0, v12);
      v11 = v21;
      if ( !v21 || *v21 != 17 )
        return 0;
    }
    v13 = *((_DWORD *)v11 + 8);
    if ( v13 <= 0x40 )
    {
      v14 = *((_QWORD *)v11 + 3);
      goto LABEL_17;
    }
    if ( v13 - (unsigned int)sub_C444A0((__int64)(v11 + 24)) <= 0x40 )
    {
      v14 = **((_QWORD **)v11 + 3);
LABEL_17:
      if ( *(_QWORD *)(a1 + 24) == v14 )
        return sub_10080A0((const void ***)(a1 + 32), *((_QWORD *)v9 - 4));
    }
    return 0;
  }
  v20 = sub_AD7630(*((_QWORD *)v7 - 4), 0, v19);
  v16 = v20;
  if ( !v20 || *v20 != 17 )
    goto LABEL_4;
LABEL_26:
  v17 = *((_DWORD *)v16 + 8);
  if ( v17 > 0x40 )
  {
    if ( v17 - (unsigned int)sub_C444A0((__int64)(v16 + 24)) > 0x40 )
    {
LABEL_4:
      v6 = (_BYTE *)*((_QWORD *)a3 - 4);
      if ( *v6 != 57 )
        return 0;
      v7 = (_BYTE *)*((_QWORD *)v6 - 8);
      if ( !v7 )
        return 0;
      v8 = (const void ***)(a1 + 8);
      goto LABEL_9;
    }
    v18 = **((_QWORD **)v16 + 3);
  }
  else
  {
    v18 = *((_QWORD *)v16 + 3);
  }
  if ( *(_QWORD *)(a1 + 24) != v18 )
    goto LABEL_4;
  result = sub_10080A0((const void ***)(a1 + 32), *((_QWORD *)v6 - 4));
  if ( !result )
    goto LABEL_4;
  return result;
}
