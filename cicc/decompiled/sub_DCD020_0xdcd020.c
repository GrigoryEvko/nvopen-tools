// Function: sub_DCD020
// Address: 0xdcd020
//
__int64 __fastcall sub_DCD020(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r13
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // r9
  bool v17; // zf
  _QWORD *v18; // rdi
  _QWORD *v19; // rsi
  _QWORD *v20; // rdi
  _QWORD *v21; // rsi
  _QWORD *v22; // rdi
  _QWORD *v23; // rsi
  __int64 v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  int v28; // eax
  __int64 v29[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = a3;
  if ( (_DWORD)a2 == 39 )
  {
    v13 = a4;
    goto LABEL_14;
  }
  if ( (unsigned int)a2 > 0x27 )
  {
    if ( (_DWORD)a2 != 41 )
    {
      if ( (unsigned __int8)sub_DCCA40(a1, a2, a3, a4) )
        return 1;
      goto LABEL_37;
    }
    a3 = a4;
    v13 = v6;
LABEL_14:
    if ( *(_WORD *)(v13 + 24) == 4 )
    {
      v14 = *(_QWORD *)(v13 + 32);
      if ( v14 )
      {
        if ( *(_WORD *)(a3 + 24) == 3 && *(_QWORD *)(a3 + 32) == v14 )
          return 1;
      }
    }
    if ( (unsigned __int8)sub_DCCA40(a1, a2, v6, a4) )
      return 1;
    if ( (_DWORD)a2 == 39 )
    {
      v16 = v6;
      v15 = a4;
      goto LABEL_21;
    }
LABEL_37:
    if ( (_DWORD)a2 != 41 )
      goto LABEL_39;
    v16 = a4;
    v15 = v6;
LABEL_21:
    v17 = *(_WORD *)(v15 + 24) == 12;
    v29[0] = v16;
    if ( v17 )
    {
      v18 = *(_QWORD **)(v15 + 32);
      v19 = &v18[*(_QWORD *)(v15 + 40)];
      if ( v19 != sub_D91070(v18, (__int64)v19, v29) )
        return 1;
    }
    v17 = *(_WORD *)(v16 + 24) == 10;
    v29[0] = v15;
    if ( !v17 )
      goto LABEL_25;
    goto LABEL_24;
  }
  if ( (_DWORD)a2 == 35 )
  {
    v8 = a4;
  }
  else
  {
    if ( (_DWORD)a2 != 37 )
    {
      if ( (unsigned __int8)sub_DCCA40(a1, a2, a3, a4) )
        return 1;
      goto LABEL_25;
    }
    a3 = a4;
    v8 = v6;
  }
  if ( *(_WORD *)(v8 + 24) == 3 )
  {
    v9 = *(_QWORD *)(v8 + 32);
    if ( v9 )
    {
      if ( *(_WORD *)(a3 + 24) == 4 && *(_QWORD *)(a3 + 32) == v9 )
        return 1;
    }
  }
  if ( (unsigned __int8)sub_DCCA40(a1, a2, v6, a4) )
    return 1;
  if ( (_DWORD)a2 == 35 )
  {
    v16 = v6;
    v15 = a4;
  }
  else
  {
    v16 = a4;
    v15 = v6;
  }
  v17 = *(_WORD *)(v15 + 24) == 11;
  v29[0] = v16;
  if ( v17 )
  {
    v22 = *(_QWORD **)(v15 + 32);
    v23 = &v22[*(_QWORD *)(v15 + 40)];
    if ( v23 != sub_D91070(v22, (__int64)v23, v29) )
      return 1;
  }
  v17 = *(_WORD *)(v16 + 24) == 9;
  v29[0] = v15;
  if ( !v17 )
    goto LABEL_25;
LABEL_24:
  v20 = *(_QWORD **)(v16 + 32);
  v21 = &v20[*(_QWORD *)(v16 + 40)];
  if ( v21 != sub_D91070(v20, (__int64)v21, v29) )
    return 1;
LABEL_25:
  if ( (unsigned int)(a2 - 32) > 1 )
  {
LABEL_39:
    if ( *(_WORD *)(v6 + 24) != 8 )
      return sub_DA34D0((__int64)a1, a2, v6, a4);
    if ( *(_WORD *)(a4 + 24) != 8 )
      return sub_DA34D0((__int64)a1, a2, v6, a4);
    if ( *(_QWORD *)(v6 + 48) != *(_QWORD *)(a4 + 48) )
      return sub_DA34D0((__int64)a1, a2, v6, a4);
    if ( *(_QWORD *)(v6 + 40) != 2 )
      return sub_DA34D0((__int64)a1, a2, v6, a4);
    if ( *(_QWORD *)(a4 + 40) != 2 )
      return sub_DA34D0((__int64)a1, a2, v6, a4);
    v24 = sub_D33D80((_QWORD *)v6, (__int64)a1, v10, v11, v15);
    if ( v24 != sub_D33D80((_QWORD *)a4, (__int64)a1, v25, v26, v27) )
      return sub_DA34D0((__int64)a1, a2, v6, a4);
    v28 = !sub_B532B0(a2) ? 2 : 4;
    if ( ((unsigned __int16)v28 & *(_WORD *)(v6 + 28)) == 0
      || ((unsigned __int16)v28 & *(_WORD *)(a4 + 28)) == 0
      || !(unsigned __int8)sub_DC3A60((__int64)a1, a2, **(_BYTE ***)(v6 + 32), **(_BYTE ***)(a4 + 32)) )
    {
      return sub_DA34D0((__int64)a1, a2, v6, a4);
    }
    return 1;
  }
  return sub_DA34D0((__int64)a1, a2, v6, a4);
}
