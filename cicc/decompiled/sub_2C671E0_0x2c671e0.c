// Function: sub_2C671E0
// Address: 0x2c671e0
//
__int64 __fastcall sub_2C671E0(__int64 a1, __int64 a2)
{
  __int64 v4; // rbx
  _DWORD *v5; // r8
  __int64 v6; // r9
  _DWORD *v7; // r11
  _DWORD *v8; // rdi
  __int64 v9; // rsi
  _DWORD *v10; // rdx
  _DWORD *v11; // rcx
  _DWORD *v12; // r13
  _DWORD *v13; // rax
  __int64 v14; // rax
  char v15; // dl
  __int64 v17; // rax

  v4 = *(_QWORD *)(a1 + 16);
  if ( !v4 )
  {
    v4 = a1 + 8;
    goto LABEL_24;
  }
  v5 = *(_DWORD **)a2;
  v6 = 4LL * *(unsigned int *)(a2 + 8);
  v7 = (_DWORD *)(*(_QWORD *)a2 + v6);
  while ( 1 )
  {
    v8 = *(_DWORD **)(v4 + 32);
    v9 = 4LL * *(unsigned int *)(v4 + 40);
    v10 = v8;
    v11 = &v5[(unsigned __int64)v9 / 4];
    v12 = &v8[(unsigned __int64)v9 / 4];
    if ( v9 >= v6 )
      v11 = v7;
    if ( v5 == v11 )
      break;
    v13 = v5;
    while ( *v13 >= *v10 )
    {
      if ( *v13 > *v10 )
        goto LABEL_13;
      ++v13;
      ++v10;
      if ( v11 == v13 )
        goto LABEL_12;
    }
LABEL_10:
    v14 = *(_QWORD *)(v4 + 16);
    v15 = 1;
    if ( !v14 )
      goto LABEL_14;
LABEL_11:
    v4 = v14;
  }
LABEL_12:
  if ( v12 != v10 )
    goto LABEL_10;
LABEL_13:
  v14 = *(_QWORD *)(v4 + 24);
  v15 = 0;
  if ( v14 )
    goto LABEL_11;
LABEL_14:
  if ( !v15 )
    goto LABEL_15;
LABEL_24:
  if ( *(_QWORD *)(a1 + 24) == v4 )
    return 0;
  v17 = sub_220EF80(v4);
  v5 = *(_DWORD **)a2;
  v8 = *(_DWORD **)(v17 + 32);
  v6 = 4LL * *(unsigned int *)(a2 + 8);
  v4 = v17;
  v9 = 4LL * *(unsigned int *)(v17 + 40);
  v7 = (_DWORD *)(*(_QWORD *)a2 + v6);
  v12 = &v8[(unsigned __int64)v9 / 4];
LABEL_15:
  if ( v6 < v9 )
    v12 = &v8[(unsigned __int64)v6 / 4];
  if ( v12 != v8 )
  {
    while ( *v8 >= *v5 )
    {
      if ( *v8 > *v5 )
        return v4;
      ++v8;
      ++v5;
      if ( v12 == v8 )
        goto LABEL_26;
    }
    return 0;
  }
LABEL_26:
  if ( v7 != v5 )
    return 0;
  return v4;
}
