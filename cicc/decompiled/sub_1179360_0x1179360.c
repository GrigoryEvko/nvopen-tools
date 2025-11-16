// Function: sub_1179360
// Address: 0x1179360
//
__int64 __fastcall sub_1179360(__int64 a1, __int64 a2, __int64 a3, char a4, char a5)
{
  char v6; // al
  char v7; // cl
  __int64 *v8; // r8
  __int64 v9; // rbx
  _QWORD *v10; // r11
  __int64 v11; // r8
  __int64 *v13; // rcx
  __int64 v14; // r10
  _QWORD *v15; // r9
  __int64 v16; // rax
  _QWORD *v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rsi
  _QWORD *v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 *v24; // rdx

  v6 = *(_BYTE *)(a3 + 7) & 0x40;
  v7 = *(_BYTE *)(a2 + 7) & 0x40;
  if ( a5 )
    goto LABEL_8;
  if ( v7 )
  {
    v8 = *(__int64 **)(a2 - 8);
    v9 = *v8;
    if ( v6 )
    {
LABEL_4:
      v10 = *(_QWORD **)(a3 - 8);
      v11 = v8[4];
      if ( v9 != *v10 )
        goto LABEL_5;
LABEL_27:
      **(_QWORD **)a1 = v11;
      if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
        v23 = *(_QWORD *)(a3 - 8);
      else
        v23 = a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
      **(_QWORD **)(a1 + 8) = *(_QWORD *)(v23 + 32);
      **(_BYTE **)(a1 + 16) = 1;
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        goto LABEL_23;
      goto LABEL_30;
    }
  }
  else
  {
    v8 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v9 = *v8;
    if ( v6 )
      goto LABEL_4;
  }
  v11 = v8[4];
  v10 = (_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
  if ( v9 == *v10 )
    goto LABEL_27;
LABEL_5:
  if ( v10[4] != v11 )
  {
    if ( !a4 )
      return 0;
LABEL_8:
    if ( v7 )
    {
      v13 = *(__int64 **)(a2 - 8);
      v14 = *v13;
      if ( v6 )
      {
LABEL_10:
        v15 = *(_QWORD **)(a3 - 8);
        v16 = v13[4];
        if ( v14 != v15[4] )
        {
LABEL_11:
          if ( *v15 != v16 )
            return 0;
          **(_QWORD **)a1 = v14;
          v17 = *(_QWORD **)(a1 + 8);
          if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
            v18 = *(_QWORD *)(a3 - 8);
          else
            v18 = a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
          v19 = *(_QWORD *)(v18 + 32);
          goto LABEL_15;
        }
LABEL_20:
        **(_QWORD **)a1 = v16;
        if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
          v21 = *(_QWORD **)(a3 - 8);
        else
          v21 = (_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
        **(_QWORD **)(a1 + 8) = *v21;
        **(_BYTE **)(a1 + 16) = 1;
        if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        {
LABEL_23:
          v22 = *(_QWORD *)(a2 - 8);
          return *(_QWORD *)v22;
        }
LABEL_30:
        v22 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
        return *(_QWORD *)v22;
      }
    }
    else
    {
      v13 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      v14 = *v13;
      if ( v6 )
        goto LABEL_10;
    }
    v15 = (_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
    v16 = v13[4];
    if ( v14 != v15[4] )
      goto LABEL_11;
    goto LABEL_20;
  }
  **(_QWORD **)a1 = v9;
  v17 = *(_QWORD **)(a1 + 8);
  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
    v24 = *(__int64 **)(a3 - 8);
  else
    v24 = (__int64 *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
  v19 = *v24;
LABEL_15:
  *v17 = v19;
  **(_BYTE **)(a1 + 16) = 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v20 = *(_QWORD *)(a2 - 8);
  else
    v20 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  return *(_QWORD *)(v20 + 32);
}
