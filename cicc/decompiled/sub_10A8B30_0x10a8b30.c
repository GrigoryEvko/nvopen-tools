// Function: sub_10A8B30
// Address: 0x10a8b30
//
__int64 __fastcall sub_10A8B30(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  _BYTE *v4; // rax
  _BYTE *v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rcx
  __int64 v8; // rax
  _QWORD *v9; // rsi
  __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // rax
  _QWORD *v18; // rsi
  __int64 v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rax
  _QWORD *v23; // r8
  __int64 v24; // r8
  __int64 v25; // rsi
  __int64 v26; // rcx
  _QWORD *v27; // r8
  __int64 v28; // r8
  __int64 v29; // rsi
  __int64 v30; // rcx
  __int64 v31; // rcx

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v4 != 42 )
    goto LABEL_4;
  v15 = *((_QWORD *)v4 - 8);
  if ( !v15 )
    goto LABEL_56;
  **a1 = v15;
  v16 = *((_QWORD *)v4 - 4);
  if ( *(_BYTE *)v16 != 86
    || ((*(_BYTE *)(v16 + 7) & 0x40) == 0
      ? (v27 = (_QWORD *)(v16 - 32LL * (*(_DWORD *)(v16 + 4) & 0x7FFFFFF)))
      : (v27 = *(_QWORD **)(v16 - 8)),
        !*v27) )
  {
LABEL_22:
    **a1 = v16;
    v17 = *((_QWORD *)v4 - 8);
    if ( *(_BYTE *)v17 == 86 )
    {
      v18 = (*(_BYTE *)(v17 + 7) & 0x40) != 0
          ? *(_QWORD **)(v17 - 8)
          : (_QWORD *)(v17 - 32LL * (*(_DWORD *)(v17 + 4) & 0x7FFFFFF));
      if ( *v18 )
      {
        *a1[1] = *v18;
        v19 = (*(_BYTE *)(v17 + 7) & 0x40) != 0 ? *(_QWORD *)(v17 - 8) : v17 - 32LL * (*(_DWORD *)(v17 + 4) & 0x7FFFFFF);
        v20 = *(_QWORD *)(v19 + 32);
        if ( v20 )
        {
          *a1[2] = v20;
          v21 = (*(_BYTE *)(v17 + 7) & 0x40) != 0
              ? *(_QWORD *)(v17 - 8)
              : v17 - 32LL * (*(_DWORD *)(v17 + 4) & 0x7FFFFFF);
          v22 = *(_QWORD *)(v21 + 64);
          if ( v22 )
          {
            *a1[3] = v22;
            goto LABEL_33;
          }
        }
      }
    }
LABEL_4:
    v5 = (_BYTE *)*((_QWORD *)a3 - 4);
    goto LABEL_5;
  }
  *a1[1] = *v27;
  v28 = (*(_BYTE *)(v16 + 7) & 0x40) != 0 ? *(_QWORD *)(v16 - 8) : v16 - 32LL * (*(_DWORD *)(v16 + 4) & 0x7FFFFFF);
  v29 = *(_QWORD *)(v28 + 32);
  if ( !v29
    || ((*a1[2] = v29, (*(_BYTE *)(v16 + 7) & 0x40) == 0)
      ? (v30 = v16 - 32LL * (*(_DWORD *)(v16 + 4) & 0x7FFFFFF))
      : (v30 = *(_QWORD *)(v16 - 8)),
        (v31 = *(_QWORD *)(v30 + 64)) == 0) )
  {
LABEL_56:
    v16 = *((_QWORD *)v4 - 4);
    if ( !v16 )
      goto LABEL_4;
    goto LABEL_22;
  }
  *a1[3] = v31;
LABEL_33:
  v5 = (_BYTE *)*((_QWORD *)a3 - 4);
  if ( !v5 )
  {
LABEL_5:
    if ( *v5 != 42 )
      return 0;
    v6 = *((_QWORD *)v5 - 8);
    if ( v6 )
    {
      **a1 = v6;
      v7 = *((_QWORD *)v5 - 4);
      if ( *(_BYTE *)v7 != 86
        || ((*(_BYTE *)(v7 + 7) & 0x40) == 0
          ? (v23 = (_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)))
          : (v23 = *(_QWORD **)(v7 - 8)),
            !*v23) )
      {
LABEL_8:
        **a1 = v7;
        v8 = *((_QWORD *)v5 - 8);
        if ( *(_BYTE *)v8 != 86 )
          return 0;
        v9 = (*(_BYTE *)(v8 + 7) & 0x40) != 0
           ? *(_QWORD **)(v8 - 8)
           : (_QWORD *)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
        if ( !*v9 )
          return 0;
        *a1[1] = *v9;
        v10 = (*(_BYTE *)(v8 + 7) & 0x40) != 0 ? *(_QWORD *)(v8 - 8) : v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF);
        v11 = *(_QWORD *)(v10 + 32);
        if ( !v11 )
          return 0;
        *a1[2] = v11;
        v12 = (*(_BYTE *)(v8 + 7) & 0x40) != 0 ? *(_QWORD *)(v8 - 8) : v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF);
        v13 = *(_QWORD *)(v12 + 64);
        if ( !v13 )
          return 0;
LABEL_18:
        *a1[3] = v13;
        v14 = *((_QWORD *)a3 - 8);
        if ( v14 )
        {
          *a1[4] = v14;
          return 1;
        }
        return 0;
      }
      *a1[1] = *v23;
      if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
        v24 = *(_QWORD *)(v7 - 8);
      else
        v24 = v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
      v25 = *(_QWORD *)(v24 + 32);
      if ( v25 )
      {
        *a1[2] = v25;
        v26 = (*(_BYTE *)(v7 + 7) & 0x40) != 0 ? *(_QWORD *)(v7 - 8) : v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
        v13 = *(_QWORD *)(v26 + 64);
        if ( v13 )
          goto LABEL_18;
      }
    }
    v7 = *((_QWORD *)v5 - 4);
    if ( !v7 )
      return 0;
    goto LABEL_8;
  }
  *a1[4] = v5;
  return 1;
}
