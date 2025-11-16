// Function: sub_171F810
// Address: 0x171f810
//
__int64 __fastcall sub_171F810(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rbx
  char v3; // al
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 *v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rdx
  _QWORD *v17; // rcx
  _BYTE *v18; // r13
  unsigned __int8 v19; // al
  unsigned int v20; // r14d
  bool v21; // al
  __int64 v22; // rax
  unsigned int v23; // r13d
  __int64 v24; // rcx
  __int64 v25; // rax
  _QWORD *v26; // rcx
  unsigned int v27; // r15d
  int v28; // r14d
  __int64 v29; // rax
  char v30; // dl
  int v31; // [rsp+Ch] [rbp-34h]

  v2 = a2;
  v3 = *(_BYTE *)(a2 + 16);
  if ( v3 != 35 )
  {
    if ( v3 != 5 || *(_WORD *)(a2 + 18) != 11 )
      return 0;
    v11 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    if ( *(_BYTE *)(v11 + 16) != 79 || (v24 = *(_QWORD *)(v11 - 72)) == 0 )
    {
      v10 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
LABEL_15:
      if ( *(_BYTE *)(v10 + 16) != 79 )
        return 0;
      v12 = *(_QWORD *)(v10 - 72);
      if ( !v12 )
        return 0;
      **a1 = v12;
      v13 = *(_QWORD *)(v10 - 48);
      if ( !v13 )
        return 0;
      v14 = a1[1];
      *v14 = v13;
      result = sub_1719260(*(_BYTE **)(v10 - 24), a2, v10, (__int64)v14);
      if ( !(_BYTE)result )
        return 0;
      v10 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      if ( !v10 )
        return 0;
      goto LABEL_12;
    }
    **a1 = v24;
    v25 = *(_QWORD *)(v11 - 48);
    if ( !v25
      || (v26 = a1[1], *v26 = v25, result = sub_1719260(*(_BYTE **)(v11 - 24), a2, v11, (__int64)v26), !(_BYTE)result) )
    {
      v10 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      goto LABEL_15;
    }
    v10 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( !v10 )
      goto LABEL_15;
LABEL_12:
    *a1[3] = v10;
    return result;
  }
  v5 = *(_QWORD *)(a2 - 48);
  if ( *(_BYTE *)(v5 + 16) != 79 )
    goto LABEL_6;
  v15 = *(_QWORD *)(v5 - 72);
  if ( !v15 )
    goto LABEL_6;
  **a1 = v15;
  v16 = *(_QWORD *)(v5 - 48);
  if ( !v16 )
    goto LABEL_6;
  v17 = a1[1];
  *v17 = v16;
  v18 = *(_BYTE **)(v5 - 24);
  v19 = v18[16];
  if ( v19 == 13 )
  {
    v20 = *((_DWORD *)v18 + 8);
    if ( v20 <= 0x40 )
      v21 = *((_QWORD *)v18 + 3) == 0;
    else
      v21 = v20 == (unsigned int)sub_16A57B0((__int64)(v18 + 24));
    goto LABEL_26;
  }
  if ( *(_BYTE *)(*(_QWORD *)v18 + 8LL) != 16 || v19 > 0x10u )
    goto LABEL_6;
  v22 = sub_15A1020(v18, a2, *(_QWORD *)v18, (__int64)v17);
  if ( v22 && *(_BYTE *)(v22 + 16) == 13 )
  {
    v23 = *(_DWORD *)(v22 + 32);
    if ( v23 <= 0x40 )
      v21 = *(_QWORD *)(v22 + 24) == 0;
    else
      v21 = v23 == (unsigned int)sub_16A57B0(v22 + 24);
LABEL_26:
    if ( v21 )
      goto LABEL_27;
LABEL_6:
    v6 = *(_QWORD *)(v2 - 24);
LABEL_7:
    if ( *(_BYTE *)(v6 + 16) != 79 )
      return 0;
    v7 = *(_QWORD *)(v6 - 72);
    if ( !v7 )
      return 0;
    **a1 = v7;
    v8 = *(_QWORD *)(v6 - 48);
    if ( !v8 )
      return 0;
    v9 = a1[1];
    *v9 = v8;
    result = sub_1719260(*(_BYTE **)(v6 - 24), a2, v8, (__int64)v9);
    if ( !(_BYTE)result )
      return 0;
    v10 = *(_QWORD *)(v2 - 48);
    if ( !v10 )
      return 0;
    goto LABEL_12;
  }
  v27 = 0;
  v28 = *(_QWORD *)(*(_QWORD *)v18 + 32LL);
  if ( v28 )
  {
    do
    {
      a2 = v27;
      v29 = sub_15A0A60((__int64)v18, v27);
      if ( !v29 )
        goto LABEL_6;
      v30 = *(_BYTE *)(v29 + 16);
      if ( v30 != 9 )
      {
        if ( v30 != 13 )
          goto LABEL_6;
        if ( *(_DWORD *)(v29 + 32) <= 0x40u )
        {
          if ( *(_QWORD *)(v29 + 24) )
            goto LABEL_6;
        }
        else
        {
          v31 = *(_DWORD *)(v29 + 32);
          if ( v31 != (unsigned int)sub_16A57B0(v29 + 24) )
            goto LABEL_6;
        }
      }
    }
    while ( v28 != ++v27 );
  }
LABEL_27:
  v6 = *(_QWORD *)(v2 - 24);
  if ( !v6 )
    goto LABEL_7;
  *a1[3] = v6;
  return 1;
}
