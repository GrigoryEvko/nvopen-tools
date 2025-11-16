// Function: sub_14B3C60
// Address: 0x14b3c60
//
__int64 __fastcall sub_14B3C60(_QWORD *a1, __int64 a2)
{
  int v2; // eax
  unsigned int v4; // r12d
  __int64 *v6; // rsi
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rcx
  int v10; // edx
  _QWORD *v11; // rdi
  int v12; // edx
  _QWORD *v13; // rcx
  __int64 v14; // r13
  int v15; // edx
  _QWORD *v16; // rax
  unsigned int v17; // r14d
  _QWORD *v18; // rax
  __int64 v19; // rdx
  _QWORD *v20; // rcx

  v2 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v2 <= 0x17u )
  {
    v4 = 0;
    if ( (_BYTE)v2 != 5 || (unsigned int)*(unsigned __int16 *)(a2 + 18) - 24 > 1 )
      return v4;
    v9 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    if ( v9 == *a1 )
      goto LABEL_25;
    v10 = *(unsigned __int8 *)(v9 + 16);
    if ( (unsigned __int8)v10 > 0x17u )
    {
      if ( (_BYTE)v10 != 69 )
      {
LABEL_20:
        v12 = v10 - 24;
LABEL_21:
        v4 = 0;
        if ( v12 != 47 )
          return v4;
        v13 = (*(_BYTE *)(v9 + 23) & 0x40) != 0
            ? *(_QWORD **)(v9 - 8)
            : (_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
        v4 = 0;
        if ( *v13 != a1[2] )
          return v4;
LABEL_25:
        v4 = 0;
        v14 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
        if ( *(_BYTE *)(v14 + 16) != 13 )
          return v4;
        goto LABEL_35;
      }
    }
    else
    {
      if ( (_BYTE)v10 != 5 )
        return v4;
      if ( *(_WORD *)(v9 + 18) != 45 )
        goto LABEL_50;
    }
    if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
      v11 = *(_QWORD **)(v9 - 8);
    else
      v11 = (_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
    if ( *v11 == a1[1] )
      goto LABEL_25;
    if ( (unsigned __int8)v10 > 0x17u )
      goto LABEL_20;
LABEL_50:
    v12 = *(unsigned __int16 *)(v9 + 18);
    goto LABEL_21;
  }
  v4 = 0;
  if ( (unsigned int)(v2 - 48) > 1 )
    return v4;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v6 = *(__int64 **)(a2 - 8);
  else
    v6 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v7 = *v6;
  if ( *v6 == *a1 )
    goto LABEL_34;
  v8 = *(unsigned __int8 *)(v7 + 16);
  if ( (unsigned __int8)v8 > 0x17u )
  {
    if ( (_BYTE)v8 != 69 )
      goto LABEL_29;
  }
  else
  {
    if ( (_BYTE)v8 != 5 )
      return 0;
    if ( *(_WORD *)(v7 + 18) != 45 )
    {
LABEL_44:
      v15 = *(unsigned __int16 *)(v7 + 18);
      goto LABEL_30;
    }
  }
  if ( (*(_BYTE *)(v7 + 23) & 0x40) != 0 )
    v20 = *(_QWORD **)(v7 - 8);
  else
    v20 = (_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
  if ( *v20 == a1[1] )
  {
LABEL_34:
    v14 = v6[3];
    v4 = 0;
    if ( *(_BYTE *)(v14 + 16) != 13 )
      return v4;
LABEL_35:
    v17 = *(_DWORD *)(v14 + 32);
    if ( v17 <= 0x40 )
    {
      v18 = (_QWORD *)a1[3];
      v19 = *(_QWORD *)(v14 + 24);
    }
    else
    {
      if ( v17 - (unsigned int)sub_16A57B0(v14 + 24) > 0x40 )
        return v4;
      v18 = (_QWORD *)a1[3];
      v19 = **(_QWORD **)(v14 + 24);
    }
    *v18 = v19;
    return 1;
  }
  if ( (unsigned __int8)v8 <= 0x17u )
    goto LABEL_44;
LABEL_29:
  v15 = v8 - 24;
LABEL_30:
  v4 = 0;
  if ( v15 == 47 )
  {
    v16 = (*(_BYTE *)(v7 + 23) & 0x40) != 0
        ? *(_QWORD **)(v7 - 8)
        : (_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
    v4 = 0;
    if ( *v16 == a1[2] )
      goto LABEL_34;
  }
  return v4;
}
