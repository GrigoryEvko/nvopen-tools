// Function: sub_14B5220
// Address: 0x14b5220
//
__int64 __fastcall sub_14B5220(_QWORD *a1, __int64 a2)
{
  char v3; // al
  unsigned int v4; // r12d
  __int64 v6; // rdi
  int v7; // eax
  int v8; // eax
  __int64 v9; // r13
  unsigned int v10; // r14d
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  int v14; // edx
  _QWORD *v15; // rdi
  int v16; // edx
  _QWORD *v17; // rcx
  _QWORD *v18; // rdx

  v3 = *(_BYTE *)(a2 + 16);
  if ( v3 == 47 )
  {
    v6 = *(_QWORD *)(a2 - 48);
    if ( v6 == *a1 )
      goto LABEL_11;
    v7 = *(unsigned __int8 *)(v6 + 16);
    if ( (unsigned __int8)v7 <= 0x17u )
    {
      if ( (_BYTE)v7 != 5 )
        return 0;
      if ( *(_WORD *)(v6 + 18) != 45 )
        goto LABEL_39;
    }
    else if ( (_BYTE)v7 != 69 )
    {
LABEL_8:
      v8 = v7 - 24;
LABEL_9:
      v4 = 0;
      if ( v8 != 47 || *(_QWORD *)sub_13CF970(v6) != a1[2] )
        return v4;
LABEL_11:
      v9 = *(_QWORD *)(a2 - 24);
      v4 = 0;
      if ( *(_BYTE *)(v9 + 16) != 13 )
        return v4;
LABEL_12:
      v10 = *(_DWORD *)(v9 + 32);
      if ( v10 > 0x40 )
      {
        if ( v10 - (unsigned int)sub_16A57B0(v9 + 24) > 0x40 )
          return v4;
        v11 = (_QWORD *)a1[3];
        v12 = **(_QWORD **)(v9 + 24);
      }
      else
      {
        v11 = (_QWORD *)a1[3];
        v12 = *(_QWORD *)(v9 + 24);
      }
      *v11 = v12;
      return 1;
    }
    if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
      v18 = *(_QWORD **)(v6 - 8);
    else
      v18 = (_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
    if ( *v18 == a1[1] )
      goto LABEL_11;
    if ( (unsigned __int8)v7 > 0x17u )
      goto LABEL_8;
LABEL_39:
    v8 = *(unsigned __int16 *)(v6 + 18);
    goto LABEL_9;
  }
  v4 = 0;
  if ( v3 != 5 || *(_WORD *)(a2 + 18) != 23 )
    return v4;
  v13 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( v13 == *a1 )
    goto LABEL_29;
  v14 = *(unsigned __int8 *)(v13 + 16);
  if ( (unsigned __int8)v14 <= 0x17u )
  {
    if ( (_BYTE)v14 != 5 )
      return v4;
    if ( *(_WORD *)(v13 + 18) != 45 )
      goto LABEL_43;
    goto LABEL_20;
  }
  if ( (_BYTE)v14 == 69 )
  {
LABEL_20:
    if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
      v15 = *(_QWORD **)(v13 - 8);
    else
      v15 = (_QWORD *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
    if ( *v15 == a1[1] )
    {
LABEL_29:
      v4 = 0;
      v9 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(v9 + 16) != 13 )
        return v4;
      goto LABEL_12;
    }
    if ( (unsigned __int8)v14 > 0x17u )
      goto LABEL_24;
LABEL_43:
    v16 = *(unsigned __int16 *)(v13 + 18);
    goto LABEL_25;
  }
LABEL_24:
  v16 = v14 - 24;
LABEL_25:
  v4 = 0;
  if ( v16 == 47 )
  {
    v17 = (*(_BYTE *)(v13 + 23) & 0x40) != 0
        ? *(_QWORD **)(v13 - 8)
        : (_QWORD *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
    v4 = 0;
    if ( *v17 == a1[2] )
      goto LABEL_29;
  }
  return v4;
}
