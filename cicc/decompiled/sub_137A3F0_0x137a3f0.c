// Function: sub_137A3F0
// Address: 0x137a3f0
//
__int64 __fastcall sub_137A3F0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // rax
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned int v11; // ebx
  int v12; // eax
  int v13; // eax
  int v14; // r14d
  int v15; // r15d
  __int64 v16; // rsi
  unsigned __int64 v17; // rax
  unsigned int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // eax
  int v24; // [rsp+18h] [rbp-38h] BYREF
  int v25[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v5 = sub_157EBA0(a2);
  if ( *(_BYTE *)(v5 + 16) != 26 )
    return 0;
  if ( (*(_DWORD *)(v5 + 20) & 0xFFFFFFF) != 3 )
    return 0;
  v7 = *(_QWORD *)(v5 - 72);
  if ( *(_BYTE *)(v7 + 16) != 75 )
    return 0;
  v8 = *(_QWORD *)(v7 - 24);
  if ( *(_BYTE *)(v8 + 16) != 13 )
    return 0;
  v9 = *(_QWORD *)(v7 - 48);
  if ( *(_BYTE *)(v9 + 16) == 50 )
  {
    v19 = (*(_BYTE *)(v9 + 23) & 0x40) != 0 ? *(_QWORD *)(v9 - 8) : v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF);
    v20 = *(_QWORD *)(v19 + 24);
    if ( *(_BYTE *)(v20 + 16) == 13 )
    {
      if ( *(_DWORD *)(v20 + 32) > 0x40u )
      {
        if ( (unsigned int)sub_16A5940(v20 + 24) == 1 )
          return 0;
      }
      else
      {
        v21 = *(_QWORD *)(v20 + 24);
        if ( v21 && (v21 & (v21 - 1)) == 0 )
          return 0;
      }
    }
  }
  v24 = 422;
  if ( a3 )
  {
    v10 = *(_QWORD *)(v7 - 48);
    if ( *(_BYTE *)(v10 + 16) == 78 )
    {
      v16 = *(_QWORD *)(v10 - 24);
      if ( !*(_BYTE *)(v16 + 16) )
      {
        sub_149CB50(*a3, v16, &v24);
        LOBYTE(v17) = 0;
        if ( (unsigned int)(v24 - 363) <= 0xB )
          v17 = (0xA09uLL >> ((unsigned __int8)v24 - 107)) & 1;
        if ( v24 == 289 || (_BYTE)v17 )
        {
          v18 = *(unsigned __int16 *)(v7 + 18);
          BYTE1(v18) &= ~0x80u;
          if ( v18 != 32 )
            goto LABEL_25;
          goto LABEL_17;
        }
      }
    }
  }
  v11 = *(_DWORD *)(v8 + 32);
  if ( v11 <= 0x40 )
  {
    if ( *(_QWORD *)(v8 + 24) )
    {
      v22 = *(_QWORD *)(v8 + 24);
      if ( v22 == 1 )
      {
        v23 = *(unsigned __int16 *)(v7 + 18);
        BYTE1(v23) &= ~0x80u;
        if ( v23 == 40 )
          goto LABEL_17;
      }
      if ( v22 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v11) )
        return 0;
LABEL_14:
      v13 = *(unsigned __int16 *)(v7 + 18);
      BYTE1(v13) &= ~0x80u;
      if ( v13 == 33 || v13 == 38 )
        goto LABEL_26;
      if ( v13 != 32 )
        return 0;
      goto LABEL_17;
    }
  }
  else if ( v11 != (unsigned int)sub_16A57B0(v8 + 24) )
  {
    if ( (unsigned int)sub_16A57B0(v8 + 24) == v11 - 1 )
    {
      v12 = *(unsigned __int16 *)(v7 + 18);
      BYTE1(v12) &= ~0x80u;
      if ( v12 == 40 )
        goto LABEL_17;
    }
    if ( v11 != (unsigned int)sub_16A58F0(v8 + 24) )
      return 0;
    goto LABEL_14;
  }
  v18 = *(unsigned __int16 *)(v7 + 18);
  BYTE1(v18) &= ~0x80u;
  if ( v18 == 38 )
    goto LABEL_26;
  if ( v18 > 0x26 )
  {
    if ( v18 != 40 )
      return 0;
  }
  else if ( v18 != 32 )
  {
LABEL_25:
    if ( v18 == 33 )
    {
LABEL_26:
      v14 = 1;
      v15 = 0;
      goto LABEL_18;
    }
    return 0;
  }
LABEL_17:
  v14 = 0;
  v15 = 1;
LABEL_18:
  sub_16AF710(v25, 20, 32);
  sub_1379150(a1, a2, v15, v25[0]);
  sub_1379150(a1, a2, v14, 0x80000000 - v25[0]);
  return 1;
}
