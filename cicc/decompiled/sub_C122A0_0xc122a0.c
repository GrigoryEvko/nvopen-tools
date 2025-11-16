// Function: sub_C122A0
// Address: 0xc122a0
//
__int64 __fastcall sub_C122A0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  __int64 v3; // rsi
  unsigned int v4; // r12d
  unsigned int v6; // eax
  _BYTE *v7; // rax
  unsigned int v8; // edx
  int v9; // eax
  char v10; // dl
  const char *v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx

  v2 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = a2 & 4;
  if ( (_DWORD)v3 )
    return *(unsigned int *)(v2 + 32);
  if ( (*(_BYTE *)(v2 + 32) & 0xF) == 1 || sub_B2FC80(v2) )
  {
    v4 = 1;
  }
  else
  {
    v4 = 0;
    if ( (*(_BYTE *)(v2 + 32) & 0x30) == 0x10 )
      v4 = ((*(_BYTE *)(v2 + 32) & 0xFu) - 7 > 1) << 9;
  }
  if ( *(_BYTE *)v2 == 3 )
  {
    v6 = v4;
    if ( (*(_BYTE *)(v2 + 80) & 1) != 0 )
    {
      BYTE1(v6) = BYTE1(v4) | 4;
      v4 = v6;
    }
  }
  v7 = (_BYTE *)sub_B32590(v2);
  if ( v7 )
  {
    v8 = v4;
    if ( (*v7 & 0xFD) == 0 )
    {
      BYTE1(v8) = BYTE1(v4) | 8;
      v4 = v8;
    }
  }
  if ( *(_BYTE *)v2 == 1 )
    v4 |= 0x20u;
  v9 = *(_BYTE *)(v2 + 32) & 0xF;
  v10 = *(_BYTE *)(v2 + 32) & 0xF;
  if ( v10 == 8 )
  {
    LOBYTE(v4) = v4 | 0x80;
    goto LABEL_30;
  }
  if ( v9 == 7 )
    goto LABEL_20;
  if ( v9 == 8 )
    goto LABEL_30;
  if ( v10 == 10 )
  {
    v4 |= 0x12u;
    goto LABEL_20;
  }
  v4 |= 2u;
  if ( v9 != 2 )
  {
LABEL_30:
    if ( (unsigned int)(v9 - 3) > 2 )
    {
      if ( v10 == 9 )
        v4 |= 4u;
    }
    else
    {
      v4 |= 4u;
    }
    goto LABEL_20;
  }
  v4 |= 4u;
LABEL_20:
  v11 = sub_BD5D20(v2);
  if ( v12 > 4 && *(_DWORD *)v11 == 1836477548 && v11[4] == 46
    || *(_BYTE *)v2 == 3
    && (*(_BYTE *)(v2 + 35) & 4) != 0
    && (v13 = sub_B31D10(v2, v3, v12), v14 == 13)
    && *(_QWORD *)v13 == 0x74656D2E6D766C6CLL
    && *(_DWORD *)(v13 + 8) == 1952539745
    && *(_BYTE *)(v13 + 12) == 97 )
  {
    LOBYTE(v4) = v4 | 0x80;
  }
  return v4;
}
