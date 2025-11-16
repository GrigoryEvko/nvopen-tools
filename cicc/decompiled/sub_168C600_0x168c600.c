// Function: sub_168C600
// Address: 0x168c600
//
__int64 __fastcall sub_168C600(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  unsigned int v3; // r12d
  unsigned int v5; // eax
  __int64 v6; // rax
  int v7; // eax
  char v8; // dl
  const char *v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int64 v13; // rdx

  v2 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (a2 & 4) != 0 )
    return *(unsigned int *)(v2 + 32);
  if ( (*(_BYTE *)(v2 + 32) & 0xF) == 1 || sub_15E4F60(v2) )
  {
    v3 = 1;
  }
  else
  {
    v3 = 0;
    if ( (*(_BYTE *)(v2 + 32) & 0x30) == 0x10 )
      v3 = ((*(_BYTE *)(v2 + 32) & 0xFu) - 7 > 1) << 9;
  }
  if ( *(_BYTE *)(v2 + 16) == 3 )
  {
    v5 = v3;
    if ( (*(_BYTE *)(v2 + 80) & 1) != 0 )
    {
      BYTE1(v5) = BYTE1(v3) | 4;
      v3 = v5;
    }
  }
  v6 = sub_15E4FA0(v2);
  if ( v6 && !*(_BYTE *)(v6 + 16) )
    v3 |= 0x800u;
  if ( *(_BYTE *)(v2 + 16) == 1 )
    v3 |= 0x20u;
  v7 = *(_BYTE *)(v2 + 32) & 0xF;
  v8 = *(_BYTE *)(v2 + 32) & 0xF;
  if ( v8 == 8 )
  {
    LOBYTE(v3) = v3 | 0x80;
  }
  else if ( v7 != 7 && v7 != 8 )
  {
    if ( v8 == 10 )
    {
      v3 |= 0x12u;
      v9 = sub_1649960(v2);
      if ( v10 <= 4 )
        goto LABEL_20;
      goto LABEL_31;
    }
    if ( (unsigned int)(v7 - 2) <= 3 || v8 == 9 )
      v3 |= 6u;
    else
      v3 |= 2u;
  }
  v9 = sub_1649960(v2);
  if ( v13 <= 4 )
    goto LABEL_20;
LABEL_31:
  if ( *(_DWORD *)v9 == 1836477548 && v9[4] == 46 )
    goto LABEL_26;
LABEL_20:
  if ( *(_BYTE *)(v2 + 16) == 3 && (*(_BYTE *)(v2 + 34) & 0x20) != 0 )
  {
    v11 = sub_15E61A0(v2);
    if ( v12 == 13
      && *(_QWORD *)v11 == 0x74656D2E6D766C6CLL
      && *(_DWORD *)(v11 + 8) == 1952539745
      && *(_BYTE *)(v11 + 12) == 97 )
    {
LABEL_26:
      LOBYTE(v3) = v3 | 0x80;
    }
  }
  return v3;
}
