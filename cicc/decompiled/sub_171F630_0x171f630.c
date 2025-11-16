// Function: sub_171F630
// Address: 0x171f630
//
__int64 __fastcall sub_171F630(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  int v7; // eax
  __int64 v8; // rax
  __int64 *v9; // rbx
  unsigned __int8 v10; // al
  __int64 v11; // rax
  unsigned int v12; // ebx
  unsigned int v13; // r14d
  int v14; // r13d
  __int64 v15; // rax
  char v16; // dl
  unsigned int v17; // r15d

  if ( *(_BYTE *)(a2 + 16) != 79 )
    return 0;
  v4 = *(_QWORD *)(a2 - 72);
  if ( *(_BYTE *)(v4 + 16) != 75 )
    return 0;
  v5 = *(_QWORD *)(v4 - 48);
  if ( !v5 )
    return 0;
  **(_QWORD **)(a1 + 8) = v5;
  v6 = *(_QWORD *)(v4 - 24);
  if ( *(_BYTE *)(v6 + 16) != 13 )
    return 0;
  **(_QWORD **)(a1 + 16) = v6;
  v7 = *(unsigned __int16 *)(v4 + 18);
  BYTE1(v7) &= ~0x80u;
  **(_DWORD **)a1 = v7;
  v8 = *(_QWORD *)(a2 - 48);
  if ( !v8 )
    return 0;
  **(_QWORD **)(a1 + 24) = v8;
  v9 = *(__int64 **)(a2 - 24);
  v10 = *((_BYTE *)v9 + 16);
  if ( v10 == 13 )
  {
    v2 = *((_DWORD *)v9 + 8);
    if ( v2 <= 0x40 )
      LOBYTE(v2) = v9[3] == 0;
    else
      LOBYTE(v2) = v2 == (unsigned int)sub_16A57B0((__int64)(v9 + 3));
    return v2;
  }
  LOBYTE(v2) = v10 <= 0x10u && *(_BYTE *)(*v9 + 8) == 16;
  if ( !(_BYTE)v2 )
    return 0;
  v11 = sub_15A1020(*(_BYTE **)(a2 - 24), a2, *v9, v6);
  if ( !v11 || *(_BYTE *)(v11 + 16) != 13 )
  {
    v13 = 0;
    v14 = *(_QWORD *)(*v9 + 32);
    if ( !v14 )
      return v2;
    while ( 1 )
    {
      v15 = sub_15A0A60((__int64)v9, v13);
      if ( !v15 )
        break;
      v16 = *(_BYTE *)(v15 + 16);
      if ( v16 != 9 )
      {
        if ( v16 != 13 )
          return 0;
        v17 = *(_DWORD *)(v15 + 32);
        if ( v17 <= 0x40 )
        {
          if ( *(_QWORD *)(v15 + 24) )
            return 0;
        }
        else if ( v17 != (unsigned int)sub_16A57B0(v15 + 24) )
        {
          return 0;
        }
      }
      if ( v14 == ++v13 )
        return v2;
    }
    return 0;
  }
  v12 = *(_DWORD *)(v11 + 32);
  if ( v12 <= 0x40 )
    LOBYTE(v2) = *(_QWORD *)(v11 + 24) == 0;
  else
    LOBYTE(v2) = v12 == (unsigned int)sub_16A57B0(v11 + 24);
  return v2;
}
