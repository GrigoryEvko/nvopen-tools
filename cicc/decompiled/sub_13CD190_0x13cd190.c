// Function: sub_13CD190
// Address: 0x13cd190
//
char __fastcall sub_13CD190(__int64 a1)
{
  unsigned int v2; // r12d
  __int64 v3; // rax
  unsigned int v4; // ebx
  int v5; // r12d
  unsigned int v6; // r13d
  __int64 v7; // rax
  char v8; // dl
  unsigned int v9; // r14d

  if ( *(_BYTE *)(a1 + 16) > 0x10u )
    return 0;
  if ( (unsigned __int8)sub_1593BB0(a1) )
    return 1;
  if ( *(_BYTE *)(a1 + 16) == 13 )
  {
    v2 = *(_DWORD *)(a1 + 32);
    if ( v2 <= 0x40 )
      return *(_QWORD *)(a1 + 24) == 0;
    else
      return v2 == (unsigned int)sub_16A57B0(a1 + 24);
  }
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 )
    return 0;
  v3 = sub_15A1020(a1);
  if ( !v3 || *(_BYTE *)(v3 + 16) != 13 )
  {
    v5 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( !v5 )
      return 1;
    v6 = 0;
    while ( 1 )
    {
      v7 = sub_15A0A60(a1, v6);
      if ( !v7 )
        break;
      v8 = *(_BYTE *)(v7 + 16);
      if ( v8 != 9 )
      {
        if ( v8 != 13 )
          return 0;
        v9 = *(_DWORD *)(v7 + 32);
        if ( v9 <= 0x40 )
        {
          if ( *(_QWORD *)(v7 + 24) )
            return 0;
        }
        else if ( v9 != (unsigned int)sub_16A57B0(v7 + 24) )
        {
          return 0;
        }
      }
      if ( v5 == ++v6 )
        return 1;
    }
    return 0;
  }
  v4 = *(_DWORD *)(v3 + 32);
  if ( v4 <= 0x40 )
    return *(_QWORD *)(v3 + 24) == 0;
  else
    return v4 == (unsigned int)sub_16A57B0(v3 + 24);
}
