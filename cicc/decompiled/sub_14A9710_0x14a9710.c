// Function: sub_14A9710
// Address: 0x14a9710
//
__int64 __fastcall sub_14A9710(__int64 a1)
{
  unsigned int v1; // r12d
  unsigned __int8 v2; // al
  __int64 v4; // rax
  unsigned int v5; // ebx
  unsigned int v6; // r15d
  int v7; // r13d
  __int64 v8; // rax
  char v9; // dl
  unsigned int v10; // r14d

  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 == 13 )
  {
    v1 = *(_DWORD *)(a1 + 32);
    if ( v1 <= 0x40 )
      LOBYTE(v1) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v1) == *(_QWORD *)(a1 + 24);
    else
      LOBYTE(v1) = v1 == (unsigned int)sub_16A58F0(a1 + 24);
    return v1;
  }
  LOBYTE(v1) = v2 <= 0x10u && *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16;
  if ( !(_BYTE)v1 )
    return 0;
  v4 = sub_15A1020(a1);
  if ( !v4 || *(_BYTE *)(v4 + 16) != 13 )
  {
    v6 = 0;
    v7 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( !v7 )
      return v1;
    while ( 1 )
    {
      v8 = sub_15A0A60(a1, v6);
      if ( !v8 )
        break;
      v9 = *(_BYTE *)(v8 + 16);
      if ( v9 != 9 )
      {
        if ( v9 != 13 )
          return 0;
        v10 = *(_DWORD *)(v8 + 32);
        if ( v10 <= 0x40 )
        {
          if ( *(_QWORD *)(v8 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v10) )
            return 0;
        }
        else if ( v10 != (unsigned int)sub_16A58F0(v8 + 24) )
        {
          return 0;
        }
      }
      if ( v7 == ++v6 )
        return v1;
    }
    return 0;
  }
  v5 = *(_DWORD *)(v4 + 32);
  if ( v5 <= 0x40 )
    LOBYTE(v1) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v5) == *(_QWORD *)(v4 + 24);
  else
    LOBYTE(v1) = v5 == (unsigned int)sub_16A58F0(v4 + 24);
  return v1;
}
