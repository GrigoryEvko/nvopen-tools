// Function: sub_13CC690
// Address: 0x13cc690
//
char __fastcall sub_13CC690(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 v3; // rax
  unsigned int v4; // ebx
  unsigned int v5; // r14d
  int v6; // r12d
  __int64 v7; // rax
  char v8; // dl
  unsigned int v9; // r15d

  if ( *(_BYTE *)(a1 + 16) == 13 )
  {
    v1 = *(_DWORD *)(a1 + 32);
    if ( v1 <= 0x40 )
      return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v1) == *(_QWORD *)(a1 + 24);
    else
      return v1 == (unsigned int)sub_16A58F0(a1 + 24);
  }
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 )
    return 0;
  v3 = sub_15A1020();
  if ( !v3 || *(_BYTE *)(v3 + 16) != 13 )
  {
    v5 = 0;
    v6 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( !v6 )
      return 1;
    while ( 1 )
    {
      v7 = sub_15A0A60(a1, v5);
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
          if ( *(_QWORD *)(v7 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9) )
            return 0;
        }
        else if ( v9 != (unsigned int)sub_16A58F0(v7 + 24) )
        {
          return 0;
        }
      }
      if ( v6 == ++v5 )
        return 1;
    }
    return 0;
  }
  v4 = *(_DWORD *)(v3 + 32);
  if ( v4 <= 0x40 )
    return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v4) == *(_QWORD *)(v3 + 24);
  else
    return v4 == (unsigned int)sub_16A58F0(v3 + 24);
}
