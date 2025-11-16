// Function: sub_14CA6E0
// Address: 0x14ca6e0
//
char __fastcall sub_14CA6E0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  unsigned int v4; // r12d
  __int64 v6; // rax
  unsigned int v7; // ebx
  unsigned int v8; // r14d
  int v9; // r12d
  __int64 v10; // rax
  char v11; // dl
  unsigned int v12; // r15d

  v2 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( !v2 )
    return 0;
  **a1 = v2;
  v3 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( *(_BYTE *)(v3 + 16) == 13 )
  {
    v4 = *(_DWORD *)(v3 + 32);
    if ( v4 <= 0x40 )
      return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v4) == *(_QWORD *)(v3 + 24);
    else
      return v4 == (unsigned int)sub_16A58F0(v3 + 24);
  }
  if ( *(_BYTE *)(*(_QWORD *)v3 + 8LL) != 16 )
    return 0;
  v6 = sub_15A1020(v3);
  if ( !v6 || *(_BYTE *)(v6 + 16) != 13 )
  {
    v8 = 0;
    v9 = *(_QWORD *)(*(_QWORD *)v3 + 32LL);
    if ( !v9 )
      return 1;
    while ( 1 )
    {
      v10 = sub_15A0A60(v3, v8);
      if ( !v10 )
        break;
      v11 = *(_BYTE *)(v10 + 16);
      if ( v11 != 9 )
      {
        if ( v11 != 13 )
          return 0;
        v12 = *(_DWORD *)(v10 + 32);
        if ( v12 <= 0x40 )
        {
          if ( *(_QWORD *)(v10 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v12) )
            return 0;
        }
        else if ( v12 != (unsigned int)sub_16A58F0(v10 + 24) )
        {
          return 0;
        }
      }
      if ( v9 == ++v8 )
        return 1;
    }
    return 0;
  }
  v7 = *(_DWORD *)(v6 + 32);
  if ( v7 <= 0x40 )
    return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v7) == *(_QWORD *)(v6 + 24);
  else
    return v7 == (unsigned int)sub_16A58F0(v6 + 24);
}
