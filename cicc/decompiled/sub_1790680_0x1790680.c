// Function: sub_1790680
// Address: 0x1790680
//
char __fastcall sub_1790680(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rax
  _BYTE *v6; // rbx
  unsigned int v7; // r12d
  __int64 v9; // rax
  unsigned int v10; // ebx
  unsigned int v11; // r14d
  int v12; // r12d
  __int64 v13; // rax
  char v14; // dl
  unsigned int v15; // r15d

  v4 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( !v4 )
    return 0;
  **a1 = v4;
  v5 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v6 = *(_BYTE **)(a2 - 24 * v5);
  if ( v6[16] == 13 )
  {
    v7 = *((_DWORD *)v6 + 8);
    if ( v7 <= 0x40 )
      return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v7) == *((_QWORD *)v6 + 3);
    else
      return v7 == (unsigned int)sub_16A58F0((__int64)(v6 + 24));
  }
  if ( *(_BYTE *)(*(_QWORD *)v6 + 8LL) != 16 )
    return 0;
  v9 = sub_15A1020(v6, a2, 4 * v5, a4);
  if ( !v9 || *(_BYTE *)(v9 + 16) != 13 )
  {
    v11 = 0;
    v12 = *(_QWORD *)(*(_QWORD *)v6 + 32LL);
    if ( !v12 )
      return 1;
    while ( 1 )
    {
      v13 = sub_15A0A60((__int64)v6, v11);
      if ( !v13 )
        break;
      v14 = *(_BYTE *)(v13 + 16);
      if ( v14 != 9 )
      {
        if ( v14 != 13 )
          return 0;
        v15 = *(_DWORD *)(v13 + 32);
        if ( v15 <= 0x40 )
        {
          if ( *(_QWORD *)(v13 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v15) )
            return 0;
        }
        else if ( v15 != (unsigned int)sub_16A58F0(v13 + 24) )
        {
          return 0;
        }
      }
      if ( v12 == ++v11 )
        return 1;
    }
    return 0;
  }
  v10 = *(_DWORD *)(v9 + 32);
  if ( v10 <= 0x40 )
    return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v10) == *(_QWORD *)(v9 + 24);
  else
    return v10 == (unsigned int)sub_16A58F0(v9 + 24);
}
