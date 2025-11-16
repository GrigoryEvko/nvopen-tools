// Function: sub_1757E30
// Address: 0x1757e30
//
char __fastcall sub_1757E30(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  __int64 v6; // rax
  unsigned int v7; // ebx
  unsigned int v8; // r14d
  int v9; // r12d
  __int64 v10; // rax
  char v11; // dl
  unsigned int v12; // r15d

  if ( a1[16] == 13 )
  {
    v4 = *((_DWORD *)a1 + 8);
    if ( v4 <= 0x40 )
      return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v4) == *((_QWORD *)a1 + 3);
    else
      return v4 == (unsigned int)sub_16A58F0((__int64)(a1 + 24));
  }
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 )
    return 0;
  v6 = sub_15A1020(a1, a2, a3, a4);
  if ( !v6 || *(_BYTE *)(v6 + 16) != 13 )
  {
    v8 = 0;
    v9 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( !v9 )
      return 1;
    while ( 1 )
    {
      v10 = sub_15A0A60((__int64)a1, v8);
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
