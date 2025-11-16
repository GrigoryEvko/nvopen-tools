// Function: sub_176A780
// Address: 0x176a780
//
char __fastcall sub_176A780(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  __int64 v6; // rax
  _BYTE *v7; // rbx
  unsigned int v8; // r12d
  __int64 v9; // rax
  unsigned int v10; // ebx
  unsigned int v11; // r14d
  int v12; // r12d
  __int64 v13; // rax
  char v14; // dl
  unsigned int v15; // r15d

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 35 )
  {
    if ( *a1 != *(_QWORD *)(a2 - 48) )
      return 0;
    return sub_1757CC0(*(_BYTE **)(a2 - 24), a2, a3, a4);
  }
  else
  {
    if ( v4 != 5 || *(_WORD *)(a2 + 18) != 11 )
      return 0;
    v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( *a1 != *(_QWORD *)(a2 - 24 * v6) )
      return 0;
    v7 = *(_BYTE **)(a2 + 24 * (1 - v6));
    if ( v7[16] == 13 )
    {
      v8 = *((_DWORD *)v7 + 8);
      if ( v8 <= 0x40 )
        return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v8) == *((_QWORD *)v7 + 3);
      else
        return v8 == (unsigned int)sub_16A58F0((__int64)(v7 + 24));
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) != 16 )
        return 0;
      v9 = sub_15A1020(v7, a2, 1 - v6, 4 * v6);
      if ( !v9 || *(_BYTE *)(v9 + 16) != 13 )
      {
        v11 = 0;
        v12 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
        if ( !v12 )
          return 1;
        while ( 1 )
        {
          v13 = sub_15A0A60((__int64)v7, v11);
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
  }
}
