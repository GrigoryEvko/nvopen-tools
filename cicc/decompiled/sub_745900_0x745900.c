// Function: sub_745900
// Address: 0x745900
//
__int64 __fastcall sub_745900(__int64 a1, __int64 a2)
{
  char v4; // al
  char i; // dl
  __int64 v6; // rcx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx

  while ( 1 )
  {
    while ( 1 )
    {
      v4 = *(_BYTE *)(a1 + 140);
      if ( v4 != 12 )
        break;
      a1 = *(_QWORD *)(a1 + 160);
    }
    for ( i = *(_BYTE *)(a2 + 140); i == 12; i = *(_BYTE *)(a2 + 140) )
      a2 = *(_QWORD *)(a2 + 160);
    if ( a1 == a2 )
      break;
    if ( dword_4F07588 )
    {
      v6 = *(_QWORD *)(a1 + 32);
      if ( *(_QWORD *)(a2 + 32) == v6 )
      {
        if ( v6 )
          break;
      }
    }
    if ( v4 != i )
      return 0;
    if ( v4 == 6 )
    {
      if ( ((*(_BYTE *)(a2 + 168) ^ *(_BYTE *)(a1 + 168)) & 1) != 0 )
        return 0;
      a2 = sub_8D46C0(a2);
      a1 = sub_8D46C0(a1);
    }
    else if ( v4 == 13 )
    {
      v8 = sub_8D4890(a1);
      v9 = sub_8D4890(a2);
      if ( v8 != v9 )
      {
        if ( !v8 )
          return 0;
        if ( !v9 )
          return 0;
        if ( !dword_4F07588 )
          return 0;
        v10 = *(_QWORD *)(v8 + 32);
        if ( *(_QWORD *)(v9 + 32) != v10 || !v10 )
          return 0;
      }
      a2 = sub_8D4870(a2);
      a1 = sub_8D4870(a1);
    }
    else
    {
      if ( dword_4F077C4 != 2
        || v4 != 8
        || (*(_WORD *)(a1 + 168) & 0x180) != 0
        || (*(_WORD *)(a2 + 168) & 0x180) != 0
        || *(_QWORD *)(a1 + 176) != *(_QWORD *)(a2 + 176) )
      {
        return 0;
      }
      a2 = sub_8D4050(a2);
      a1 = sub_8D4050(a1);
    }
  }
  return 1;
}
