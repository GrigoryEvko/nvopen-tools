// Function: sub_8DAAE0
// Address: 0x8daae0
//
__int64 __fastcall sub_8DAAE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // rbx
  unsigned int v7; // r8d
  char v9; // al

  v5 = a2;
  v6 = a1;
  if ( a1 == a2 || (v7 = sub_8D97D0(a1, a2, 0, a4, a5)) != 0 )
  {
    v7 = 1;
    if ( dword_4F077C4 != 2 )
    {
      while ( 1 )
      {
        v9 = *(_BYTE *)(v6 + 140);
        if ( v9 != 12 )
          break;
        v6 = *(_QWORD *)(v6 + 160);
      }
      if ( *(_BYTE *)(a2 + 140) == 12 )
      {
        do
          v5 = *(_QWORD *)(v5 + 160);
        while ( *(_BYTE *)(v5 + 140) == 12 );
      }
      v7 = 1;
      if ( v9 == 2 && (*(_BYTE *)(v6 + 161) & 8) == 0 && (*(_BYTE *)(v5 + 161) & 8) == 0 )
        return *(_QWORD *)(v6 + 168) == *(_QWORD *)(v5 + 168);
    }
  }
  return v7;
}
