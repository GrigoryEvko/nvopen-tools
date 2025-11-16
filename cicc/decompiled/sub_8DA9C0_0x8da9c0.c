// Function: sub_8DA9C0
// Address: 0x8da9c0
//
__int64 __fastcall sub_8DA9C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  unsigned int v5; // r8d
  __int64 v7; // rdi

  while ( 1 )
  {
    v4 = *(_BYTE *)(a1 + 140);
    if ( v4 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  v5 = 0;
  if ( v4 != 6 )
    return v5;
  if ( (*(_BYTE *)(a1 + 168) & 1) != 0 )
    return v5;
  v7 = *(_QWORD *)(a1 + 160);
  v5 = 1;
  if ( v7 == a2 )
    return v5;
  else
    return (unsigned int)sub_8D97D0(v7, a2, 0, a4, 1) != 0;
}
