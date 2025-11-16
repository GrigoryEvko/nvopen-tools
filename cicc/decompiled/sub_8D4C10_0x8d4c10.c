// Function: sub_8D4C10
// Address: 0x8d4c10
//
__int64 __fastcall sub_8D4C10(__int64 a1, int a2)
{
  unsigned int v2; // r8d
  __int64 v3; // rdx
  char v4; // al
  unsigned __int64 v6; // rax
  char v7; // al

  v2 = 0;
  v3 = 6338;
  while ( 1 )
  {
    while ( 1 )
    {
      v4 = *(_BYTE *)(a1 + 140);
      if ( v4 == 12 )
        break;
      if ( v4 == 8 && !a2 )
      {
        a1 = *(_QWORD *)(a1 + 160);
        if ( a1 )
          continue;
      }
      return v2;
    }
    if ( (*(_BYTE *)(a1 + 186) & 8) != 0 )
    {
      v6 = *(unsigned __int8 *)(a1 + 184);
      if ( (unsigned __int8)v6 > 0xCu || !_bittest64(&v3, v6) )
        break;
    }
    v7 = *(_BYTE *)(a1 + 185);
    a1 = *(_QWORD *)(a1 + 160);
    v2 |= v7 & 0x7F;
  }
  return v2;
}
