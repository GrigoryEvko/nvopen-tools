// Function: sub_FFECD0
// Address: 0xffecd0
//
__int64 __fastcall sub_FFECD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned __int8 v5; // al
  unsigned __int8 v6; // dl

  v5 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 > 0x1Cu )
  {
    a5 = 0;
    if ( v5 == 60 )
      return sub_B4D040(a1);
    return a5;
  }
  if ( v5 <= 3u )
  {
    v6 = *(_BYTE *)(a1 + 32);
    if ( (v6 & 0xFu) - 7 <= 1 || (((v6 & 0x30) - 16) & 0xE0) == 0 || (a5 = 0, v6 >> 6 == 2) )
      LOBYTE(a5) = (*(_BYTE *)(a1 + 33) & 0x1C) == 0;
    return a5;
  }
  if ( v5 != 22 )
    return 0;
  return sub_B2D680(a1);
}
