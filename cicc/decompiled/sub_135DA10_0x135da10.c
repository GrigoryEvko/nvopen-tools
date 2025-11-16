// Function: sub_135DA10
// Address: 0x135da10
//
__int64 __fastcall sub_135DA10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned __int8 v5; // al
  bool v7; // dl

  v5 = *(_BYTE *)(a1 + 16);
  if ( v5 <= 0x17u )
  {
    v7 = 0;
  }
  else
  {
    if ( v5 != 78 && v5 != 29 )
      goto LABEL_4;
    v7 = (a1 & 0xFFFFFFFFFFFFFFF8LL) != 0;
  }
  if ( !v7 && v5 != 17 )
  {
LABEL_4:
    LOBYTE(a5) = v5 == 54;
    return a5;
  }
  LOBYTE(a5) = v7 || v5 == 17;
  return a5;
}
