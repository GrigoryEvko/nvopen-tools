// Function: sub_BCAF30
// Address: 0xbcaf30
//
__int64 __fastcall sub_BCAF30(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  unsigned __int8 v4; // di
  int v6; // r8d
  char v7; // al
  char v8; // dl
  char v9; // dl
  __int64 v10; // [rsp+10h] [rbp-20h]
  char v11; // [rsp+18h] [rbp-18h]

  if ( a1 == a2 )
    return 1;
  v4 = *(_BYTE *)(a1 + 8);
  LOBYTE(v2) = v4 != 13 && v4 != 7;
  if ( !(_BYTE)v2 )
    return v2;
  v6 = *(unsigned __int8 *)(a2 + 8);
  v7 = *(_BYTE *)(a2 + 8);
  LOBYTE(v2) = (_BYTE)v6 != 7 && (_BYTE)v6 != 13;
  if ( !(_BYTE)v2 )
    return v2;
  if ( (unsigned int)v4 - 17 <= 1 && (unsigned int)(v6 - 17) <= 1 )
  {
    v2 = 0;
    v10 = sub_BCAE30(a2);
    v11 = v8;
    if ( sub_BCAE30(a1) == v10 )
      LOBYTE(v2) = v9 == v11;
    return v2;
  }
  LOBYTE(v2) = v7 == 10 && v4 == 17;
  if ( (_BYTE)v2 )
  {
    if ( sub_BCAE30(a1) == 0x2000 )
      return v2;
  }
  else
  {
    LOBYTE(v2) = v7 == 17 && v4 == 10;
    if ( (_BYTE)v2 && sub_BCAE30(a2) == 0x2000 )
      return v2;
  }
  return 0;
}
