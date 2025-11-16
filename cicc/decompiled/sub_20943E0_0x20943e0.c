// Function: sub_20943E0
// Address: 0x20943e0
//
char __fastcall sub_20943E0(__int64 a1)
{
  int v1; // eax
  __int64 v3; // rcx

  v1 = *(unsigned __int16 *)(a1 + 24);
  if ( (unsigned __int16)(v1 - 185) <= 0x35u )
  {
    v3 = 0x3FFFFD00000003LL;
    if ( !_bittest64(&v3, (unsigned int)(v1 - 185)) )
      return (__int16)v1 > 658;
    return 1;
  }
  else
  {
    if ( (unsigned __int16)(v1 - 44) > 1u )
      return (__int16)v1 > 658;
    return (*(_BYTE *)(a1 + 26) & 2) != 0;
  }
}
