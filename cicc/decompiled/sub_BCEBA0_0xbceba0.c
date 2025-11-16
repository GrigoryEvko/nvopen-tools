// Function: sub_BCEBA0
// Address: 0xbceba0
//
__int64 __fastcall sub_BCEBA0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  int v3; // edx
  int v5; // edx
  unsigned __int8 v6; // al

  v2 = *(_BYTE *)(a1 + 8);
  while ( v2 == 16 || (unsigned int)v2 - 17 <= 1 )
  {
    a1 = *(_QWORD *)(a1 + 24);
    v3 = *(unsigned __int8 *)(a1 + 8);
    v2 = *(_BYTE *)(a1 + 8);
    if ( (_BYTE)v3 == 12 || v2 <= 3u || v2 == 5 || (v2 & 0xFD) == 4 || (v2 & 0xFB) == 0xA )
      return 1;
    if ( (unsigned __int8)(v2 - 15) > 3u && v3 != 20 )
      return 0;
  }
  if ( v2 != 20 )
    return ((__int64 (*)(void))sub_BCED00)();
  while ( 1 )
  {
    a1 = sub_BCE9B0(a1);
    v6 = *(_BYTE *)(a1 + 8);
    v5 = v6;
    if ( v6 == 12 || v6 <= 3u || v6 == 5 || (v6 & 0xFD) == 4 || (v6 & 0xFB) == 0xA )
      return 1;
    if ( (unsigned __int8)(v6 - 15) > 3u )
      break;
LABEL_22:
    while ( v6 == 16 || (unsigned int)v6 - 17 <= 1 )
    {
      a1 = *(_QWORD *)(a1 + 24);
      v5 = *(unsigned __int8 *)(a1 + 8);
      v6 = *(_BYTE *)(a1 + 8);
      if ( (_BYTE)v5 == 12 || v6 <= 3u || v6 == 5 || (v6 & 0xFD) == 4 || (v6 & 0xFB) == 0xA )
        return 1;
      if ( (unsigned __int8)(v6 - 15) > 3u )
        goto LABEL_21;
    }
    if ( v6 != 20 )
      return sub_BCED00(a1, a2);
  }
LABEL_21:
  if ( v5 == 20 )
    goto LABEL_22;
  return 0;
}
