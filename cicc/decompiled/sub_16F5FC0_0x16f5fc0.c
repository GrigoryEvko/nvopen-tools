// Function: sub_16F5FC0
// Address: 0x16f5fc0
//
__int64 __fastcall sub_16F5FC0(__int64 a1, unsigned __int64 a2)
{
  int v3; // eax

  if ( a2 <= 4 )
  {
    if ( a2 > 2 && *(_WORD *)a1 == 29281 && *(_BYTE *)(a1 + 2) == 109 )
      return (unsigned int)(*(_WORD *)(a1 + a2 - 2) == 25189) + 1;
    return 0;
  }
  if ( *(_DWORD *)a1 == 1701671521 && *(_BYTE *)(a1 + 4) == 98 )
    return 2;
  if ( a2 <= 6 )
    goto LABEL_35;
  if ( *(_DWORD *)a1 == 1836410996 && *(_WORD *)(a1 + 4) == 25954 && *(_BYTE *)(a1 + 6) == 98 )
    return 2;
  if ( a2 <= 9 )
  {
LABEL_35:
    if ( !memcmp((const void *)a1, &unk_3F8856D, 3u) || *(_DWORD *)a1 == 1836410996 && *(_BYTE *)(a1 + 4) == 98 )
      return (unsigned int)(*(_WORD *)(a1 + a2 - 2) == 25189) + 1;
    if ( a2 <= 6 )
      return 0;
  }
  else
  {
    if ( *(_QWORD *)a1 == 0x5F34366863726161LL && *(_WORD *)(a1 + 8) == 25954 )
      return 2;
    if ( *(_WORD *)a1 == 29281 && *(_BYTE *)(a1 + 2) == 109 || *(_DWORD *)a1 == 1836410996 && *(_BYTE *)(a1 + 4) == 98 )
      return (unsigned int)(*(_WORD *)(a1 + a2 - 2) == 25189) + 1;
  }
  if ( *(_DWORD *)a1 != 1668440417 || *(_WORD *)(a1 + 4) != 13928 || (v3 = 0, *(_BYTE *)(a1 + 6) != 52) )
    v3 = 1;
  return v3 ^ 1u;
}
