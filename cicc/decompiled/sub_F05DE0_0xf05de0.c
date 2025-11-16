// Function: sub_F05DE0
// Address: 0xf05de0
//
__int64 __fastcall sub_F05DE0(__int64 a1, unsigned __int64 a2)
{
  if ( a2 <= 4 )
  {
    if ( a2 > 2 && *(_WORD *)a1 == 29281 && *(_BYTE *)(a1 + 2) == 109 )
      return (unsigned int)(*(_WORD *)(a1 + a2 - 2) == 25189) + 1;
    return 0;
  }
  if ( *(_DWORD *)a1 == 1701671521 && *(_BYTE *)(a1 + 4) == 98 )
    return 2;
  if ( a2 <= 6 )
    goto LABEL_37;
  if ( *(_DWORD *)a1 == 1836410996 && *(_WORD *)(a1 + 4) == 25954 && *(_BYTE *)(a1 + 6) == 98 )
    return 2;
  if ( a2 <= 9 )
  {
LABEL_37:
    if ( !memcmp((const void *)a1, &unk_3F8856D, 3u) || *(_DWORD *)a1 == 1836410996 && *(_BYTE *)(a1 + 4) == 98 )
      return (unsigned int)(*(_WORD *)(a1 + a2 - 2) == 25189) + 1;
    if ( a2 > 6 )
    {
LABEL_9:
      if ( *(_DWORD *)a1 == 1668440417 && *(_WORD *)(a1 + 4) == 13928 && *(_BYTE *)(a1 + 6) == 52
        || a2 > 9 && *(_QWORD *)a1 == 0x5F34366863726161LL && *(_WORD *)(a1 + 8) == 12851 )
      {
        return 1;
      }
    }
    return 0;
  }
  if ( *(_QWORD *)a1 != 0x5F34366863726161LL || *(_WORD *)(a1 + 8) != 25954 )
  {
    if ( *(_WORD *)a1 == 29281 && *(_BYTE *)(a1 + 2) == 109 || *(_DWORD *)a1 == 1836410996 && *(_BYTE *)(a1 + 4) == 98 )
      return (unsigned int)(*(_WORD *)(a1 + a2 - 2) == 25189) + 1;
    goto LABEL_9;
  }
  return 2;
}
