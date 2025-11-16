// Function: sub_2306580
// Address: 0x2306580
//
__int64 __fastcall sub_2306580(__int64 a1, unsigned __int64 a2)
{
  if ( a2 <= 6 )
  {
    if ( a2 <= 2 )
      return 0;
    goto LABEL_4;
  }
  if ( (*(_DWORD *)a1 != 1634100580 || *(_WORD *)(a1 + 4) != 27765 || *(_BYTE *)(a1 + 6) != 116)
    && (*(_DWORD *)a1 != 1852401780 || *(_WORD *)(a1 + 4) != 29804 || *(_BYTE *)(a1 + 6) != 111) )
  {
LABEL_4:
    if ( *(_WORD *)a1 != 29804 || *(_BYTE *)(a1 + 2) != 111 )
      return 0;
  }
  return 1;
}
