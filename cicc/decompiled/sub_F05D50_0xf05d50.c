// Function: sub_F05D50
// Address: 0xf05d50
//
__int64 __fastcall sub_F05D50(__int64 a1, unsigned __int64 a2)
{
  if ( a2 > 6 )
  {
    if ( *(_DWORD *)a1 == 1668440417 && *(_WORD *)(a1 + 4) == 13928 && *(_BYTE *)(a1 + 6) == 52 )
      return 3;
LABEL_3:
    if ( *(_DWORD *)a1 == 913142369 && *(_BYTE *)(a1 + 4) == 52 )
      return 3;
    if ( *(_DWORD *)a1 == 1836410996 && *(_BYTE *)(a1 + 4) == 98 )
      return 2;
    return *(_WORD *)a1 == 29281 && *(_BYTE *)(a1 + 2) == 109;
  }
  if ( a2 > 4 )
    goto LABEL_3;
  if ( a2 <= 2 )
    return 0;
  return *(_WORD *)a1 == 29281 && *(_BYTE *)(a1 + 2) == 109;
}
