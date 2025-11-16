// Function: sub_6EA100
// Address: 0x6ea100
//
_BOOL8 __fastcall sub_6EA100(_BYTE *a1)
{
  _BOOL8 result; // rax

  result = 0;
  if ( (a1[176] & 8) == 0 && a1[136] <= 2u && (a1[140] & 1) == 0 && (dword_4F077C4 == 2 || (a1[169] & 8) != 0) )
  {
    result = 0;
    if ( (*((_WORD *)a1 + 78) & 0x101) != 0x101 )
    {
      result = 1;
      if ( (a1[156] & 2) != 0 )
        return unk_4D046C8 != 0;
    }
  }
  return result;
}
