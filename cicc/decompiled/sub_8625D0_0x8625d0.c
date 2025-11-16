// Function: sub_8625D0
// Address: 0x8625d0
//
_BOOL8 __fastcall sub_8625D0(__int64 a1)
{
  _BOOL8 result; // rax

  result = 1;
  if ( (*(_BYTE *)(a1 + 193) & 2) == 0 )
  {
    result = 0;
    if ( *(char *)(a1 + 192) < 0 )
      return dword_4D04380 != 0;
  }
  return result;
}
