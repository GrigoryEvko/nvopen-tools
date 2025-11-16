// Function: sub_130CCD0
// Address: 0x130ccd0
//
int __fastcall sub_130CCD0(_BYTE *a1, _BYTE *a2)
{
  size_t v3; // rsi
  int result; // eax

  if ( a1 )
  {
    if ( a2 )
    {
      v3 = a2 + 4096 - a1;
      if ( v3 <= 0x4000 )
        return mprotect(a1, v3, 3);
    }
    result = mprotect(a1, 0x1000u, 3);
  }
  if ( a2 )
    return mprotect(a2, 0x1000u, 3);
  return result;
}
