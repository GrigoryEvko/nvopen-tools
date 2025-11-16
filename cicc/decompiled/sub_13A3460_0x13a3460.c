// Function: sub_13A3460
// Address: 0x13a3460
//
_BOOL8 __fastcall sub_13A3460(__int64 a1)
{
  char v1; // dl
  unsigned int v2; // edx
  _BOOL8 result; // rax
  unsigned int v4; // ecx

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 == 54 )
  {
    v2 = *(unsigned __int16 *)(a1 + 18);
    result = !(v2 & 1);
    if ( ((v2 >> 7) & 6) != 0 )
      return 0;
  }
  else
  {
    result = 0;
    if ( v1 == 55 )
    {
      v4 = *(unsigned __int16 *)(a1 + 18);
      if ( ((v4 >> 7) & 6) == 0 )
        return !(v4 & 1);
    }
  }
  return result;
}
