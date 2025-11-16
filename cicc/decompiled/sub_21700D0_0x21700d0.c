// Function: sub_21700D0
// Address: 0x21700d0
//
__int64 __fastcall sub_21700D0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // edx
  __int64 result; // rax

  v2 = 8 * sub_15A9520(a1, a2);
  if ( v2 == 32 )
    return 5;
  if ( v2 > 0x20 )
  {
    result = 6;
    if ( v2 != 64 )
    {
      result = 0;
      if ( v2 == 128 )
        return 7;
    }
  }
  else
  {
    result = 3;
    if ( v2 != 8 )
    {
      LOBYTE(result) = v2 == 16;
      return (unsigned int)(4 * result);
    }
  }
  return result;
}
