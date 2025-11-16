// Function: sub_1D13A20
// Address: 0x1d13a20
//
__int64 __fastcall sub_1D13A20(__int64 a1, __int64 a2)
{
  unsigned int v2; // edx
  __int64 result; // rax

  v2 = 8 * sub_15A9520(a2, 0);
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
