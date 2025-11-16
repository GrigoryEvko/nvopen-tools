// Function: sub_15B11B0
// Address: 0x15b11b0
//
__int64 __fastcall sub_15B11B0(unsigned __int64 **a1)
{
  unsigned __int64 v1; // rdx
  __int64 result; // rax

  v1 = **a1;
  result = 3;
  if ( v1 != 4096 )
  {
    result = 1;
    if ( v1 <= 0x1000 )
    {
      result = 2;
      if ( v1 != 35 )
      {
        if ( v1 <= 0x23 )
          return (unsigned int)(v1 == 16) + 1;
        else
          return (unsigned int)(v1 - 147 < 3) + 1;
      }
    }
  }
  return result;
}
