// Function: sub_2B0A110
// Address: 0x2b0a110
//
unsigned __int8 **__fastcall sub_2B0A110(unsigned __int8 **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  unsigned __int8 **v4; // rax
  unsigned __int8 **result; // rax

  v2 = (a2 - (__int64)a1) >> 5;
  v3 = (a2 - (__int64)a1) >> 3;
  if ( v2 <= 0 )
  {
LABEL_9:
    if ( v3 != 2 )
    {
      if ( v3 != 3 )
      {
        if ( v3 != 1 )
          return (unsigned __int8 **)a2;
LABEL_22:
        result = a1;
        if ( (unsigned int)**a1 - 12 < 2 )
          return (unsigned __int8 **)a2;
        return result;
      }
      result = a1;
      if ( (unsigned int)**a1 - 12 > 1 )
        return result;
      ++a1;
    }
    result = a1;
    if ( (unsigned int)**a1 - 12 > 1 )
      return result;
    ++a1;
    goto LABEL_22;
  }
  v4 = &a1[4 * v2];
  while ( 1 )
  {
    if ( (unsigned int)**a1 - 12 > 1 )
      return a1;
    if ( (unsigned int)*a1[1] - 12 > 1 )
      return a1 + 1;
    if ( (unsigned int)*a1[2] - 12 > 1 )
      return a1 + 2;
    if ( (unsigned int)*a1[3] - 12 > 1 )
      return a1 + 3;
    a1 += 4;
    if ( v4 == a1 )
    {
      v3 = (a2 - (__int64)a1) >> 3;
      goto LABEL_9;
    }
  }
}
