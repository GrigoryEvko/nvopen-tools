// Function: sub_5EA830
// Address: 0x5ea830
//
__int64 **__fastcall sub_5EA830(__int64 ***a1, __int64 *a2, __int64 *a3)
{
  __int64 **result; // rax
  __int64 *v4; // rdx

  result = *a1;
  if ( a2 )
  {
    for ( ; result; result = (__int64 **)*result )
    {
      if ( ((_BYTE)result[4] & 3) == 0 && result[1] == a2 )
        break;
    }
  }
  else if ( a3 )
  {
    for ( ; result; result = (__int64 **)*result )
    {
      if ( ((_BYTE)result[4] & 1) != 0 && a3 == result[3] )
        break;
      if ( ((_BYTE)result[4] & 2) != 0 && a3 == result[1] )
        break;
    }
  }
  else
  {
    while ( result )
    {
      if ( ((_BYTE)result[4] & 4) != 0 )
        break;
      if ( ((_WORD)result[4] & 0x201) == 0 )
      {
        v4 = result[2];
        if ( v4 )
        {
          if ( (*((_BYTE *)v4 + 145) & 1) != 0 )
            break;
        }
      }
      result = (__int64 **)*result;
    }
  }
  return result;
}
