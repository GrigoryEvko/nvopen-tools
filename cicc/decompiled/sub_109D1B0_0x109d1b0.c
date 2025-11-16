// Function: sub_109D1B0
// Address: 0x109d1b0
//
__int64 __fastcall sub_109D1B0(unsigned __int8 *a1)
{
  unsigned __int8 v1; // cl
  __int64 result; // rax
  int v3; // edx

  v1 = *a1;
  if ( *a1 > 0x1Cu )
  {
    if ( v1 > 0x36u )
    {
      if ( v1 == 58 )
        return (a1[1] & 2) != 0;
    }
    else
    {
      result = ((0x40540000000000uLL >> v1) & 1) == 0;
      if ( ((0x40540000000000uLL >> v1) & 1) != 0 )
      {
        if ( v1 != 42 )
          return result;
        if ( (a1[1] & 4) != 0 )
          return 1;
      }
    }
    return 0;
  }
  result = 0;
  if ( v1 == 5 )
  {
    v3 = *((unsigned __int16 *)a1 + 1);
    result = v3 & 0xFFFFFFFD;
    LOBYTE(result) = (*((_WORD *)a1 + 1) & 0xFFF7) == 17 || (v3 & 0xFFFD) == 13;
    if ( (_BYTE)result )
    {
      if ( (_WORD)v3 != 13 )
        return 0;
      if ( (a1[1] & 4) == 0 )
        return 0;
    }
  }
  return result;
}
