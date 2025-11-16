// Function: sub_9B76D0
// Address: 0x9b76d0
//
char __fastcall sub_9B76D0(__int64 a1, unsigned int a2, __int64 a3)
{
  if ( a3 && (unsigned __int8)sub_B60C40(a1) )
    return sub_DFAA70(a3, (unsigned int)a1, a2);
  if ( (unsigned __int8)sub_B5B010((unsigned int)a1) )
    return a2 + 1 <= 1;
  if ( (_DWORD)a1 == 285 )
    return ((a2 + 1) & 0xFFFFFFFD) == 0;
  if ( (unsigned int)a1 <= 0x11D )
  {
    if ( (_DWORD)a1 != 212 )
    {
      if ( (unsigned int)a1 <= 0xD4 )
      {
        if ( (unsigned int)a1 > 0xB0 )
        {
          if ( (_DWORD)a1 == 207 )
            return a2 == 0;
          return a2 == -1;
        }
        if ( (unsigned int)a1 <= 0xAE )
          return a2 == -1;
      }
      else if ( (_DWORD)a1 != 223 )
      {
        if ( (_DWORD)a1 != 249 )
          return a2 == -1;
        return a2 == 0;
      }
    }
    return a2 + 1 <= 1;
  }
  if ( (_DWORD)a1 == 436 )
    return a2 == 0;
  if ( (unsigned int)a1 > 0x1B4 )
  {
    if ( (a1 & 0xFFFFFFFD) != 0x1B5 )
      return a2 == -1;
    return a2 + 1 <= 1;
  }
  if ( (unsigned int)a1 <= 0x147 )
  {
    if ( (unsigned int)a1 <= 0x145 )
    {
      if ( (_DWORD)a1 != 313 )
        return a2 == -1;
      return a2 + 1 <= 1;
    }
    return a2 == 0;
  }
  if ( (_DWORD)a1 != 362 )
    return a2 == -1;
  return a2 + 1 <= 1;
}
