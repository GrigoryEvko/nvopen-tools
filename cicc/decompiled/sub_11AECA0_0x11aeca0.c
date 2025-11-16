// Function: sub_11AECA0
// Address: 0x11aeca0
//
_DWORD *__fastcall sub_11AECA0(_DWORD *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  _DWORD *v4; // rdx
  _DWORD *result; // rax

  v2 = (a2 - (__int64)a1) >> 4;
  v3 = (a2 - (__int64)a1) >> 2;
  if ( v2 <= 0 )
  {
LABEL_11:
    if ( v3 != 2 )
    {
      if ( v3 != 3 )
      {
        if ( v3 != 1 )
          return (_DWORD *)a2;
LABEL_19:
        result = a1;
        if ( (unsigned int)(*a1 + 1) < 2 )
          return (_DWORD *)a2;
        return result;
      }
      result = a1;
      if ( (unsigned int)(*a1 + 1) > 1 )
        return result;
      ++a1;
    }
    result = a1;
    if ( (unsigned int)(*a1 + 1) > 1 )
      return result;
    ++a1;
    goto LABEL_19;
  }
  v4 = &a1[4 * v2];
  while ( 1 )
  {
    if ( (unsigned int)(*a1 + 1) > 1 )
      return a1;
    if ( (unsigned int)(a1[1] + 1) > 1 )
      return a1 + 1;
    if ( (unsigned int)(a1[2] + 1) > 1 )
      return a1 + 2;
    if ( (unsigned int)(a1[3] + 1) > 1 )
      return a1 + 3;
    a1 += 4;
    if ( v4 == a1 )
    {
      v3 = (a2 - (__int64)a1) >> 2;
      goto LABEL_11;
    }
  }
}
