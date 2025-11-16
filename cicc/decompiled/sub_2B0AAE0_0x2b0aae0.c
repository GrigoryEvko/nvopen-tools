// Function: sub_2B0AAE0
// Address: 0x2b0aae0
//
_BYTE **__fastcall sub_2B0AAE0(_BYTE **a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  _BYTE **v4; // rdx
  _BYTE **result; // rax

  v2 = (a2 - (__int64)a1) >> 5;
  v3 = (a2 - (__int64)a1) >> 3;
  if ( v2 > 0 )
  {
    v4 = &a1[4 * v2];
    while ( **a1 != 86 )
    {
      if ( *a1[1] == 86 )
        return a1 + 1;
      if ( *a1[2] == 86 )
        return a1 + 2;
      if ( *a1[3] == 86 )
        return a1 + 3;
      a1 += 4;
      if ( a1 == v4 )
      {
        v3 = (a2 - (__int64)a1) >> 3;
        goto LABEL_9;
      }
    }
    return a1;
  }
LABEL_9:
  if ( v3 != 2 )
  {
    if ( v3 != 3 )
    {
      if ( v3 != 1 )
        return (_BYTE **)a2;
      goto LABEL_21;
    }
    if ( **a1 == 86 )
      return a1;
    ++a1;
  }
  if ( **a1 == 86 )
    return a1;
  ++a1;
LABEL_21:
  result = a1;
  if ( **a1 != 86 )
    return (_BYTE **)a2;
  return result;
}
