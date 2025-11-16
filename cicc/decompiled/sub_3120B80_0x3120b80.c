// Function: sub_3120B80
// Address: 0x3120b80
//
_BYTE **__fastcall sub_3120B80(_BYTE **a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  _BYTE **v4; // rdx
  _BYTE **result; // rax

  v2 = (a2 - (__int64)a1) >> 5;
  v3 = (a2 - (__int64)a1) >> 3;
  if ( v2 <= 0 )
  {
LABEL_11:
    if ( v3 != 2 )
    {
      if ( v3 != 3 )
      {
        if ( v3 != 1 )
          return (_BYTE **)a2;
LABEL_19:
        result = a1;
        if ( **a1 < 0x16u )
          return (_BYTE **)a2;
        return result;
      }
      result = a1;
      if ( **a1 > 0x15u )
        return result;
      ++a1;
    }
    result = a1;
    if ( **a1 > 0x15u )
      return result;
    ++a1;
    goto LABEL_19;
  }
  v4 = &a1[4 * v2];
  while ( 1 )
  {
    if ( **a1 > 0x15u )
      return a1;
    if ( *a1[1] > 0x15u )
      return a1 + 1;
    if ( *a1[2] > 0x15u )
      return a1 + 2;
    if ( *a1[3] > 0x15u )
      return a1 + 3;
    a1 += 4;
    if ( a1 == v4 )
    {
      v3 = (a2 - (__int64)a1) >> 3;
      goto LABEL_11;
    }
  }
}
