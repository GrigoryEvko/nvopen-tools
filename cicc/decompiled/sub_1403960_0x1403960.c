// Function: sub_1403960
// Address: 0x1403960
//
_QWORD *__fastcall sub_1403960(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *result; // rax

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
          return (_QWORD *)a2;
LABEL_19:
        result = a1;
        if ( !*a1 )
          return (_QWORD *)a2;
        return result;
      }
      result = a1;
      if ( *a1 )
        return result;
      ++a1;
    }
    result = a1;
    if ( *a1 )
      return result;
    ++a1;
    goto LABEL_19;
  }
  v4 = &a1[4 * v2];
  while ( 1 )
  {
    if ( *a1 )
      return a1;
    if ( a1[1] )
      return a1 + 1;
    if ( a1[2] )
      return a1 + 2;
    if ( a1[3] )
      return a1 + 3;
    a1 += 4;
    if ( a1 == v4 )
    {
      v3 = (a2 - (__int64)a1) >> 3;
      goto LABEL_11;
    }
  }
}
