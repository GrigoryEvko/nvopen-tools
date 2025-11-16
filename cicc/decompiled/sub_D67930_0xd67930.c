// Function: sub_D67930
// Address: 0xd67930
//
_QWORD *__fastcall sub_D67930(_QWORD *a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // rdx
  _QWORD *v7; // rcx
  _QWORD *result; // rax
  __int64 v9; // rax
  bool v10; // zf

  v4 = (a2 - (__int64)a1) >> 5;
  v5 = (a2 - (__int64)a1) >> 3;
  if ( v4 > 0 )
  {
    v6 = *a3;
    v7 = &a1[4 * v4];
    while ( *a1 != v6 )
    {
      if ( v6 == a1[1] )
        return a1 + 1;
      if ( v6 == a1[2] )
        return a1 + 2;
      if ( v6 == a1[3] )
        return a1 + 3;
      a1 += 4;
      if ( a1 == v7 )
      {
        v5 = (a2 - (__int64)a1) >> 3;
        goto LABEL_10;
      }
    }
    return a1;
  }
LABEL_10:
  if ( v5 == 2 )
  {
    v9 = *a3;
    goto LABEL_21;
  }
  if ( v5 == 3 )
  {
    v9 = *a3;
    if ( *a1 == v9 )
      return a1;
    ++a1;
LABEL_21:
    if ( *a1 != v9 )
    {
      ++a1;
      goto LABEL_17;
    }
    return a1;
  }
  if ( v5 != 1 )
    return (_QWORD *)a2;
  v9 = *a3;
LABEL_17:
  v10 = *a1 == v9;
  result = a1;
  if ( !v10 )
    return (_QWORD *)a2;
  return result;
}
