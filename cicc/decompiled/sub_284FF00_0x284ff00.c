// Function: sub_284FF00
// Address: 0x284ff00
//
_QWORD *__fastcall sub_284FF00(_QWORD *a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // rdx
  _QWORD *v7; // rcx
  _QWORD *result; // rax
  __int64 v9; // rax
  bool v10; // zf

  v4 = (a2 - (__int64)a1) >> 7;
  v5 = (a2 - (__int64)a1) >> 5;
  if ( v4 > 0 )
  {
    v6 = *a3;
    v7 = &a1[16 * v4];
    while ( *a1 != v6 )
    {
      if ( v6 == a1[4] )
        return a1 + 4;
      if ( v6 == a1[8] )
        return a1 + 8;
      if ( v6 == a1[12] )
        return a1 + 12;
      a1 += 16;
      if ( a1 == v7 )
      {
        v5 = (a2 - (__int64)a1) >> 5;
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
    a1 += 4;
LABEL_21:
    if ( *a1 != v9 )
    {
      a1 += 4;
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
