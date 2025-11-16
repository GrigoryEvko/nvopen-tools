// Function: sub_D46490
// Address: 0xd46490
//
_QWORD *__fastcall sub_D46490(_QWORD *a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // rdx
  _QWORD *v7; // rcx
  __int64 v9; // rax

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
      if ( v7 == a1 )
      {
        v5 = (a2 - (__int64)a1) >> 3;
        goto LABEL_10;
      }
    }
    return a1;
  }
LABEL_10:
  switch ( v5 )
  {
    case 2LL:
      v9 = *a3;
LABEL_20:
      if ( *a1 == v9 )
        return a1;
      ++a1;
      goto LABEL_17;
    case 3LL:
      v9 = *a3;
      if ( *a1 == v9 )
        return a1;
      ++a1;
      goto LABEL_20;
    case 1LL:
      v9 = *a3;
LABEL_17:
      if ( *a1 != v9 )
        return (_QWORD *)a2;
      return a1;
  }
  return (_QWORD *)a2;
}
