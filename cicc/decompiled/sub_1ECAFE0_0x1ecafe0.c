// Function: sub_1ECAFE0
// Address: 0x1ecafe0
//
_DWORD *__fastcall sub_1ECAFE0(_DWORD *a1, __int64 a2, int *a3)
{
  __int64 v4; // rcx
  __int64 v5; // rdx
  int v6; // edx
  _DWORD *v7; // rcx
  int v9; // eax

  v4 = (a2 - (__int64)a1) >> 4;
  v5 = (a2 - (__int64)a1) >> 2;
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
        v5 = (a2 - (__int64)a1) >> 2;
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
        return (_DWORD *)a2;
      return a1;
  }
  return (_DWORD *)a2;
}
