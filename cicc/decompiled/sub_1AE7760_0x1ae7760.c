// Function: sub_1AE7760
// Address: 0x1ae7760
//
_DWORD *__fastcall sub_1AE7760(_DWORD *a1, __int64 a2, int *a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  int v6; // eax
  _DWORD *v7; // rdx
  _DWORD *result; // rax
  int v9; // edx

  v4 = (a2 - (__int64)a1) >> 4;
  v5 = (a2 - (__int64)a1) >> 2;
  if ( v4 <= 0 )
  {
LABEL_11:
    switch ( v5 )
    {
      case 2LL:
        v9 = *a3;
        break;
      case 3LL:
        v9 = *a3;
        result = a1;
        if ( *a1 == *a3 )
          return result;
        ++a1;
        break;
      case 1LL:
        v9 = *a3;
LABEL_18:
        result = a1;
        if ( *a1 != v9 )
          return (_DWORD *)a2;
        return result;
      default:
        return (_DWORD *)a2;
    }
    result = a1;
    if ( *a1 == v9 )
      return result;
    ++a1;
    goto LABEL_18;
  }
  v6 = *a3;
  v7 = &a1[4 * v4];
  while ( 1 )
  {
    if ( *a1 == v6 )
      return a1;
    if ( v6 == a1[1] )
      return a1 + 1;
    if ( v6 == a1[2] )
      return a1 + 2;
    if ( v6 == a1[3] )
      return a1 + 3;
    a1 += 4;
    if ( a1 == v7 )
    {
      v5 = (a2 - (__int64)a1) >> 2;
      goto LABEL_11;
    }
  }
}
