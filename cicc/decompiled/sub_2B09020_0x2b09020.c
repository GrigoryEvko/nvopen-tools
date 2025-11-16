// Function: sub_2B09020
// Address: 0x2b09020
//
_DWORD *__fastcall sub_2B09020(_DWORD *a1, __int64 a2, int a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  _DWORD *v5; // rax
  _DWORD *result; // rax

  v3 = (a2 - (__int64)a1) >> 4;
  v4 = (a2 - (__int64)a1) >> 2;
  if ( v3 <= 0 )
  {
LABEL_11:
    switch ( v4 )
    {
      case 2LL:
        result = a1;
        break;
      case 3LL:
        result = a1;
        if ( *a1 >= a3 )
          return result;
        result = a1 + 1;
        break;
      case 1LL:
LABEL_18:
        result = a1;
        if ( *a1 < a3 )
          return (_DWORD *)a2;
        return result;
      default:
        return (_DWORD *)a2;
    }
    if ( *result >= a3 )
      return result;
    a1 = result + 1;
    goto LABEL_18;
  }
  v5 = &a1[4 * v3];
  while ( 1 )
  {
    if ( a3 <= *a1 )
      return a1;
    if ( a3 <= a1[1] )
      return a1 + 1;
    if ( a3 <= a1[2] )
      return a1 + 2;
    if ( a3 <= a1[3] )
      return a1 + 3;
    a1 += 4;
    if ( a1 == v5 )
    {
      v4 = (a2 - (__int64)a1) >> 2;
      goto LABEL_11;
    }
  }
}
