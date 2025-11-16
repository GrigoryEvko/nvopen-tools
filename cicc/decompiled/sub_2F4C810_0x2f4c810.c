// Function: sub_2F4C810
// Address: 0x2f4c810
//
unsigned __int16 *__fastcall sub_2F4C810(unsigned __int16 *a1, __int64 a2, int *a3)
{
  __int64 v4; // rcx
  __int64 v5; // rdx
  int v6; // eax
  unsigned __int16 *v7; // rcx
  unsigned __int16 *result; // rax
  int v9; // edx

  v4 = (a2 - (__int64)a1) >> 1;
  v5 = (a2 - (__int64)a1) >> 3;
  if ( v5 <= 0 )
  {
LABEL_11:
    switch ( v4 )
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
          return (unsigned __int16 *)a2;
        return result;
      default:
        return (unsigned __int16 *)a2;
    }
    result = a1;
    if ( *a1 == v9 )
      return result;
    ++a1;
    goto LABEL_18;
  }
  v6 = *a3;
  v7 = &a1[4 * v5];
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
      v4 = (a2 - (__int64)a1) >> 1;
      goto LABEL_11;
    }
  }
}
