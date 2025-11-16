// Function: sub_D91130
// Address: 0xd91130
//
char *__fastcall sub_D91130(char *a1, char *a2, __int64 *a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  char *v7; // rdx
  char *result; // rax
  __int64 v9; // rdx

  v4 = (a2 - a1) >> 6;
  v5 = (a2 - a1) >> 4;
  if ( v4 <= 0 )
  {
LABEL_9:
    switch ( v5 )
    {
      case 2LL:
        v9 = *a3;
        break;
      case 3LL:
        v9 = *a3;
        if ( *(_QWORD *)a1 == *a3 )
        {
          result = a1;
          if ( *((_QWORD *)a1 + 1) == a3[1] )
            return result;
        }
        a1 += 16;
        break;
      case 1LL:
        v9 = *a3;
LABEL_23:
        result = a2;
        if ( *(_QWORD *)a1 == v9 && *((_QWORD *)a1 + 1) == a3[1] )
          return a1;
        return result;
      default:
        return a2;
    }
    if ( *(_QWORD *)a1 == v9 )
    {
      result = a1;
      if ( *((_QWORD *)a1 + 1) == a3[1] )
        return result;
    }
    a1 += 16;
    goto LABEL_23;
  }
  v6 = *a3;
  v7 = &a1[64 * v4];
  while ( 1 )
  {
    if ( *(_QWORD *)a1 == v6 && *((_QWORD *)a1 + 1) == a3[1] )
      return a1;
    if ( v6 == *((_QWORD *)a1 + 2) && *((_QWORD *)a1 + 3) == a3[1] )
      return a1 + 16;
    if ( v6 == *((_QWORD *)a1 + 4) && *((_QWORD *)a1 + 5) == a3[1] )
      return a1 + 32;
    if ( v6 == *((_QWORD *)a1 + 6) && *((_QWORD *)a1 + 7) == a3[1] )
      return a1 + 48;
    a1 += 64;
    if ( a1 == v7 )
    {
      v5 = (a2 - a1) >> 4;
      goto LABEL_9;
    }
  }
}
