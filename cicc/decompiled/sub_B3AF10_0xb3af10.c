// Function: sub_B3AF10
// Address: 0xb3af10
//
_BYTE *__fastcall sub_B3AF10(_BYTE *a1, __int64 a2, char *a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  char v6; // al
  _BYTE *v7; // rdx
  _BYTE *result; // rax
  char v9; // dl

  v4 = a2 - (_QWORD)a1;
  v5 = (a2 - (__int64)a1) >> 2;
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
          return (_BYTE *)a2;
        return result;
      default:
        return (_BYTE *)a2;
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
      v4 = a2 - (_QWORD)a1;
      goto LABEL_11;
    }
  }
}
