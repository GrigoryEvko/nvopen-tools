// Function: sub_29BF110
// Address: 0x29bf110
//
char *__fastcall sub_29BF110(char *src, char *a2, char *a3, char *a4, _QWORD *a5, _QWORD *a6)
{
  char *v11; // r8
  __int64 v12; // rcx
  char *v13; // r13

  if ( src == a2 )
  {
LABEL_7:
    v13 = (char *)a5 + a4 - a3;
    if ( a3 != a4 )
    {
      memmove(a5, a3, a4 - a3);
      return (char *)a5 + a4 - a3;
    }
  }
  else
  {
    v11 = src;
    while ( a3 != a4 )
    {
      v12 = *(_QWORD *)v11;
      if ( *(_DWORD *)(*a6 + 16LL * *(_QWORD *)a3) < *(_DWORD *)(*a6 + 16LL * *(_QWORD *)v11) )
      {
        *a5 = *(_QWORD *)a3;
        a3 += 8;
        ++a5;
        if ( a2 == v11 )
          goto LABEL_7;
      }
      else
      {
        v11 += 8;
        *a5++ = v12;
        if ( a2 == v11 )
          goto LABEL_7;
      }
    }
    return (char *)memmove(a5, v11, a2 - v11) + a2 - v11;
  }
  return v13;
}
