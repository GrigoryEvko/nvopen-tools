// Function: sub_29BF1E0
// Address: 0x29bf1e0
//
char *__fastcall sub_29BF1E0(char *src, char *a2, char *a3, char *a4, _QWORD *a5, _QWORD *a6)
{
  char *v6; // r10
  __int64 v9; // rcx
  signed __int64 v10; // r13
  char *v11; // r8

  v6 = src;
  if ( src != a2 )
  {
    while ( a3 != a4 )
    {
      v9 = *(_QWORD *)v6;
      if ( *(_DWORD *)(*a6 + 16LL * *(_QWORD *)a3) < *(_DWORD *)(*a6 + 16LL * *(_QWORD *)v6) )
      {
        *a5 = *(_QWORD *)a3;
        a3 += 8;
        ++a5;
        if ( v6 == a2 )
          break;
      }
      else
      {
        v6 += 8;
        *a5++ = v9;
        if ( v6 == a2 )
          break;
      }
    }
  }
  v10 = a2 - v6;
  if ( a2 != v6 )
    a5 = memmove(a5, v6, a2 - v6);
  v11 = (char *)a5 + v10;
  if ( a3 != a4 )
    v11 = (char *)memmove(v11, a3, a4 - a3);
  return &v11[a4 - a3];
}
