// Function: sub_F1AF10
// Address: 0xf1af10
//
char *__fastcall sub_F1AF10(
        char *src,
        char *a2,
        char *a3,
        char *a4,
        _QWORD *a5,
        unsigned __int8 (__fastcall *a6)(_QWORD, _QWORD))
{
  char *v7; // r13
  char *v8; // r12
  __int64 v10; // rax
  __int64 v11; // rax
  signed __int64 v12; // r15
  char *v13; // r8

  v7 = a3;
  v8 = src;
  if ( src != a2 && a3 != a4 )
  {
    do
    {
      if ( a6(*(_QWORD *)v7, *(_QWORD *)v8) )
      {
        v10 = *(_QWORD *)v7;
        ++a5;
        v7 += 8;
        *(a5 - 1) = v10;
        if ( v8 == a2 )
          break;
      }
      else
      {
        v11 = *(_QWORD *)v8;
        v8 += 8;
        *a5++ = v11;
        if ( v8 == a2 )
          break;
      }
    }
    while ( v7 != a4 );
  }
  v12 = a2 - v8;
  if ( a2 != v8 )
    memmove(a5, v8, a2 - v8);
  v13 = (char *)a5 + v12;
  if ( a4 != v7 )
    v13 = (char *)memmove((char *)a5 + v12, v7, a4 - v7);
  return &v13[a4 - v7];
}
