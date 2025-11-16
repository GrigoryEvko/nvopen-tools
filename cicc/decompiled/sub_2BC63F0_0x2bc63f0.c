// Function: sub_2BC63F0
// Address: 0x2bc63f0
//
char *__fastcall sub_2BC63F0(
        char *src,
        char *a2,
        char *a3,
        char *a4,
        _QWORD *a5,
        __int64 a6,
        unsigned __int8 (__fastcall *a7)(__int64, _QWORD, _QWORD),
        __int64 a8)
{
  char *v8; // r13
  char *v9; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  char *v13; // r8

  v8 = a3;
  v9 = src;
  if ( src != a2 && a3 != a4 )
  {
    do
    {
      if ( a7(a8, *(_QWORD *)v8, *(_QWORD *)v9) )
      {
        v11 = *(_QWORD *)v8;
        ++a5;
        v8 += 8;
        *(a5 - 1) = v11;
        if ( v9 == a2 )
          break;
      }
      else
      {
        v12 = *(_QWORD *)v9;
        ++a5;
        v9 += 8;
        *(a5 - 1) = v12;
        if ( v9 == a2 )
          break;
      }
    }
    while ( v8 != a4 );
  }
  if ( a2 != v9 )
    memmove(a5, v9, a2 - v9);
  v13 = (char *)a5 + a2 - v9;
  if ( a4 != v8 )
    v13 = (char *)memmove(v13, v8, a4 - v8);
  return &v13[a4 - v8];
}
