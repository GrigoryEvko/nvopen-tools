// Function: sub_39BB2D0
// Address: 0x39bb2d0
//
char *__fastcall sub_39BB2D0(char *src, char *a2, char *a3, char *a4, _QWORD *a5)
{
  char *v6; // r13
  char *v7; // r12
  __int64 v8; // rax
  __int64 v9; // r14
  unsigned __int64 v10; // rbx
  __int64 v11; // rax
  char *v12; // r8

  v6 = a3;
  v7 = src;
  if ( a2 != src && a4 != a3 )
  {
    do
    {
      v9 = *(_QWORD *)v7;
      v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)v6 + 16LL))(*(_QWORD *)v6);
      if ( v10 < (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 16LL))(v9) )
      {
        v8 = *(_QWORD *)v6;
        ++a5;
        v6 += 8;
        *(a5 - 1) = v8;
        if ( a2 == v7 )
          break;
      }
      else
      {
        v11 = *(_QWORD *)v7;
        ++a5;
        v7 += 8;
        *(a5 - 1) = v11;
        if ( a2 == v7 )
          break;
      }
    }
    while ( a4 != v6 );
  }
  if ( a2 != v7 )
    memmove(a5, v7, a2 - v7);
  v12 = (char *)a5 + a2 - v7;
  if ( a4 != v6 )
    v12 = (char *)memmove(v12, v6, a4 - v6);
  return &v12[a4 - v6];
}
