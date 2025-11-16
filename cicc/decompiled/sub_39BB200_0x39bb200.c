// Function: sub_39BB200
// Address: 0x39bb200
//
char *__fastcall sub_39BB200(char *src, char *a2, char *a3, char *a4, char *a5)
{
  char *v7; // r13
  __int64 v8; // rax
  __int64 v9; // r14
  unsigned __int64 v10; // rbx
  __int64 v11; // rax
  signed __int64 v12; // rbx
  void *v13; // rdi

  if ( src == a2 )
  {
LABEL_7:
    v12 = a4 - a3;
    if ( a4 != a3 )
      memmove(a5, a3, a4 - a3);
  }
  else
  {
    v7 = src;
    while ( a4 != a3 )
    {
      v9 = *(_QWORD *)v7;
      v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)a3 + 16LL))(*(_QWORD *)a3);
      if ( v10 < (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 16LL))(v9) )
      {
        v8 = *(_QWORD *)a3;
        a5 += 8;
        a3 += 8;
        *((_QWORD *)a5 - 1) = v8;
        if ( v7 == a2 )
          goto LABEL_7;
      }
      else
      {
        v11 = *(_QWORD *)v7;
        a5 += 8;
        v7 += 8;
        *((_QWORD *)a5 - 1) = v11;
        if ( v7 == a2 )
          goto LABEL_7;
      }
    }
    v13 = a5;
    v12 = 0;
    a5 += a2 - v7;
    memmove(v13, v7, a2 - v7);
  }
  return &a5[v12];
}
