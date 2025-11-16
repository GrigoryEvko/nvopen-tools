// Function: sub_14E9D50
// Address: 0x14e9d50
//
_QWORD *__fastcall sub_14E9D50(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rbx
  char *v5; // rcx
  const void *v6; // rax
  const void *v7; // rsi
  __int64 v8; // rbx

  v4 = *(_QWORD *)(a2 + 1792) - *(_QWORD *)(a2 + 1784);
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( v4 )
  {
    if ( v4 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a1, a2, a3);
    v5 = (char *)sub_22077B0(v4);
  }
  else
  {
    v5 = 0;
  }
  *a1 = v5;
  a1[2] = &v5[v4];
  a1[1] = v5;
  v6 = *(const void **)(a2 + 1792);
  v7 = *(const void **)(a2 + 1784);
  v8 = *(_QWORD *)(a2 + 1792) - (_QWORD)v7;
  if ( v6 != v7 )
    v5 = (char *)memmove(v5, v7, *(_QWORD *)(a2 + 1792) - (_QWORD)v7);
  a1[1] = &v5[v8];
  return a1;
}
