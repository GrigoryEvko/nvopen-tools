// Function: sub_9C3390
// Address: 0x9c3390
//
_QWORD *__fastcall sub_9C3390(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // rbx
  char *v6; // rcx
  const void *v7; // rax
  const void *v8; // rsi
  __int64 v9; // rbx

  v5 = *(_QWORD *)(a2 + 2016) - *(_QWORD *)(a2 + 2008);
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( v5 )
  {
    if ( v5 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a1, a2, a3, a4);
    v6 = (char *)sub_22077B0(v5);
  }
  else
  {
    v6 = 0;
  }
  *a1 = v6;
  a1[2] = &v6[v5];
  a1[1] = v6;
  v7 = *(const void **)(a2 + 2016);
  v8 = *(const void **)(a2 + 2008);
  v9 = *(_QWORD *)(a2 + 2016) - (_QWORD)v8;
  if ( v7 != v8 )
    v6 = (char *)memmove(v6, v8, *(_QWORD *)(a2 + 2016) - (_QWORD)v8);
  a1[1] = &v6[v9];
  return a1;
}
