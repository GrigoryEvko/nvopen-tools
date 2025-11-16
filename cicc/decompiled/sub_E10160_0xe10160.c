// Function: sub_E10160
// Address: 0xe10160
//
char *__fastcall sub_E10160(__int64 a1, void **a2)
{
  char *v3; // rsi
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // rax
  char *v7; // rax
  char *result; // rax

  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 16) + 32LL))(*(_QWORD *)(a1 + 16));
  v3 = (char *)a2[1];
  v4 = (unsigned __int64)a2[2];
  if ( (unsigned __int64)(v3 + 3) <= v4 )
  {
    v7 = (char *)*a2;
  }
  else
  {
    v5 = (unsigned __int64)(v3 + 995);
    v6 = 2 * v4;
    if ( v5 > v6 )
      a2[2] = (void *)v5;
    else
      a2[2] = (void *)v6;
    v7 = (char *)realloc(*a2);
    *a2 = v7;
    if ( !v7 )
      abort();
    v3 = (char *)a2[1];
  }
  result = &v7[(_QWORD)v3];
  *(_WORD *)result = 11822;
  result[2] = 46;
  a2[1] = (char *)a2[1] + 3;
  return result;
}
