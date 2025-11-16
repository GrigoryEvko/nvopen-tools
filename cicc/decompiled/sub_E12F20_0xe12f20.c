// Function: sub_E12F20
// Address: 0xe12f20
//
__int64 *__fastcall sub_E12F20(__int64 *a1, size_t a2, const void *a3)
{
  char *v5; // rax
  size_t v7; // rdx
  char *v9; // rdi
  char *v10; // rsi
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rdx
  __int64 v13; // rax

  if ( !a2 )
    return a1;
  v5 = (char *)a1[1];
  v7 = a1[2];
  v9 = (char *)*a1;
  v10 = &v5[a2];
  if ( (unsigned __int64)v10 > v7 )
  {
    v11 = (unsigned __int64)(v10 + 992);
    v12 = 2 * v7;
    if ( v11 > v12 )
      a1[2] = v11;
    else
      a1[2] = v12;
    v13 = realloc(v9);
    *a1 = v13;
    v9 = (char *)v13;
    if ( !v13 )
      abort();
    v5 = (char *)a1[1];
  }
  memcpy(&v9[(_QWORD)v5], a3, a2);
  a1[1] += a2;
  return a1;
}
