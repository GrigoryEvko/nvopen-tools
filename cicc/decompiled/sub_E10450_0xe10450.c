// Function: sub_E10450
// Address: 0xe10450
//
void *__fastcall sub_E10450(__int64 a1, __int64 *a2)
{
  void *result; // rax
  size_t v4; // r13
  char *v5; // rax
  size_t v6; // rdx
  const void *v7; // r12
  char *v8; // rdi
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rdx
  __int64 v11; // rax

  result = (void *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 16) + 32LL))(*(_QWORD *)(a1 + 16));
  v4 = *(_QWORD *)(a1 + 24);
  if ( v4 )
  {
    v5 = (char *)a2[1];
    v6 = a2[2];
    v7 = *(const void **)(a1 + 32);
    v8 = (char *)*a2;
    if ( (unsigned __int64)&v5[v4] > v6 )
    {
      v9 = (unsigned __int64)&v5[v4 + 992];
      v10 = 2 * v6;
      if ( v9 > v10 )
        a2[2] = v9;
      else
        a2[2] = v10;
      v11 = realloc(v8);
      *a2 = v11;
      v8 = (char *)v11;
      if ( !v11 )
        abort();
      v5 = (char *)a2[1];
    }
    result = memcpy(&v8[(_QWORD)v5], v7, v4);
    a2[1] += v4;
  }
  return result;
}
