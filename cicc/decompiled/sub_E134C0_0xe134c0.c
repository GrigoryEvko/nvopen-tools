// Function: sub_E134C0
// Address: 0xe134c0
//
void *__fastcall sub_E134C0(__int64 a1, __int64 *a2)
{
  void *result; // rax
  size_t v4; // r12
  const void *v5; // rdx
  const void *v6; // r13
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  char *v9; // rdi
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rdx
  __int64 v12; // rax

  if ( *(_BYTE *)(a1 + 24) )
    sub_E12F20(a2, 1u, "~");
  result = (void *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 16) + 56LL))(*(_QWORD *)(a1 + 16));
  v4 = (size_t)result;
  v6 = v5;
  if ( result )
  {
    v7 = a2[1];
    v8 = a2[2];
    v9 = (char *)*a2;
    if ( v7 + v4 > v8 )
    {
      v10 = v7 + v4 + 992;
      v11 = 2 * v8;
      if ( v10 > v11 )
        a2[2] = v10;
      else
        a2[2] = v11;
      v12 = realloc(v9);
      *a2 = v12;
      v9 = (char *)v12;
      if ( !v12 )
        abort();
      v7 = a2[1];
    }
    result = memcpy(&v9[v7], v6, v4);
    a2[1] += v4;
  }
  return result;
}
