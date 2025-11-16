// Function: sub_E100C0
// Address: 0xe100c0
//
void __fastcall sub_E100C0(__int64 a1, __int64 *a2)
{
  size_t v2; // r12
  char *v4; // rax
  const void *v5; // r13
  size_t v6; // rdx
  char *v7; // rdi
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rdx
  __int64 v10; // rax

  v2 = *(_QWORD *)(a1 + 16);
  if ( v2 )
  {
    v4 = (char *)a2[1];
    v5 = *(const void **)(a1 + 24);
    v6 = a2[2];
    v7 = (char *)*a2;
    if ( (unsigned __int64)&v4[v2] > v6 )
    {
      v8 = (unsigned __int64)&v4[v2 + 992];
      v9 = 2 * v6;
      if ( v8 > v9 )
        a2[2] = v8;
      else
        a2[2] = v9;
      v10 = realloc(v7);
      *a2 = v10;
      v7 = (char *)v10;
      if ( !v10 )
        abort();
      v4 = (char *)a2[1];
    }
    memcpy(&v7[(_QWORD)v4], v5, v2);
    a2[1] += v2;
  }
}
