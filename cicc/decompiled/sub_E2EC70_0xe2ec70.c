// Function: sub_E2EC70
// Address: 0xe2ec70
//
void __fastcall sub_E2EC70(__int64 a1, __int64 *a2, unsigned int a3)
{
  size_t v6; // rbx
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  const void *v9; // r15
  char *v10; // rdi
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rdx
  __int64 v13; // rax

  v6 = *(_QWORD *)(a1 + 24);
  if ( v6 )
  {
    v7 = a2[1];
    v8 = a2[2];
    v9 = *(const void **)(a1 + 32);
    v10 = (char *)*a2;
    if ( v7 + v6 > v8 )
    {
      v11 = v7 + v6 + 992;
      v12 = 2 * v8;
      if ( v11 > v12 )
        a2[2] = v11;
      else
        a2[2] = v12;
      v13 = realloc(v10);
      *a2 = v13;
      v10 = (char *)v13;
      if ( !v13 )
        abort();
      v7 = a2[1];
    }
    memcpy(&v10[v7], v9, v6);
    a2[1] += v6;
  }
  sub_E2EB40(a1, (__int64)a2, a3);
}
