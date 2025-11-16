// Function: sub_E9F7C0
// Address: 0xe9f7c0
//
const char **__fastcall sub_E9F7C0(const char **a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  const char **v4; // r13
  __int64 v5; // r12
  size_t v6; // r14
  const char **v7; // rbx
  size_t v8; // rax
  const char *v9; // rdi
  size_t v10; // rdx
  void *v11; // rcx
  int v12; // eax
  void *s2; // [rsp+0h] [rbp-40h]
  const char *s1; // [rsp+8h] [rbp-38h]
  void *s1a; // [rsp+8h] [rbp-38h]

  v3 = a2 - (_QWORD)a1;
  v4 = a1;
  v5 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 5);
  if ( v3 > 0 )
  {
    v6 = *(_QWORD *)(a3 + 8);
    s2 = *(void **)a3;
    while ( 1 )
    {
      v7 = &(&v4[4 * (v5 >> 1)])[4 * (v5 & 0xFFFFFFFFFFFFFFFELL)];
      if ( !*v7 )
        break;
      s1 = *v7;
      v8 = strlen(*v7);
      v9 = s1;
      v10 = v8;
      v11 = (void *)v8;
      if ( v6 <= v8 )
        v10 = v6;
      if ( v10 && (s1a = (void *)v8, v12 = memcmp(v9, s2, v10), v11 = s1a, v12) )
      {
        if ( v12 >= 0 )
        {
          v5 >>= 1;
          goto LABEL_12;
        }
LABEL_3:
        v4 = v7 + 12;
        v5 = v5 - (v5 >> 1) - 1;
        if ( v5 <= 0 )
          return v4;
      }
      else
      {
LABEL_9:
        if ( v11 != (void *)v6 && (unsigned __int64)v11 < v6 )
          goto LABEL_3;
        v5 >>= 1;
LABEL_12:
        if ( v5 <= 0 )
          return v4;
      }
    }
    v11 = 0;
    goto LABEL_9;
  }
  return v4;
}
