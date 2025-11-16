// Function: sub_E9F950
// Address: 0xe9f950
//
const char **__fastcall sub_E9F950(const void *a1, size_t a2, const char **a3, __int64 a4)
{
  __int64 v4; // r15
  const char **v5; // r14
  const char **v6; // rbx
  size_t v7; // rax
  size_t v8; // rdx
  const char *v9; // rdi
  void *v10; // r8
  int v11; // eax
  const char *v12; // r12
  const char **v14; // [rsp+8h] [rbp-48h]
  const char *s1; // [rsp+18h] [rbp-38h]
  void *s1a; // [rsp+18h] [rbp-38h]

  v4 = a4 << 6 >> 6;
  v5 = a3;
  v14 = &a3[8 * a4];
  if ( a4 << 6 > 0 )
  {
    while ( 1 )
    {
      v6 = &v5[8 * (v4 >> 1)];
      if ( !*v6 )
        break;
      s1 = *v6;
      v7 = strlen(*v6);
      v8 = a2;
      v9 = s1;
      v10 = (void *)v7;
      if ( v7 <= a2 )
        v8 = v7;
      if ( v8 && (s1a = (void *)v7, v11 = memcmp(v9, a1, v8), v10 = s1a, v11) )
      {
        if ( v11 >= 0 )
        {
          v4 >>= 1;
          goto LABEL_12;
        }
LABEL_3:
        v5 = v6 + 8;
        v4 = v4 - (v4 >> 1) - 1;
        if ( v4 <= 0 )
          goto LABEL_13;
      }
      else
      {
LABEL_9:
        if ( v10 != (void *)a2 && (unsigned __int64)v10 < a2 )
          goto LABEL_3;
        v4 >>= 1;
LABEL_12:
        if ( v4 <= 0 )
          goto LABEL_13;
      }
    }
    v10 = 0;
    goto LABEL_9;
  }
LABEL_13:
  if ( v14 == v5 )
    return 0;
  v12 = *v5;
  if ( !*v5 )
  {
    if ( !a2 )
      return v5;
    return 0;
  }
  if ( strlen(*v5) != a2 || a2 && memcmp(v12, a1, a2) )
    return 0;
  return v5;
}
