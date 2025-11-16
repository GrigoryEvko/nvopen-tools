// Function: sub_1680B50
// Address: 0x1680b50
//
const char **__fastcall sub_1680B50(const void *a1, size_t a2, const char **a3, __int64 a4)
{
  __int64 v4; // rcx
  __int64 v5; // r15
  const char **v6; // r14
  int v7; // eax
  void *v8; // rcx
  __int64 v9; // r12
  const char **v10; // rbx
  size_t v11; // rax
  const char *v12; // rdi
  const char *v13; // r12
  const char *v15; // rdi
  const char **v16; // [rsp+8h] [rbp-48h]
  void *s1; // [rsp+18h] [rbp-38h]
  const char *s1a; // [rsp+18h] [rbp-38h]
  void *s1b; // [rsp+18h] [rbp-38h]

  v4 = a4 << 6;
  v5 = v4 >> 6;
  v6 = a3;
  v16 = (const char **)((char *)a3 + v4);
  if ( v4 > 0 )
  {
    while ( 1 )
    {
      v9 = v5 >> 1;
      v10 = &v6[8 * (v5 >> 1)];
      if ( !*v10 )
        break;
      s1a = *v10;
      v11 = strlen(*v10);
      v12 = s1a;
      v8 = (void *)v11;
      if ( a2 >= v11 )
      {
        if ( !v11 )
          goto LABEL_5;
        s1 = (void *)v11;
        v7 = memcmp(v12, a1, v11);
        v8 = s1;
        if ( !v7 )
          goto LABEL_5;
LABEL_12:
        if ( v7 >= 0 )
          goto LABEL_13;
LABEL_7:
        v6 = v10 + 8;
        v5 = v5 - v9 - 1;
        if ( v5 <= 0 )
          goto LABEL_14;
      }
      else
      {
        if ( !a2 )
          goto LABEL_13;
        s1b = (void *)v11;
        v7 = memcmp(v12, a1, a2);
        v8 = s1b;
        if ( v7 )
          goto LABEL_12;
LABEL_6:
        if ( a2 > (unsigned __int64)v8 )
          goto LABEL_7;
LABEL_13:
        v5 >>= 1;
        if ( v9 <= 0 )
          goto LABEL_14;
      }
    }
    v8 = 0;
LABEL_5:
    if ( (void *)a2 == v8 )
      goto LABEL_13;
    goto LABEL_6;
  }
LABEL_14:
  if ( v16 == v6 )
    return 0;
  v13 = *v6;
  if ( *v6 )
  {
    if ( strlen(*v6) != a2 )
      return 0;
    if ( a2 )
    {
      v15 = v13;
      v13 = 0;
      if ( !memcmp(v15, a1, a2) )
        return v6;
    }
    else
    {
      return v6;
    }
  }
  else if ( !a2 )
  {
    return v6;
  }
  return (const char **)v13;
}
