// Function: sub_A16990
// Address: 0xa16990
//
unsigned int __fastcall sub_A16990(const void **a1, const void **a2)
{
  const void *v2; // r12
  size_t v3; // rbx
  const void *v4; // r14
  const void *v5; // r15
  size_t v6; // r13
  int v7; // eax
  bool v8; // sf
  unsigned int result; // eax

  v2 = a1[1];
  v3 = (size_t)a2[1];
  v4 = *a1;
  v5 = *a2;
  v6 = v3;
  if ( (unsigned __int64)v2 <= v3 )
    v6 = (size_t)a1[1];
  if ( !v6 )
  {
    if ( v2 == (const void *)v3 )
      return 0;
LABEL_6:
    result = -1;
    if ( (unsigned __int64)v2 < v3 )
      return result;
    if ( !v6 )
      return (unsigned __int64)v2 > v3;
    result = memcmp(v5, v4, v6);
    if ( !result )
      return (unsigned __int64)v2 > v3;
LABEL_18:
    result >>= 31;
    return result;
  }
  v7 = memcmp(*a1, *a2, v6);
  v8 = v7 < 0;
  if ( v7 )
  {
    result = -1;
    if ( !v8 )
    {
      result = memcmp(v5, v4, v6);
      if ( result )
        goto LABEL_18;
      if ( v2 != (const void *)v3 )
        return (unsigned __int64)v2 > v3;
    }
  }
  else
  {
    if ( v2 != (const void *)v3 )
      goto LABEL_6;
    result = memcmp(v5, v4, v6);
    if ( result )
      goto LABEL_18;
  }
  return result;
}
