// Function: sub_E9F8B0
// Address: 0xe9f8b0
//
const char **__fastcall sub_E9F8B0(void *s2, __int64 a2, const char **a3, __int64 a4)
{
  __int64 v4; // rbx
  const char **v5; // rax
  const char *v6; // r15
  const char **v7; // r14
  size_t v8; // rax
  _QWORD v10[8]; // [rsp+0h] [rbp-40h] BYREF

  v4 = (__int64)&a3[12 * a4];
  v10[0] = s2;
  v10[1] = a2;
  v5 = sub_E9F7C0(a3, v4, (__int64)v10);
  if ( (const char **)v4 == v5 )
    return 0;
  v6 = *v5;
  v7 = v5;
  if ( !*v5 )
  {
    if ( !a2 )
      return v7;
    return 0;
  }
  v8 = strlen(*v5);
  if ( v8 != a2 || v8 && memcmp(v6, s2, v8) )
    return 0;
  return v7;
}
