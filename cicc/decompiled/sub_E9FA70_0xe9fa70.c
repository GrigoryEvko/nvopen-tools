// Function: sub_E9FA70
// Address: 0xe9fa70
//
__int64 __fastcall sub_E9FA70(__int64 a1, const void *a2, __int64 a3)
{
  __int64 v4; // rax
  const char **v5; // r8
  __int64 v6; // rbx
  const char **v7; // rax
  unsigned int v8; // r8d
  const char *v9; // r14
  size_t v10; // rax
  _QWORD v12[6]; // [rsp+0h] [rbp-30h] BYREF

  v4 = *(_QWORD *)(a1 + 168);
  v5 = *(const char ***)(a1 + 160);
  v12[0] = a2;
  v12[1] = a3;
  v6 = (__int64)&v5[12 * v4];
  v7 = sub_E9F7C0(v5, v6, (__int64)v12);
  v8 = 0;
  if ( (const char **)v6 == v7 )
    return v8;
  v9 = *v7;
  if ( *v7 )
  {
    v10 = strlen(*v7);
    if ( v10 == a3 )
    {
      if ( v10 )
      {
        LOBYTE(v8) = memcmp(v9, a2, v10) == 0;
        return v8;
      }
      return 1;
    }
    return 0;
  }
  return !a3;
}
