// Function: sub_BB97F0
// Address: 0xbb97f0
//
__int64 __fastcall sub_BB97F0(_QWORD *a1, const void **a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rdx
  __int64 ***v6; // rax
  __int64 **v7; // rbx
  __int64 *v8; // rcx
  const void **v9; // r8
  unsigned __int64 v10; // r15
  __int64 v11; // r14
  size_t v12; // rdx
  int v13; // eax
  const void **v15; // [rsp+8h] [rbp-38h]

  v2 = sub_22076E0(*a2, a2[1], 3339675911LL);
  v3 = a1[1];
  v4 = v2;
  v5 = v2 % v3;
  v6 = *(__int64 ****)(*a1 + 8 * (v2 % v3));
  if ( v6 )
  {
    v7 = *v6;
    if ( *v6 )
    {
      v8 = v7[5];
      v9 = a2;
      v10 = v5;
      v11 = 0;
      while ( 1 )
      {
        if ( v8 != (__int64 *)v4
          || (v12 = (size_t)v9[1], (__int64 *)v12 != v7[2])
          || v12 && (v15 = v9, v13 = memcmp(*v9, v7[1], v12), v9 = v15, v13) )
        {
          if ( v11 )
            return v11;
          v7 = (__int64 **)*v7;
          if ( !v7 )
            return v11;
        }
        else
        {
          v7 = (__int64 **)*v7;
          ++v11;
          if ( !v7 )
            return v11;
        }
        v8 = v7[5];
        if ( v10 != (unsigned __int64)v8 % v3 )
          return v11;
      }
    }
  }
  return 0;
}
