// Function: sub_3118890
// Address: 0x3118890
//
__int64 __fastcall sub_3118890(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // rbx
  __int64 v8; // r12
  __int64 v9; // r13
  size_t v10; // rcx
  size_t v11; // r8
  __int64 *v12; // rdi
  size_t v13; // rdx
  int v14; // eax
  __int64 v15; // rcx
  size_t v17; // [rsp+10h] [rbp-B0h]
  size_t v18; // [rsp+18h] [rbp-A8h]
  __int64 *v19; // [rsp+20h] [rbp-A0h]
  int v20; // [rsp+20h] [rbp-A0h]
  int v21; // [rsp+20h] [rbp-A0h]
  __int64 v22; // [rsp+28h] [rbp-98h]
  void *s1; // [rsp+30h] [rbp-90h] BYREF
  size_t n; // [rsp+38h] [rbp-88h]
  __int64 v25; // [rsp+40h] [rbp-80h] BYREF
  char v26; // [rsp+50h] [rbp-70h]
  void *s2; // [rsp+60h] [rbp-60h] BYREF
  size_t v28; // [rsp+68h] [rbp-58h]
  __int64 v29; // [rsp+70h] [rbp-50h] BYREF
  char v30; // [rsp+80h] [rbp-40h]

  v4 = a2 - a1;
  v5 = v4 >> 3;
  if ( v4 <= 0 )
    return a1;
  v22 = a1;
  do
  {
    while ( 1 )
    {
      v8 = v5 >> 1;
      v9 = v22 + 8 * (v5 >> 1);
      sub_31185E0((__int64)&s2, a4, *(_DWORD *)(*(_QWORD *)v9 + 12LL));
      sub_31185E0((__int64)&s1, a4, *(_DWORD *)(*(_QWORD *)a3 + 12LL));
      v10 = n;
      v11 = v28;
      v12 = (__int64 *)s1;
      v13 = v28;
      if ( n <= v28 )
        v13 = n;
      if ( !v13
        || (v17 = v28, v18 = n, v19 = (__int64 *)s1, v14 = memcmp(s1, s2, v13), v12 = v19, v10 = v18, v11 = v17, !v14) )
      {
        v15 = v10 - v11;
        v14 = 0x7FFFFFFF;
        if ( v15 < 0x80000000LL )
        {
          v14 = 0x80000000;
          if ( v15 > (__int64)0xFFFFFFFF7FFFFFFFLL )
            v14 = v15;
        }
      }
      if ( v26 )
      {
        v26 = 0;
        if ( v12 != &v25 )
        {
          v21 = v14;
          j_j___libc_free_0((unsigned __int64)v12);
          v14 = v21;
        }
      }
      if ( v30 )
      {
        v30 = 0;
        if ( s2 != &v29 )
        {
          v20 = v14;
          j_j___libc_free_0((unsigned __int64)s2);
          v14 = v20;
        }
      }
      if ( v14 < 0 )
        break;
      v5 = v5 - v8 - 1;
      v22 = v9 + 8;
      if ( v5 <= 0 )
        return v22;
    }
    v5 >>= 1;
  }
  while ( v8 > 0 );
  return v22;
}
