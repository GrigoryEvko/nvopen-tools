// Function: sub_CB0720
// Address: 0xcb0720
//
__int64 __fastcall sub_CB0720(__int64 a1, const void *a2, __int64 a3, unsigned int a4)
{
  unsigned int v4; // r14d
  __int64 *v5; // rax
  _QWORD *v8; // r15
  __m128i s2; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v11[8]; // [rsp+10h] [rbp-40h] BYREF

  v4 = 0;
  v5 = *(__int64 **)(a1 + 672);
  if ( v5 )
  {
    sub_CA95D0(&s2, *v5);
    v8 = (_QWORD *)s2.m128i_i64[0];
    if ( s2.m128i_i64[1] )
    {
      if ( s2.m128i_i64[1] == a3 )
      {
        LOBYTE(v4) = memcmp(a2, (const void *)s2.m128i_i64[0], s2.m128i_u64[1]) == 0;
        if ( v8 == v11 )
          return v4;
LABEL_8:
        j_j___libc_free_0(v8, v11[0] + 1LL);
        return v4;
      }
    }
    else
    {
      v4 = a4;
    }
    if ( (_QWORD *)s2.m128i_i64[0] == v11 )
      return v4;
    goto LABEL_8;
  }
  return v4;
}
