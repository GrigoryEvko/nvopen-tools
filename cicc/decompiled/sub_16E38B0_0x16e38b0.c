// Function: sub_16E38B0
// Address: 0x16e38b0
//
__int64 __fastcall sub_16E38B0(__int64 a1, const void *a2, __int64 a3, unsigned int a4)
{
  __int64 *v6; // r14
  void *s2; // [rsp+0h] [rbp-40h] BYREF
  size_t n; // [rsp+8h] [rbp-38h]
  __int64 v10; // [rsp+10h] [rbp-30h] BYREF

  sub_16FDB20(&s2, *(_QWORD *)(*(_QWORD *)(a1 + 264) + 8LL));
  v6 = (__int64 *)s2;
  if ( n )
  {
    a4 = 0;
    if ( n == a3 )
      LOBYTE(a4) = memcmp(a2, s2, n) == 0;
  }
  if ( v6 != &v10 )
    j_j___libc_free_0(v6, v10 + 1);
  return a4;
}
