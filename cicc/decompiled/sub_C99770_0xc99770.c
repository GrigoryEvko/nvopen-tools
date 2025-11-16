// Function: sub_C99770
// Address: 0xc99770
//
__int64 __fastcall sub_C99770(_BYTE *a1, __int64 a2, void (__fastcall *a3)(__m128i **, __int64), __int64 a4)
{
  __int64 *v6; // rax
  __int64 v7; // r12
  __m128i v9; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v10[8]; // [rsp+20h] [rbp-40h] BYREF

  v6 = (__int64 *)sub_C94E20((__int64)&qword_4F84F00);
  v7 = qword_4F84F10;
  if ( v6 )
    v7 = *v6;
  if ( v7 )
  {
    v9.m128i_i64[0] = (__int64)v10;
    sub_C95DE0(v9.m128i_i64, a1, (__int64)&a1[a2]);
    v7 = sub_C96F60(v7, &v9, a3, a4, 0);
    if ( (_QWORD *)v9.m128i_i64[0] != v10 )
      j_j___libc_free_0(v9.m128i_i64[0], v10[0] + 1LL);
  }
  return v7;
}
