// Function: sub_C996C0
// Address: 0xc996c0
//
__int64 __fastcall sub_C996C0(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // rax
  __int64 v5; // r12
  _QWORD v7[3]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD *v8; // [rsp+28h] [rbp-58h] BYREF
  __m128i v9; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v10[8]; // [rsp+40h] [rbp-40h] BYREF

  v7[0] = a3;
  v7[1] = a4;
  v4 = (__int64 *)sub_C94E20((__int64)&qword_4F84F00);
  v5 = qword_4F84F10;
  if ( v4 )
    v5 = *v4;
  if ( v5 )
  {
    v8 = v7;
    v9.m128i_i64[0] = (__int64)v10;
    sub_C95DE0(v9.m128i_i64, a1, (__int64)&a1[a2]);
    v5 = sub_C96F60(v5, &v9, (void (__fastcall *)(__m128i **, __int64))sub_C95F60, (__int64)&v8, 0);
    if ( (_QWORD *)v9.m128i_i64[0] != v10 )
      j_j___libc_free_0(v9.m128i_i64[0], v10[0] + 1LL);
  }
  return v5;
}
