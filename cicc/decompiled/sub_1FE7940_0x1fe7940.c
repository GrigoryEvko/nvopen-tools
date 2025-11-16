// Function: sub_1FE7940
// Address: 0x1fe7940
//
_QWORD *__fastcall sub_1FE7940(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rbx
  __int64 v3; // rsi
  __int64 v4; // r14
  _QWORD *v5; // rax
  _QWORD *v6; // r12
  __int64 v8; // [rsp+8h] [rbp-58h] BYREF
  __m128i v9; // [rsp+10h] [rbp-50h] BYREF
  __int64 v10; // [rsp+20h] [rbp-40h]
  __int64 v11; // [rsp+28h] [rbp-38h]

  v2 = *a2;
  v3 = a2[1];
  v8 = v3;
  if ( v3 )
    sub_1623A60((__int64)&v8, v3, 2);
  v4 = *a1;
  v5 = sub_1E0B640(*a1, *(_QWORD *)(a1[2] + 8) + 832LL, &v8, 0);
  v11 = v2;
  v9.m128i_i64[0] = 14;
  v6 = v5;
  v10 = 0;
  sub_1E1A9C0((__int64)v5, v4, &v9);
  if ( v8 )
    sub_161E7C0((__int64)&v8, v8);
  return v6;
}
