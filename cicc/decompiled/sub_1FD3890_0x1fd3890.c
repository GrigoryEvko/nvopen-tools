// Function: sub_1FD3890
// Address: 0x1fd3890
//
__int64 __fastcall sub_1FD3890(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int32 a5)
{
  __int64 v6; // r14
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __m128i v12; // [rsp+0h] [rbp-60h] BYREF
  __int64 v13; // [rsp+10h] [rbp-50h]
  __int64 v14; // [rsp+18h] [rbp-48h]
  __int64 v15; // [rsp+20h] [rbp-40h]

  v6 = *(_QWORD *)(a1 + 56);
  v7 = (__int64)sub_1E0B640(v6, a4, a3, 0);
  sub_1DD5BA0((__int64 *)(a1 + 16), v7);
  v8 = *(_QWORD *)v7;
  v9 = *a2;
  *(_QWORD *)(v7 + 8) = a2;
  v9 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v7 = v9 | v8 & 7;
  *(_QWORD *)(v9 + 8) = v7;
  v10 = *a2;
  v12.m128i_i32[1] = 0;
  v12.m128i_i32[2] = a5;
  v13 = 0;
  *a2 = v7 | v10 & 7;
  v14 = 0;
  v15 = 0;
  v12.m128i_i32[0] = 0x10000000;
  sub_1E1A9C0(v7, v6, &v12);
  return v6;
}
