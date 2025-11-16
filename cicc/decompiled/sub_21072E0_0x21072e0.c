// Function: sub_21072E0
// Address: 0x21072e0
//
__int64 __fastcall sub_21072E0(unsigned int a1, __int64 a2, __int64 *a3, __int64 a4, size_t a5, __int64 a6)
{
  __int32 v9; // eax
  __int64 v10; // rsi
  __int64 v11; // r15
  __int32 v12; // r14d
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v18; // [rsp+18h] [rbp-68h] BYREF
  __m128i v19; // [rsp+20h] [rbp-60h] BYREF
  __int64 v20; // [rsp+30h] [rbp-50h]
  __int64 v21; // [rsp+38h] [rbp-48h]
  __int64 v22; // [rsp+40h] [rbp-40h]

  v9 = sub_1E6B9A0(a5, a4, (unsigned __int8 *)byte_3F871B3, 0, a5, a6);
  v10 = *(_QWORD *)(a6 + 8) + ((unsigned __int64)a1 << 6);
  v11 = *(_QWORD *)(a2 + 56);
  v12 = v9;
  v18 = 0;
  v13 = (__int64)sub_1E0B640(v11, v10, &v18, 0);
  sub_1DD5BA0((__int64 *)(a2 + 16), v13);
  v14 = *(_QWORD *)v13;
  v15 = *a3;
  *(_QWORD *)(v13 + 8) = a3;
  v15 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v13 = v15 | v14 & 7;
  *(_QWORD *)(v15 + 8) = v13;
  v16 = *a3;
  v19.m128i_i32[1] = 0;
  v20 = 0;
  v19.m128i_i32[2] = v12;
  *a3 = v13 | v16 & 7;
  v21 = 0;
  v22 = 0;
  v19.m128i_i32[0] = 0x10000000;
  sub_1E1A9C0(v13, v11, &v19);
  if ( v18 )
    sub_161E7C0((__int64)&v18, v18);
  return v11;
}
