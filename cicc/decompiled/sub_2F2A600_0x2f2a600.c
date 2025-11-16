// Function: sub_2F2A600
// Address: 0x2f2a600
//
_QWORD *__fastcall sub_2F2A600(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int32 a5)
{
  __int64 v8; // rsi
  bool v9; // zf
  _QWORD *v10; // r14
  __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v15; // rdx
  __int64 v16; // rax
  __m128i v18; // [rsp+10h] [rbp-60h] BYREF
  __int64 v19; // [rsp+20h] [rbp-50h]
  __int64 v20; // [rsp+28h] [rbp-48h]
  __int64 v21; // [rsp+30h] [rbp-40h]

  v8 = *a3;
  v9 = (*(_BYTE *)(a2 + 44) & 4) == 0;
  v10 = *(_QWORD **)(a1 + 32);
  v18.m128i_i64[0] = *a3;
  if ( v9 )
  {
    if ( v8 )
      sub_B96E90((__int64)&v18, v8, 1);
    v11 = (__int64)sub_2E7B380(v10, a4, (unsigned __int8 **)&v18, 0);
    if ( v18.m128i_i64[0] )
      sub_B91220((__int64)&v18, v18.m128i_i64[0]);
    sub_2E31040((__int64 *)(a1 + 40), v11);
    v15 = *(_QWORD *)a2;
    v16 = *(_QWORD *)v11;
    *(_QWORD *)(v11 + 8) = a2;
    v15 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v11 = v15 | v16 & 7;
    *(_QWORD *)(v15 + 8) = v11;
    *(_QWORD *)a2 = v11 | *(_QWORD *)a2 & 7LL;
    v12 = a3[1];
    if ( v12 )
      goto LABEL_7;
  }
  else
  {
    if ( v8 )
      sub_B96E90((__int64)&v18, v8, 1);
    v11 = (__int64)sub_2E7B380(v10, a4, (unsigned __int8 **)&v18, 0);
    if ( v18.m128i_i64[0] )
      sub_B91220((__int64)&v18, v18.m128i_i64[0]);
    sub_2E326B0(a1, (__int64 *)a2, v11);
    v12 = a3[1];
    if ( v12 )
LABEL_7:
      sub_2E882B0(v11, (__int64)v10, v12);
  }
  v13 = a3[2];
  if ( v13 )
    sub_2E88680(v11, (__int64)v10, v13);
  v18.m128i_i64[0] = 0x10000000;
  v18.m128i_i32[2] = a5;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  sub_2E8EAD0(v11, (__int64)v10, &v18);
  return v10;
}
