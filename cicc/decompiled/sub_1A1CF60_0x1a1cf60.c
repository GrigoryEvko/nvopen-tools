// Function: sub_1A1CF60
// Address: 0x1a1cf60
//
_QWORD *__fastcall sub_1A1CF60(__int64 *a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  _QWORD *v6; // rax
  _QWORD *v7; // r12
  __int64 v8; // r13
  unsigned __int64 *v9; // r14
  __m128i v10; // xmm0
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rsi
  unsigned __int8 *v15; // rsi
  __m128i v17; // [rsp+0h] [rbp-90h] BYREF
  __int64 v18; // [rsp+10h] [rbp-80h]
  __m128i v19; // [rsp+20h] [rbp-70h] BYREF
  __int64 v20; // [rsp+30h] [rbp-60h]
  _QWORD v21[2]; // [rsp+40h] [rbp-50h] BYREF

  LOWORD(v18) = 257;
  v6 = sub_1648A60(64, 2u);
  v7 = v6;
  if ( v6 )
    sub_15F9650((__int64)v6, a2, a3, a4, 0);
  v8 = a1[1];
  v9 = (unsigned __int64 *)a1[2];
  v10 = _mm_loadu_si128(&v17);
  v20 = v18;
  v19 = v10;
  if ( v8 )
  {
    sub_157E9D0(v8 + 40, (__int64)v7);
    v11 = *v9;
    v12 = v7[3];
    v7[4] = v9;
    v11 &= 0xFFFFFFFFFFFFFFF8LL;
    v7[3] = v11 | v12 & 7;
    *(_QWORD *)(v11 + 8) = v7 + 3;
    *v9 = *v9 & 7 | (unsigned __int64)(v7 + 3);
  }
  sub_164B780((__int64)v7, v19.m128i_i64);
  v13 = *a1;
  if ( *a1 )
  {
    v21[0] = *a1;
    sub_1623A60((__int64)v21, v13, 2);
    v14 = v7[6];
    if ( v14 )
      sub_161E7C0((__int64)(v7 + 6), v14);
    v15 = (unsigned __int8 *)v21[0];
    v7[6] = v21[0];
    if ( v15 )
      sub_1623210((__int64)v21, v15, (__int64)(v7 + 6));
  }
  return v7;
}
