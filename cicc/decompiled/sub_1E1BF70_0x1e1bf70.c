// Function: sub_1E1BF70
// Address: 0x1e1bf70
//
__int64 __fastcall sub_1E1BF70(__int64 a1, __int64 *a2, __int64 a3, int a4)
{
  _QWORD *v5; // rax
  __int64 v6; // r15
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  _QWORD *v13; // [rsp+8h] [rbp-68h]
  __m128i v14; // [rsp+10h] [rbp-60h] BYREF
  __int64 v15; // [rsp+20h] [rbp-50h]
  __int64 v16; // [rsp+28h] [rbp-48h]

  v5 = sub_1E16520(a3);
  v6 = *(_QWORD *)(a1 + 56);
  v13 = v5;
  v7 = (__int64)sub_1E0B640(v6, *(_QWORD *)(a3 + 16), (__int64 *)(a3 + 64), 0);
  sub_1DD5BA0((__int64 *)(a1 + 16), v7);
  v8 = *(_QWORD *)v7;
  v9 = *a2;
  *(_QWORD *)(v7 + 8) = a2;
  v9 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v7 = v9 | v8 & 7;
  *(_QWORD *)(v9 + 8) = v7;
  v10 = *a2;
  LODWORD(v16) = a4;
  v14.m128i_i64[0] = 5;
  v15 = 0;
  *a2 = v7 | v10 & 7;
  sub_1E1A9C0(v7, v6, &v14);
  v14.m128i_i64[0] = 1;
  v15 = 0;
  v16 = 0;
  sub_1E1A9C0(v7, v6, &v14);
  v16 = sub_1E16500(a3);
  v14.m128i_i64[0] = 14;
  v15 = 0;
  sub_1E1A9C0(v7, v6, &v14);
  v14.m128i_i64[0] = 14;
  v15 = 0;
  v16 = (__int64)v13;
  sub_1E1A9C0(v7, v6, &v14);
  return v7;
}
