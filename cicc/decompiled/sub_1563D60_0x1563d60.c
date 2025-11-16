// Function: sub_1563D60
// Address: 0x1563d60
//
__int64 __fastcall sub_1563D60(__int64 *a1, __int64 *a2, __int32 a3, __int64 a4)
{
  __int64 v5; // r12
  __m128i v7; // [rsp+0h] [rbp-80h] BYREF
  int v8; // [rsp+10h] [rbp-70h] BYREF
  _QWORD *v9; // [rsp+18h] [rbp-68h]
  int *v10; // [rsp+20h] [rbp-60h]
  int *v11; // [rsp+28h] [rbp-58h]
  __int64 v12; // [rsp+30h] [rbp-50h]
  __int64 v13; // [rsp+38h] [rbp-48h]
  __int64 v14; // [rsp+40h] [rbp-40h]
  __int64 v15; // [rsp+48h] [rbp-38h]
  __int64 v16; // [rsp+50h] [rbp-30h]
  __int64 v17; // [rsp+58h] [rbp-28h]

  v7.m128i_i64[0] = 0;
  v8 = 0;
  v9 = 0;
  v10 = &v8;
  v11 = &v8;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  sub_1562E30(&v7, a4);
  v5 = sub_15637E0(a1, a2, a3, &v7);
  sub_155CC10(v9);
  return v5;
}
