// Function: sub_169E320
// Address: 0x169e320
//
__int64 __fastcall sub_169E320(_QWORD *a1, __int64 *a2, __int16 *a3)
{
  __int16 *v4; // rax
  __int16 *v6; // r12
  __int64 v7; // [rsp+8h] [rbp-98h]
  _BYTE v8[32]; // [rsp+10h] [rbp-90h] BYREF
  char v9[8]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v10[3]; // [rsp+38h] [rbp-68h] BYREF
  char v11[8]; // [rsp+50h] [rbp-50h] BYREF
  _QWORD v12[9]; // [rsp+58h] [rbp-48h] BYREF

  v4 = (__int16 *)sub_16982C0();
  if ( a3 != v4 )
    return sub_1698450((__int64)a1, (__int64)a2);
  v6 = v4;
  v7 = *a2;
  sub_1698450((__int64)v8, (__int64)a2);
  sub_1698450((__int64)v11, (__int64)v8);
  sub_169E320(v10, v11, v7);
  sub_1698460((__int64)v11);
  if ( v6 == word_42AE9D0 )
    sub_169C4E0(v12, (__int64)v6);
  else
    sub_1698360((__int64)v12, (__int64)word_42AE9D0);
  sub_169C810(a1, (__int64)v6, (__int64)v9, (__int64)v11);
  sub_127D120(v12);
  sub_127D120(v10);
  return sub_1698460((__int64)v8);
}
