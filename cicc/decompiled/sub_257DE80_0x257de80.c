// Function: sub_257DE80
// Address: 0x257de80
//
__int64 __fastcall sub_257DE80(__int64 a1, __int64 a2)
{
  int v2; // eax
  int v3; // r13d
  int v4; // r13d
  unsigned int v5; // r8d
  unsigned __int64 v7; // rax
  int v8; // [rsp+10h] [rbp-98h]
  int v9; // [rsp+14h] [rbp-94h]
  char v10; // [rsp+22h] [rbp-86h] BYREF
  char v11; // [rsp+23h] [rbp-85h] BYREF
  int v12; // [rsp+24h] [rbp-84h] BYREF
  _QWORD v13[2]; // [rsp+28h] [rbp-80h] BYREF
  _QWORD v14[2]; // [rsp+38h] [rbp-70h] BYREF
  _QWORD v15[2]; // [rsp+48h] [rbp-60h] BYREF
  _QWORD v16[2]; // [rsp+58h] [rbp-50h] BYREF
  __m128i v17[4]; // [rsp+68h] [rbp-40h] BYREF

  v2 = *(_DWORD *)(a1 + 124);
  v3 = *(_DWORD *)(a1 + 220);
  v13[0] = a1;
  v4 = v3 - *(_DWORD *)(a1 + 224);
  v13[1] = a2;
  v9 = v2;
  v8 = *(_DWORD *)(a1 + 128);
  v14[0] = a1;
  v14[1] = a2;
  v15[0] = a1;
  v15[1] = a2;
  v16[0] = a2;
  v16[1] = a1;
  v17[0].m128i_i64[0] = 0x2100000020LL;
  v10 = 0;
  v17[0].m128i_i64[1] = 0x2500000024LL;
  sub_2526370(
    a2,
    (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_25591C0,
    (__int64)v13,
    a1,
    v17[0].m128i_i32,
    4,
    &v10,
    1,
    0);
  v17[0].m128i_i32[0] = 2;
  sub_2526370(
    a2,
    (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_25590F0,
    (__int64)v14,
    a1,
    v17[0].m128i_i32,
    1,
    &v10,
    1,
    0);
  v17[0].m128i_i64[0] = 0xB00000005LL;
  v17[0].m128i_i32[2] = 56;
  sub_2526370(
    a2,
    (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_25941E0,
    (__int64)v15,
    a1,
    v17[0].m128i_i32,
    3,
    &v10,
    0,
    0);
  if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(sub_25096F0((_QWORD *)(a1 + 72)) + 24) + 16LL) + 8LL) != 7 )
  {
    v7 = sub_25096F0((_QWORD *)(a1 + 72));
    sub_250D230((unsigned __int64 *)v17, v7, 2, 0);
    if ( !(unsigned __int8)sub_251C230(a2, v17[0].m128i_i64, a1, 0, &v10, 0, 1) )
    {
      sub_257DDD0(a2, a1, v17, 2, &v11, 0, 0);
      if ( v11 )
      {
        v12 = 1;
        sub_2526370(
          a2,
          (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_258F980,
          (__int64)v16,
          a1,
          &v12,
          1,
          &v10,
          1,
          0);
      }
    }
  }
  v5 = 0;
  if ( v4 == *(_DWORD *)(a1 + 220) - *(_DWORD *)(a1 + 224) )
    return *(_DWORD *)(a1 + 124) - *(_DWORD *)(a1 + 128) == v9 - v8;
  return v5;
}
