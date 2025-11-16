// Function: sub_17D2EF0
// Address: 0x17d2ef0
//
_QWORD *__fastcall sub_17D2EF0(__int64 *a1, int a2, __int64 *a3, __int64 a4, __int64 *a5, char a6, char a7)
{
  __int64 v10; // rax
  __int64 v11; // rdi
  _QWORD *v12; // r12
  unsigned __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rsi
  unsigned __int8 *v17; // rsi
  unsigned __int64 *v19; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v20; // [rsp+18h] [rbp-58h] BYREF
  char v21[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v22; // [rsp+30h] [rbp-40h]

  v22 = 257;
  v10 = sub_15FB440(a2, a3, a4, (__int64)v21, 0);
  v11 = a1[1];
  v12 = (_QWORD *)v10;
  if ( v11 )
  {
    v19 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v11 + 40, v10);
    v13 = *v19;
    v14 = v12[3] & 7LL;
    v12[4] = v19;
    v13 &= 0xFFFFFFFFFFFFFFF8LL;
    v12[3] = v13 | v14;
    *(_QWORD *)(v13 + 8) = v12 + 3;
    *v19 = *v19 & 7 | (unsigned __int64)(v12 + 3);
  }
  sub_164B780((__int64)v12, a5);
  v15 = *a1;
  if ( *a1 )
  {
    v20 = (unsigned __int8 *)*a1;
    sub_1623A60((__int64)&v20, v15, 2);
    v16 = v12[6];
    if ( v16 )
      sub_161E7C0((__int64)(v12 + 6), v16);
    v17 = v20;
    v12[6] = v20;
    if ( v17 )
      sub_1623210((__int64)&v20, v17, (__int64)(v12 + 6));
  }
  if ( a6 )
    sub_15F2310((__int64)v12, 1);
  if ( a7 )
    sub_15F2330((__int64)v12, 1);
  return v12;
}
