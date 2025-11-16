// Function: sub_1F3D1E0
// Address: 0x1f3d1e0
//
_QWORD *__fastcall sub_1F3D1E0(__int64 a1, __int64 *a2, __int64 a3, int a4)
{
  __int16 v5; // r13
  _QWORD *v6; // rax
  _QWORD *v7; // r12
  __int64 v8; // rdi
  unsigned __int64 *v9; // r13
  __int64 v10; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rsi
  unsigned __int8 *v14; // rsi
  unsigned __int8 *v16; // [rsp+8h] [rbp-48h] BYREF
  __int64 v17; // [rsp+10h] [rbp-40h] BYREF
  __int16 v18; // [rsp+20h] [rbp-30h]

  if ( !byte_428C1E0[8 * a4 + 5] )
    return 0;
  v5 = a4;
  if ( !(unsigned __int8)sub_15F3310(a3) )
    return 0;
  v18 = 257;
  v6 = sub_1648A60(64, 0);
  v7 = v6;
  if ( v6 )
    sub_15F9C80((__int64)v6, a2[3], v5, 1, 0);
  v8 = a2[1];
  if ( v8 )
  {
    v9 = (unsigned __int64 *)a2[2];
    sub_157E9D0(v8 + 40, (__int64)v7);
    v10 = v7[3];
    v11 = *v9;
    v7[4] = v9;
    v11 &= 0xFFFFFFFFFFFFFFF8LL;
    v7[3] = v11 | v10 & 7;
    *(_QWORD *)(v11 + 8) = v7 + 3;
    *v9 = *v9 & 7 | (unsigned __int64)(v7 + 3);
  }
  sub_164B780((__int64)v7, &v17);
  v12 = *a2;
  if ( *a2 )
  {
    v16 = (unsigned __int8 *)*a2;
    sub_1623A60((__int64)&v16, v12, 2);
    v13 = v7[6];
    if ( v13 )
      sub_161E7C0((__int64)(v7 + 6), v13);
    v14 = v16;
    v7[6] = v16;
    if ( v14 )
    {
      sub_1623210((__int64)&v16, v14, (__int64)(v7 + 6));
      return v7;
    }
  }
  return v7;
}
