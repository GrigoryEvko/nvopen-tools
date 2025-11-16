// Function: sub_18179B0
// Address: 0x18179b0
//
__int64 __fastcall sub_18179B0(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 v3; // rax
  unsigned __int8 *v4; // rsi
  __int64 v5; // rax
  unsigned __int8 *v6; // rsi
  __int64 v7; // r12
  unsigned __int8 *v9[2]; // [rsp-98h] [rbp-98h] BYREF
  __int16 v10; // [rsp-88h] [rbp-88h]
  unsigned __int8 *v11; // [rsp-78h] [rbp-78h] BYREF
  __int64 v12; // [rsp-70h] [rbp-70h]
  __int64 v13; // [rsp-68h] [rbp-68h]
  __int64 v14; // [rsp-60h] [rbp-60h]
  __int64 v15; // [rsp-58h] [rbp-58h]
  int v16; // [rsp-50h] [rbp-50h]
  __int64 v17; // [rsp-48h] [rbp-48h]
  __int64 v18; // [rsp-40h] [rbp-40h]

  v1 = *(_QWORD *)(a1[1] + 80LL);
  if ( !v1 )
    BUG();
  v2 = *(_QWORD *)(v1 + 24);
  if ( !v2 )
  {
    v11 = 0;
    v13 = 0;
    v14 = sub_16498A0(0);
    v15 = 0;
    v16 = 0;
    v17 = 0;
    v18 = 0;
    v12 = 0;
    BUG();
  }
  v15 = 0;
  v14 = sub_16498A0(v2 - 24);
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v3 = *(_QWORD *)(v2 + 16);
  v4 = *(unsigned __int8 **)(v2 + 24);
  v13 = v2;
  v11 = 0;
  v12 = v3;
  v9[0] = v4;
  if ( v4 )
  {
    sub_1623A60((__int64)v9, (__int64)v4, 2);
    if ( v11 )
      sub_161E7C0((__int64)&v11, (__int64)v11);
    v11 = v9[0];
    if ( v9[0] )
      sub_1623210((__int64)v9, v9[0], (__int64)&v11);
  }
  v10 = 257;
  v5 = sub_1285290(
         (__int64 *)&v11,
         *(_QWORD *)(**(_QWORD **)(*a1 + 256LL) + 24LL),
         *(_QWORD *)(*a1 + 256LL),
         0,
         0,
         (__int64)v9,
         0);
  v6 = v11;
  a1[13] = v5;
  v7 = v5;
  if ( v6 )
    sub_161E7C0((__int64)&v11, (__int64)v6);
  return v7;
}
