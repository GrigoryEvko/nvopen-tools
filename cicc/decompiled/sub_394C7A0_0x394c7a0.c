// Function: sub_394C7A0
// Address: 0x394c7a0
//
__int64 __fastcall sub_394C7A0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        unsigned __int8 a5,
        double a6,
        double a7,
        double a8)
{
  __int64 v11; // rax
  __int64 v12; // rdi
  _QWORD *v13; // r12
  unsigned __int64 *v14; // r14
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rsi
  unsigned __int8 *v19; // rsi
  unsigned __int8 *v20; // [rsp+8h] [rbp-48h] BYREF
  _BYTE v21[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v22; // [rsp+20h] [rbp-30h]

  if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(a3 + 16) <= 0x10u )
    return sub_15A2C70((__int64 *)a2, a3, a5, a6, a7, a8);
  v22 = 257;
  if ( a5 )
  {
    v13 = (_QWORD *)sub_15FB440(17, (__int64 *)a2, a3, (__int64)v21, 0);
    sub_15F2350((__int64)v13, 1);
    v12 = a1[1];
    if ( !v12 )
      goto LABEL_7;
    goto LABEL_6;
  }
  v11 = sub_15FB440(17, (__int64 *)a2, a3, (__int64)v21, 0);
  v12 = a1[1];
  v13 = (_QWORD *)v11;
  if ( v12 )
  {
LABEL_6:
    v14 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v12 + 40, (__int64)v13);
    v15 = v13[3];
    v16 = *v14;
    v13[4] = v14;
    v16 &= 0xFFFFFFFFFFFFFFF8LL;
    v13[3] = v16 | v15 & 7;
    *(_QWORD *)(v16 + 8) = v13 + 3;
    *v14 = *v14 & 7 | (unsigned __int64)(v13 + 3);
  }
LABEL_7:
  sub_164B780((__int64)v13, a4);
  v17 = *a1;
  if ( *a1 )
  {
    v20 = (unsigned __int8 *)*a1;
    sub_1623A60((__int64)&v20, v17, 2);
    v18 = v13[6];
    if ( v18 )
      sub_161E7C0((__int64)(v13 + 6), v18);
    v19 = v20;
    v13[6] = v20;
    if ( v19 )
      sub_1623210((__int64)&v20, v19, (__int64)(v13 + 6));
  }
  return (__int64)v13;
}
