// Function: sub_182D270
// Address: 0x182d270
//
__int64 __fastcall sub_182D270(char a1, __int64 *a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r15
  _QWORD *v14; // r12
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int64 *v18; // r13
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rsi
  __int64 v22; // rsi
  unsigned __int8 *v23; // rsi
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-68h] BYREF
  __int64 v26; // [rsp+10h] [rbp-60h] BYREF
  __int16 v27; // [rsp+20h] [rbp-50h]
  _BYTE v28[16]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v29; // [rsp+40h] [rbp-30h]

  v9 = *(_QWORD *)a3;
  if ( !a1 )
  {
    v29 = 257;
    v24 = sub_15A0680(v9, 0xFFFFFFFFFFFFFFLL, 0);
    return sub_1281C00(a2, a3, v24, (__int64)v28);
  }
  v27 = 257;
  v10 = sub_15A0680(v9, 0xFF00000000000000LL, 0);
  v13 = v10;
  if ( *(_BYTE *)(v10 + 16) <= 0x10u )
  {
    v14 = (_QWORD *)a3;
    if ( sub_1593BB0(v10, 0xFF00000000000000LL, v11, v12) )
      return (__int64)v14;
    if ( *(_BYTE *)(a3 + 16) <= 0x10u )
      return sub_15A2D10((__int64 *)a3, v13, a4, a5, a6);
  }
  v29 = 257;
  v16 = sub_15FB440(27, (__int64 *)a3, v13, (__int64)v28, 0);
  v17 = a2[1];
  v14 = (_QWORD *)v16;
  if ( v17 )
  {
    v18 = (unsigned __int64 *)a2[2];
    sub_157E9D0(v17 + 40, v16);
    v19 = v14[3];
    v20 = *v18;
    v14[4] = v18;
    v20 &= 0xFFFFFFFFFFFFFFF8LL;
    v14[3] = v20 | v19 & 7;
    *(_QWORD *)(v20 + 8) = v14 + 3;
    *v18 = *v18 & 7 | (unsigned __int64)(v14 + 3);
  }
  sub_164B780((__int64)v14, &v26);
  v21 = *a2;
  if ( *a2 )
  {
    v25 = *a2;
    sub_1623A60((__int64)&v25, v21, 2);
    v22 = v14[6];
    if ( v22 )
      sub_161E7C0((__int64)(v14 + 6), v22);
    v23 = (unsigned __int8 *)v25;
    v14[6] = v25;
    if ( v23 )
      sub_1623210((__int64)&v25, v23, (__int64)(v14 + 6));
  }
  return (__int64)v14;
}
