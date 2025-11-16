// Function: sub_2127C00
// Address: 0x2127c00
//
__int64 __fastcall sub_2127C00(__int64 *a1, unsigned __int64 a2, double a3, double a4, double a5)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  char v8; // r12
  __int64 v9; // rax
  char v10; // r12
  __int64 *v11; // r15
  __int64 v12; // r12
  __int64 v14; // [rsp+10h] [rbp-70h] BYREF
  __int64 v15; // [rsp+18h] [rbp-68h]
  __int64 v16; // [rsp+20h] [rbp-60h] BYREF
  int v17; // [rsp+28h] [rbp-58h]
  _BYTE v18[16]; // [rsp+30h] [rbp-50h] BYREF
  const void **v19; // [rsp+40h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 40);
  v7 = *(_QWORD *)(a2 + 72);
  v8 = *(_BYTE *)v6;
  v9 = *(_QWORD *)(v6 + 8);
  v16 = v7;
  LOBYTE(v14) = v8;
  v15 = v9;
  if ( v7 )
    sub_1623A60((__int64)&v16, v7, 2);
  v17 = *(_DWORD *)(a2 + 64);
  if ( v8 )
    v10 = sub_2127930(v8);
  else
    v10 = sub_1F58D40((__int64)&v14);
  v11 = (__int64 *)a1[1];
  sub_1F40D10((__int64)v18, *a1, v11[6], v14, v15);
  v12 = sub_1D309E0(v11, 142 - ((unsigned int)((v10 & 7) == 0) - 1), (__int64)&v16, v18[8], v19, 0, a3, a4, a5, a2);
  if ( v16 )
    sub_161E7C0((__int64)&v16, v16);
  return v12;
}
