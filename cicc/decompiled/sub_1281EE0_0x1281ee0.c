// Function: sub_1281EE0
// Address: 0x1281ee0
//
__int64 __fastcall sub_1281EE0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5, unsigned __int8 a6)
{
  __int64 v10; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  _QWORD *v14; // r12
  unsigned __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rsi
  unsigned __int64 *v19; // [rsp+8h] [rbp-68h]
  __int64 v20; // [rsp+18h] [rbp-58h] BYREF
  char v21[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v22; // [rsp+30h] [rbp-40h]

  v10 = sub_15A0680(*(_QWORD *)a2, a3, 0);
  if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(v10 + 16) <= 0x10u )
    return sub_15A2D50(a2, v10, a5, a6);
  v22 = 257;
  v12 = sub_15FB440(23, a2, v10, v21, 0);
  v13 = a1[1];
  v14 = (_QWORD *)v12;
  if ( v13 )
  {
    v19 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v13 + 40, v12);
    v15 = *v19;
    v16 = v14[3] & 7LL;
    v14[4] = v19;
    v15 &= 0xFFFFFFFFFFFFFFF8LL;
    v14[3] = v15 | v16;
    *(_QWORD *)(v15 + 8) = v14 + 3;
    *v19 = *v19 & 7 | (unsigned __int64)(v14 + 3);
  }
  sub_164B780(v14, a4);
  v17 = *a1;
  if ( *a1 )
  {
    v20 = *a1;
    sub_1623A60(&v20, v17, 2);
    if ( v14[6] )
      sub_161E7C0(v14 + 6);
    v18 = v20;
    v14[6] = v20;
    if ( v18 )
      sub_1623210(&v20, v18, v14 + 6);
  }
  if ( a5 )
    sub_15F2310(v14, 1);
  if ( a6 )
    sub_15F2330(v14, 1);
  return (__int64)v14;
}
