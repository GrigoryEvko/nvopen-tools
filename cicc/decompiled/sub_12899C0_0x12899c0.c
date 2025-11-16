// Function: sub_12899C0
// Address: 0x12899c0
//
__int64 __fastcall sub_12899C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5, unsigned __int8 a6)
{
  __int64 v11; // rax
  __int64 v12; // rdi
  _QWORD *v13; // r13
  unsigned __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rsi
  unsigned __int64 *v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+18h] [rbp-58h] BYREF
  char v20[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v21; // [rsp+30h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(a3 + 16) <= 0x10u )
    return sub_15A2B30(a2, a3, a5, a6);
  v21 = 257;
  v11 = sub_15FB440(11, a2, a3, v20, 0);
  v12 = a1[1];
  v13 = (_QWORD *)v11;
  if ( v12 )
  {
    v18 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v12 + 40, v11);
    v14 = *v18;
    v15 = v13[3] & 7LL;
    v13[4] = v18;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    v13[3] = v14 | v15;
    *(_QWORD *)(v14 + 8) = v13 + 3;
    *v18 = *v18 & 7 | (unsigned __int64)(v13 + 3);
  }
  sub_164B780(v13, a4);
  v16 = *a1;
  if ( *a1 )
  {
    v19 = *a1;
    sub_1623A60(&v19, v16, 2);
    if ( v13[6] )
      sub_161E7C0(v13 + 6);
    v17 = v19;
    v13[6] = v19;
    if ( v17 )
      sub_1623210(&v19, v17, v13 + 6);
  }
  if ( a5 )
    sub_15F2310(v13, 1);
  if ( a6 )
    sub_15F2330(v13, 1);
  return (__int64)v13;
}
