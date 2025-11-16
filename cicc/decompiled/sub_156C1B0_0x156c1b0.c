// Function: sub_156C1B0
// Address: 0x156c1b0
//
__int64 __fastcall sub_156C1B0(__int64 *a1, __int64 a2, __int64 a3, unsigned __int8 a4, unsigned __int8 a5)
{
  __int64 v10; // rax
  __int64 v11; // rdi
  _QWORD *v12; // r15
  unsigned __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rsi
  unsigned __int64 *v17; // [rsp+8h] [rbp-68h]
  __int64 v18; // [rsp+18h] [rbp-58h] BYREF
  char v19[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v20; // [rsp+30h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    return sub_15A2B90(a2, a4, a5);
  v20 = 257;
  v10 = sub_15FB530(a2, v19, 0);
  v11 = a1[1];
  v12 = (_QWORD *)v10;
  if ( v11 )
  {
    v17 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v11 + 40, v10);
    v13 = *v17;
    v14 = v12[3] & 7LL;
    v12[4] = v17;
    v13 &= 0xFFFFFFFFFFFFFFF8LL;
    v12[3] = v13 | v14;
    *(_QWORD *)(v13 + 8) = v12 + 3;
    *v17 = *v17 & 7 | (unsigned __int64)(v12 + 3);
  }
  sub_164B780(v12, a3);
  v15 = *a1;
  if ( *a1 )
  {
    v18 = *a1;
    sub_1623A60(&v18, v15, 2);
    if ( v12[6] )
      sub_161E7C0(v12 + 6);
    v16 = v18;
    v12[6] = v18;
    if ( v16 )
      sub_1623210(&v18, v16, v12 + 6);
  }
  if ( a4 )
    sub_15F2310(v12, 1);
  if ( a5 )
    sub_15F2330(v12, 1);
  return (__int64)v12;
}
