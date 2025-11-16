// Function: sub_156A7D0
// Address: 0x156a7d0
//
__int64 __fastcall sub_156A7D0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 v11; // rax
  _QWORD *v12; // r12
  __int64 v13; // rdi
  unsigned __int64 *v14; // r13
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // [rsp+8h] [rbp-68h]
  __int64 v20; // [rsp+18h] [rbp-58h] BYREF
  char v21[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v22; // [rsp+30h] [rbp-40h]

  v9 = sub_1599580(a1[3], a4, a5, a4);
  if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(a3 + 16) <= 0x10u && *(_BYTE *)(v9 + 16) <= 0x10u )
    return sub_15A3950(a2, a3, v9, 0);
  v19 = v9;
  v22 = 257;
  v11 = sub_1648A60(56, 3);
  v12 = (_QWORD *)v11;
  if ( v11 )
    sub_15FA660(v11, a2, a3, v19, v21, 0);
  v13 = a1[1];
  if ( v13 )
  {
    v14 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v13 + 40, v12);
    v15 = v12[3];
    v16 = *v14;
    v12[4] = v14;
    v16 &= 0xFFFFFFFFFFFFFFF8LL;
    v12[3] = v16 | v15 & 7;
    *(_QWORD *)(v16 + 8) = v12 + 3;
    *v14 = *v14 & 7 | (unsigned __int64)(v12 + 3);
  }
  sub_164B780(v12, a6);
  v17 = *a1;
  if ( *a1 )
  {
    v20 = *a1;
    sub_1623A60(&v20, v17, 2);
    if ( v12[6] )
      sub_161E7C0(v12 + 6);
    v18 = v20;
    v12[6] = v20;
    if ( v18 )
      sub_1623210(&v20, v18, v12 + 6);
  }
  return (__int64)v12;
}
