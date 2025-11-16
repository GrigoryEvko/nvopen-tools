// Function: sub_17DAD20
// Address: 0x17dad20
//
unsigned __int64 __fastcall sub_17DAD20(__int128 a1, double a2, double a3, double a4)
{
  __int128 v4; // kr00_16
  _QWORD *v5; // rax
  __int64 v6; // rax
  __int64 *v7; // rax
  __int64 v8; // rdi
  _QWORD *v9; // r14
  __int64 v10; // rdi
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 result; // rax
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rax
  unsigned __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-E8h]
  __int64 *v27; // [rsp+8h] [rbp-E8h]
  __int64 v28; // [rsp+10h] [rbp-E0h]
  __int64 v29; // [rsp+10h] [rbp-E0h]
  __int64 v30; // [rsp+10h] [rbp-E0h]
  __int64 v31; // [rsp+10h] [rbp-E0h]
  unsigned __int64 *v32; // [rsp+18h] [rbp-D8h]
  __int64 ***v33; // [rsp+20h] [rbp-D0h]
  __int64 *v34; // [rsp+28h] [rbp-C8h]
  __int64 v35[2]; // [rsp+30h] [rbp-C0h] BYREF
  __int16 v36; // [rsp+40h] [rbp-B0h]
  __int64 v37[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v38; // [rsp+60h] [rbp-90h]
  __int64 v39; // [rsp+70h] [rbp-80h] BYREF
  __int64 v40; // [rsp+78h] [rbp-78h]
  unsigned __int64 *v41; // [rsp+80h] [rbp-70h]

  v4 = a1;
  sub_17CE510((__int64)&v39, *((__int64 *)&a1 + 1), 0, 0, 0);
  if ( (*(_BYTE *)(*((_QWORD *)&a1 + 1) + 23LL) & 0x40) != 0 )
    v5 = *(_QWORD **)(*((_QWORD *)&a1 + 1) - 8LL);
  else
    v5 = (_QWORD *)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF));
  *((_QWORD *)&a1 + 1) = *v5;
  v34 = sub_17D4DA0(a1);
  if ( (*(_BYTE *)(*((_QWORD *)&v4 + 1) + 23LL) & 0x40) != 0 )
    v6 = *(_QWORD *)(*((_QWORD *)&v4 + 1) - 8LL);
  else
    v6 = *((_QWORD *)&v4 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&v4 + 1) + 20LL) & 0xFFFFFFF);
  *((_QWORD *)&a1 + 1) = *(_QWORD *)(v6 + 24);
  v7 = sub_17D4DA0(a1);
  v8 = *(_QWORD *)(*((_QWORD *)&v4 + 1) - 48LL);
  v33 = (__int64 ***)v7;
  v36 = 257;
  if ( *(_BYTE *)(v8 + 16) > 0x10u )
  {
    v38 = 257;
    v23 = sub_15FB630((__int64 *)v8, (__int64)v37, 0);
    v9 = (_QWORD *)v23;
    if ( v40 )
    {
      v32 = v41;
      sub_157E9D0(v40 + 40, v23);
      v24 = *v32;
      v25 = v9[3] & 7LL;
      v9[4] = v32;
      v24 &= 0xFFFFFFFFFFFFFFF8LL;
      v9[3] = v24 | v25;
      *(_QWORD *)(v24 + 8) = v9 + 3;
      *v32 = *v32 & 7 | (unsigned __int64)(v9 + 3);
    }
    sub_164B780((__int64)v9, v35);
    sub_12A86E0(&v39, (__int64)v9);
  }
  else
  {
    v9 = (_QWORD *)sub_15A2B00((__int64 *)v8, a2, a3, a4);
  }
  v10 = *(_QWORD *)(*((_QWORD *)&v4 + 1) - 24LL);
  v36 = 257;
  if ( *(_BYTE *)(v10 + 16) > 0x10u )
  {
    v38 = 257;
    v19 = sub_15FB630((__int64 *)v10, (__int64)v37, 0);
    v20 = v19;
    if ( v40 )
    {
      v30 = v19;
      v27 = (__int64 *)v41;
      sub_157E9D0(v40 + 40, v19);
      v20 = v30;
      v21 = *(_QWORD *)(v30 + 24);
      v22 = *v27;
      *(_QWORD *)(v30 + 32) = v27;
      v22 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v30 + 24) = v22 | v21 & 7;
      *(_QWORD *)(v22 + 8) = v30 + 24;
      *v27 = *v27 & 7 | (v30 + 24);
    }
    v31 = v20;
    sub_164B780(v20, v35);
    sub_12A86E0(&v39, v31);
    v11 = v31;
  }
  else
  {
    v11 = sub_15A2B00((__int64 *)v10, a2, a3, a4);
  }
  if ( *v9 != *v34 )
  {
    v28 = v11;
    v38 = 257;
    v9 = (_QWORD *)sub_17CE200(&v39, (__int64)v9, (__int64 **)*v34, 0, v37);
    v38 = 257;
    v11 = sub_17CE200(&v39, v28, *v33, 0, v37);
  }
  v26 = v11;
  v38 = 257;
  v12 = sub_1281C00(&v39, (__int64)v34, (__int64)v33, (__int64)v37);
  v38 = 257;
  v29 = v12;
  v13 = sub_1281C00(&v39, (__int64)v9, (__int64)v33, (__int64)v37);
  v38 = 257;
  v14 = v13;
  v15 = sub_1281C00(&v39, (__int64)v34, v26, (__int64)v37);
  v38 = 257;
  v36 = 257;
  v16 = sub_156D390(&v39, v14, v15, (__int64)v35);
  v17 = sub_156D390(&v39, v29, v16, (__int64)v37);
  sub_17D4920(v4, *((__int64 **)&v4 + 1), v17);
  result = *(_QWORD *)(v4 + 8);
  if ( *(_DWORD *)(result + 156) )
    result = sub_17D9C10((_QWORD *)v4, *((__int64 *)&v4 + 1));
  if ( v39 )
    return sub_161E7C0((__int64)&v39, v39);
  return result;
}
