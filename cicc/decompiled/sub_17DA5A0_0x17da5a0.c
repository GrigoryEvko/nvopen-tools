// Function: sub_17DA5A0
// Address: 0x17da5a0
//
unsigned __int64 __fastcall sub_17DA5A0(__int128 a1, double a2, double a3, double a4)
{
  __int64 *v4; // r15
  __int64 v5; // r13
  __int64 *v6; // rax
  __int64 v7; // rax
  __int64 *v8; // r9
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 ***v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r8
  _BYTE *v18; // rax
  __int64 v19; // rax
  __int64 v20; // r13
  unsigned __int64 result; // rax
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // [rsp+0h] [rbp-150h]
  __int64 *v29; // [rsp+8h] [rbp-148h]
  __int64 *v30; // [rsp+10h] [rbp-140h]
  __int64 v31; // [rsp+18h] [rbp-138h]
  __int64 v32; // [rsp+18h] [rbp-138h]
  __int64 v33; // [rsp+18h] [rbp-138h]
  __int64 *v34; // [rsp+20h] [rbp-130h]
  __int64 v35; // [rsp+20h] [rbp-130h]
  __int64 v36; // [rsp+20h] [rbp-130h]
  __int64 v37; // [rsp+20h] [rbp-130h]
  __int64 v38; // [rsp+20h] [rbp-130h]
  __int64 *v39; // [rsp+28h] [rbp-128h]
  __int64 v40; // [rsp+28h] [rbp-128h]
  __int64 v41; // [rsp+30h] [rbp-120h] BYREF
  __int16 v42; // [rsp+40h] [rbp-110h]
  char v43[16]; // [rsp+50h] [rbp-100h] BYREF
  __int16 v44; // [rsp+60h] [rbp-F0h]
  char v45[16]; // [rsp+70h] [rbp-E0h] BYREF
  __int16 v46; // [rsp+80h] [rbp-D0h]
  __int64 v47[2]; // [rsp+90h] [rbp-C0h] BYREF
  __int16 v48; // [rsp+A0h] [rbp-B0h]
  __int64 v49[2]; // [rsp+B0h] [rbp-A0h] BYREF
  __int16 v50; // [rsp+C0h] [rbp-90h]
  __int64 v51; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v52; // [rsp+D8h] [rbp-78h]
  __int64 *v53; // [rsp+E0h] [rbp-70h]

  v4 = (__int64 *)*((_QWORD *)&a1 + 1);
  sub_17CE510((__int64)&v51, *((__int64 *)&a1 + 1), 0, 0, 0);
  v5 = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 24LL);
  *((_QWORD *)&a1 + 1) = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 48LL);
  v31 = *(v4 - 6);
  v6 = sub_17D4DA0(a1);
  *((_QWORD *)&a1 + 1) = v5;
  v39 = v6;
  v50 = 257;
  v34 = sub_17D4DA0(a1);
  v32 = sub_12A95D0(&v51, v31, *v39, (__int64)v49);
  v50 = 257;
  v7 = sub_12A95D0(&v51, v5, *v34, (__int64)v49);
  v8 = (__int64 *)v32;
  v9 = v7;
  v48 = 257;
  if ( *(_BYTE *)(v32 + 16) > 0x10u
    || *(_BYTE *)(v7 + 16) > 0x10u
    || (v28 = v7,
        v10 = sub_15A2A30((__int64 *)0x1C, (__int64 *)v32, v7, 0, 0, a2, a3, a4),
        v8 = (__int64 *)v32,
        v9 = v28,
        (v33 = v10) == 0) )
  {
    v50 = 257;
    v33 = sub_15FB440(28, v8, v9, (__int64)v49, 0);
    if ( v52 )
    {
      v30 = v53;
      sub_157E9D0(v52 + 40, v33);
      v26 = *v30;
      v27 = *(_QWORD *)(v33 + 24) & 7LL;
      *(_QWORD *)(v33 + 32) = v30;
      v26 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v33 + 24) = v26 | v27;
      *(_QWORD *)(v26 + 8) = v33 + 24;
      *v30 = *v30 & 7 | (v33 + 24);
    }
    sub_164B780(v33, v47);
    sub_12A86E0(&v51, v33);
  }
  v50 = 257;
  v11 = (__int64 ***)sub_156D390(&v51, (__int64)v39, (__int64)v34, (__int64)v49);
  v40 = sub_15A06D0(*v11, (__int64)v39, v12, v13);
  v14 = sub_15A04A0(*v11);
  v48 = 257;
  v15 = v14;
  v46 = 257;
  v44 = 257;
  v42 = 257;
  if ( *((_BYTE *)v11 + 16) > 0x10u
    || *(_BYTE *)(v14 + 16) > 0x10u
    || (v35 = v14, v16 = sub_15A2A30((__int64 *)0x1C, (__int64 *)v11, v14, 0, 0, a2, a3, a4),
                   v15 = v35,
                   (v17 = v16) == 0) )
  {
    v50 = 257;
    v22 = sub_15FB440(28, (__int64 *)v11, v15, (__int64)v49, 0);
    v23 = v22;
    if ( v52 )
    {
      v37 = v22;
      v29 = v53;
      sub_157E9D0(v52 + 40, v22);
      v23 = v37;
      v24 = *(_QWORD *)(v37 + 24);
      v25 = *v29;
      *(_QWORD *)(v37 + 32) = v29;
      v25 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v37 + 24) = v25 | v24 & 7;
      *(_QWORD *)(v25 + 8) = v37 + 24;
      *v29 = *v29 & 7 | (v37 + 24);
    }
    v38 = v23;
    sub_164B780(v23, &v41);
    sub_12A86E0(&v51, v38);
    v17 = v38;
  }
  v18 = (_BYTE *)sub_1281C00(&v51, v17, v33, (__int64)v43);
  v36 = sub_12AA0C0(&v51, 0x20u, v18, v40, (__int64)v45);
  v50 = 257;
  v19 = sub_12AA0C0(&v51, 0x21u, v11, v40, (__int64)v49);
  v20 = sub_1281C00(&v51, v19, v36, (__int64)v47);
  v50 = 259;
  v49[0] = (__int64)"_msprop_icmp";
  sub_164B780(v20, v49);
  sub_17D4920(a1, v4, v20);
  result = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(result + 156) )
    result = sub_17D9C10((_QWORD *)a1, (__int64)v4);
  if ( v51 )
    return sub_161E7C0((__int64)&v51, v51);
  return result;
}
