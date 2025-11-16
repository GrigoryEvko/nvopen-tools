// Function: sub_17DBC30
// Address: 0x17dbc30
//
unsigned __int64 __fastcall sub_17DBC30(__int128 a1, double a2, double a3, double a4)
{
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 *v7; // rax
  __int64 *v8; // r14
  __int64 v9; // rax
  __int64 v10; // r15
  bool v11; // al
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 *v16; // rdi
  __int64 v17; // r8
  __int64 v18; // rax
  _BYTE *v19; // rax
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // r14
  __int64 *v26; // rax
  __int64 v27; // r8
  __int64 v28; // rax
  _BYTE *v29; // rdx
  _BYTE *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r15
  __int64 v34; // r14
  __int64 v36; // rax
  __int64 *v37; // rbx
  __int64 v38; // rax
  __int64 v39; // rcx
  _QWORD *v40; // rax
  _QWORD *v41; // rax
  _QWORD *v42; // [rsp+18h] [rbp-148h]
  __int64 v43; // [rsp+20h] [rbp-140h]
  __int64 v44; // [rsp+28h] [rbp-138h]
  __int64 v45; // [rsp+28h] [rbp-138h]
  __int64 v46; // [rsp+28h] [rbp-138h]
  __int64 v47; // [rsp+28h] [rbp-138h]
  __int64 v48; // [rsp+30h] [rbp-130h]
  __int64 v49; // [rsp+30h] [rbp-130h]
  _BYTE v50[16]; // [rsp+40h] [rbp-120h] BYREF
  __int16 v51; // [rsp+50h] [rbp-110h]
  _BYTE v52[16]; // [rsp+60h] [rbp-100h] BYREF
  __int16 v53; // [rsp+70h] [rbp-F0h]
  __int64 v54[2]; // [rsp+80h] [rbp-E0h] BYREF
  __int16 v55; // [rsp+90h] [rbp-D0h]
  __int64 v56[2]; // [rsp+A0h] [rbp-C0h] BYREF
  __int16 v57; // [rsp+B0h] [rbp-B0h]
  _BYTE v58[16]; // [rsp+C0h] [rbp-A0h] BYREF
  __int16 v59; // [rsp+D0h] [rbp-90h]
  __int64 v60; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v61; // [rsp+E8h] [rbp-78h]
  __int64 *v62; // [rsp+F0h] [rbp-70h]

  v4 = *((_QWORD *)&a1 + 1);
  v42 = (_QWORD *)a1;
  sub_17CE510((__int64)&v60, *((__int64 *)&a1 + 1), 0, 0, 0);
  v6 = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 24LL);
  *((_QWORD *)&a1 + 1) = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 48LL);
  v5 = *((_QWORD *)&a1 + 1);
  v7 = sub_17D4DA0(a1);
  *((_QWORD *)&a1 + 1) = v6;
  v48 = (__int64)v7;
  v8 = sub_17D4DA0(a1);
  v59 = 257;
  v9 = sub_12A95D0(&v60, v5, *(_QWORD *)v48, (__int64)v58);
  v59 = 257;
  v43 = v9;
  v10 = sub_12A95D0(&v60, v6, *v8, (__int64)v58);
  v11 = sub_15FF7F0(*(_WORD *)(v4 + 18) & 0x7FFF);
  v51 = 257;
  if ( v11 )
  {
    v55 = 257;
    v57 = 257;
    v12 = sub_15A0680(*v8, 1, 0);
    if ( *((_BYTE *)v8 + 16) > 0x10u || *(_BYTE *)(v12 + 16) > 0x10u )
      v13 = (__int64)sub_17D2EF0(&v60, 23, v8, v12, v56, 0, 0);
    else
      v13 = sub_15A2D50(v8, v12, 0, 0, a2, a3, a4);
    v14 = sub_15A0680(*(_QWORD *)v13, 1, 0);
    if ( *(_BYTE *)(v13 + 16) > 0x10u || *(_BYTE *)(v14 + 16) > 0x10u )
    {
      v59 = 257;
      v41 = (_QWORD *)sub_15FB440(24, (__int64 *)v13, v14, (__int64)v58, 0);
      v15 = (__int64)sub_17CF870(&v60, v41, v54);
    }
    else
    {
      v15 = sub_15A2D80((__int64 *)v13, v14, 0, a2, a3, a4);
    }
    v59 = 257;
    v16 = sub_156D4C0(&v60, (__int64)v8, v15, (__int64)v58);
    v53 = 257;
    v55 = 257;
    v57 = 257;
    if ( *((_BYTE *)v16 + 16) > 0x10u )
    {
      v59 = 257;
      v47 = sub_15FB630(v16, (__int64)v58, 0);
      sub_17CCC80(v47, v56, v61, v62);
      sub_12A86E0(&v60, v47);
      v17 = v47;
    }
    else
    {
      v17 = sub_15A2B00(v16, a2, a3, a4);
    }
    v18 = sub_1281C00(&v60, v10, v17, (__int64)v54);
    v44 = sub_156D390(&v60, v18, v15, (__int64)v52);
    v19 = (_BYTE *)sub_17D3460((__int64)&v60, v43, v48, 1, a2, a3, a4);
    v45 = sub_12AA0C0(&v60, *(_WORD *)(v4 + 18) & 0x7FFF, v19, v44, (__int64)v50);
    v51 = 257;
    v20 = sub_17D3460((__int64)&v60, v10, (__int64)v8, 1, a2, a3, a4);
    v55 = 257;
    v21 = v20;
    v57 = 257;
    v22 = sub_15A0680(*(_QWORD *)v48, 1, 0);
    if ( *(_BYTE *)(v48 + 16) > 0x10u || *(_BYTE *)(v22 + 16) > 0x10u )
      v23 = (__int64)sub_17D2EF0(&v60, 23, (__int64 *)v48, v22, v56, 0, 0);
    else
      v23 = sub_15A2D50((__int64 *)v48, v22, 0, 0, a2, a3, a4);
    v24 = sub_15A0680(*(_QWORD *)v23, 1, 0);
    if ( *(_BYTE *)(v23 + 16) > 0x10u || *(_BYTE *)(v24 + 16) > 0x10u )
    {
      v59 = 257;
      v40 = (_QWORD *)sub_15FB440(24, (__int64 *)v23, v24, (__int64)v58, 0);
      v25 = (__int64)sub_17CF870(&v60, v40, v54);
    }
    else
    {
      v25 = sub_15A2D80((__int64 *)v23, v24, 0, a2, a3, a4);
    }
    v59 = 257;
    v26 = sub_156D4C0(&v60, v48, v25, (__int64)v58);
    v53 = 257;
    v55 = 257;
    v57 = 257;
    if ( *((_BYTE *)v26 + 16) > 0x10u )
    {
      v59 = 257;
      v49 = sub_15FB630(v26, (__int64)v58, 0);
      sub_17CCC80(v49, v56, v61, v62);
      sub_12A86E0(&v60, v49);
      v27 = v49;
    }
    else
    {
      v27 = sub_15A2B00(v26, a2, a3, a4);
    }
    v28 = sub_1281C00(&v60, v43, v27, (__int64)v54);
    v29 = (_BYTE *)sub_156D390(&v60, v28, v25, (__int64)v52);
  }
  else
  {
    v59 = 257;
    v46 = sub_156D390(&v60, v10, (__int64)v8, (__int64)v58);
    v30 = (_BYTE *)sub_17D3460((__int64)&v60, v43, v48, 0, a2, a3, a4);
    v31 = sub_12AA0C0(&v60, *(_WORD *)(v4 + 18) & 0x7FFF, v30, v46, (__int64)v50);
    v51 = 257;
    v45 = v31;
    v32 = sub_17D3460((__int64)&v60, v10, (__int64)v8, 0, a2, a3, a4);
    v59 = 257;
    v21 = v32;
    v29 = (_BYTE *)sub_156D390(&v60, v43, v48, (__int64)v58);
  }
  v33 = sub_12AA0C0(&v60, *(_WORD *)(v4 + 18) & 0x7FFF, v29, v21, (__int64)v50);
  v57 = 257;
  if ( *(_BYTE *)(v45 + 16) > 0x10u
    || *(_BYTE *)(v33 + 16) > 0x10u
    || (v34 = sub_15A2A30((__int64 *)0x1C, (__int64 *)v45, v33, 0, 0, a2, a3, a4)) == 0 )
  {
    v59 = 257;
    v36 = sub_15FB440(28, (__int64 *)v45, v33, (__int64)v58, 0);
    v34 = v36;
    if ( v61 )
    {
      v37 = v62;
      sub_157E9D0(v61 + 40, v36);
      v38 = *(_QWORD *)(v34 + 24);
      v39 = *v37;
      *(_QWORD *)(v34 + 32) = v37;
      v39 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v34 + 24) = v39 | v38 & 7;
      *(_QWORD *)(v39 + 8) = v34 + 24;
      *v37 = *v37 & 7 | (v34 + 24);
    }
    sub_164B780(v34, v56);
    sub_12A86E0(&v60, v34);
  }
  sub_17D4920((__int64)v42, (__int64 *)v4, v34);
  if ( *(_DWORD *)(v42[1] + 156LL) )
    sub_17D9C10(v42, v4);
  return sub_17CD270(&v60);
}
