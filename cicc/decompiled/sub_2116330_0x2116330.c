// Function: sub_2116330
// Address: 0x2116330
//
void __fastcall sub_2116330(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rax
  unsigned __int8 *v5; // rsi
  __int64 v6; // rax
  _QWORD *v7; // rax
  __int64 v8; // r12
  __int64 v9; // rax
  _BYTE *v10; // rbx
  bool v11; // cc
  __int64 v12; // r11
  _QWORD *v13; // r12
  _QWORD *v14; // rax
  __int64 v15; // rax
  __int64 v16; // r13
  _QWORD *v17; // rax
  _QWORD *v18; // rbx
  unsigned __int64 *v19; // r12
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  __int64 v22; // rsi
  unsigned __int8 *v23; // rsi
  _QWORD *v24; // rax
  __int64 v25; // r11
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 v28; // rax
  __int64 v29; // r11
  __int64 *v30; // r10
  __int64 v31; // rax
  unsigned __int64 *v32; // r15
  __int64 v33; // rax
  unsigned __int64 v34; // rcx
  __int64 v35; // rsi
  unsigned __int8 *v36; // rsi
  __int64 *v37; // rax
  __int64 v38; // rax
  __int64 v39; // [rsp+0h] [rbp-100h]
  __int64 v40; // [rsp+0h] [rbp-100h]
  __int64 v41; // [rsp+8h] [rbp-F8h]
  __int64 v42; // [rsp+10h] [rbp-F0h]
  __int64 v43; // [rsp+10h] [rbp-F0h]
  __int64 v44; // [rsp+10h] [rbp-F0h]
  int v45; // [rsp+18h] [rbp-E8h]
  unsigned __int8 *v47; // [rsp+28h] [rbp-D8h] BYREF
  __int64 *v48; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v49; // [rsp+38h] [rbp-C8h]
  __int64 v50[2]; // [rsp+40h] [rbp-C0h] BYREF
  char v51; // [rsp+50h] [rbp-B0h]
  char v52; // [rsp+51h] [rbp-AFh]
  unsigned __int8 *v53[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int16 v54; // [rsp+70h] [rbp-90h]
  unsigned __int8 *v55; // [rsp+80h] [rbp-80h] BYREF
  __int64 v56; // [rsp+88h] [rbp-78h]
  unsigned __int64 *v57; // [rsp+90h] [rbp-70h]
  __int64 v58; // [rsp+98h] [rbp-68h]
  __int64 v59; // [rsp+A0h] [rbp-60h]
  int v60; // [rsp+A8h] [rbp-58h]
  __int64 v61; // [rsp+B0h] [rbp-50h]
  __int64 v62; // [rsp+B8h] [rbp-48h]

  v4 = sub_16498A0(a2);
  v5 = *(unsigned __int8 **)(a2 + 48);
  v55 = 0;
  v58 = v4;
  v6 = *(_QWORD *)(a2 + 40);
  v59 = 0;
  v56 = v6;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v57 = (unsigned __int64 *)(a2 + 24);
  v53[0] = v5;
  if ( v5 )
  {
    sub_1623A60((__int64)v53, (__int64)v5, 2);
    if ( v55 )
      sub_161E7C0((__int64)&v55, (__int64)v55);
    v55 = v53[0];
    if ( v53[0] )
      sub_1623210((__int64)v53, v53[0], (__int64)&v55);
  }
  v7 = (_QWORD *)sub_16498A0(a2);
  v8 = sub_1643350(v7);
  v42 = sub_15A0680(v8, 0, 0);
  v9 = sub_15A0680(v8, 1, 0);
  v50[0] = (__int64)"call_site";
  v10 = *(_BYTE **)(a1 + 256);
  v52 = 1;
  v51 = 3;
  v11 = v10[16] <= 0x10u;
  v48 = (__int64 *)v42;
  v12 = *(_QWORD *)(a1 + 176);
  v49 = v9;
  if ( v11 && *(_BYTE *)(v42 + 16) <= 0x10u && *(_BYTE *)(v9 + 16) <= 0x10u )
  {
    BYTE4(v53[0]) = 0;
    v13 = (_QWORD *)sub_15A2E80(v12, (__int64)v10, &v48, 2u, 0, (__int64)v53, 0);
  }
  else
  {
    v54 = 257;
    if ( !v12 )
    {
      v38 = *(_QWORD *)v10;
      if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 16 )
        v38 = **(_QWORD **)(v38 + 16);
      v12 = *(_QWORD *)(v38 + 24);
    }
    v43 = v12;
    v24 = sub_1648A60(72, 3u);
    v25 = v43;
    v13 = v24;
    if ( v24 )
    {
      v44 = (__int64)v24;
      v41 = (__int64)(v24 - 9);
      v26 = *(_QWORD *)v10;
      if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 16 )
        v26 = **(_QWORD **)(v26 + 16);
      v39 = v25;
      v45 = *(_DWORD *)(v26 + 8) >> 8;
      v27 = (__int64 *)sub_15F9F50(v25, (__int64)&v48, 2);
      v28 = sub_1646BA0(v27, v45);
      v29 = v39;
      v30 = (__int64 *)v28;
      v31 = *(_QWORD *)v10;
      if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 16
        || (v31 = *v48, *(_BYTE *)(*v48 + 8) == 16)
        || (v31 = *(_QWORD *)v49, *(_BYTE *)(*(_QWORD *)v49 + 8LL) == 16) )
      {
        v37 = sub_16463B0(v30, *(_QWORD *)(v31 + 32));
        v29 = v39;
        v30 = v37;
      }
      v40 = v29;
      sub_15F1EA0((__int64)v13, (__int64)v30, 32, v41, 3, 0);
      v13[7] = v40;
      v13[8] = sub_15F9F50(v40, (__int64)&v48, 2);
      sub_15F9CE0((__int64)v13, (__int64)v10, (__int64 *)&v48, 2, (__int64)v53);
    }
    else
    {
      v44 = 0;
    }
    if ( v56 )
    {
      v32 = v57;
      sub_157E9D0(v56 + 40, (__int64)v13);
      v33 = v13[3];
      v34 = *v32;
      v13[4] = v32;
      v34 &= 0xFFFFFFFFFFFFFFF8LL;
      v13[3] = v34 | v33 & 7;
      *(_QWORD *)(v34 + 8) = v13 + 3;
      *v32 = *v32 & 7 | (unsigned __int64)(v13 + 3);
    }
    sub_164B780(v44, v50);
    if ( v55 )
    {
      v47 = v55;
      sub_1623A60((__int64)&v47, (__int64)v55, 2);
      v35 = v13[6];
      if ( v35 )
        sub_161E7C0((__int64)(v13 + 6), v35);
      v36 = v47;
      v13[6] = v47;
      if ( v36 )
        sub_1623210((__int64)&v47, v36, (__int64)(v13 + 6));
    }
  }
  v14 = (_QWORD *)sub_16498A0(a2);
  v15 = sub_1643350(v14);
  v16 = sub_159C470(v15, a3, 0);
  v54 = 257;
  v17 = sub_1648A60(64, 2u);
  v18 = v17;
  if ( v17 )
    sub_15F9650((__int64)v17, v16, (__int64)v13, 1u, 0);
  if ( v56 )
  {
    v19 = v57;
    sub_157E9D0(v56 + 40, (__int64)v18);
    v20 = v18[3];
    v21 = *v19;
    v18[4] = v19;
    v21 &= 0xFFFFFFFFFFFFFFF8LL;
    v18[3] = v21 | v20 & 7;
    *(_QWORD *)(v21 + 8) = v18 + 3;
    *v19 = *v19 & 7 | (unsigned __int64)(v18 + 3);
  }
  sub_164B780((__int64)v18, (__int64 *)v53);
  if ( v55 )
  {
    v50[0] = (__int64)v55;
    sub_1623A60((__int64)v50, (__int64)v55, 2);
    v22 = v18[6];
    if ( v22 )
      sub_161E7C0((__int64)(v18 + 6), v22);
    v23 = (unsigned __int8 *)v50[0];
    v18[6] = v50[0];
    if ( v23 )
      sub_1623210((__int64)v50, v23, (__int64)(v18 + 6));
    if ( v55 )
      sub_161E7C0((__int64)&v55, (__int64)v55);
  }
}
