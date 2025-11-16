// Function: sub_394C8F0
// Address: 0x394c8f0
//
__int64 *__fastcall sub_394C8F0(__int64 *a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r15
  _QWORD *v12; // rax
  __int64 v13; // rbx
  __int64 v14; // r15
  __int64 v15; // rax
  unsigned __int8 *v16; // rsi
  __int64 v17; // rax
  __int64 *v18; // rdx
  char v19; // al
  __int64 v20; // r13
  __int64 v21; // rbx
  __int64 v22; // rdx
  __int64 v23; // rax
  bool v24; // cc
  __int64 v25; // rcx
  _QWORD *v26; // rax
  _QWORD *v27; // rbx
  unsigned __int64 *v28; // r13
  __int64 v29; // rax
  unsigned __int64 v30; // rcx
  __int64 v31; // rsi
  unsigned __int8 *v32; // rsi
  __int64 v34; // rcx
  __int64 *v35; // rsi
  int v36; // edi
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rcx
  __int64 *v40; // rbx
  __int64 v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // r13
  unsigned __int8 *v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rcx
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rsi
  __int64 v51; // rdx
  unsigned __int8 *v52; // rsi
  __int64 *v53; // [rsp+10h] [rbp-E0h]
  __int64 v54; // [rsp+18h] [rbp-D8h]
  __int64 v55; // [rsp+18h] [rbp-D8h]
  __int64 v56; // [rsp+18h] [rbp-D8h]
  __int64 v57; // [rsp+18h] [rbp-D8h]
  __int64 v58; // [rsp+18h] [rbp-D8h]
  __int64 v59; // [rsp+18h] [rbp-D8h]
  __int64 v60; // [rsp+18h] [rbp-D8h]
  unsigned __int8 *v61; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v62[2]; // [rsp+30h] [rbp-C0h] BYREF
  __int16 v63; // [rsp+40h] [rbp-B0h]
  unsigned __int8 *v64[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v65; // [rsp+60h] [rbp-90h]
  unsigned __int8 *v66; // [rsp+70h] [rbp-80h] BYREF
  __int64 v67; // [rsp+78h] [rbp-78h]
  __int64 *v68; // [rsp+80h] [rbp-70h]
  __int64 v69; // [rsp+88h] [rbp-68h]
  __int64 v70; // [rsp+90h] [rbp-60h]
  int v71; // [rsp+98h] [rbp-58h]
  __int64 v72; // [rsp+A0h] [rbp-50h]
  __int64 v73; // [rsp+A8h] [rbp-48h]

  v9 = *(_QWORD *)(a2 + 24);
  a1[1] = 0;
  a1[2] = 0;
  v10 = *(_QWORD *)(v9 + 56);
  LOWORD(v68) = 257;
  v54 = v10;
  v11 = sub_15E0530(v10);
  v12 = (_QWORD *)sub_22077B0(0x40u);
  v13 = (__int64)v12;
  if ( v12 )
    sub_157FB60(v12, v11, (__int64)&v66, v54, a3);
  v14 = *(_QWORD *)(v13 + 48);
  *a1 = v13;
  v15 = sub_157E9C0(v13);
  v67 = v13;
  v66 = 0;
  v69 = v15;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v68 = (__int64 *)v14;
  if ( v14 != v13 + 40 )
  {
    if ( !v14 )
      BUG();
    v16 = *(unsigned __int8 **)(v14 + 24);
    v64[0] = v16;
    if ( v16 )
    {
      sub_1623A60((__int64)v64, (__int64)v16, 2);
      if ( v66 )
        sub_161E7C0((__int64)&v66, (__int64)v66);
      v66 = v64[0];
      if ( v64[0] )
        sub_1623210((__int64)v64, v64[0], (__int64)&v66);
    }
  }
  v17 = *(_QWORD *)(a2 + 8);
  if ( (*(_BYTE *)(v17 + 23) & 0x40) != 0 )
    v18 = *(__int64 **)(v17 - 8);
  else
    v18 = (__int64 *)(v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF));
  v19 = *(_BYTE *)(v17 + 16);
  v20 = *v18;
  v21 = v18[3];
  if ( v19 != 45 && v19 != 42 )
  {
    v22 = v18[3];
    v65 = 257;
    v23 = sub_394C7A0((__int64 *)&v66, v20, v22, (__int64 *)v64, 0, a4, a5, a6);
    v63 = 257;
    v24 = *(_BYTE *)(v20 + 16) <= 0x10u;
    a1[1] = v23;
    if ( v24 && *(_BYTE *)(v21 + 16) <= 0x10u )
    {
      v25 = sub_15A2A30((__int64 *)0x14, (__int64 *)v20, v21, 0, 0, a4, a5, a6);
      if ( v25 )
        goto LABEL_17;
    }
    v37 = v21;
    v35 = (__int64 *)v20;
    v65 = 257;
    v36 = 20;
    goto LABEL_36;
  }
  v63 = 257;
  if ( *(_BYTE *)(v20 + 16) > 0x10u || *(_BYTE *)(v21 + 16) > 0x10u )
  {
    v65 = 257;
    v46 = sub_15FB440(18, (__int64 *)v20, v21, (__int64)v64, 0);
    v47 = v46;
    if ( v67 )
    {
      v58 = v46;
      v53 = v68;
      sub_157E9D0(v67 + 40, v46);
      v47 = v58;
      v48 = *(_QWORD *)(v58 + 24);
      v49 = *v53;
      *(_QWORD *)(v58 + 32) = v53;
      v49 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v58 + 24) = v49 | v48 & 7;
      *(_QWORD *)(v49 + 8) = v58 + 24;
      *v53 = *v53 & 7 | (v58 + 24);
    }
    v59 = v47;
    sub_164B780(v47, v62);
    v34 = v59;
    if ( v66 )
    {
      v61 = v66;
      sub_1623A60((__int64)&v61, (__int64)v66, 2);
      v34 = v59;
      v50 = *(_QWORD *)(v59 + 48);
      v51 = v59 + 48;
      if ( v50 )
      {
        sub_161E7C0(v59 + 48, v50);
        v34 = v59;
        v51 = v59 + 48;
      }
      v52 = v61;
      *(_QWORD *)(v34 + 48) = v61;
      if ( v52 )
      {
        v60 = v34;
        sub_1623210((__int64)&v61, v52, v51);
        v34 = v60;
      }
    }
  }
  else
  {
    v34 = sub_15A2C90((__int64 *)v20, v21, 0, a4, a5, a6);
  }
  a1[1] = v34;
  v63 = 257;
  if ( *(_BYTE *)(v20 + 16) > 0x10u
    || *(_BYTE *)(v21 + 16) > 0x10u
    || (v25 = sub_15A2A30((__int64 *)0x15, (__int64 *)v20, v21, 0, 0, a4, a5, a6)) == 0 )
  {
    v35 = (__int64 *)v20;
    v65 = 257;
    v36 = 21;
    v37 = v21;
LABEL_36:
    v38 = sub_15FB440(v36, v35, v37, (__int64)v64, 0);
    v39 = v38;
    if ( v67 )
    {
      v40 = v68;
      v55 = v38;
      sub_157E9D0(v67 + 40, v38);
      v39 = v55;
      v41 = *v40;
      v42 = *(_QWORD *)(v55 + 24);
      *(_QWORD *)(v55 + 32) = v40;
      v41 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v55 + 24) = v41 | v42 & 7;
      *(_QWORD *)(v41 + 8) = v55 + 24;
      *v40 = *v40 & 7 | (v55 + 24);
    }
    v56 = v39;
    sub_164B780(v39, v62);
    v25 = v56;
    if ( v66 )
    {
      v61 = v66;
      sub_1623A60((__int64)&v61, (__int64)v66, 2);
      v25 = v56;
      v43 = *(_QWORD *)(v56 + 48);
      v44 = v56 + 48;
      if ( v43 )
      {
        sub_161E7C0(v56 + 48, v43);
        v25 = v56;
      }
      v45 = v61;
      *(_QWORD *)(v25 + 48) = v61;
      if ( v45 )
      {
        v57 = v25;
        sub_1623210((__int64)&v61, v45, v44);
        v25 = v57;
      }
    }
  }
LABEL_17:
  a1[2] = v25;
  v65 = 257;
  v26 = sub_1648A60(56, 1u);
  v27 = v26;
  if ( v26 )
    sub_15F8320((__int64)v26, a3, 0);
  if ( v67 )
  {
    v28 = (unsigned __int64 *)v68;
    sub_157E9D0(v67 + 40, (__int64)v27);
    v29 = v27[3];
    v30 = *v28;
    v27[4] = v28;
    v30 &= 0xFFFFFFFFFFFFFFF8LL;
    v27[3] = v30 | v29 & 7;
    *(_QWORD *)(v30 + 8) = v27 + 3;
    *v28 = *v28 & 7 | (unsigned __int64)(v27 + 3);
  }
  sub_164B780((__int64)v27, (__int64 *)v64);
  if ( v66 )
  {
    v62[0] = (__int64)v66;
    sub_1623A60((__int64)v62, (__int64)v66, 2);
    v31 = v27[6];
    if ( v31 )
      sub_161E7C0((__int64)(v27 + 6), v31);
    v32 = (unsigned __int8 *)v62[0];
    v27[6] = v62[0];
    if ( v32 )
      sub_1623210((__int64)v62, v32, (__int64)(v27 + 6));
    if ( v66 )
      sub_161E7C0((__int64)&v66, (__int64)v66);
  }
  return a1;
}
