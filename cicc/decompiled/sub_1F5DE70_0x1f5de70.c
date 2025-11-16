// Function: sub_1F5DE70
// Address: 0x1f5de70
//
char __fastcall sub_1F5DE70(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int8 *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdi
  _QWORD *v17; // r15
  __int64 v18; // r13
  __int64 v19; // rax
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 v22; // r14
  __int64 v23; // rbx
  _QWORD *v24; // rax
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 v27; // rcx
  int v28; // eax
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int8 *v31; // rsi
  unsigned __int8 *v32; // rsi
  __int64 v33; // r14
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // r15
  _QWORD *v38; // rax
  _QWORD *v39; // r14
  unsigned __int64 *v40; // r15
  __int64 v41; // rax
  unsigned __int64 v42; // rcx
  __int64 v43; // rsi
  unsigned __int8 *v44; // rsi
  __int64 v45; // rax
  __m128 v46; // xmm0
  __int64 *v47; // rax
  __int64 v48; // r13
  __int64 v49; // r14
  _QWORD *v50; // rax
  __int64 v51; // rbx
  __int64 v52; // r15
  char v53; // al
  int v54; // r13d
  __int64 *v55; // r13
  __int64 v56; // rax
  __int64 v57; // rcx
  __int64 v58; // rsi
  unsigned __int8 *v59; // rsi
  __int64 *v60; // rax
  __int64 v61; // r14
  _QWORD *v62; // r13
  unsigned __int64 *v63; // rbx
  __int64 v64; // rax
  unsigned __int64 v65; // rcx
  double v66; // xmm4_8
  double v67; // xmm5_8
  __int64 v68; // rsi
  unsigned __int8 *v69; // rsi
  __int64 v70; // r15
  _QWORD *v71; // rax
  _QWORD *v72; // r14
  unsigned __int64 *v73; // r15
  __int64 v74; // rax
  unsigned __int64 v75; // rcx
  __int64 v76; // rsi
  unsigned __int8 *v77; // rsi
  __int64 v78; // rsi
  __int64 v79; // rdx
  __int64 v80; // rcx
  __int64 v83; // [rsp+28h] [rbp-158h]
  __int64 v84; // [rsp+28h] [rbp-158h]
  _QWORD *v85; // [rsp+38h] [rbp-148h]
  __int64 v87; // [rsp+58h] [rbp-128h] BYREF
  __int64 v88; // [rsp+60h] [rbp-120h] BYREF
  __int16 v89; // [rsp+70h] [rbp-110h]
  _QWORD v90[2]; // [rsp+80h] [rbp-100h] BYREF
  __int16 v91; // [rsp+90h] [rbp-F0h]
  _QWORD v92[2]; // [rsp+A0h] [rbp-E0h] BYREF
  __m128i v93; // [rsp+B0h] [rbp-D0h] BYREF
  unsigned __int8 *v94[2]; // [rsp+C0h] [rbp-C0h] BYREF
  __m128 v95; // [rsp+D0h] [rbp-B0h] BYREF
  char *v96; // [rsp+E0h] [rbp-A0h]
  char *v97; // [rsp+E8h] [rbp-98h]
  char *v98; // [rsp+F0h] [rbp-90h]
  unsigned __int8 *v99; // [rsp+100h] [rbp-80h] BYREF
  __int64 v100; // [rsp+108h] [rbp-78h]
  unsigned __int64 *v101; // [rsp+110h] [rbp-70h]
  _QWORD *v102; // [rsp+118h] [rbp-68h]
  __int64 v103; // [rsp+120h] [rbp-60h]
  int v104; // [rsp+128h] [rbp-58h]
  __int64 v105; // [rsp+130h] [rbp-50h]
  __int64 v106; // [rsp+138h] [rbp-48h]

  v99 = 0;
  v101 = 0;
  v102 = (_QWORD *)sub_157E9C0(a2);
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v100 = 0;
  v12 = sub_157EE30(a2);
  if ( !v12 )
    BUG();
  v13 = *(_QWORD *)(v12 + 16);
  v14 = *(unsigned __int8 **)(v12 + 24);
  v101 = (unsigned __int64 *)v12;
  v100 = v13;
  v94[0] = v14;
  if ( v14 )
  {
    sub_1623A60((__int64)v94, (__int64)v14, 2);
    if ( v99 )
      sub_161E7C0((__int64)&v99, (__int64)v99);
    v99 = v94[0];
    if ( v94[0] )
      sub_1623210((__int64)v94, v94[0], (__int64)&v99);
  }
  v94[0] = "exn";
  v95.m128_i16[0] = 259;
  v15 = sub_1643350(v102);
  v92[0] = sub_159C470(v15, 0, 0);
  v16 = a2;
  v17 = 0;
  v18 = sub_1285290(
          (__int64 *)&v99,
          *(_QWORD *)(*(_QWORD *)(a1 + 200) + 24LL),
          *(_QWORD *)(a1 + 200),
          (int)v92,
          1,
          (__int64)v94,
          0);
  v19 = sub_157ED20(v16);
  v85 = 0;
  v22 = *(_QWORD *)(v19 + 8);
  v23 = v19;
  if ( !v22 )
  {
    v78 = v18;
    sub_164D160(0, v18, a4, a5, a6, a7, v20, v21, a10, a11);
    sub_15F20C0(0);
    v28 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
    if ( v28 == 1 )
      goto LABEL_67;
LABEL_16:
    if ( v28 == 2 )
    {
      LOBYTE(v28) = sub_1593BB0(*(_QWORD *)(v23 - 48), v78, v79, v80);
      if ( (_BYTE)v28 )
        goto LABEL_73;
    }
    v29 = *(_QWORD *)(v18 + 32);
    if ( v29 == *(_QWORD *)(v18 + 40) + 40LL || !v29 )
      BUG();
    v30 = *(_QWORD *)(v29 + 16);
    v31 = *(unsigned __int8 **)(v29 + 24);
    v101 = *(unsigned __int64 **)(v18 + 32);
    v100 = v30;
    v94[0] = v31;
    if ( v31 )
    {
      sub_1623A60((__int64)v94, (__int64)v31, 2);
      v32 = v99;
      if ( !v99 )
        goto LABEL_23;
    }
    else
    {
      v32 = v99;
      if ( !v99 )
        goto LABEL_25;
    }
    sub_161E7C0((__int64)&v99, (__int64)v32);
LABEL_23:
    v99 = v94[0];
    if ( v94[0] )
      sub_1623210((__int64)v94, v94[0], (__int64)&v99);
LABEL_25:
    v33 = a3;
    v95.m128_i16[0] = 257;
    v34 = sub_1643350(v102);
    v92[0] = sub_159C470(v34, a3, 0);
    sub_1285290(
      (__int64 *)&v99,
      *(_QWORD *)(*(_QWORD *)(a1 + 208) + 24LL),
      *(_QWORD *)(a1 + 208),
      (int)v92,
      1,
      (__int64)v94,
      0);
    v83 = *(_QWORD *)(a1 + 176);
    v35 = sub_1643350(v102);
    v36 = sub_159C470(v35, v33, 0);
    v95.m128_i16[0] = 257;
    v37 = v36;
    v38 = sub_1648A60(64, 2u);
    v39 = v38;
    if ( v38 )
      sub_15F9650((__int64)v38, v37, v83, 0, 0);
    if ( v100 )
    {
      v40 = v101;
      sub_157E9D0(v100 + 40, (__int64)v39);
      v41 = v39[3];
      v42 = *v40;
      v39[4] = v40;
      v42 &= 0xFFFFFFFFFFFFFFF8LL;
      v39[3] = v42 | v41 & 7;
      *(_QWORD *)(v42 + 8) = v39 + 3;
      *v40 = *v40 & 7 | (unsigned __int64)(v39 + 3);
    }
    sub_164B780((__int64)v39, (__int64 *)v94);
    if ( v99 )
    {
      v92[0] = v99;
      sub_1623A60((__int64)v92, (__int64)v99, 2);
      v43 = v39[6];
      if ( v43 )
        sub_161E7C0((__int64)(v39 + 6), v43);
      v44 = (unsigned __int8 *)v92[0];
      v39[6] = v92[0];
      if ( v44 )
        sub_1623210((__int64)v92, v44, (__int64)(v39 + 6));
    }
    v45 = *(_QWORD *)(v23 - 24);
    if ( (*(_BYTE *)(v45 + 23) & 0x40) != 0 )
    {
      if ( *(_BYTE *)(**(_QWORD **)(v45 - 8) + 16LL) != 16 )
      {
LABEL_36:
        v93.m128i_i64[0] = 0x74656C636E7566LL;
        v46 = (__m128)_mm_load_si128(&v93);
        v92[0] = &v93;
        v89 = 257;
        v95 = v46;
        v94[0] = (unsigned __int8 *)&v95;
        v94[1] = (unsigned __int8 *)7;
        v92[1] = 0;
        v93.m128i_i8[0] = 0;
        v96 = 0;
        v97 = 0;
        v98 = 0;
        v47 = (__int64 *)sub_22077B0(8);
        v87 = v18;
        v96 = (char *)v47;
        *v47 = v23;
        v98 = (char *)(v47 + 1);
        v48 = *(_QWORD *)(a1 + 240);
        v97 = (char *)(v47 + 1);
        v91 = 257;
        v49 = *(_QWORD *)(*(_QWORD *)v48 + 24LL);
        v50 = sub_1648AB0(72, 3u, 0x10u);
        v51 = (__int64)v50;
        if ( v50 )
        {
          v52 = (__int64)v50;
          sub_15F1EA0(
            (__int64)v50,
            **(_QWORD **)(v49 + 16),
            54,
            (__int64)&v50[-3 * (unsigned int)((v97 - v96) >> 3) - 6],
            ((v97 - v96) >> 3) + 2,
            0);
          *(_QWORD *)(v51 + 56) = 0;
          sub_15F5B40(v51, v49, v48, &v87, 1, (__int64)v90, (__int64 *)v94, 1);
        }
        else
        {
          v52 = 0;
        }
        v53 = *(_BYTE *)(*(_QWORD *)v51 + 8LL);
        if ( v53 == 16 )
          v53 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v51 + 16LL) + 8LL);
        if ( (unsigned __int8)(v53 - 1) <= 5u || *(_BYTE *)(v51 + 16) == 76 )
        {
          v54 = v104;
          if ( v103 )
            sub_1625C10(v51, 3, v103);
          sub_15F2440(v51, v54);
        }
        if ( v100 )
        {
          v55 = (__int64 *)v101;
          sub_157E9D0(v100 + 40, v51);
          v56 = *(_QWORD *)(v51 + 24);
          v57 = *v55;
          *(_QWORD *)(v51 + 32) = v55;
          v57 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v51 + 24) = v57 | v56 & 7;
          *(_QWORD *)(v57 + 8) = v51 + 24;
          *v55 = *v55 & 7 | (v51 + 24);
        }
        sub_164B780(v52, &v88);
        if ( v99 )
        {
          v90[0] = v99;
          sub_1623A60((__int64)v90, (__int64)v99, 2);
          v58 = *(_QWORD *)(v51 + 48);
          if ( v58 )
            sub_161E7C0(v51 + 48, v58);
          v59 = (unsigned __int8 *)v90[0];
          *(_QWORD *)(v51 + 48) = v90[0];
          if ( v59 )
            sub_1623210((__int64)v90, v59, v51 + 48);
        }
        if ( v96 )
          j_j___libc_free_0(v96, v98 - v96);
        if ( (__m128 *)v94[0] != &v95 )
          j_j___libc_free_0(v94[0], v95.m128_u64[0] + 1);
        if ( (__m128i *)v92[0] != &v93 )
          j_j___libc_free_0(v92[0], v93.m128i_i64[0] + 1);
        v94[0] = *(unsigned __int8 **)(v51 + 56);
        v60 = (__int64 *)sub_16498A0(v52);
        *(_QWORD *)(v51 + 56) = sub_1563AB0((__int64 *)v94, v60, -1, 30);
        v61 = *(_QWORD *)(a1 + 192);
        v95.m128_i16[0] = 259;
        v94[0] = (unsigned __int8 *)"selector";
        v62 = sub_1648A60(64, 1u);
        if ( v62 )
          sub_15F9210((__int64)v62, *(_QWORD *)(*(_QWORD *)v61 + 24LL), v61, 0, 0, 0);
        if ( v100 )
        {
          v63 = v101;
          sub_157E9D0(v100 + 40, (__int64)v62);
          v64 = v62[3];
          v65 = *v63;
          v62[4] = v63;
          v65 &= 0xFFFFFFFFFFFFFFF8LL;
          v62[3] = v65 | v64 & 7;
          *(_QWORD *)(v65 + 8) = v62 + 3;
          *v63 = *v63 & 7 | (unsigned __int64)(v62 + 3);
        }
        sub_164B780((__int64)v62, (__int64 *)v94);
        if ( v99 )
        {
          v92[0] = v99;
          sub_1623A60((__int64)v92, (__int64)v99, 2);
          v68 = v62[6];
          if ( v68 )
            sub_161E7C0((__int64)(v62 + 6), v68);
          v69 = (unsigned __int8 *)v92[0];
          v62[6] = v92[0];
          if ( v69 )
            sub_1623210((__int64)v92, v69, (__int64)(v62 + 6));
        }
        sub_164D160((__int64)v85, (__int64)v62, v46, a5, a6, a7, v66, v67, a10, a11);
        LOBYTE(v28) = sub_15F20C0(v85);
        goto LABEL_67;
      }
    }
    else if ( *(_BYTE *)(*(_QWORD *)(v45 - 24LL * (*(_DWORD *)(v45 + 20) & 0xFFFFFFF)) + 16LL) != 16 )
    {
      goto LABEL_36;
    }
    v93.m128i_i16[0] = 257;
    v70 = *(_QWORD *)(a1 + 184);
    v84 = sub_1285290(
            (__int64 *)&v99,
            *(_QWORD *)(**(_QWORD **)(a1 + 216) + 24LL),
            *(_QWORD *)(a1 + 216),
            0,
            0,
            (__int64)v92,
            0);
    v95.m128_i16[0] = 257;
    v71 = sub_1648A60(64, 2u);
    v72 = v71;
    if ( v71 )
      sub_15F9650((__int64)v71, v84, v70, 0, 0);
    if ( v100 )
    {
      v73 = v101;
      sub_157E9D0(v100 + 40, (__int64)v72);
      v74 = v72[3];
      v75 = *v73;
      v72[4] = v73;
      v75 &= 0xFFFFFFFFFFFFFFF8LL;
      v72[3] = v75 | v74 & 7;
      *(_QWORD *)(v75 + 8) = v72 + 3;
      *v73 = *v73 & 7 | (unsigned __int64)(v72 + 3);
    }
    sub_164B780((__int64)v72, (__int64 *)v94);
    if ( v99 )
    {
      v90[0] = v99;
      sub_1623A60((__int64)v90, (__int64)v99, 2);
      v76 = v72[6];
      if ( v76 )
        sub_161E7C0((__int64)(v72 + 6), v76);
      v77 = (unsigned __int8 *)v90[0];
      v72[6] = v90[0];
      if ( v77 )
        sub_1623210((__int64)v90, v77, (__int64)(v72 + 6));
    }
    goto LABEL_36;
  }
  do
  {
    while ( 1 )
    {
      v24 = sub_1648700(v22);
      if ( *((_BYTE *)v24 + 16) == 78 )
        break;
LABEL_9:
      v22 = *(_QWORD *)(v22 + 8);
      if ( !v22 )
        goto LABEL_15;
    }
    v27 = *(v24 - 3);
    if ( *(_QWORD *)(a1 + 224) == v27 )
    {
      v17 = v24;
      goto LABEL_9;
    }
    v22 = *(_QWORD *)(v22 + 8);
    if ( *(_QWORD *)(a1 + 232) != v27 )
      v24 = v85;
    v85 = v24;
  }
  while ( v22 );
LABEL_15:
  v78 = v18;
  sub_164D160((__int64)v17, v18, a4, a5, a6, a7, v25, v26, a10, a11);
  sub_15F20C0(v17);
  v28 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
  if ( v28 != 1 )
    goto LABEL_16;
LABEL_73:
  if ( v85 )
    LOBYTE(v28) = sub_15F20C0(v85);
LABEL_67:
  if ( v99 )
    LOBYTE(v28) = sub_161E7C0((__int64)&v99, (__int64)v99);
  return v28;
}
