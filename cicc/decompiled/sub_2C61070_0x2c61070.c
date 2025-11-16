// Function: sub_2C61070
// Address: 0x2c61070
//
__int64 __fastcall sub_2C61070(__int64 a1, unsigned __int8 *a2)
{
  unsigned int v4; // r13d
  unsigned __int8 *v6; // rax
  _BYTE *v7; // rdi
  _BYTE *v8; // r15
  __int64 v9; // rbx
  unsigned __int64 v10; // r13
  int v11; // edx
  __int64 v12; // rdx
  __int64 *v13; // r11
  __int64 v14; // r13
  __int64 **v15; // rdx
  unsigned int v16; // r15d
  __int64 v17; // rax
  int v18; // edx
  __int64 v19; // rax
  int v20; // edx
  __int64 v21; // rax
  int v22; // edx
  int v23; // edx
  __int64 v24; // rcx
  __int64 v25; // rsi
  int v26; // eax
  unsigned __int64 v27; // rax
  int v28; // edi
  unsigned __int64 v29; // rax
  bool v30; // zf
  int v31; // edx
  unsigned __int64 v32; // rax
  _QWORD **v33; // rdi
  signed int v34; // eax
  int v35; // ecx
  __int64 v36; // rax
  int v37; // edx
  __int64 v38; // r9
  __int64 *v39; // r11
  __int64 v40; // r15
  int v41; // r10d
  unsigned __int64 v42; // rax
  int v43; // r8d
  __int64 v44; // rax
  __int64 v45; // rax
  int v46; // r10d
  int v47; // edx
  unsigned __int64 v48; // r15
  __int64 v49; // rax
  int v50; // r10d
  int v51; // edx
  unsigned __int64 v52; // r9
  __int64 v53; // rax
  int v54; // r10d
  __int64 v55; // rcx
  int v56; // edx
  unsigned __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rdx
  bool v60; // sf
  bool v61; // of
  __int64 v62; // rax
  unsigned __int64 v63; // rdx
  __int64 v64; // r11
  __int64 v65; // r9
  int v66; // r8d
  __int64 *v67; // rax
  __int64 *v68; // rdx
  __int64 v69; // rax
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rax
  unsigned __int8 *v73; // r11
  unsigned __int8 *v74; // rcx
  char v75; // si
  _BYTE *v76; // rax
  __int64 v77; // rax
  __int64 *v78; // rax
  __int64 *v79; // rdx
  bool v80; // cc
  unsigned __int64 v81; // rax
  __int64 v82; // [rsp+0h] [rbp-2D0h]
  __int64 *v83; // [rsp+8h] [rbp-2C8h]
  __int64 v84; // [rsp+8h] [rbp-2C8h]
  int v85; // [rsp+8h] [rbp-2C8h]
  int v86; // [rsp+8h] [rbp-2C8h]
  int v87; // [rsp+10h] [rbp-2C0h]
  int v88; // [rsp+10h] [rbp-2C0h]
  unsigned __int64 v89; // [rsp+10h] [rbp-2C0h]
  int v90; // [rsp+10h] [rbp-2C0h]
  unsigned __int64 v91; // [rsp+18h] [rbp-2B8h]
  int v92; // [rsp+20h] [rbp-2B0h]
  int v93; // [rsp+28h] [rbp-2A8h]
  unsigned __int64 v94; // [rsp+30h] [rbp-2A0h]
  __int64 *v95; // [rsp+38h] [rbp-298h]
  __int64 v96; // [rsp+40h] [rbp-290h]
  __int64 *v97; // [rsp+48h] [rbp-288h]
  signed int v98; // [rsp+48h] [rbp-288h]
  __int64 v99; // [rsp+50h] [rbp-280h]
  __int64 v100; // [rsp+58h] [rbp-278h]
  int v101; // [rsp+60h] [rbp-270h]
  unsigned int v102; // [rsp+64h] [rbp-26Ch]
  __int64 *v103; // [rsp+68h] [rbp-268h]
  int v104; // [rsp+68h] [rbp-268h]
  __int64 v105; // [rsp+68h] [rbp-268h]
  int v106; // [rsp+68h] [rbp-268h]
  __int64 v107; // [rsp+70h] [rbp-260h]
  __int64 v108; // [rsp+70h] [rbp-260h]
  __int64 v109; // [rsp+78h] [rbp-258h]
  __int64 *v110; // [rsp+78h] [rbp-258h]
  int v111; // [rsp+80h] [rbp-250h]
  __int64 *v112; // [rsp+80h] [rbp-250h]
  signed __int64 v113; // [rsp+80h] [rbp-250h]
  int v114; // [rsp+80h] [rbp-250h]
  int v115; // [rsp+80h] [rbp-250h]
  __int64 v116; // [rsp+80h] [rbp-250h]
  unsigned __int64 v117; // [rsp+80h] [rbp-250h]
  _BYTE *v118; // [rsp+88h] [rbp-248h]
  _BYTE *v119; // [rsp+90h] [rbp-240h]
  __int64 v120; // [rsp+90h] [rbp-240h]
  __int64 v121; // [rsp+98h] [rbp-238h]
  unsigned __int64 v122; // [rsp+A0h] [rbp-230h]
  __int64 v123; // [rsp+B0h] [rbp-220h] BYREF
  int v124; // [rsp+BCh] [rbp-214h]
  __int64 v125; // [rsp+C4h] [rbp-20Ch]
  int v126; // [rsp+CCh] [rbp-204h]
  _BYTE v127[32]; // [rsp+D0h] [rbp-200h] BYREF
  __int16 v128; // [rsp+F0h] [rbp-1E0h]
  void *s; // [rsp+100h] [rbp-1D0h] BYREF
  __int64 v130; // [rsp+108h] [rbp-1C8h]
  _BYTE v131[128]; // [rsp+110h] [rbp-1C0h] BYREF
  __int64 *v132; // [rsp+190h] [rbp-140h] BYREF
  __int64 v133; // [rsp+198h] [rbp-138h] BYREF
  _QWORD v134[38]; // [rsp+1A0h] [rbp-130h] BYREF

  if ( (unsigned int)*a2 - 42 > 0x11 || !sub_BCAC40(*((_QWORD *)a2 + 1), 1) )
    return 0;
  v6 = (a2[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a2 - 1) : &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v7 = *(_BYTE **)v6;
  v8 = (_BYTE *)*((_QWORD *)v6 + 4);
  if ( (unsigned __int8)(**(_BYTE **)v6 - 82) > 1u )
    return 0;
  v9 = *((_QWORD *)v7 - 8);
  if ( *(_BYTE *)v9 <= 0x1Cu )
    return 0;
  v119 = (_BYTE *)*((_QWORD *)v7 - 4);
  if ( *v119 > 0x15u )
    return 0;
  v10 = sub_B53900((__int64)v7) & 0xFFFFFFFFFFLL;
  if ( (unsigned __int8)(*v8 - 82) > 1u )
    return 0;
  v121 = *((_QWORD *)v8 - 8);
  if ( *(_BYTE *)v121 <= 0x1Cu )
    return 0;
  v118 = (_BYTE *)*((_QWORD *)v8 - 4);
  if ( *v118 > 0x15u )
    return 0;
  v122 = sub_B53900((__int64)v8);
  v125 = sub_B53630(v10, v122 & 0xFFFFFFFFFFLL);
  v126 = v11;
  if ( !(_BYTE)v11 || *(_BYTE *)v9 != 90 )
    return 0;
  if ( (*(_BYTE *)(v9 + 7) & 0x40) != 0 )
  {
    v12 = *(_QWORD *)(v9 - 8);
    v13 = *(__int64 **)v12;
    if ( !*(_QWORD *)v12 )
      return 0;
  }
  else
  {
    v12 = v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF);
    v13 = *(__int64 **)v12;
    if ( !*(_QWORD *)v12 )
      return 0;
  }
  v14 = *(_QWORD *)(v12 + 32);
  if ( *(_BYTE *)v14 != 17 )
    return 0;
  if ( *(_DWORD *)(v14 + 32) > 0x40u )
  {
    v110 = v13;
    v111 = *(_DWORD *)(v14 + 32);
    if ( v111 - (unsigned int)sub_C444A0(v14 + 24) <= 0x40 )
    {
      v13 = v110;
      v109 = **(_QWORD **)(v14 + 24);
      goto LABEL_20;
    }
    return 0;
  }
  v109 = *(_QWORD *)(v14 + 24);
LABEL_20:
  v132 = v13;
  v133 = (__int64)&v123;
  if ( *(_BYTE *)v121 != 90 )
    return 0;
  v15 = (*(_BYTE *)(v121 + 7) & 0x40) != 0
      ? *(__int64 ***)(v121 - 8)
      : (__int64 **)(v121 - 32LL * (*(_DWORD *)(v121 + 4) & 0x7FFFFFF));
  if ( *v15 != v13 )
    return 0;
  v112 = v13;
  v4 = sub_11B1B00((_QWORD **)&v133, (__int64)v15[4]);
  if ( !(_BYTE)v4 )
    return 0;
  v100 = sub_2C4CBB0(a1, v9, v121, *(_DWORD *)(a1 + 192));
  if ( !v100 )
    return 0;
  v97 = v112;
  v102 = v125;
  v16 = ((unsigned int)v125 < 0x10) + 53;
  if ( *(_BYTE *)(v112[1] + 8) != 17 )
    return 0;
  v99 = v112[1];
  v17 = sub_DFD3F0(*(_QWORD *)(a1 + 152));
  v93 = v18;
  v96 = v17;
  v19 = sub_DFD3F0(*(_QWORD *)(a1 + 152));
  v92 = v20;
  v103 = *(__int64 **)(a1 + 152);
  v107 = v19;
  sub_1001990(*(_QWORD ***)(v9 + 8));
  v21 = sub_DFD2D0(v103, v16, *(_QWORD *)(v9 + 8));
  v104 = v22;
  v113 = v21;
  v24 = sub_DFD800(
          *(_QWORD *)(a1 + 152),
          (unsigned int)*a2 - 29,
          *((_QWORD *)a2 + 1),
          *(_DWORD *)(a1 + 192),
          0,
          0,
          0,
          0,
          0,
          0);
  v25 = 2 * v113;
  if ( !is_mul_ok(2u, v113) )
  {
    v25 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v113 <= 0 )
      v25 = 0x8000000000000000LL;
  }
  v26 = 1;
  if ( v92 != 1 )
    v26 = v93;
  v114 = v26;
  v27 = v107 + v96;
  if ( __OFADD__(v107, v96) )
  {
    v27 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v107 <= 0 )
      v27 = 0x8000000000000000LL;
  }
  v28 = 1;
  if ( v104 != 1 )
    v28 = v114;
  v61 = __OFADD__(v25, v27);
  v29 = v25 + v27;
  if ( v61 )
  {
    v29 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v25 <= 0 )
      v29 = 0x8000000000000000LL;
  }
  v30 = v23 == 1;
  v31 = 1;
  if ( !v30 )
    v31 = v28;
  v61 = __OFADD__(v24, v29);
  v32 = v24 + v29;
  v115 = v31;
  if ( v61 )
  {
    v32 = 0x8000000000000000LL;
    if ( v24 > 0 )
      v32 = 0x7FFFFFFFFFFFFFFFLL;
  }
  v33 = (_QWORD **)v97[1];
  v94 = v32;
  v34 = v123;
  v95 = v97;
  v35 = v109;
  if ( v100 != v9 )
  {
    v35 = v123;
    v34 = v109;
  }
  v101 = v35;
  v98 = v34;
  v105 = sub_1001990(v33);
  v83 = *(__int64 **)(a1 + 152);
  sub_1001990((_QWORD **)v95[1]);
  v36 = sub_DFD2D0(v83, v16, v95[1]);
  v39 = v95;
  v40 = v36;
  v41 = v37;
  s = v131;
  v42 = *(unsigned int *)(v99 + 32);
  v130 = 0x2000000000LL;
  v43 = v42;
  if ( (unsigned int)v42 > 0x20 )
  {
    v86 = v37;
    v90 = v42;
    v91 = v42;
    sub_C8D5F0((__int64)&s, v131, v42, 4u, v42, v38);
    memset(s, 255, 4 * v91);
    v41 = v86;
    v39 = v95;
    LODWORD(v130) = v90;
  }
  else
  {
    if ( v42 )
    {
      v44 = 4 * v42;
      if ( v44 )
        memset(v131, -1, (unsigned int)v44);
    }
    LODWORD(v130) = v43;
  }
  v84 = (__int64)v39;
  v87 = v41;
  *((_DWORD *)s + v98) = v101;
  v45 = sub_DFBC30(
          *(__int64 **)(a1 + 152),
          7,
          v105,
          (__int64)s,
          (unsigned int)v130,
          *(unsigned int *)(a1 + 192),
          0,
          0,
          0,
          0,
          0);
  v46 = v87;
  if ( v47 == 1 )
    v46 = 1;
  if ( __OFADD__(v45, v40) )
  {
    v80 = v45 <= 0;
    v81 = 0x8000000000000000LL;
    if ( !v80 )
      v81 = 0x7FFFFFFFFFFFFFFFLL;
    v48 = v81;
  }
  else
  {
    v48 = v45 + v40;
  }
  v88 = v46;
  v49 = sub_DFD800(*(_QWORD *)(a1 + 152), (unsigned int)*a2 - 29, v105, *(_DWORD *)(a1 + 192), 0, 0, 0, 0, 0, 0);
  v50 = v88;
  if ( v51 == 1 )
    v50 = 1;
  v52 = v49 + v48;
  if ( __OFADD__(v49, v48) )
  {
    v52 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v49 <= 0 )
      v52 = 0x8000000000000000LL;
  }
  v82 = v84;
  v85 = v50;
  v89 = v52;
  v53 = sub_DFD3F0(*(_QWORD *)(a1 + 152));
  v54 = v85;
  v55 = v53;
  if ( v56 == 1 )
    v54 = 1;
  v57 = v53 + v89;
  if ( __OFADD__(v55, v89) )
  {
    v57 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v55 <= 0 )
      v57 = 0x8000000000000000LL;
  }
  v58 = *(_QWORD *)(v9 + 16);
  if ( !v58 || *(_QWORD *)(v58 + 8) )
  {
    if ( v93 == 1 )
      v54 = 1;
    v61 = __OFADD__(v96, v57);
    v57 += v96;
    if ( v61 )
    {
      v57 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v96 <= 0 )
        v57 = 0x8000000000000000LL;
    }
  }
  v59 = *(_QWORD *)(v121 + 16);
  if ( !v59 || *(_QWORD *)(v59 + 8) )
  {
    if ( v92 == 1 )
      v54 = 1;
    v61 = __OFADD__(v107, v57);
    v57 += v107;
    if ( v61 )
    {
      v57 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v107 <= 0 )
        v57 = 0x8000000000000000LL;
    }
  }
  v61 = __OFSUB__(v115, v54);
  v60 = v115 - v54 < 0;
  if ( v115 == v54 )
  {
    v61 = __OFSUB__(v94, v57);
    v60 = (__int64)(v94 - v57) < 0;
  }
  if ( v54 || v60 != v61 )
  {
    v4 = 0;
  }
  else
  {
    v62 = sub_ACADE0(*(__int64 ***)(v99 + 24));
    v63 = *(unsigned int *)(v99 + 32);
    v64 = v82;
    v65 = v62;
    v66 = v63;
    v132 = v134;
    v133 = 0x2000000000LL;
    if ( (unsigned int)v63 > 0x20 )
    {
      v106 = v63;
      v108 = v62;
      v117 = v63;
      sub_C8D5F0((__int64)&v132, v134, v63, 8u, v63, v62);
      v78 = v132;
      v66 = v106;
      v64 = v82;
      v79 = &v132[v117];
      while ( v79 != v78 )
        *v78++ = v108;
    }
    else if ( v63 )
    {
      v67 = v134;
      v68 = &v134[v63];
      while ( v68 != v67 )
        *v67++ = v65;
    }
    LODWORD(v133) = v66;
    v116 = v64;
    v132[v109] = (__int64)v119;
    v132[v123] = (__int64)v118;
    v128 = 257;
    v69 = sub_AD3730(v132, (unsigned int)v133);
    v120 = sub_2B22A00(a1 + 8, v102, v116, v69, (__int64)v127, 0);
    v72 = sub_2C4FD50(v120, v101, v98, (__int64 *)(a1 + 8), v70, v71);
    v73 = (unsigned __int8 *)v120;
    v74 = (unsigned __int8 *)v72;
    if ( v100 == v9 )
    {
      v73 = (unsigned __int8 *)v72;
      v74 = (unsigned __int8 *)v120;
    }
    v75 = *a2 - 29;
    v128 = 257;
    v76 = (_BYTE *)sub_2C51350((__int64 *)(a1 + 8), v75, v73, v74, v124, 0, (__int64)v127, 0);
    v128 = 257;
    v77 = sub_A83900((unsigned int **)(a1 + 8), v76, v98, (__int64)v127);
    sub_2C535E0(a1, a2, v77);
    if ( v132 != v134 )
      _libc_free((unsigned __int64)v132);
  }
  if ( s != v131 )
    _libc_free((unsigned __int64)s);
  return v4;
}
