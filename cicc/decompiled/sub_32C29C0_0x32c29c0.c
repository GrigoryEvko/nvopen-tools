// Function: sub_32C29C0
// Address: 0x32c29c0
//
__int64 __fastcall sub_32C29C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r14d
  __int64 v7; // r12
  unsigned __int16 *v8; // rdx
  int v9; // eax
  __int64 v10; // rdx
  unsigned __int16 v11; // ax
  __int64 v12; // rdx
  __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  char v16; // al
  int v17; // r13d
  __int64 v18; // rdx
  unsigned __int64 v19; // rdx
  char v20; // al
  unsigned int v21; // ebx
  _BYTE *v22; // rax
  _BYTE *v23; // rbx
  unsigned __int64 v24; // r13
  unsigned __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned int v30; // eax
  int v31; // eax
  __int64 v32; // rdx
  __int64 v33; // rcx
  unsigned int *v34; // rbx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // r9
  __int64 v38; // rdx
  __int64 v39; // r8
  __int64 *v40; // rdx
  _BYTE *v41; // rax
  __int64 v42; // r14
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // r14
  __int64 v49; // rcx
  __int64 v50; // rsi
  __int64 v51; // r9
  __int64 v52; // rax
  __int16 *v53; // rdx
  __int16 v54; // ax
  __int64 v55; // rdx
  int v56; // ecx
  unsigned int v57; // r14d
  unsigned __int16 v58; // ax
  int v59; // r8d
  unsigned __int16 v60; // bx
  __int64 v61; // rsi
  __int64 v62; // r14
  unsigned __int64 v63; // r9
  __int64 v64; // r15
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rsi
  int v68; // eax
  __int64 v69; // r12
  unsigned __int64 v70; // r14
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 v75; // rax
  unsigned __int64 v76; // rdx
  __int64 *v77; // rax
  __int64 v78; // rdi
  __int64 v79; // rax
  __int64 v80; // rdx
  unsigned int v81; // ebx
  __int64 *v82; // r14
  unsigned __int16 v83; // ax
  int v84; // r9d
  int v85; // r8d
  unsigned int v86; // eax
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // r13
  __int64 v90; // rcx
  __int64 v91; // rax
  __int64 v92; // rax
  int v93; // edx
  int v94; // edx
  __int128 v95; // [rsp-10h] [rbp-210h]
  __int128 v96; // [rsp-10h] [rbp-210h]
  __int128 v97; // [rsp-10h] [rbp-210h]
  __int64 v98; // [rsp+8h] [rbp-1F8h]
  int v99; // [rsp+10h] [rbp-1F0h]
  __int64 v100; // [rsp+10h] [rbp-1F0h]
  __int64 v101; // [rsp+10h] [rbp-1F0h]
  __int64 v102; // [rsp+18h] [rbp-1E8h]
  __int64 v103; // [rsp+20h] [rbp-1E0h]
  unsigned int *i; // [rsp+28h] [rbp-1D8h]
  __int64 *v105; // [rsp+28h] [rbp-1D8h]
  int v106; // [rsp+28h] [rbp-1D8h]
  __int64 v107; // [rsp+30h] [rbp-1D0h]
  unsigned int v108; // [rsp+30h] [rbp-1D0h]
  unsigned __int64 v109; // [rsp+30h] [rbp-1D0h]
  __int64 v110; // [rsp+30h] [rbp-1D0h]
  __int64 v111; // [rsp+30h] [rbp-1D0h]
  __int64 v112; // [rsp+30h] [rbp-1D0h]
  __int64 v113; // [rsp+30h] [rbp-1D0h]
  __int64 v114; // [rsp+38h] [rbp-1C8h]
  __int64 v115; // [rsp+38h] [rbp-1C8h]
  __int64 v116; // [rsp+40h] [rbp-1C0h] BYREF
  __int64 v117; // [rsp+48h] [rbp-1B8h]
  __int64 v118; // [rsp+50h] [rbp-1B0h] BYREF
  __int64 v119; // [rsp+58h] [rbp-1A8h]
  __int64 v120; // [rsp+60h] [rbp-1A0h] BYREF
  int v121; // [rsp+68h] [rbp-198h]
  __int64 v122; // [rsp+70h] [rbp-190h]
  __int64 v123; // [rsp+78h] [rbp-188h]
  __int64 v124; // [rsp+80h] [rbp-180h]
  __int64 v125; // [rsp+88h] [rbp-178h]
  __int64 v126; // [rsp+90h] [rbp-170h]
  __int64 v127; // [rsp+98h] [rbp-168h]
  __int64 v128; // [rsp+A0h] [rbp-160h] BYREF
  int v129; // [rsp+A8h] [rbp-158h]
  _BYTE *v130; // [rsp+B0h] [rbp-150h] BYREF
  __int64 v131; // [rsp+B8h] [rbp-148h]
  _BYTE v132[48]; // [rsp+C0h] [rbp-140h] BYREF
  _BYTE *v133; // [rsp+F0h] [rbp-110h] BYREF
  __int64 v134; // [rsp+F8h] [rbp-108h]
  _BYTE v135[48]; // [rsp+100h] [rbp-100h] BYREF
  int v136; // [rsp+130h] [rbp-D0h]
  _BYTE *v137; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v138; // [rsp+148h] [rbp-B8h]
  _BYTE v139[176]; // [rsp+150h] [rbp-B0h] BYREF

  v7 = a2;
  v116 = a3;
  v8 = *(unsigned __int16 **)(a2 + 48);
  v117 = a4;
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v137) = v9;
  v138 = v10;
  if ( (_WORD)v9 )
  {
    v11 = word_4456580[v9 - 1];
    v12 = 0;
  }
  else
  {
    v11 = sub_3009970((__int64)&v137, a2, v10, a4, a5);
  }
  LOWORD(v118) = v11;
  v119 = v12;
  if ( (_WORD)v116 == v11 )
  {
    if ( v11 || v12 == v117 )
      return v7;
    goto LABEL_7;
  }
  if ( !v11 )
  {
LABEL_7:
    v124 = sub_3007260((__int64)&v118);
    v125 = v14;
    v15 = v124;
    v16 = v125;
    goto LABEL_8;
  }
  if ( v11 == 1 || (unsigned __int16)(v11 - 504) <= 7u )
    goto LABEL_139;
  v65 = 16LL * (v11 - 1);
  v15 = *(_QWORD *)&byte_444C4A0[v65];
  v16 = byte_444C4A0[v65 + 8];
LABEL_8:
  LOBYTE(v138) = v16;
  v137 = (_BYTE *)v15;
  v17 = sub_CA1930(&v137);
  if ( (_WORD)v116 )
  {
    if ( (_WORD)v116 == 1 || (unsigned __int16)(v116 - 504) <= 7u )
      goto LABEL_139;
    v66 = 16LL * ((unsigned __int16)v116 - 1);
    v19 = *(_QWORD *)&byte_444C4A0[v66];
    v20 = byte_444C4A0[v66 + 8];
  }
  else
  {
    v122 = sub_3007260((__int64)&v116);
    v123 = v18;
    v19 = v122;
    v20 = v123;
  }
  v137 = (_BYTE *)v19;
  LOBYTE(v138) = v20;
  v21 = sub_CA1930(&v137);
  if ( v17 == v21 )
  {
    v34 = *(unsigned int **)(a2 + 40);
    v137 = v139;
    v138 = 0x800000000LL;
    for ( i = &v34[10 * *(unsigned int *)(a2 + 64)]; i != v34; ++*((_DWORD *)a1 + 12) )
    {
      while ( 1 )
      {
        v45 = v34[2];
        v46 = *(_QWORD *)v34;
        v47 = *(_QWORD *)v34;
        v48 = *((_QWORD *)v34 + 1);
        v49 = *(_QWORD *)(*(_QWORD *)v34 + 48LL) + 16 * v45;
        if ( *(_WORD *)v49 != (_WORD)v118 || *(_QWORD *)(v49 + 8) != v119 && !*(_WORD *)v49 )
        {
          v50 = *(_QWORD *)(v7 + 80);
          v51 = *a1;
          v133 = (_BYTE *)v50;
          if ( v50 )
          {
            v99 = v51;
            sub_B96E90((__int64)&v133, v50, 1);
            LODWORD(v51) = v99;
          }
          *((_QWORD *)&v95 + 1) = v48;
          *(_QWORD *)&v95 = v47;
          LODWORD(v134) = *(_DWORD *)(v7 + 72);
          v46 = sub_33FAF80(v51, 216, (unsigned int)&v133, v118, v119, v51, v95);
          v45 = (unsigned int)v45;
          if ( v133 )
          {
            v98 = (unsigned int)v45;
            v100 = v46;
            sub_B91220((__int64)&v133, (__int64)v133);
            v45 = v98;
            v46 = v100;
          }
        }
        a2 = (unsigned int)v116;
        v35 = sub_33FB890(*a1, (unsigned int)v116, v117, v46, v45 | v48 & 0xFFFFFFFF00000000LL);
        v37 = v36;
        v38 = (unsigned int)v138;
        v39 = v35;
        if ( (unsigned __int64)(unsigned int)v138 + 1 > HIDWORD(v138) )
        {
          a2 = (__int64)v139;
          v101 = v35;
          v102 = v37;
          sub_C8D5F0((__int64)&v137, v139, (unsigned int)v138 + 1LL, 0x10u, v35, v37);
          v38 = (unsigned int)v138;
          v39 = v101;
          v37 = v102;
        }
        v40 = (__int64 *)&v137[16 * v38];
        *v40 = v39;
        v40[1] = v37;
        LODWORD(v138) = v138 + 1;
        v41 = &v137[16 * (unsigned int)v138];
        v42 = *((_QWORD *)v41 - 2);
        if ( *(_DWORD *)(v42 + 24) != 328 )
        {
          a2 = (__int64)&v133;
          v133 = (_BYTE *)*((_QWORD *)v41 - 2);
          sub_32B3B20((__int64)(a1 + 71), (__int64 *)&v133);
          if ( *(int *)(v42 + 88) < 0 )
            break;
        }
        v34 += 10;
        if ( i == v34 )
          goto LABEL_67;
      }
      *(_DWORD *)(v42 + 88) = *((_DWORD *)a1 + 12);
      v52 = *((unsigned int *)a1 + 12);
      if ( v52 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
      {
        a2 = (__int64)(a1 + 7);
        sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v52 + 1, 8u, v43, v44);
        v52 = *((unsigned int *)a1 + 12);
      }
      v34 += 10;
      *(_QWORD *)(a1[5] + 8 * v52) = v42;
    }
LABEL_67:
    v53 = *(__int16 **)(v7 + 48);
    v54 = *v53;
    v55 = *((_QWORD *)v53 + 1);
    LOWORD(v133) = v54;
    v134 = v55;
    if ( v54 )
    {
      if ( (unsigned __int16)(v54 - 176) > 0x34u )
      {
LABEL_69:
        v56 = word_4456340[(unsigned __int16)v133 - 1];
        goto LABEL_70;
      }
    }
    else if ( !sub_3007100((__int64)&v133) )
    {
      goto LABEL_87;
    }
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( (_WORD)v133 )
    {
      if ( (unsigned __int16)((_WORD)v133 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_69;
    }
LABEL_87:
    v56 = sub_3007130((__int64)&v133, a2);
LABEL_70:
    v57 = v116;
    v108 = v56;
    v105 = *(__int64 **)(*a1 + 64);
    v103 = v117;
    v58 = sub_2D43050(v116, v56);
    v59 = 0;
    v60 = v58;
    if ( !v58 )
    {
      v60 = sub_3009400(v105, v57, v103, v108, 0);
      v59 = v93;
    }
    v61 = *(_QWORD *)(v7 + 80);
    v62 = *a1;
    v63 = (unsigned __int64)v137;
    v64 = (unsigned int)v138;
    v133 = (_BYTE *)v61;
    if ( v61 )
    {
      v106 = v59;
      v109 = (unsigned __int64)v137;
      sub_B96E90((__int64)&v133, v61, 1);
      v59 = v106;
      v63 = v109;
    }
    *((_QWORD *)&v96 + 1) = v64;
    *(_QWORD *)&v96 = v63;
    LODWORD(v134) = *(_DWORD *)(v7 + 72);
    v7 = sub_33FC220(v62, 156, (unsigned int)&v133, v60, v59, v63, v96);
    if ( v133 )
      sub_B91220((__int64)&v133, (__int64)v133);
    v25 = (unsigned __int64)v137;
    if ( v137 != v139 )
      goto LABEL_28;
    return v7;
  }
  if ( !(_WORD)v118 )
  {
    if ( !(unsigned __int8)sub_3007030((__int64)&v118) )
      goto LABEL_13;
    v26 = sub_3007260((__int64)&v118);
    v126 = v26;
    v127 = v27;
LABEL_107:
    v137 = (_BYTE *)v26;
    LOBYTE(v138) = v27;
    v86 = sub_CA1930(&v137);
    switch ( v86 )
    {
      case 1u:
        LOWORD(v87) = 2;
        break;
      case 2u:
        LOWORD(v87) = 3;
        break;
      case 4u:
        LOWORD(v87) = 4;
        break;
      case 8u:
        LOWORD(v87) = 5;
        break;
      case 0x10u:
        LOWORD(v87) = 6;
        break;
      case 0x20u:
        LOWORD(v87) = 7;
        break;
      case 0x40u:
        LOWORD(v87) = 8;
        break;
      case 0x80u:
        LOWORD(v87) = 9;
        break;
      default:
        v87 = sub_3007020(*(_QWORD **)(*a1 + 64), v86);
        v107 = v87;
        v89 = v88;
LABEL_116:
        v90 = v107;
        LOWORD(v90) = v87;
        v113 = v90;
        v91 = sub_32C29C0(a1, a2, (unsigned int)v90, v89);
        v119 = v89;
        v7 = v91;
        v118 = v113;
        goto LABEL_13;
    }
    v89 = 0;
    goto LABEL_116;
  }
  if ( (unsigned __int16)(v118 - 10) <= 6u
    || (unsigned __int16)(v118 - 126) <= 0x31u
    || (unsigned __int16)(v118 - 208) <= 0x14u )
  {
    if ( (_WORD)v118 == 1 || (unsigned __int16)(v118 - 504) <= 7u )
      goto LABEL_139;
    v27 = 16LL * ((unsigned __int16)v118 - 1);
    v26 = *(_QWORD *)&byte_444C4A0[v27];
    LOBYTE(v27) = byte_444C4A0[v27 + 8];
    goto LABEL_107;
  }
LABEL_13:
  if ( !(_WORD)v116 )
  {
    if ( !(unsigned __int8)sub_3007030((__int64)&v116) )
      goto LABEL_17;
    v28 = sub_3007260((__int64)&v116);
    v138 = v29;
    v137 = (_BYTE *)v28;
LABEL_41:
    v133 = (_BYTE *)v28;
    LOBYTE(v134) = v29;
    v30 = sub_CA1930(&v133);
    switch ( v30 )
    {
      case 1u:
        LOWORD(v31) = 2;
        break;
      case 2u:
        LOWORD(v31) = 3;
        break;
      case 4u:
        LOWORD(v31) = 4;
        break;
      case 8u:
        LOWORD(v31) = 5;
        break;
      case 0x10u:
        LOWORD(v31) = 6;
        break;
      case 0x20u:
        LOWORD(v31) = 7;
        break;
      case 0x40u:
        LOWORD(v31) = 8;
        break;
      case 0x80u:
        LOWORD(v31) = 9;
        break;
      default:
        v31 = sub_3007020(*(_QWORD **)(*a1 + 64), v30);
        HIWORD(v5) = HIWORD(v31);
        v33 = v32;
LABEL_122:
        LOWORD(v5) = v31;
        v92 = sub_32C29C0(a1, v7, v5, v33);
        return sub_32C29C0(a1, v92, (unsigned int)v116, v117);
    }
    v33 = 0;
    goto LABEL_122;
  }
  if ( (unsigned __int16)(v116 - 10) <= 6u
    || (unsigned __int16)(v116 - 126) <= 0x31u
    || (unsigned __int16)(v116 - 208) <= 0x14u )
  {
    if ( (_WORD)v116 != 1 && (unsigned __int16)(v116 - 504) > 7u )
    {
      v29 = 16LL * ((unsigned __int16)v116 - 1);
      v28 = *(_QWORD *)&byte_444C4A0[v29];
      LOBYTE(v29) = byte_444C4A0[v29 + 8];
      goto LABEL_41;
    }
LABEL_139:
    BUG();
  }
LABEL_17:
  v136 = 0;
  v133 = v135;
  v134 = 0x600000000LL;
  v130 = v132;
  v131 = 0x300000000LL;
  v22 = (_BYTE *)sub_2E79000(*(__int64 **)(*a1 + 40));
  if ( !(unsigned __int8)sub_33E3F20(v7, *v22 ^ 1u, v21, &v130, &v133) )
  {
    v7 = 0;
    goto LABEL_19;
  }
  v67 = *(_QWORD *)(v7 + 80);
  v120 = v67;
  if ( v67 )
    sub_B96E90((__int64)&v120, v67, 1);
  v68 = *(_DWORD *)(v7 + 72);
  v69 = (unsigned int)v131;
  v121 = v68;
  v137 = v139;
  v138 = 0x800000000LL;
  if ( (_DWORD)v131 )
  {
    v70 = 0;
    while ( 1 )
    {
      v78 = *a1;
      v79 = *(_QWORD *)&v133[8 * ((unsigned int)v70 >> 6)];
      if ( _bittest64(&v79, v70) )
        break;
      v73 = sub_34007B0(v78, (int)v130 + 16 * (int)v70, (unsigned int)&v120, v116, v117, 0, 0);
      v75 = (unsigned int)v138;
      v74 = v80;
      v76 = (unsigned int)v138 + 1LL;
      if ( v76 > HIDWORD(v138) )
        goto LABEL_98;
LABEL_95:
      ++v70;
      v77 = (__int64 *)&v137[16 * v75];
      *v77 = v73;
      v77[1] = v74;
      LODWORD(v138) = v138 + 1;
      if ( v69 == v70 )
      {
        LODWORD(v69) = v138;
        goto LABEL_100;
      }
    }
    v128 = 0;
    v129 = 0;
    v71 = sub_33F17F0(v78, 51, &v128, v116, v117);
    v73 = v71;
    v74 = v72;
    if ( v128 )
    {
      v110 = v71;
      v114 = v72;
      sub_B91220((__int64)&v128, v128);
      v73 = v110;
      v74 = v114;
    }
    v75 = (unsigned int)v138;
    v76 = (unsigned int)v138 + 1LL;
    if ( v76 <= HIDWORD(v138) )
      goto LABEL_95;
LABEL_98:
    v111 = v73;
    v115 = v74;
    sub_C8D5F0((__int64)&v137, v139, v76, 0x10u, v73, v74);
    v75 = (unsigned int)v138;
    v73 = v111;
    v74 = v115;
    goto LABEL_95;
  }
LABEL_100:
  v81 = v116;
  v82 = *(__int64 **)(*a1 + 64);
  v112 = v117;
  v83 = sub_2D43050(v116, v69);
  v85 = 0;
  if ( !v83 )
  {
    v83 = sub_3009400(v82, v81, v112, (unsigned int)v69, 0);
    v85 = v94;
  }
  *((_QWORD *)&v97 + 1) = (unsigned int)v138;
  *(_QWORD *)&v97 = v137;
  v7 = sub_33FC220(*a1, 156, (unsigned int)&v120, v83, v85, v84, v97);
  if ( v137 != v139 )
    _libc_free((unsigned __int64)v137);
  if ( v120 )
    sub_B91220((__int64)&v120, v120);
LABEL_19:
  v23 = v130;
  v24 = (unsigned __int64)&v130[16 * (unsigned int)v131];
  if ( v130 != (_BYTE *)v24 )
  {
    do
    {
      v24 -= 16LL;
      if ( *(_DWORD *)(v24 + 8) > 0x40u && *(_QWORD *)v24 )
        j_j___libc_free_0_0(*(_QWORD *)v24);
    }
    while ( v23 != (_BYTE *)v24 );
    v24 = (unsigned __int64)v130;
  }
  if ( (_BYTE *)v24 != v132 )
    _libc_free(v24);
  v25 = (unsigned __int64)v133;
  if ( v133 != v135 )
LABEL_28:
    _libc_free(v25);
  return v7;
}
