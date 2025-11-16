// Function: sub_1366100
// Address: 0x1366100
//
__int64 __fastcall sub_1366100(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        __int64 a7,
        __int64 a8,
        unsigned __int64 a9)
{
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // rcx
  char v15; // r13
  __int64 v16; // rax
  __int64 v17; // r11
  __int64 v18; // r8
  char v19; // r13
  unsigned __int8 v20; // al
  unsigned int v21; // r15d
  _BYTE *v22; // rdi
  bool v24; // al
  _QWORD *v25; // rax
  __int64 v26; // r15
  _QWORD *v27; // rax
  bool v28; // al
  int v29; // r10d
  unsigned __int64 v30; // rax
  __int64 v31; // rdi
  unsigned __int64 v32; // rbx
  __int64 v33; // rdx
  unsigned int v34; // r12d
  __int64 v35; // r15
  char v36; // r13
  __int64 v37; // rbx
  __int64 *v38; // rax
  unsigned int v39; // eax
  unsigned int v40; // ecx
  unsigned __int64 v41; // r10
  unsigned int v42; // r8d
  __int64 v43; // r9
  _BYTE *v44; // rbx
  bool v45; // r11
  char v46; // si
  unsigned int v47; // eax
  _QWORD *v48; // rcx
  _BYTE *v49; // rsi
  _QWORD *v50; // rdx
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rbx
  __int64 v53; // r15
  unsigned __int64 v54; // rax
  int v55; // eax
  __int64 v56; // rdx
  _QWORD **v57; // rax
  int v58; // esi
  __int64 v59; // rcx
  __int64 v60; // rcx
  unsigned int v61; // edi
  __int64 *v62; // rax
  __int64 v63; // rax
  unsigned int v64; // r8d
  __int64 *v65; // rdi
  __int64 v66; // rcx
  __int64 v67; // rax
  unsigned int v68; // eax
  __int64 *v69; // r13
  unsigned int v70; // r12d
  __int64 v71; // rdx
  __int64 v72; // r15
  __int64 v73; // rax
  __int64 v74; // r15
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rsi
  char v78; // al
  unsigned __int64 *v79; // rax
  unsigned __int64 v80; // rcx
  unsigned __int64 *v81; // rsi
  __int64 v82; // rax
  unsigned __int64 v83; // rax
  _QWORD *v84; // rdx
  unsigned __int64 v85; // rdx
  unsigned __int64 v86; // rax
  __int64 v87; // rsi
  __int64 v88; // r15
  __int64 v89; // rax
  unsigned __int64 v90; // rax
  unsigned int v91; // edx
  __int64 v92; // rsi
  unsigned int v93; // eax
  unsigned __int64 v94; // rdi
  __int64 v95; // rdi
  __int64 v96; // r9
  __int64 v97; // rdi
  __int64 v98; // rax
  unsigned __int64 v99; // r12
  __int64 v100; // rbx
  __int64 v101; // rax
  __int128 v102; // [rsp-28h] [rbp-288h]
  __int64 v103; // [rsp-18h] [rbp-278h]
  unsigned __int64 v104; // [rsp+8h] [rbp-258h]
  unsigned __int64 v105; // [rsp+18h] [rbp-248h]
  char v106; // [rsp+20h] [rbp-240h]
  __int64 v107; // [rsp+20h] [rbp-240h]
  __int64 v108; // [rsp+20h] [rbp-240h]
  __int64 v109; // [rsp+30h] [rbp-230h]
  bool v110; // [rsp+30h] [rbp-230h]
  __int64 v111; // [rsp+38h] [rbp-228h]
  bool v112; // [rsp+38h] [rbp-228h]
  _QWORD *v113; // [rsp+38h] [rbp-228h]
  unsigned __int64 v114; // [rsp+38h] [rbp-228h]
  __int64 v116; // [rsp+40h] [rbp-220h]
  __int64 v117; // [rsp+40h] [rbp-220h]
  __int64 v118; // [rsp+48h] [rbp-218h]
  __int64 v119; // [rsp+50h] [rbp-210h]
  __int64 v120; // [rsp+50h] [rbp-210h]
  __int64 v121; // [rsp+50h] [rbp-210h]
  char v124; // [rsp+78h] [rbp-1E8h]
  __int64 v125; // [rsp+78h] [rbp-1E8h]
  int v126; // [rsp+78h] [rbp-1E8h]
  __int64 v127; // [rsp+78h] [rbp-1E8h]
  __int64 v128; // [rsp+80h] [rbp-1E0h] BYREF
  unsigned int v129; // [rsp+88h] [rbp-1D8h]
  __int64 v130; // [rsp+90h] [rbp-1D0h] BYREF
  unsigned int v131; // [rsp+98h] [rbp-1C8h]
  __int64 v132[2]; // [rsp+A0h] [rbp-1C0h] BYREF
  __int64 v133[2]; // [rsp+B0h] [rbp-1B0h] BYREF
  __int64 *v134; // [rsp+C0h] [rbp-1A0h] BYREF
  __int64 v135; // [rsp+C8h] [rbp-198h]
  __int64 v136; // [rsp+D0h] [rbp-190h] BYREF
  unsigned int v137; // [rsp+D8h] [rbp-188h]
  _QWORD v138[3]; // [rsp+110h] [rbp-150h] BYREF
  _BYTE *v139; // [rsp+128h] [rbp-138h] BYREF
  __int64 v140; // [rsp+130h] [rbp-130h]
  _BYTE v141[104]; // [rsp+138h] [rbp-128h] BYREF
  __int64 v142; // [rsp+1A0h] [rbp-C0h] BYREF
  __int64 v143; // [rsp+1A8h] [rbp-B8h]
  __int64 v144; // [rsp+1B0h] [rbp-B0h]
  _BYTE *v145; // [rsp+1B8h] [rbp-A8h] BYREF
  __int64 v146; // [rsp+1C0h] [rbp-A0h]
  _BYTE v147[152]; // [rsp+1C8h] [rbp-98h] BYREF

  v12 = a1[5];
  v13 = a1[1];
  v14 = a1[4];
  v139 = v141;
  v145 = v147;
  v140 = 0x400000000LL;
  v146 = 0x400000000LL;
  v124 = sub_135EAB0(a2, (__int64)v138, v13, v14, v12);
  v15 = sub_135EAB0(a5, (__int64)&v142, a1[1], a1[4], a1[5]);
  v138[0] = sub_1CCAE90(v138[0], 0);
  v16 = sub_1CCAE90(v142, 0);
  v17 = v143;
  v18 = v144;
  v142 = v16;
  v19 = v124 | v15;
  v118 = v138[1] + v138[2];
  if ( !v19 )
  {
    v109 = v144;
    v111 = v143;
    v28 = sub_1360E30(a2, (__int64)v138, (__int64)&v142, a6);
    v17 = v111;
    v18 = v109;
    if ( v28 )
      goto LABEL_27;
  }
  v20 = *(_BYTE *)(a5 + 16);
  if ( v20 <= 0x17u )
  {
    if ( v20 != 5 || *(_WORD *)(a5 + 18) != 32 )
      goto LABEL_4;
  }
  else if ( v20 != 56 )
  {
LABEL_4:
    if ( (a3 & a6) == 0xFFFFFFFFFFFFFFFFLL )
      goto LABEL_31;
    v135 = 0;
    v103 = *(_QWORD *)(a7 + 16);
    *((_QWORD *)&v102 + 1) = *(_QWORD *)(a7 + 8);
    v136 = 0;
    *(_QWORD *)&v102 = *(_QWORD *)a7;
    v134 = 0;
    v21 = sub_1362890(a1, a8, 0xFFFFFFFFFFFFFFFFLL, a5, 0xFFFFFFFFFFFFFFFFLL, 0, 0, 0, v102, v103, a9);
    if ( (_BYTE)v21 != 3 )
    {
LABEL_6:
      v22 = v145;
      goto LABEL_7;
    }
    if ( v124 )
      goto LABEL_31;
    goto LABEL_33;
  }
  if ( !v19 )
  {
    v119 = v18;
    v125 = v17;
    v24 = sub_1360E30(a5, (__int64)&v142, (__int64)v138, a3);
    v17 = v125;
    v18 = v119;
    if ( v24 )
    {
LABEL_27:
      v22 = v145;
      v21 = 0;
      goto LABEL_7;
    }
  }
  v135 = 0;
  v136 = 0;
  v132[1] = 0;
  v133[0] = 0;
  v120 = v17 + v18;
  v134 = 0;
  v132[0] = 0;
  v21 = sub_1362890(a1, a8, 0xFFFFFFFFFFFFFFFFLL, a9, 0xFFFFFFFFFFFFFFFFLL, 0, 0, 0, 0, 0, 0);
  if ( (_BYTE)v21 == 1 && a6 == a3 )
  {
    v47 = sub_1362890(
            a1,
            a8,
            a3,
            a9,
            a6,
            0,
            *(_OWORD *)a4,
            *(_QWORD *)(a4 + 16),
            *(_OWORD *)a7,
            *(_QWORD *)(a7 + 16),
            0);
    v22 = v145;
    v21 = v47;
    if ( !(_BYTE)v47 && !v19 && v118 == v120 && (unsigned int)v140 == (unsigned __int64)(unsigned int)v146 )
    {
      v48 = v139;
      v49 = &v139[24 * (unsigned int)v140];
      if ( v139 == v49 )
        goto LABEL_7;
      v50 = v145;
      while ( *v48 == *v50 && v48[1] == v50[1] && v48[2] == v50[2] )
      {
        v48 += 3;
        v50 += 3;
        if ( v49 == (_BYTE *)v48 )
          goto LABEL_7;
      }
    }
    goto LABEL_73;
  }
  if ( (_BYTE)v21 != 3 )
    goto LABEL_6;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v25 = *(_QWORD **)(a2 - 8);
  else
    v25 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v26 = sub_164AA50(*v25);
  if ( (*(_BYTE *)(a5 + 23) & 0x40) != 0 )
    v27 = *(_QWORD **)(a5 - 8);
  else
    v27 = (_QWORD *)(a5 - 24LL * (*(_DWORD *)(a5 + 20) & 0xFFFFFFF));
  if ( v26 != sub_164AA50(*v27) )
    goto LABEL_24;
  v56 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v57 = (*(_BYTE *)(a5 + 23) & 0x40) != 0
      ? *(_QWORD ***)(a5 - 8)
      : (_QWORD **)(a5 - 24LL * (*(_DWORD *)(a5 + 20) & 0xFFFFFFF));
  if ( **v57 != **(_QWORD **)v56 )
    goto LABEL_24;
  v58 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( v58 != (*(_DWORD *)(a5 + 20) & 0xFFFFFFF) )
    goto LABEL_24;
  v59 = (unsigned int)(v58 - 1);
  if ( (unsigned int)v59 <= 1 )
    goto LABEL_24;
  v110 = a6 == -1 || a3 == -1;
  if ( v110 )
    goto LABEL_24;
  v60 = 3 * v59;
  v116 = *(_QWORD *)(v56 + v60 * 8);
  v113 = v57[v60];
  if ( *(_BYTE *)(v116 + 16) != 13 )
  {
    v116 = 0;
    if ( *((_BYTE *)v113 + 16) == 13 )
      goto LABEL_102;
    goto LABEL_142;
  }
  if ( *((_BYTE *)v113 + 16) != 13 )
  {
LABEL_142:
    v113 = 0;
LABEL_102:
    v67 = a1[1];
    v134 = &v136;
    v127 = v67;
    v136 = *(_QWORD *)(v56 + 24);
    v135 = 0x800000001LL;
    if ( v58 != 3 )
    {
      v106 = v19;
      v68 = 1;
      v69 = &v136;
      v104 = a5;
      v70 = 1;
      while ( 1 )
      {
        v74 = v68;
        v75 = sub_16348C0(a2);
        if ( *(_BYTE *)(sub_15F9F50(v75, v69, v74) + 8) != 14 )
          break;
        ++v70;
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v71 = *(_QWORD *)(a2 - 8);
        else
          v71 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v72 = *(_QWORD *)(v71 + 24LL * v70);
        v73 = (unsigned int)v135;
        if ( (unsigned int)v135 >= HIDWORD(v135) )
        {
          sub_16CD150(&v134, &v136, 0, 8);
          v73 = (unsigned int)v135;
        }
        v134[v73] = v72;
        v68 = v135 + 1;
        LODWORD(v135) = v135 + 1;
        if ( v58 - 2 == v70 )
        {
          v19 = v106;
          a5 = v104;
          goto LABEL_121;
        }
        v69 = v134;
      }
      v19 = v106;
      goto LABEL_139;
    }
LABEL_121:
    v105 = (unsigned __int64)v134;
    v107 = (unsigned int)v135;
    v76 = sub_16348C0(a2);
    v77 = sub_15F9F50(v76, v105, v107);
    v78 = *(_BYTE *)(v77 + 8);
    if ( v78 == 13 )
    {
      if ( v116 && v113 )
      {
        v79 = (unsigned __int64 *)sub_15A9930(v127, v77);
        v80 = *v79;
        v81 = v79;
        if ( *(_DWORD *)(v116 + 32) <= 0x40u )
          v82 = *(_QWORD *)(v116 + 24);
        else
          v82 = **(_QWORD **)(v116 + 24);
        v83 = v81[(unsigned int)v82 + 2];
        v84 = (_QWORD *)v113[3];
        if ( *((_DWORD *)v113 + 8) > 0x40u )
          v84 = (_QWORD *)*v84;
        v85 = v81[(unsigned int)v84 + 2];
        if ( v83 < v85 && v85 >= a3 + v83 && (v80 >= a6 + v85 || v83 >= a6 + v85 - v80) )
          goto LABEL_136;
        if ( v83 > v85 && v83 >= a6 + v85 )
        {
          v86 = a3 + v83;
          if ( v80 >= v86 || v85 >= v86 - v80 )
            goto LABEL_136;
        }
      }
LABEL_139:
      if ( v134 != &v136 )
        _libc_free((unsigned __int64)v134);
      goto LABEL_24;
    }
    if ( ((v78 - 14) & 0xFD) != 0 )
      goto LABEL_139;
    v87 = *(_QWORD *)(v77 + 24);
    v88 = 1;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v87 + 8) )
      {
        case 1:
          v89 = 16;
          break;
        case 2:
          v89 = 32;
          break;
        case 3:
        case 9:
          v89 = 64;
          break;
        case 4:
          v89 = 80;
          break;
        case 5:
        case 6:
          v89 = 128;
          break;
        case 7:
          v89 = 8 * (unsigned int)sub_15A9520(v127, 0);
          break;
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v101 = *(_QWORD *)(v87 + 32);
          v87 = *(_QWORD *)(v87 + 24);
          v88 *= v101;
          continue;
        case 0xB:
          v89 = *(_DWORD *)(v87 + 8) >> 8;
          break;
        case 0xD:
          v89 = 8LL * *(_QWORD *)sub_15A9930(v127, v87);
          break;
        case 0xE:
          v117 = *(_QWORD *)(v87 + 32);
          v108 = *(_QWORD *)(v87 + 24);
          v114 = (unsigned int)sub_15A9FE0(v127, v108);
          v89 = 8 * v117 * v114 * ((v114 + ((unsigned __int64)(sub_127FA20(v127, v108) + 7) >> 3) - 1) / v114);
          break;
        case 0xF:
          v89 = 8 * (unsigned int)sub_15A9520(v127, *(_DWORD *)(v87 + 8) >> 8);
          break;
      }
      break;
    }
    v90 = (unsigned __int64)(v88 * v89 + 7) >> 3;
    if ( a3 != v90 || a6 != v90 )
      goto LABEL_139;
    v91 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v92 = a2 - 24LL * v91;
    v93 = 0;
    while ( v91 - 2 != v93 )
    {
      ++v93;
      v95 = a2 - 24LL * v91;
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v95 = *(_QWORD *)(a2 - 8);
      v96 = *(_QWORD *)(v95 + 24LL * v93);
      if ( (*(_BYTE *)(a5 + 23) & 0x40) != 0 )
        v94 = *(_QWORD *)(a5 - 8);
      else
        v94 = a5 - 24LL * (*(_DWORD *)(a5 + 20) & 0xFFFFFFF);
      if ( v96 != *(_QWORD *)(v94 + 24LL * v93) )
        goto LABEL_139;
    }
    if ( v110 )
    {
LABEL_136:
      if ( v134 != &v136 )
      {
        _libc_free((unsigned __int64)v134);
        v21 = 0;
        v22 = v145;
        goto LABEL_7;
      }
      goto LABEL_27;
    }
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v92 = *(_QWORD *)(a2 - 8);
    v97 = *(_QWORD *)(v92 + 24LL * (v91 - 1));
    v98 = *(_DWORD *)(a5 + 20) & 0xFFFFFFF;
    if ( (*(_BYTE *)(a5 + 23) & 0x40) != 0 )
      v99 = *(_QWORD *)(a5 - 8);
    else
      v99 = a5 - 24 * v98;
    v100 = *(_QWORD *)(v99 + 24LL * (unsigned int)(v98 - 1));
    if ( *(_BYTE *)(v97 + 16) != 77 && *(_BYTE *)(v100 + 16) != 77 )
    {
      if ( (unsigned __int8)sub_14C00B0(v97, *(_QWORD *)(v99 + 24LL * (unsigned int)(v98 - 1)), v127, 0, 0, 0) )
        goto LABEL_136;
      goto LABEL_139;
    }
    if ( v97 == v100 || *(_QWORD *)v97 != *(_QWORD *)v100 )
      goto LABEL_139;
    sub_14C2530((unsigned int)&v128, v97, v127, 0, 0, 0, 0, 0);
    sub_14C2530((unsigned int)v132, v100, v127, 0, 0, 0, 0, 0);
    if ( v129 <= 0x40 )
    {
      if ( (v133[0] & v128) != 0 )
        goto LABEL_176;
    }
    else if ( (unsigned __int8)sub_16A59B0(&v128, v133) )
    {
      goto LABEL_176;
    }
    if ( v131 <= 0x40 )
    {
      if ( (v132[0] & v130) != 0 )
        goto LABEL_176;
    }
    else if ( (unsigned __int8)sub_16A59B0(&v130, v132) )
    {
LABEL_176:
      sub_135E100(v133);
      sub_135E100(v132);
      sub_135E100(&v130);
      sub_135E100(&v128);
      goto LABEL_136;
    }
    sub_135E100(v133);
    sub_135E100(v132);
    sub_135E100(&v130);
    sub_135E100(&v128);
    goto LABEL_139;
  }
  v61 = *(_DWORD *)(v116 + 32);
  v62 = *(__int64 **)(v116 + 24);
  if ( v61 > 0x40 )
    v63 = *v62;
  else
    v63 = (__int64)((_QWORD)v62 << (64 - (unsigned __int8)v61)) >> (64 - (unsigned __int8)v61);
  v64 = *((_DWORD *)v113 + 8);
  v65 = (__int64 *)v113[3];
  if ( v64 > 0x40 )
    v66 = *v65;
  else
    v66 = (__int64)((_QWORD)v65 << (64 - (unsigned __int8)v64)) >> (64 - (unsigned __int8)v64);
  if ( v66 != v63 )
  {
    v110 = 1;
    goto LABEL_102;
  }
LABEL_24:
  if ( v19 )
    goto LABEL_31;
  v124 = ((unsigned int)v146 | (unsigned int)v140) != 0;
  v118 -= v120;
  sub_1360FA0((__int64)a1, &v139, (__int64)&v145);
LABEL_33:
  if ( v118 )
  {
    if ( *a1 && *(_BYTE *)*a1 && v124 )
      goto LABEL_31;
    v29 = v140;
    if ( (_DWORD)v140 )
      goto LABEL_38;
    v51 = v118;
    v22 = v145;
    if ( v118 >= 0 )
    {
      v52 = a6;
      v21 = 1;
      if ( a6 == -1 )
        goto LABEL_7;
      goto LABEL_77;
    }
    v52 = a3;
    if ( a3 != -1 && a6 != -1 )
    {
      v51 = -v118;
LABEL_77:
      LOBYTE(v51) = v51 < v52;
      v21 = 2 * v51;
      goto LABEL_7;
    }
LABEL_73:
    v21 = 1;
    goto LABEL_7;
  }
  v29 = v140;
  if ( !(_DWORD)v140 )
  {
    v22 = v145;
    v21 = 3;
    goto LABEL_7;
  }
  if ( *a1 && *(_BYTE *)*a1 && v124 )
  {
LABEL_31:
    v22 = v145;
    v21 = 1;
    goto LABEL_7;
  }
LABEL_38:
  v112 = a6 != -1 && a3 != -1;
  if ( v112 )
  {
    v30 = (unsigned __int64)v139;
    v31 = abs64(*((_QWORD *)v139 + 2));
    if ( v29 != 1 )
    {
      v32 = 1;
      while ( 1 )
      {
        v33 = 3 * v32++;
        v31 = sub_1CCB0A0(v31, abs64(*(_QWORD *)(v30 + 8 * v33 + 16)));
        if ( v32 >= (unsigned int)v140 )
          break;
        v30 = (unsigned __int64)v139;
      }
    }
    if ( !(unsigned __int8)sub_1CCB0D0(v31, v118, a3, 0, a6) )
      goto LABEL_27;
    v29 = v140;
    if ( !(_DWORD)v140 )
      goto LABEL_31;
  }
  v126 = v29;
  v34 = 0;
  v35 = 0;
  v36 = 1;
  do
  {
    v37 = 24LL * v34;
    v38 = (__int64 *)&v139[v37];
    v35 |= *(_QWORD *)&v139[v37 + 16];
    if ( v36 )
    {
      v121 = *v38;
      sub_14C2530((unsigned int)&v134, *v38, a1[1], 0, a1[4], 0, a1[5], 0);
      v39 = v135;
      v40 = v135 - 1;
      if ( (unsigned int)v135 > 0x40 )
        v41 = v134[v40 >> 6];
      else
        v41 = (unsigned __int64)v134;
      v42 = v137 - 1;
      v43 = v136;
      if ( v137 > 0x40 )
        v43 = *(_QWORD *)(v136 + 8LL * (v42 >> 6));
      v44 = &v139[v37];
      v45 = 0;
      if ( !*((_DWORD *)v44 + 2) )
      {
        v46 = *(_BYTE *)(v121 + 16);
        v36 = v46 == 61 || (v41 & (1LL << v40)) != 0;
        v45 = v46 != 61;
      }
      v36 &= *((_QWORD *)v44 + 2) >= 0LL;
      if ( !v36 )
        v36 = v45 && (v43 & (1LL << v42)) != 0 && *((__int64 *)v44 + 2) < 0;
      if ( v137 > 0x40 && v136 )
      {
        j_j___libc_free_0_0(v136);
        v39 = v135;
      }
      if ( v39 > 0x40 && v134 )
        j_j___libc_free_0_0(v134);
    }
    ++v34;
  }
  while ( v34 != v126 );
  v53 = -v35 & v35;
  v54 = v118 & (v53 - 1);
  if ( a6 <= v54 && v112 && v53 - v54 >= a3 || v118 >= a6 && v118 > 0 && v36 )
    goto LABEL_27;
  v55 = sub_1361180((__int64)a1, &v139, a3, a6, v118, a1[4], a1[5]);
  v22 = v145;
  v21 = v55 ^ 1;
LABEL_7:
  if ( v22 != v147 )
    _libc_free((unsigned __int64)v22);
  if ( v139 != v141 )
    _libc_free((unsigned __int64)v139);
  return v21;
}
