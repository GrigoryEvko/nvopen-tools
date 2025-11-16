// Function: sub_14D44C0
// Address: 0x14d44c0
//
__int64 __fastcall sub_14D44C0(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v3; // r15
  __int64 v4; // r12
  bool v5; // zf
  char v6; // al
  char v7; // dl
  unsigned __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // r14
  unsigned int v11; // ebx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  int v15; // ebx
  char v16; // al
  __int64 v17; // rbx
  __int64 v18; // r14
  __int64 v19; // r13
  unsigned int v20; // r13d
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v25; // r13
  int v26; // eax
  int v27; // ebx
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  char v33; // cl
  int v34; // eax
  int v35; // esi
  unsigned int v36; // ebx
  _QWORD *v37; // r13
  unsigned int v38; // r14d
  int v39; // r12d
  __int64 v40; // rax
  __int64 v41; // rdi
  char v42; // al
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  int v46; // eax
  int v47; // r13d
  __int64 v48; // rcx
  unsigned int v49; // ebx
  _BYTE *v50; // r14
  __int64 v51; // rax
  __int64 v52; // rdx
  unsigned __int64 v53; // rcx
  char v55; // al
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  unsigned int v59; // eax
  __int64 v60; // r9
  unsigned __int64 v61; // r13
  __int64 v62; // rbx
  __int64 v63; // r14
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // rax
  int v75; // eax
  _QWORD *v76; // rax
  int v77; // eax
  __int64 v78; // rax
  __int64 v79; // rax
  unsigned __int64 v80; // rax
  __int64 v81; // r12
  __int64 v82; // r13
  __int64 v83; // rax
  __int64 v84; // r12
  __int64 v85; // r14
  __int64 v86; // rdx
  __int64 v87; // r15
  __int64 v88; // r13
  __int64 v89; // rsi
  __int64 v90; // rdx
  int v91; // eax
  int v92; // esi
  __int64 v93; // rax
  _QWORD *v94; // rbx
  char v95; // al
  unsigned int v96; // r14d
  int v97; // r12d
  __int64 v98; // r8
  __int64 v99; // rax
  __int64 v100; // rsi
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 v103; // rax
  __int64 v104; // rax
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // rdx
  __int64 v108; // rbx
  int v109; // eax
  __int64 *v110; // rax
  __int64 *v111; // rdx
  __int64 v112; // [rsp+8h] [rbp-1A8h]
  int v113; // [rsp+18h] [rbp-198h]
  int v114; // [rsp+1Ch] [rbp-194h]
  int v115; // [rsp+1Ch] [rbp-194h]
  __int64 v116; // [rsp+28h] [rbp-188h]
  unsigned int v117; // [rsp+28h] [rbp-188h]
  char v118; // [rsp+30h] [rbp-180h]
  int v119; // [rsp+30h] [rbp-180h]
  __int64 v120; // [rsp+30h] [rbp-180h]
  __int64 v121; // [rsp+30h] [rbp-180h]
  __int64 v122; // [rsp+30h] [rbp-180h]
  __int64 v123; // [rsp+38h] [rbp-178h]
  __int64 v124; // [rsp+38h] [rbp-178h]
  _QWORD *v125; // [rsp+38h] [rbp-178h]
  char v126; // [rsp+38h] [rbp-178h]
  unsigned int v127; // [rsp+40h] [rbp-170h]
  unsigned __int64 v128; // [rsp+40h] [rbp-170h]
  int v129; // [rsp+48h] [rbp-168h]
  __int64 v130; // [rsp+48h] [rbp-168h]
  unsigned int v131; // [rsp+48h] [rbp-168h]
  unsigned __int64 v132; // [rsp+48h] [rbp-168h]
  __int64 v133; // [rsp+48h] [rbp-168h]
  __int64 v134; // [rsp+48h] [rbp-168h]
  __int64 v135; // [rsp+48h] [rbp-168h]
  __int64 v136; // [rsp+48h] [rbp-168h]
  __int64 v137; // [rsp+48h] [rbp-168h]
  __int64 v138; // [rsp+48h] [rbp-168h]
  int v139; // [rsp+50h] [rbp-160h]
  __int64 v140; // [rsp+50h] [rbp-160h]
  __int64 v141; // [rsp+50h] [rbp-160h]
  __int64 v142; // [rsp+50h] [rbp-160h]
  unsigned int v143; // [rsp+50h] [rbp-160h]
  unsigned int v145; // [rsp+58h] [rbp-158h]
  __int64 v146; // [rsp+58h] [rbp-158h]
  __int64 v147; // [rsp+58h] [rbp-158h]
  unsigned int v148; // [rsp+58h] [rbp-158h]
  unsigned __int64 v149; // [rsp+60h] [rbp-150h] BYREF
  unsigned int v150; // [rsp+68h] [rbp-148h]
  __int64 v151; // [rsp+70h] [rbp-140h] BYREF
  __int64 v152; // [rsp+78h] [rbp-138h] BYREF
  _QWORD v153[38]; // [rsp+80h] [rbp-130h] BYREF

  v3 = a1;
  v4 = a2;
  v5 = (unsigned __int8)sub_1596070(a1) == 0;
  v6 = *(_BYTE *)(a2 + 8);
  if ( !v5 )
  {
    if ( v6 == 9 )
      return sub_15A4510(a1, a2, 0);
    v7 = *(_BYTE *)(a2 + 8);
    if ( v6 == 16 )
      v7 = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
    if ( v7 != 15 )
      return sub_15A04A0(a2);
  }
  v8 = *(_QWORD *)a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 || v6 != 11 && (unsigned __int8)(v6 - 1) > 5u )
  {
    if ( v6 != 16 )
      return sub_15A4510(a1, a2, 0);
    v16 = *(_BYTE *)(a1 + 16);
    if ( (unsigned __int8)(v16 - 13) <= 1u )
    {
      v151 = a1;
      v25 = sub_15A01B0(&v151, 1);
      if ( (unsigned __int8)sub_1593BB0(v25) && *(_BYTE *)(a2 + 8) != 9 )
        return sub_15A06D0(a2);
      else
        return sub_14D44C0(v25, a2, a3);
    }
    if ( (v16 & 0xFB) != 8 )
      return sub_15A4510(a1, a2, 0);
    v17 = *(_QWORD *)(a2 + 32);
    v18 = *(_QWORD *)(v8 + 32);
    v139 = v17;
    v129 = v18;
    if ( (_DWORD)v17 == (_DWORD)v18 )
      return sub_15A4510(a1, a2, 0);
    v19 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int8)(*(_BYTE *)(v19 + 8) - 1) <= 5u )
    {
      v20 = sub_1643030(*(_QWORD *)(a2 + 24));
      v21 = sub_16498A0(a1);
      v22 = sub_1644900(v21, v20);
      v23 = sub_16463B0(v22, (unsigned int)v17);
      if ( !(unsigned __int8)sub_1593BB0(a1) || *(_BYTE *)(v23 + 8) == 9 )
        a1 = sub_14D44C0(a1, v23, a3);
      else
        a1 = sub_15A06D0(v23);
      return sub_15A4510(a1, a2, 0);
    }
    v28 = **(_QWORD **)(v8 + 16);
    if ( (unsigned __int8)(*(_BYTE *)(v28 + 8) - 1) <= 5u )
    {
      v123 = **(_QWORD **)(v8 + 16);
      v127 = sub_1643030(v123);
      v29 = sub_16498A0(a1);
      v30 = sub_1644900(v29, v127);
      v31 = sub_16463B0(v30, (unsigned int)v18);
      v32 = sub_15A4510(a1, v31, 0);
      v28 = v123;
      v3 = v32;
      if ( (*(_BYTE *)(v32 + 16) & 0xFB) != 8 )
        return v3;
    }
    v124 = v28;
    v33 = *a3;
    v151 = (__int64)v153;
    v152 = 0x2000000000LL;
    if ( (unsigned int)v17 < (unsigned int)v18 )
    {
      v118 = v33;
      v116 = sub_15A06D0(v19);
      v145 = (unsigned int)v18 / (unsigned int)v17;
      v34 = sub_1643030(v124);
      v35 = 0;
      v114 = 0;
      v130 = v4;
      if ( v118 )
      {
        v35 = v34 * (v145 - 1);
        v34 = -v34;
      }
      v36 = 0;
      v119 = v34;
      while ( 1 )
      {
        v37 = (_QWORD *)v116;
        v38 = v35;
        v39 = 0;
        do
        {
          v40 = sub_15A0A60(v3, v36);
          v41 = v40;
          if ( !v40 )
          {
LABEL_129:
            v81 = v130;
LABEL_130:
            v3 = sub_15A4510(v3, v81, 0);
LABEL_48:
            if ( (_QWORD *)v151 != v153 )
              _libc_free(v151);
            return v3;
          }
          v42 = *(_BYTE *)(v40 + 16);
          if ( v42 == 9 )
          {
            v41 = sub_15A06D0(**(_QWORD **)(*(_QWORD *)v3 + 16LL));
            if ( !v41 )
              goto LABEL_129;
          }
          else if ( v42 != 13 )
          {
            goto LABEL_129;
          }
          ++v36;
          ++v39;
          v125 = (_QWORD *)sub_15A3CB0(v41, *v37, 0);
          v43 = sub_15A0680(*v125, v38, 0);
          v44 = sub_15A2D50(v125, v43, 0, 0);
          v38 += v119;
          v37 = (_QWORD *)sub_15A2D10(v37, v44);
        }
        while ( v145 != v39 );
        v45 = (unsigned int)v152;
        if ( (unsigned int)v152 >= HIDWORD(v152) )
        {
          sub_16CD150(&v151, v153, 0, 8);
          v45 = (unsigned int)v152;
        }
        ++v114;
        *(_QWORD *)(v151 + 8 * v45) = v37;
        LODWORD(v152) = v152 + 1;
        if ( v139 == v114 )
        {
LABEL_47:
          v3 = sub_15A01B0(v151, (unsigned int)v152);
          goto LABEL_48;
        }
      }
    }
    v126 = v33;
    v143 = (unsigned int)v17 / (unsigned int)v18;
    v91 = sub_127FA20((__int64)a3, v19);
    v117 = v91;
    v92 = 0;
    v148 = 0;
    v112 = v4;
    if ( v126 )
    {
      v92 = v91 * (v143 - 1);
      v91 = -v91;
    }
    v113 = v92;
    v115 = v91;
    while ( 1 )
    {
      v93 = sub_15A0A60(v3, v148);
      v94 = (_QWORD *)v93;
      if ( !v93 )
        goto LABEL_161;
      v95 = *(_BYTE *)(v93 + 16);
      if ( v95 != 9 )
        break;
      v106 = sub_1599EF0(v19);
      v107 = (unsigned int)v152;
      v108 = v106;
      v109 = v152;
      if ( v143 > HIDWORD(v152) - (unsigned __int64)(unsigned int)v152 )
      {
        sub_16CD150(&v151, v153, v143 + (unsigned __int64)(unsigned int)v152, 8);
        v107 = (unsigned int)v152;
        v109 = v152;
      }
      if ( v143 )
      {
        v110 = (__int64 *)(v151 + 8 * v107);
        v111 = &v110[v143];
        do
          *v110++ = v108;
        while ( v111 != v110 );
        v109 = v152;
      }
      LODWORD(v152) = v143 + v109;
LABEL_159:
      if ( v129 == ++v148 )
        goto LABEL_47;
    }
    if ( v95 != 13 )
    {
LABEL_161:
      v81 = v112;
      goto LABEL_130;
    }
    v96 = v113;
    v97 = 0;
    while ( 1 )
    {
      v100 = v96;
      v96 += v115;
      v101 = sub_159C470(*v94, v100, 0);
      v102 = sub_15A2D80(v94, v101, 0);
      if ( *(_BYTE *)(v19 + 8) == 15 )
      {
        v121 = v102;
        v103 = sub_16498A0(v3);
        v104 = sub_1644C60(v103, v117);
        v105 = sub_15A43B0(v121, v104, 0);
        v98 = sub_15A3BA0(v105, v19, 0);
        v99 = (unsigned int)v152;
        if ( (unsigned int)v152 >= HIDWORD(v152) )
        {
LABEL_158:
          v122 = v98;
          sub_16CD150(&v151, v153, 0, 8);
          v99 = (unsigned int)v152;
          v98 = v122;
        }
      }
      else
      {
        v98 = sub_15A43B0(v102, v19, 0);
        v99 = (unsigned int)v152;
        if ( (unsigned int)v152 >= HIDWORD(v152) )
          goto LABEL_158;
      }
      ++v97;
      *(_QWORD *)(v151 + 8 * v99) = v98;
      LODWORD(v152) = v152 + 1;
      if ( v143 == v97 )
        goto LABEL_159;
    }
  }
  v9 = *(_QWORD *)(v8 + 24);
  v10 = *(_QWORD *)(v8 + 32);
  if ( (unsigned __int8)(*(_BYTE *)(v9 + 8) - 1) <= 5u )
  {
    v11 = sub_1643030(*(_QWORD *)(v8 + 24));
    v12 = sub_16498A0(a1);
    v13 = sub_1644900(v12, v11);
    v14 = sub_16463B0(v13, (unsigned int)v10);
    v3 = sub_15A4510(a1, v14, 0);
  }
  v15 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(a2 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v57 = *(_QWORD *)(a2 + 32);
        a2 = *(_QWORD *)(a2 + 24);
        v15 *= (_DWORD)v57;
        continue;
      case 1:
        v26 = 16;
        break;
      case 2:
        v26 = 32;
        break;
      case 3:
      case 9:
        v26 = 64;
        break;
      case 4:
        v26 = 80;
        break;
      case 5:
      case 6:
        v26 = 128;
        break;
      case 7:
        a1 = (__int64)a3;
        a2 = 0;
        v26 = 8 * sub_15A9520(a3, 0);
        break;
      case 0xB:
        v26 = *(_DWORD *)(a2 + 8) >> 8;
        break;
      case 0xD:
        a1 = (__int64)a3;
        v26 = 8 * *(_QWORD *)sub_15A9930(a3, a2);
        break;
      case 0xE:
        v140 = *(_QWORD *)(a2 + 32);
        a2 = *(_QWORD *)(a2 + 24);
        a1 = (__int64)a3;
        v132 = (unsigned int)sub_15A9FE0(a3, a2);
        v56 = sub_127FA20((__int64)a3, a2);
        v8 = v132 * v140;
        v26 = 8 * v132 * v140 * ((v132 + ((unsigned __int64)(v56 + 7) >> 3) - 1) / v132);
        break;
      case 0xF:
        a1 = (__int64)a3;
        a2 = *(_DWORD *)(a2 + 8) >> 8;
        v26 = 8 * sub_15A9520(a3, a2);
        break;
    }
    break;
  }
  v150 = v15 * v26;
  if ( (unsigned int)(v15 * v26) > 0x40 )
  {
    a1 = (__int64)&v149;
    a2 = 0;
    sub_16A4EF0(&v149, 0, 0);
  }
  else
  {
    v149 = 0;
  }
  v27 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v9 + 8) )
    {
      case 1:
        v46 = 16;
        goto LABEL_53;
      case 2:
        v46 = 32;
        goto LABEL_53;
      case 3:
      case 9:
        v46 = 64;
        goto LABEL_53;
      case 4:
        v46 = 80;
        goto LABEL_53;
      case 5:
      case 6:
        v46 = 128;
        goto LABEL_53;
      case 7:
        a1 = (__int64)a3;
        a2 = 0;
        v46 = 8 * sub_15A9520(a3, 0);
        goto LABEL_53;
      case 0xB:
        v46 = *(_DWORD *)(v9 + 8) >> 8;
        goto LABEL_53;
      case 0xD:
        a1 = (__int64)a3;
        a2 = v9;
        v46 = 8 * *(_QWORD *)sub_15A9930(a3, v9);
        goto LABEL_53;
      case 0xE:
        a1 = (__int64)a3;
        v133 = *(_QWORD *)(v9 + 24);
        v141 = *(_QWORD *)(v9 + 32);
        v59 = sub_15A9FE0(a3, v133);
        a2 = v133;
        v60 = 1;
        v61 = v59;
        while ( 2 )
        {
          switch ( *(_BYTE *)(a2 + 8) )
          {
            case 1:
              v74 = 16;
              goto LABEL_107;
            case 2:
              v74 = 32;
              goto LABEL_107;
            case 3:
            case 9:
              v74 = 64;
              goto LABEL_107;
            case 4:
              v74 = 80;
              goto LABEL_107;
            case 5:
            case 6:
              v74 = 128;
              goto LABEL_107;
            case 7:
              a1 = (__int64)a3;
              a2 = 0;
              v134 = v60;
              v75 = sub_15A9520(a3, 0);
              v60 = v134;
              v74 = (unsigned int)(8 * v75);
              goto LABEL_107;
            case 0xB:
              v74 = *(_DWORD *)(a2 + 8) >> 8;
              goto LABEL_107;
            case 0xD:
              a1 = (__int64)a3;
              v135 = v60;
              v76 = (_QWORD *)sub_15A9930(a3, a2);
              v60 = v135;
              v74 = 8LL * *v76;
              goto LABEL_107;
            case 0xE:
              v120 = v60;
              v137 = *(_QWORD *)(a2 + 32);
              a2 = *(_QWORD *)(a2 + 24);
              a1 = (__int64)a3;
              v128 = (unsigned int)sub_15A9FE0(a3, a2);
              v78 = sub_127FA20((__int64)a3, a2);
              v60 = v120;
              v74 = 8 * v137 * v128 * ((v128 + ((unsigned __int64)(v78 + 7) >> 3) - 1) / v128);
              goto LABEL_107;
            case 0xF:
              a1 = (__int64)a3;
              v136 = v60;
              a2 = *(_DWORD *)(a2 + 8) >> 8;
              v77 = sub_15A9520(a3, a2);
              v60 = v136;
              v74 = (unsigned int)(8 * v77);
LABEL_107:
              v8 = v61 * v141;
              v46 = 8 * v61 * v141 * ((v61 + ((unsigned __int64)(v74 * v60 + 7) >> 3) - 1) / v61);
              goto LABEL_53;
            case 0x10:
              v79 = *(_QWORD *)(a2 + 32);
              a2 = *(_QWORD *)(a2 + 24);
              v60 *= v79;
              continue;
            default:
              goto LABEL_173;
          }
        }
      case 0xF:
        a1 = (__int64)a3;
        a2 = *(_DWORD *)(v9 + 8) >> 8;
        v46 = 8 * sub_15A9520(a3, a2);
LABEL_53:
        v47 = v10 - 1;
        v48 = (unsigned int)(v46 * v27);
        v49 = v10 - 1;
        v131 = v48;
        if ( !(_DWORD)v10 )
          goto LABEL_69;
        v50 = a3;
        break;
      case 0x10:
        v58 = *(_QWORD *)(v9 + 32);
        v9 = *(_QWORD *)(v9 + 24);
        v27 *= (_DWORD)v58;
        continue;
      default:
LABEL_173:
        JUMPOUT(0x419798);
    }
    break;
  }
  while ( 1 )
  {
    a2 = v47 - v49;
    if ( *v50 )
      break;
    a2 = v49;
    v51 = sub_15A0A60(v3, v49);
    if ( !v51 )
      goto LABEL_119;
LABEL_57:
    v8 = *(unsigned __int8 *)(v51 + 16);
    if ( (_BYTE)v8 == 9 )
    {
      v48 = v150;
      if ( v150 > 0x40 )
      {
        a2 = v131;
        a1 = (__int64)&v149;
        sub_16A7DC0(&v149, v131);
      }
      else
      {
        a1 = v131;
        v80 = 0;
        if ( v131 != v150 )
        {
          v48 = v131;
          v8 = v149 << v131;
          v80 = (v149 << v131) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v150);
        }
        v149 = v80;
      }
    }
    else
    {
      if ( (_BYTE)v8 != 13 )
        goto LABEL_119;
      v52 = v150;
      if ( v150 > 0x40 )
      {
        v146 = v51;
        sub_16A7DC0(&v149, v131);
        v52 = v150;
        v51 = v146;
      }
      else
      {
        v53 = 0;
        if ( v131 != v150 )
          v53 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v150) & (v149 << v131);
        v149 = v53;
      }
      a1 = (__int64)&v151;
      a2 = v51 + 24;
      sub_16A5DD0(&v151, v51 + 24, v52);
      if ( v150 > 0x40 )
      {
        a2 = (__int64)&v151;
        a1 = (__int64)&v149;
        sub_16A89F0(&v149, &v151);
      }
      else
      {
        v149 |= v151;
      }
      if ( (unsigned int)v152 > 0x40 )
      {
        a1 = v151;
        if ( v151 )
          j_j___libc_free_0_0(v151);
      }
    }
    if ( v49-- == 0 )
      goto LABEL_69;
  }
  v51 = sub_15A0A60(v3, a2);
  if ( v51 )
    goto LABEL_57;
LABEL_119:
  a1 = v3;
  a2 = v4;
  v3 = sub_15A4510(v3, v4, 0);
  if ( v3 )
    goto LABEL_95;
LABEL_69:
  v55 = *(_BYTE *)(v4 + 8);
  if ( v55 != 11 )
  {
    switch ( v55 )
    {
      case 1:
        v63 = sub_1698260(a1, a2, v8, v48);
        v62 = sub_16982C0(a1, a2, v72, v73);
        goto LABEL_99;
      case 2:
        v63 = sub_1698270(a1, a2);
        v62 = sub_16982C0(a1, a2, v70, v71);
        goto LABEL_99;
      case 3:
        v63 = sub_1698280(a1);
        v62 = sub_16982C0(a1, a2, v68, v69);
        goto LABEL_99;
      case 4:
        v63 = sub_16982A0();
        v62 = sub_16982C0(a1, a2, v66, v67);
        goto LABEL_99;
      case 5:
        v63 = sub_1698290();
        v62 = sub_16982C0(a1, a2, v64, v65);
LABEL_99:
        if ( v62 == v63 )
          goto LABEL_92;
        sub_169D050(&v152, v63, &v149);
        goto LABEL_93;
      case 6:
        v62 = sub_16982C0(a1, a2, v8, v48);
        v63 = v62;
LABEL_92:
        sub_169D060(&v152, v63, &v149);
LABEL_93:
        v3 = sub_159CCF0(*(_QWORD *)v4, &v151);
        if ( v152 == v62 )
        {
          v82 = v153[0];
          if ( v153[0] )
          {
            v83 = 32LL * *(_QWORD *)(v153[0] - 8LL);
            v84 = v153[0] + v83;
            if ( v153[0] != v153[0] + v83 )
            {
              v142 = v153[0];
              v138 = v3;
              do
              {
                v84 -= 32;
                if ( *(_QWORD *)(v84 + 8) == v62 )
                {
                  v85 = *(_QWORD *)(v84 + 16);
                  if ( v85 )
                  {
                    v86 = 32LL * *(_QWORD *)(v85 - 8);
                    v87 = v85 + v86;
                    while ( v85 != v87 )
                    {
                      v87 -= 32;
                      if ( *(_QWORD *)(v87 + 8) == v62 )
                      {
                        v88 = *(_QWORD *)(v87 + 16);
                        if ( v88 )
                        {
                          v89 = 32LL * *(_QWORD *)(v88 - 8);
                          v90 = v88 + v89;
                          if ( v88 != v88 + v89 )
                          {
                            do
                            {
                              v147 = v90 - 32;
                              sub_127D120((_QWORD *)(v90 - 24));
                              v90 = v147;
                            }
                            while ( v88 != v147 );
                          }
                          j_j_j___libc_free_0_0(v88 - 8);
                        }
                      }
                      else
                      {
                        sub_1698460(v87 + 8);
                      }
                    }
                    j_j_j___libc_free_0_0(v85 - 8);
                  }
                }
                else
                {
                  sub_1698460(v84 + 8);
                }
              }
              while ( v142 != v84 );
              v82 = v142;
              v3 = v138;
            }
            j_j_j___libc_free_0_0(v82 - 8);
          }
        }
        else
        {
          sub_1698460(&v152);
        }
        goto LABEL_95;
      default:
        goto LABEL_173;
    }
  }
  v3 = sub_15A1070(v4, &v149);
LABEL_95:
  if ( v150 > 0x40 )
  {
    if ( v149 )
      j_j___libc_free_0_0(v149);
  }
  return v3;
}
