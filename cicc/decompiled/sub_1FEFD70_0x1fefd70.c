// Function: sub_1FEFD70
// Address: 0x1fefd70
//
__int64 *__fastcall sub_1FEFD70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, double a6, __m128i a7)
{
  __int64 v8; // r12
  __int64 v9; // rax
  char v10; // dl
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned int v13; // eax
  char v14; // r12
  const void **v15; // rdx
  unsigned __int64 v16; // r15
  unsigned int v17; // eax
  unsigned int v18; // r12d
  int v19; // ebx
  unsigned int v20; // edx
  __int128 v21; // rax
  unsigned int v22; // edx
  __int64 *v23; // r14
  unsigned int v24; // ecx
  unsigned __int64 v25; // r15
  unsigned __int64 v26; // rax
  __int128 v27; // rax
  unsigned int v28; // edx
  unsigned int v29; // edx
  __int64 *v30; // rdi
  __int128 v31; // rax
  unsigned int v32; // r13d
  __int64 v33; // r15
  unsigned int v34; // edi
  unsigned __int64 v35; // r15
  unsigned int v36; // r13d
  __int64 v37; // r15
  unsigned int v38; // edi
  unsigned __int64 v39; // r15
  unsigned int v40; // r13d
  __int64 v41; // r15
  unsigned int v42; // edi
  unsigned __int64 v43; // r15
  unsigned int v44; // r13d
  __int64 v45; // r15
  unsigned int v46; // edi
  unsigned __int64 v47; // rdx
  unsigned int v48; // eax
  __int64 v49; // r13
  unsigned int v50; // edi
  unsigned __int64 v51; // rdx
  __int64 v52; // r13
  unsigned int v53; // esi
  unsigned __int64 v54; // r13
  unsigned int v55; // edx
  __int64 *v56; // r12
  const void **v58; // rdx
  unsigned int v59; // edx
  __int128 v60; // rax
  __int64 *v61; // r14
  __int64 *v62; // r12
  unsigned int v63; // edx
  unsigned __int64 v64; // r15
  __int128 v65; // rax
  __int64 *v66; // r12
  unsigned int v67; // edx
  __int128 v68; // rax
  __int64 *v69; // r14
  unsigned int v70; // edx
  __int64 *v71; // r12
  unsigned __int64 v72; // r15
  __int128 v73; // rax
  unsigned int v74; // edx
  __int64 *v75; // r12
  unsigned int v76; // edx
  __int128 v77; // rax
  __int64 *v78; // r14
  __int64 *v79; // r12
  unsigned int v80; // edx
  unsigned __int64 v81; // r15
  __int128 v82; // rax
  __int64 *v83; // r12
  unsigned int v84; // edx
  __int128 v85; // rax
  __int64 *v86; // r14
  __int64 *v87; // r12
  unsigned int v88; // edx
  unsigned __int64 v89; // r15
  __int128 v90; // rax
  unsigned int v91; // edx
  __int64 *v92; // r12
  unsigned int v93; // edx
  __int128 v94; // rax
  __int64 *v95; // r14
  __int64 *v96; // r12
  unsigned int v97; // edx
  unsigned __int64 v98; // r15
  __int128 v99; // rax
  __int64 *v100; // r12
  unsigned int v101; // edx
  __int128 v102; // rax
  __int64 *v103; // r14
  __int64 *v104; // r12
  unsigned int v105; // edx
  unsigned __int64 v106; // r15
  __int128 v107; // rax
  unsigned int v108; // edx
  __int128 v109; // [rsp-10h] [rbp-270h]
  __int128 v110; // [rsp-10h] [rbp-270h]
  __int128 v111; // [rsp-10h] [rbp-270h]
  __int128 v112; // [rsp-10h] [rbp-270h]
  __int64 v113; // [rsp+8h] [rbp-258h]
  unsigned __int64 v114; // [rsp+10h] [rbp-250h]
  __int64 v115; // [rsp+18h] [rbp-248h]
  unsigned int v116; // [rsp+20h] [rbp-240h]
  __int128 v117; // [rsp+30h] [rbp-230h]
  const void **v118; // [rsp+40h] [rbp-220h]
  unsigned __int64 v119; // [rsp+40h] [rbp-220h]
  __int64 *v120; // [rsp+50h] [rbp-210h]
  unsigned int v121; // [rsp+58h] [rbp-208h]
  __int64 v122; // [rsp+60h] [rbp-200h]
  unsigned int v123; // [rsp+68h] [rbp-1F8h]
  unsigned int v124; // [rsp+70h] [rbp-1F0h]
  __int64 *v125; // [rsp+70h] [rbp-1F0h]
  __int64 *v126; // [rsp+70h] [rbp-1F0h]
  __int64 *v127; // [rsp+70h] [rbp-1F0h]
  __int64 *v128; // [rsp+70h] [rbp-1F0h]
  unsigned __int64 v129; // [rsp+78h] [rbp-1E8h]
  unsigned __int64 v130; // [rsp+78h] [rbp-1E8h]
  unsigned __int64 v131; // [rsp+78h] [rbp-1E8h]
  unsigned __int64 v132; // [rsp+78h] [rbp-1E8h]
  unsigned __int64 v133; // [rsp+78h] [rbp-1E8h]
  unsigned int v134; // [rsp+80h] [rbp-1E0h]
  __int64 *v135; // [rsp+80h] [rbp-1E0h]
  unsigned int v137; // [rsp+90h] [rbp-1D0h]
  unsigned __int64 v138; // [rsp+90h] [rbp-1D0h]
  unsigned int v139; // [rsp+90h] [rbp-1D0h]
  __int64 *v140; // [rsp+90h] [rbp-1D0h]
  __int64 *v141; // [rsp+90h] [rbp-1D0h]
  unsigned __int64 v142; // [rsp+98h] [rbp-1C8h]
  unsigned __int64 v143; // [rsp+98h] [rbp-1C8h]
  unsigned __int64 v144; // [rsp+98h] [rbp-1C8h]
  unsigned __int64 v145; // [rsp+98h] [rbp-1C8h]
  __int64 *v146; // [rsp+A0h] [rbp-1C0h]
  __int64 *v147; // [rsp+120h] [rbp-140h]
  __int64 *v148; // [rsp+170h] [rbp-F0h]
  unsigned int v149; // [rsp+1B0h] [rbp-B0h] BYREF
  const void **v150; // [rsp+1B8h] [rbp-A8h]
  unsigned __int64 v151; // [rsp+1C0h] [rbp-A0h] BYREF
  unsigned int v152; // [rsp+1C8h] [rbp-98h]
  unsigned __int64 v153; // [rsp+1D0h] [rbp-90h] BYREF
  unsigned int v154; // [rsp+1D8h] [rbp-88h]
  unsigned __int64 v155; // [rsp+1E0h] [rbp-80h] BYREF
  unsigned int v156; // [rsp+1E8h] [rbp-78h]
  unsigned __int64 v157; // [rsp+1F0h] [rbp-70h] BYREF
  unsigned int v158; // [rsp+1F8h] [rbp-68h]
  unsigned __int64 v159; // [rsp+200h] [rbp-60h] BYREF
  unsigned int v160; // [rsp+208h] [rbp-58h]
  unsigned __int64 v161; // [rsp+210h] [rbp-50h] BYREF
  unsigned int v162; // [rsp+218h] [rbp-48h]
  unsigned __int64 v163; // [rsp+220h] [rbp-40h] BYREF
  const void **v164; // [rsp+228h] [rbp-38h]

  *(_QWORD *)&v117 = a2;
  v8 = *(_QWORD *)(a1 + 8);
  *((_QWORD *)&v117 + 1) = a3;
  v115 = a2;
  v114 = (unsigned int)a3;
  v9 = *(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3;
  v10 = *(_BYTE *)v9;
  v150 = *(const void ***)(v9 + 8);
  v11 = *(_QWORD *)(a1 + 16);
  LOBYTE(v149) = v10;
  v12 = sub_1E0A0C0(*(_QWORD *)(v11 + 32));
  v13 = sub_1F40B60(v8, v149, (__int64)v150, v12, 1);
  v14 = v149;
  v118 = v15;
  v116 = v13;
  if ( (_BYTE)v149 )
  {
    if ( (unsigned __int8)(v149 - 14) <= 0x5Fu )
    {
      switch ( (char)v149 )
      {
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 62:
        case 63:
        case 64:
        case 65:
        case 66:
        case 67:
          v14 = 3;
          break;
        case 33:
        case 34:
        case 35:
        case 36:
        case 37:
        case 38:
        case 39:
        case 40:
        case 68:
        case 69:
        case 70:
        case 71:
        case 72:
        case 73:
          v14 = 4;
          break;
        case 41:
        case 42:
        case 43:
        case 44:
        case 45:
        case 46:
        case 47:
        case 48:
        case 74:
        case 75:
        case 76:
        case 77:
        case 78:
        case 79:
          v14 = 5;
          break;
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 80:
        case 81:
        case 82:
        case 83:
        case 84:
        case 85:
          v14 = 6;
          break;
        case 55:
          v14 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v14 = 8;
          break;
        case 89:
        case 90:
        case 91:
        case 92:
        case 93:
        case 101:
        case 102:
        case 103:
        case 104:
        case 105:
          v14 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v14 = 10;
          break;
        default:
          v14 = 2;
          break;
      }
      goto LABEL_82;
    }
  }
  else if ( sub_1F58D20((__int64)&v149) )
  {
    v14 = sub_1F596B0((__int64)&v149);
    v164 = v58;
    LOBYTE(v163) = v14;
    if ( !v14 )
      goto LABEL_4;
LABEL_82:
    v134 = sub_1FEB8F0(v14);
    goto LABEL_5;
  }
  LOBYTE(v163) = v14;
  v164 = v150;
  if ( v14 )
    goto LABEL_82;
LABEL_4:
  v134 = sub_1F58D40((__int64)&v163);
LABEL_5:
  v16 = 0;
  v17 = v134;
  v18 = v134 - 1;
  if ( v134 <= 7 )
  {
    v122 = sub_1D38BB0(*(_QWORD *)(a1 + 16), 0, a4, v149, v150, 0, a5, a6, a7, 0);
    v123 = v55;
    v142 = v55;
    if ( !v134 )
      return (__int64 *)v122;
    goto LABEL_8;
  }
  v19 = v18 & v134;
  if ( (v18 & v134) != 0 )
  {
    v122 = sub_1D38BB0(*(_QWORD *)(a1 + 16), 0, a4, v149, v150, 0, a5, a6, a7, 0);
    v123 = v20;
    v142 = v20;
LABEL_8:
    v121 = v134 - 1;
    v124 = 0;
    while ( 1 )
    {
      v30 = *(__int64 **)(a1 + 16);
      if ( v124 < v18 )
      {
        *(_QWORD *)&v21 = sub_1D38BB0((__int64)v30, v121, a4, v116, v118, 0, a5, a6, a7, 0);
        v23 = sub_1D332F0(
                v30,
                122,
                a4,
                v149,
                v150,
                0,
                *(double *)a5.m128i_i64,
                a6,
                a7,
                a2,
                *((unsigned __int64 *)&v117 + 1),
                v21);
      }
      else
      {
        *(_QWORD *)&v31 = sub_1D38BB0((__int64)v30, -v121, a4, v116, v118, 0, a5, a6, a7, 0);
        v23 = sub_1D332F0(
                v30,
                124,
                a4,
                v149,
                v150,
                0,
                *(double *)a5.m128i_i64,
                a6,
                a7,
                a2,
                *((unsigned __int64 *)&v117 + 1),
                v31);
      }
      v24 = v134;
      LODWORD(v164) = v134;
      v25 = v22 | v16 & 0xFFFFFFFF00000000LL;
      if ( v134 <= 0x40 )
        break;
      sub_16A4EF0((__int64)&v163, 1, 0);
      v24 = (unsigned int)v164;
      if ( (unsigned int)v164 <= 0x40 )
        goto LABEL_12;
      sub_16A7DC0((__int64 *)&v163, v18);
LABEL_15:
      v120 = *(__int64 **)(a1 + 16);
      *(_QWORD *)&v27 = sub_1D38970((__int64)v120, (__int64)&v163, a4, v149, v150, 0, a5, a6, a7, 0);
      v146 = sub_1D332F0(v120, 118, a4, v149, v150, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v23, v25, v27);
      v16 = v28 | v25 & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v109 + 1) = v16;
      *(_QWORD *)&v109 = v146;
      v143 = v123 | v142 & 0xFFFFFFFF00000000LL;
      v122 = (__int64)sub_1D332F0(
                        *(__int64 **)(a1 + 16),
                        119,
                        a4,
                        v149,
                        v150,
                        0,
                        *(double *)a5.m128i_i64,
                        a6,
                        a7,
                        v122,
                        v143,
                        v109);
      v123 = v29;
      v142 = v29 | v143 & 0xFFFFFFFF00000000LL;
      if ( (unsigned int)v164 > 0x40 && v163 )
        j_j___libc_free_0_0(v163);
      ++v124;
      --v18;
      v121 -= 2;
      if ( v124 >= v134 )
        return (__int64 *)v122;
    }
    v163 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v134) & 1;
LABEL_12:
    v26 = 0;
    if ( v18 != v24 )
      v26 = (v163 << v18) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v24);
    v163 = v26;
    goto LABEL_15;
  }
  v152 = v134;
  if ( v134 <= 0x40 )
  {
    v156 = v134;
    v151 = 0;
    v154 = v134;
    v153 = 0;
    v155 = 0;
    v158 = v134;
    v157 = 0;
    v160 = v134;
    v159 = 0;
    v162 = v134;
    v161 = 0;
  }
  else
  {
    sub_16A4EF0((__int64)&v151, 0, 0);
    v154 = v134;
    sub_16A4EF0((__int64)&v153, 0, 0);
    v156 = v134;
    sub_16A4EF0((__int64)&v155, 0, 0);
    v158 = v134;
    sub_16A4EF0((__int64)&v157, 0, 0);
    v160 = v134;
    sub_16A4EF0((__int64)&v159, 0, 0);
    v162 = v134;
    sub_16A4EF0((__int64)&v161, 0, 0);
    v17 = v152;
  }
  v113 = a4;
  while ( 1 )
  {
    LODWORD(v164) = v17;
    v52 = 240LL << v19;
    if ( v17 > 0x40 )
    {
      sub_16A4FD0((__int64)&v163, (const void **)&v151);
      v17 = (unsigned int)v164;
      if ( (unsigned int)v164 > 0x40 )
      {
        *(_QWORD *)v163 |= v52;
        v17 = (unsigned int)v164;
        v54 = v163;
        v53 = v152;
        goto LABEL_73;
      }
      v53 = v152;
    }
    else
    {
      v53 = v17;
      v163 = v151;
    }
    v163 = (v52 | v163) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v17);
    v54 = v163;
LABEL_73:
    LODWORD(v164) = 0;
    if ( v53 > 0x40 && v151 )
    {
      v137 = v17;
      j_j___libc_free_0_0(v151);
      v151 = v54;
      v152 = v137;
      if ( (unsigned int)v164 > 0x40 && v163 )
        j_j___libc_free_0_0(v163);
    }
    else
    {
      v151 = v54;
      v152 = v17;
    }
    v32 = v158;
    LODWORD(v164) = v158;
    v33 = 15LL << v19;
    if ( v158 > 0x40 )
    {
      sub_16A4FD0((__int64)&v163, (const void **)&v157);
      v32 = (unsigned int)v164;
      if ( (unsigned int)v164 > 0x40 )
      {
        *(_QWORD *)v163 |= v33;
        v32 = (unsigned int)v164;
        v35 = v163;
        v34 = v158;
        goto LABEL_31;
      }
      v34 = v158;
    }
    else
    {
      v34 = v158;
      v163 = v157;
    }
    v163 = (v33 | v163) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v32);
    v35 = v163;
LABEL_31:
    LODWORD(v164) = 0;
    if ( v34 > 0x40 && v157 )
    {
      j_j___libc_free_0_0(v157);
      v157 = v35;
      v158 = v32;
      if ( (unsigned int)v164 > 0x40 && v163 )
        j_j___libc_free_0_0(v163);
    }
    else
    {
      v157 = v35;
      v158 = v32;
    }
    v36 = v154;
    LODWORD(v164) = v154;
    v37 = 204LL << v19;
    if ( v154 > 0x40 )
    {
      sub_16A4FD0((__int64)&v163, (const void **)&v153);
      v36 = (unsigned int)v164;
      if ( (unsigned int)v164 > 0x40 )
      {
        *(_QWORD *)v163 |= v37;
        v36 = (unsigned int)v164;
        v39 = v163;
        v38 = v154;
        goto LABEL_39;
      }
      v38 = v154;
    }
    else
    {
      v38 = v154;
      v163 = v153;
    }
    v163 = (v37 | v163) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v36);
    v39 = v163;
LABEL_39:
    LODWORD(v164) = 0;
    if ( v38 > 0x40 && v153 )
    {
      j_j___libc_free_0_0(v153);
      v153 = v39;
      v154 = v36;
      if ( (unsigned int)v164 > 0x40 && v163 )
        j_j___libc_free_0_0(v163);
    }
    else
    {
      v153 = v39;
      v154 = v36;
    }
    v40 = v160;
    LODWORD(v164) = v160;
    v41 = 51LL << v19;
    if ( v160 > 0x40 )
    {
      sub_16A4FD0((__int64)&v163, (const void **)&v159);
      v40 = (unsigned int)v164;
      if ( (unsigned int)v164 > 0x40 )
      {
        *(_QWORD *)v163 |= v41;
        v40 = (unsigned int)v164;
        v43 = v163;
        v42 = v160;
        goto LABEL_47;
      }
      v42 = v160;
    }
    else
    {
      v42 = v160;
      v163 = v159;
    }
    v163 = (v41 | v163) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v40);
    v43 = v163;
LABEL_47:
    LODWORD(v164) = 0;
    if ( v42 > 0x40 && v159 )
    {
      j_j___libc_free_0_0(v159);
      v159 = v43;
      v160 = v40;
      if ( (unsigned int)v164 > 0x40 && v163 )
        j_j___libc_free_0_0(v163);
    }
    else
    {
      v159 = v43;
      v160 = v40;
    }
    v44 = v156;
    LODWORD(v164) = v156;
    v45 = 170LL << v19;
    if ( v156 > 0x40 )
    {
      sub_16A4FD0((__int64)&v163, (const void **)&v155);
      v44 = (unsigned int)v164;
      if ( (unsigned int)v164 > 0x40 )
      {
        *(_QWORD *)v163 |= v45;
        v44 = (unsigned int)v164;
        v47 = v163;
        v46 = v156;
        goto LABEL_55;
      }
      v46 = v156;
    }
    else
    {
      v46 = v156;
      v163 = v155;
    }
    v47 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v44) & (v45 | v163);
    v163 = v47;
LABEL_55:
    LODWORD(v164) = 0;
    if ( v46 > 0x40 && v155 )
    {
      v138 = v47;
      j_j___libc_free_0_0(v155);
      v156 = v44;
      v155 = v138;
      if ( (unsigned int)v164 > 0x40 && v163 )
        j_j___libc_free_0_0(v163);
    }
    else
    {
      v155 = v47;
      v156 = v44;
    }
    v48 = v162;
    LODWORD(v164) = v162;
    v49 = 85LL << v19;
    if ( v162 > 0x40 )
    {
      sub_16A4FD0((__int64)&v163, (const void **)&v161);
      v48 = (unsigned int)v164;
      if ( (unsigned int)v164 > 0x40 )
      {
        *(_QWORD *)v163 |= v49;
        v48 = (unsigned int)v164;
        v51 = v163;
        v50 = v162;
        goto LABEL_63;
      }
      v50 = v162;
    }
    else
    {
      v50 = v162;
      v163 = v161;
    }
    v51 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v48) & (v49 | v163);
    v163 = v51;
LABEL_63:
    LODWORD(v164) = 0;
    if ( v50 > 0x40 )
    {
      if ( v161 )
        break;
    }
    v161 = v51;
    v19 += 8;
    v162 = v48;
    if ( v19 == v134 )
      goto LABEL_90;
LABEL_69:
    v17 = v152;
  }
  v119 = v51;
  v139 = v48;
  j_j___libc_free_0_0(v161);
  v161 = v119;
  v162 = v139;
  if ( (unsigned int)v164 > 0x40 && v163 )
    j_j___libc_free_0_0(v163);
  v19 += 8;
  if ( v19 != v134 )
    goto LABEL_69;
LABEL_90:
  if ( v19 != 8 )
  {
    v115 = sub_1D309E0(
             *(__int64 **)(a1 + 16),
             127,
             v113,
             v149,
             v150,
             0,
             *(double *)a5.m128i_i64,
             a6,
             *(double *)a7.m128i_i64,
             v117);
    v114 = v59;
  }
  v135 = *(__int64 **)(a1 + 16);
  *(_QWORD *)&v60 = sub_1D38970((__int64)v135, (__int64)&v151, v113, v149, v150, 0, a5, a6, a7, 0);
  v61 = sub_1D332F0(v135, 118, v113, v149, v150, 0, *(double *)a5.m128i_i64, a6, a7, v115, v114, v60);
  v62 = *(__int64 **)(a1 + 16);
  v64 = v63;
  *(_QWORD *)&v65 = sub_1D38970((__int64)v62, (__int64)&v157, v113, v149, v150, 0, a5, a6, a7, 0);
  v125 = sub_1D332F0(v62, 118, v113, v149, v150, 0, *(double *)a5.m128i_i64, a6, a7, v115, v114, v65);
  v66 = *(__int64 **)(a1 + 16);
  v129 = v67;
  *(_QWORD *)&v68 = sub_1D38BB0((__int64)v66, 4, v113, v149, v150, 0, a5, a6, a7, 0);
  v69 = sub_1D332F0(v66, 124, v113, v149, v150, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v61, v64, v68);
  v71 = *(__int64 **)(a1 + 16);
  v72 = v70 | v64 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v73 = sub_1D38BB0((__int64)v71, 4, v113, v149, v150, 0, a5, a6, a7, 0);
  v148 = sub_1D332F0(v71, 122, v113, v149, v150, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v125, v129, v73);
  v130 = v74 | v129 & 0xFFFFFFFF00000000LL;
  *((_QWORD *)&v110 + 1) = v130;
  *(_QWORD *)&v110 = v148;
  v140 = sub_1D332F0(
           *(__int64 **)(a1 + 16),
           119,
           v113,
           v149,
           v150,
           0,
           *(double *)a5.m128i_i64,
           a6,
           a7,
           (__int64)v69,
           v72,
           v110);
  v75 = *(__int64 **)(a1 + 16);
  v144 = v76 | v114 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v77 = sub_1D38970((__int64)v75, (__int64)&v153, v113, v149, v150, 0, a5, a6, a7, 0);
  v78 = sub_1D332F0(v75, 118, v113, v149, v150, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v140, v144, v77);
  v79 = *(__int64 **)(a1 + 16);
  v81 = v80 | v72 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v82 = sub_1D38970((__int64)v79, (__int64)&v159, v113, v149, v150, 0, a5, a6, a7, 0);
  v126 = sub_1D332F0(v79, 118, v113, v149, v150, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v140, v144, v82);
  v83 = *(__int64 **)(a1 + 16);
  v131 = v84 | v130 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v85 = sub_1D38BB0((__int64)v83, 2, v113, v149, v150, 0, a5, a6, a7, 0);
  v86 = sub_1D332F0(v83, 124, v113, v149, v150, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v78, v81, v85);
  v87 = *(__int64 **)(a1 + 16);
  v89 = v88 | v81 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v90 = sub_1D38BB0((__int64)v87, 2, v113, v149, v150, 0, a5, a6, a7, 0);
  v147 = sub_1D332F0(v87, 122, v113, v149, v150, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v126, v131, v90);
  v132 = v91 | v131 & 0xFFFFFFFF00000000LL;
  *((_QWORD *)&v111 + 1) = v132;
  *(_QWORD *)&v111 = v147;
  v141 = sub_1D332F0(
           *(__int64 **)(a1 + 16),
           119,
           v113,
           v149,
           v150,
           0,
           *(double *)a5.m128i_i64,
           a6,
           a7,
           (__int64)v86,
           v89,
           v111);
  v92 = *(__int64 **)(a1 + 16);
  v145 = v93 | v144 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v94 = sub_1D38970((__int64)v92, (__int64)&v155, v113, v149, v150, 0, a5, a6, a7, 0);
  v95 = sub_1D332F0(v92, 118, v113, v149, v150, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v141, v145, v94);
  v96 = *(__int64 **)(a1 + 16);
  v98 = v97 | v89 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v99 = sub_1D38970((__int64)v96, (__int64)&v161, v113, v149, v150, 0, a5, a6, a7, 0);
  v127 = sub_1D332F0(v96, 118, v113, v149, v150, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v141, v145, v99);
  v100 = *(__int64 **)(a1 + 16);
  v133 = v101 | v132 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v102 = sub_1D38BB0((__int64)v100, 1, v113, v149, v150, 0, a5, a6, a7, 0);
  v103 = sub_1D332F0(v100, 124, v113, v149, v150, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v95, v98, v102);
  v104 = *(__int64 **)(a1 + 16);
  v106 = v105 | v98 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v107 = sub_1D38BB0((__int64)v104, 1, v113, v149, v150, 0, a5, a6, a7, 0);
  v128 = sub_1D332F0(v104, 122, v113, v149, v150, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v127, v133, v107);
  *((_QWORD *)&v112 + 1) = v108 | v133 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v112 = v128;
  v56 = sub_1D332F0(
          *(__int64 **)(a1 + 16),
          119,
          v113,
          v149,
          v150,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          (__int64)v103,
          v106,
          v112);
  if ( v162 > 0x40 && v161 )
    j_j___libc_free_0_0(v161);
  if ( v160 > 0x40 && v159 )
    j_j___libc_free_0_0(v159);
  if ( v158 > 0x40 && v157 )
    j_j___libc_free_0_0(v157);
  if ( v156 > 0x40 && v155 )
    j_j___libc_free_0_0(v155);
  if ( v154 > 0x40 && v153 )
    j_j___libc_free_0_0(v153);
  if ( v152 > 0x40 && v151 )
    j_j___libc_free_0_0(v151);
  return v56;
}
