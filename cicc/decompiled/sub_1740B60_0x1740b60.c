// Function: sub_1740B60
// Address: 0x1740b60
//
__int64 __fastcall sub_1740B60(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // r13
  unsigned int v13; // eax
  unsigned int v14; // r14d
  unsigned __int64 v15; // r15
  __int64 v16; // r12
  __int64 v17; // rbx
  char v18; // bl
  _QWORD *v19; // rax
  __int64 v20; // rax
  unsigned __int64 v22; // rbx
  unsigned __int64 v23; // rsi
  unsigned int v24; // eax
  __int64 v25; // r14
  __int64 v26; // rbx
  __int64 v27; // r12
  __int64 v28; // rax
  _QWORD *v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rdi
  unsigned __int64 v32; // rax
  __int64 v33; // r13
  unsigned __int64 v34; // rax
  __int64 v35; // r12
  __int64 v36; // rbx
  unsigned __int64 v37; // r14
  __int64 v38; // rax
  int v39; // r8d
  int v40; // r9d
  int v41; // edi
  unsigned int v42; // edx
  int v43; // ecx
  unsigned __int64 v44; // rax
  __int64 v45; // rsi
  unsigned __int64 v46; // rax
  __int64 v47; // rcx
  __int64 v48; // r8
  int v49; // r9d
  __int64 v50; // rsi
  __int64 v51; // rax
  __int64 *v52; // r13
  char v53; // dl
  unsigned __int8 *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // r15
  unsigned int v58; // ebx
  __int64 v59; // rcx
  __int64 v60; // rdi
  __int64 v61; // rsi
  char v62; // dl
  char v63; // dl
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // r14
  __int64 v67; // r8
  __int64 v68; // rax
  __int64 v69; // rax
  int v70; // eax
  __int64 *v71; // rbx
  int v72; // r13d
  __int64 v73; // rax
  __int64 v74; // r8
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 **v78; // r12
  int v79; // eax
  __int64 v80; // rdi
  __int64 v81; // r9
  __int64 *v82; // rax
  unsigned __int64 v83; // rbx
  _QWORD *v84; // r14
  __int64 *v85; // r12
  __int64 v86; // rax
  __int64 v87; // rbx
  bool v88; // zf
  __int64 v89; // rdi
  __int64 ***v90; // r12
  __int16 v91; // dx
  unsigned int v92; // eax
  __int16 v93; // cx
  __int64 v94; // rax
  __int64 v95; // rsi
  __int64 *v96; // r13
  __int64 v97; // rdx
  __int64 v98; // rax
  __int64 v99; // r13
  __int64 v100; // rbx
  _QWORD *v101; // rax
  __int64 v102; // r13
  __int64 v103; // rbx
  _QWORD *v104; // rax
  double v105; // xmm4_8
  double v106; // xmm5_8
  __int64 *v107; // r12
  __int64 *v108; // rbx
  __int64 v109; // rdi
  unsigned __int8 *v110; // r8
  __int64 v111; // rsi
  unsigned __int8 *v112; // rsi
  __int64 v113; // [rsp+0h] [rbp-260h]
  unsigned __int8 *v114; // [rsp+0h] [rbp-260h]
  unsigned __int64 v115; // [rsp+8h] [rbp-258h]
  _QWORD *v116; // [rsp+10h] [rbp-250h]
  _QWORD *v117; // [rsp+18h] [rbp-248h]
  int v118; // [rsp+24h] [rbp-23Ch]
  unsigned int v119; // [rsp+28h] [rbp-238h]
  __int64 v120; // [rsp+28h] [rbp-238h]
  unsigned __int8 *v121; // [rsp+28h] [rbp-238h]
  unsigned int v122; // [rsp+30h] [rbp-230h]
  unsigned __int64 v123; // [rsp+48h] [rbp-218h]
  __int64 v124; // [rsp+50h] [rbp-210h]
  __int64 v125; // [rsp+58h] [rbp-208h]
  __int64 **v126; // [rsp+60h] [rbp-200h]
  unsigned int v127; // [rsp+68h] [rbp-1F8h]
  __int64 v128; // [rsp+68h] [rbp-1F8h]
  unsigned __int64 v129; // [rsp+70h] [rbp-1F0h]
  char v130; // [rsp+70h] [rbp-1F0h]
  unsigned __int64 v131; // [rsp+70h] [rbp-1F0h]
  unsigned __int64 v132; // [rsp+70h] [rbp-1F0h]
  int v133; // [rsp+78h] [rbp-1E8h]
  __int64 **v134; // [rsp+78h] [rbp-1E8h]
  _QWORD *v135; // [rsp+78h] [rbp-1E8h]
  __int64 v136; // [rsp+78h] [rbp-1E8h]
  __int64 v137; // [rsp+80h] [rbp-1E0h]
  __int64 v138; // [rsp+80h] [rbp-1E0h]
  __int64 v139; // [rsp+80h] [rbp-1E0h]
  __int64 v141; // [rsp+98h] [rbp-1C8h] BYREF
  __int64 v142; // [rsp+A0h] [rbp-1C0h] BYREF
  __int64 v143; // [rsp+A8h] [rbp-1B8h] BYREF
  __int64 v144[2]; // [rsp+B0h] [rbp-1B0h] BYREF
  __int16 v145; // [rsp+C0h] [rbp-1A0h]
  __int64 *v146; // [rsp+D0h] [rbp-190h] BYREF
  __int64 v147; // [rsp+D8h] [rbp-188h]
  _BYTE v148[64]; // [rsp+E0h] [rbp-180h] BYREF
  _BYTE *v149; // [rsp+120h] [rbp-140h] BYREF
  __int64 v150; // [rsp+128h] [rbp-138h]
  _BYTE v151[64]; // [rsp+130h] [rbp-130h] BYREF
  __m128i v152; // [rsp+170h] [rbp-F0h] BYREF
  _QWORD *v153; // [rsp+188h] [rbp-D8h]
  __int64 *v154; // [rsp+1D0h] [rbp-90h] BYREF
  __int64 v155; // [rsp+1D8h] [rbp-88h]
  __int16 v156; // [rsp+1E0h] [rbp-80h] BYREF
  _QWORD *v157; // [rsp+1E8h] [rbp-78h]

  v10 = (__int64 *)((a2 & 0xFFFFFFFFFFFFFFF8LL) - 72);
  v141 = a2;
  if ( (a2 & 4) != 0 )
    v10 = (__int64 *)((a2 & 0xFFFFFFFFFFFFFFF8LL) - 24);
  v11 = sub_1649C60(*v10);
  if ( *(_BYTE *)(v11 + 16) )
    return 0;
  v12 = (_QWORD *)v11;
  LOBYTE(v13) = sub_15602E0((_QWORD *)(v11 + 112), "thunk", 5u);
  v14 = v13;
  if ( (_BYTE)v13 )
    return 0;
  v15 = v141 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v141 & 4) != 0 && (*(_WORD *)(v15 + 18) & 3) == 2 )
    return 0;
  v16 = v12[3];
  v142 = *(_QWORD *)(v15 + 56);
  v126 = *(__int64 ***)v15;
  v137 = **(_QWORD **)(v16 + 16);
  if ( *(_QWORD *)v15 != v137 )
  {
    if ( *(_BYTE *)(v137 + 8) == 13 )
      return 0;
    if ( sub_15FBDD0(v137, (__int64)v126, a1[333]) )
    {
      v17 = *(_QWORD *)(v15 + 8);
      if ( !v142 )
        goto LABEL_14;
      if ( !v17 )
        goto LABEL_23;
LABEL_12:
      sub_1562F70(&v152, v142, 0);
      sub_1560E30((__int64)&v154, v137);
      v18 = sub_1561CE0(&v152, &v154);
      sub_173D510(v157);
      if ( v18 )
      {
        sub_173D510(v153);
        return v14;
      }
      sub_173D510(v153);
      v17 = *(_QWORD *)(v15 + 8);
LABEL_14:
      if ( !v17 )
        goto LABEL_23;
      goto LABEL_15;
    }
    if ( sub_15E4F60((__int64)v12) )
      return 0;
    v17 = *(_QWORD *)(v15 + 8);
    if ( v17 )
    {
      if ( *(_BYTE *)(v137 + 8) )
        return 0;
      if ( !v142 )
      {
LABEL_15:
        if ( *(_BYTE *)(v15 + 16) == 29 )
        {
          while ( 1 )
          {
            v19 = sub_1648700(v17);
            if ( *((_BYTE *)v19 + 16) == 77 )
            {
              v20 = v19[5];
              if ( v20 == *(_QWORD *)(v15 - 48) || *(_QWORD *)(v15 - 24) == v20 )
                return 0;
            }
            v17 = *(_QWORD *)(v17 + 8);
            if ( !v17 )
              goto LABEL_23;
          }
        }
        goto LABEL_23;
      }
      goto LABEL_12;
    }
  }
LABEL_23:
  v129 = sub_1389B50(&v141);
  v22 = v141 & 0xFFFFFFFFFFFFFFF8LL;
  v127 = *(_DWORD *)((v141 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF;
  v133 = *(_DWORD *)(v16 + 12);
  v152.m128i_i64[0] = v12[14];
  if ( (unsigned __int8)sub_1560490(&v152, 11, 0) )
    return v14;
  v154 = (__int64 *)v12[14];
  if ( (unsigned __int8)sub_1560490(&v154, 6, 0) )
    return v14;
  v23 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v129 + 24LL * v127 - v22) >> 3);
  v119 = v23;
  v118 = -1431655765 * ((__int64)(v129 + 24LL * v127 - v22) >> 3);
  v24 = v133 - 1;
  if ( v133 - 1 > (unsigned int)v23 )
    v24 = -1431655765 * ((__int64)(v129 + 24LL * v127 - v22) >> 3);
  v122 = v24;
  v134 = (__int64 **)((v141 & 0xFFFFFFFFFFFFFFF8LL)
                    - 24LL * (*(_DWORD *)((v141 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
  if ( v24 )
  {
    v124 = v16;
    v25 = 0;
    v117 = v12;
    while ( 1 )
    {
      v128 = v25 + 1;
      v26 = *(_QWORD *)(*(_QWORD *)(v124 + 16) + 8 * (v25 + 1));
      v27 = **v134;
      if ( !sub_15FBDD0(v27, v26, a1[333]) )
        return 0;
      v28 = sub_1560230(&v142, v25);
      sub_1563030(&v152, v28);
      sub_1560E30((__int64)&v154, v26);
      v130 = sub_1561CE0(&v152, &v154);
      sub_173D510(v157);
      sub_173D510(v153);
      if ( v130 )
        return 0;
      v131 = v141 & 0xFFFFFFFFFFFFFFF8LL;
      v29 = (_QWORD *)((v141 & 0xFFFFFFFFFFFFFFF8LL) + 56);
      if ( (v141 & 4) == 0 )
        break;
      if ( (unsigned __int8)sub_1560290(v29, v25, 11) )
        return 0;
      v30 = *(_QWORD *)(v131 - 24);
      if ( !*(_BYTE *)(v30 + 16) )
        goto LABEL_34;
LABEL_35:
      if ( v26 != v27 && (unsigned __int8)sub_1560290(&v142, v25, 6) )
      {
        if ( *(_BYTE *)(v26 + 8) != 15 )
          return 0;
        v31 = *(_QWORD *)(v26 + 24);
        v32 = *(unsigned __int8 *)(v31 + 8);
        if ( (unsigned __int8)v32 > 0xFu || (v65 = 35454, !_bittest64(&v65, v32)) )
        {
          if ( (unsigned int)(v32 - 13) > 1 && (_DWORD)v32 != 16 || !sub_16435F0(v31, 0) )
            return 0;
        }
        v33 = 1;
        v34 = sub_12BE0A0(a1[333], **(_QWORD **)(v27 + 16));
        v35 = a1[333];
        v36 = *(_QWORD *)(v26 + 24);
        v132 = v34;
        v37 = (unsigned int)sub_15A9FE0(v35, v36);
        while ( 2 )
        {
          switch ( *(_BYTE *)(v36 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v64 = *(_QWORD *)(v36 + 32);
              v36 = *(_QWORD *)(v36 + 24);
              v33 *= v64;
              continue;
            case 1:
              v38 = 16;
              break;
            case 2:
              v38 = 32;
              break;
            case 3:
            case 9:
              v38 = 64;
              break;
            case 4:
              v38 = 80;
              break;
            case 5:
            case 6:
              v38 = 128;
              break;
            case 7:
              v38 = 8 * (unsigned int)sub_15A9520(v35, 0);
              break;
            case 0xB:
              v38 = *(_DWORD *)(v36 + 8) >> 8;
              break;
            case 0xD:
              v38 = 8LL * *(_QWORD *)sub_15A9930(v35, v36);
              break;
            case 0xE:
              v125 = *(_QWORD *)(v36 + 32);
              v38 = 8 * sub_12BE0A0(v35, *(_QWORD *)(v36 + 24)) * v125;
              break;
            case 0xF:
              v38 = 8 * (unsigned int)sub_15A9520(v35, *(_DWORD *)(v36 + 8) >> 8);
              break;
          }
          break;
        }
        if ( v132 != v37 * ((v37 + ((unsigned __int64)(v38 * v33 + 7) >> 3) - 1) / v37) )
          return 0;
      }
      v134 += 3;
      v25 = v128;
      if ( v122 == v128 )
      {
        v16 = v124;
        v12 = v117;
        goto LABEL_53;
      }
    }
    if ( (unsigned __int8)sub_1560290(v29, v25, 11) )
      return 0;
    v30 = *(_QWORD *)(v131 - 72);
    if ( *(_BYTE *)(v30 + 16) )
      goto LABEL_35;
LABEL_34:
    v154 = *(__int64 **)(v30 + 112);
    if ( (unsigned __int8)sub_1560290(&v154, v25, 11) )
      return 0;
    goto LABEL_35;
  }
LABEL_53:
  if ( sub_15E4F60((__int64)v12) )
  {
    v41 = *(_DWORD *)(v16 + 12);
    v42 = v41 - 1;
    v43 = *(_DWORD *)(v16 + 8) >> 8;
    if ( v41 - 1 < (unsigned int)v23 && !v43 )
      return 0;
    v39 = (v141 & 0xFFFFFFF8) - 24;
    v44 = (v141 & 0xFFFFFFFFFFFFFFF8LL) - 72;
    if ( (v141 & 4) != 0 )
      v44 = (v141 & 0xFFFFFFFFFFFFFFF8LL) - 24;
    v45 = *(_QWORD *)(**(_QWORD **)v44 + 24LL);
    LOBYTE(v39) = *(_DWORD *)(v45 + 8) >> 8 != 0;
    if ( (_BYTE)v39 != (v43 != 0) || *(_DWORD *)(v45 + 8) >> 8 && v41 != *(_DWORD *)(v45 + 12) )
      return 0;
  }
  else
  {
    v42 = *(_DWORD *)(v16 + 12) - 1;
  }
  if ( v119 > v42
    && *(_DWORD *)(v16 + 8) >> 8
    && v142
    && (unsigned __int8)sub_1560490(&v142, 53, (int *)&v154)
    && (unsigned int)v154 > *(_DWORD *)(v16 + 12) - 1 )
  {
    return 0;
  }
  v146 = (__int64 *)v148;
  v147 = 0x800000000LL;
  v150 = 0x800000000LL;
  v46 = 8;
  v149 = v151;
  if ( v119 > 8uLL )
  {
    sub_16CD150((__int64)&v146, v148, v119, 8, v39, v40);
    v46 = HIDWORD(v150);
  }
  if ( v119 > v46 )
    sub_16CD150((__int64)&v149, v151, v119, 8, v39, v40);
  sub_1562F70(&v152, v142, 0);
  sub_1560E30((__int64)&v154, v137);
  sub_1561FA0((__int64)&v152, &v154);
  sub_173D510(v157);
  v50 = (__int64)&v149;
  v123 = (v141 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v141 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
  v51 = 0;
  if ( v122 )
  {
    v116 = v12;
    v52 = (__int64 *)((v141 & 0xFFFFFFFFFFFFFFF8LL)
                    - 24LL * (*(_DWORD *)((v141 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
    v115 = v15;
    while ( 1 )
    {
      v57 = v51 + 1;
      v58 = v51;
      v59 = *(_QWORD *)(*(_QWORD *)(v16 + 16) + 8 * (v51 + 1));
      v54 = (unsigned __int8 *)*v52;
      if ( *(_QWORD *)*v52 == v59 )
        goto LABEL_76;
      v49 = 257;
      v156 = 257;
      v60 = a1[1];
      v54 = (unsigned __int8 *)*v52;
      v61 = *(_QWORD *)*v52;
      if ( v59 == v61 )
        goto LABEL_76;
      v62 = *(_BYTE *)(v61 + 8);
      if ( v62 == 16 )
        v62 = *(_BYTE *)(**(_QWORD **)(v61 + 16) + 8LL);
      if ( v62 == 15 )
      {
        v63 = *(_BYTE *)(v59 + 8);
        if ( v63 == 16 )
          v63 = *(_BYTE *)(**(_QWORD **)(v59 + 16) + 8LL);
        if ( v63 == 11 )
        {
          v54 = sub_1708970(v60, 45, *v52, (__int64 **)v59, (__int64 *)&v154);
          goto LABEL_76;
        }
      }
      else if ( v62 == 11 )
      {
        v53 = *(_BYTE *)(v59 + 8);
        if ( v53 == 16 )
          v53 = *(_BYTE *)(**(_QWORD **)(v59 + 16) + 8LL);
        if ( v53 == 15 )
        {
          v54 = sub_1708970(v60, 46, *v52, (__int64 **)v59, (__int64 *)&v154);
          goto LABEL_76;
        }
      }
      v54 = sub_1708970(v60, 47, *v52, (__int64 **)v59, (__int64 *)&v154);
LABEL_76:
      v55 = (unsigned int)v147;
      if ( (unsigned int)v147 >= HIDWORD(v147) )
      {
        v114 = v54;
        sub_16CD150((__int64)&v146, v148, 0, 8, v48, v49);
        v55 = (unsigned int)v147;
        v54 = v114;
      }
      v50 = v58;
      v146[v55] = (__int64)v54;
      LODWORD(v147) = v147 + 1;
      v48 = sub_1560230(&v142, v58);
      v56 = (unsigned int)v150;
      if ( (unsigned int)v150 >= HIDWORD(v150) )
      {
        v50 = (__int64)v151;
        v113 = v48;
        sub_16CD150((__int64)&v149, v151, 0, 8, v48, v49);
        v56 = (unsigned int)v150;
        v48 = v113;
      }
      v52 += 3;
      *(_QWORD *)&v149[8 * v56] = v48;
      v51 = v57;
      LODWORD(v150) = v150 + 1;
      if ( v57 == v122 )
      {
        v12 = v116;
        v15 = v115;
        v123 += 24 * v51;
        break;
      }
    }
  }
  LODWORD(v66) = v122;
  if ( *(_DWORD *)(v16 + 12) - 1 != v122 )
  {
    do
    {
      v66 = (unsigned int)(v66 + 1);
      v67 = sub_15A06D0(*(__int64 ***)(*(_QWORD *)(v16 + 16) + 8 * v66), v50, v66, v47);
      v68 = (unsigned int)v147;
      if ( (unsigned int)v147 >= HIDWORD(v147) )
      {
        v50 = (__int64)v148;
        v136 = v67;
        sub_16CD150((__int64)&v146, v148, 0, 8, v67, v49);
        v68 = (unsigned int)v147;
        v67 = v136;
      }
      v146[v68] = v67;
      v69 = (unsigned int)v150;
      LODWORD(v147) = v147 + 1;
      if ( (unsigned int)v150 >= HIDWORD(v150) )
      {
        v50 = (__int64)v151;
        sub_16CD150((__int64)&v149, v151, 0, 8, v67, v49);
        v69 = (unsigned int)v150;
      }
      *(_QWORD *)&v149[8 * v69] = 0;
      v70 = *(_DWORD *)(v16 + 12);
      LODWORD(v150) = v150 + 1;
    }
    while ( v70 - 1 != (_DWORD)v66 );
    v122 = v66;
  }
  if ( v119 > v122 && *(_DWORD *)(v16 + 8) >> 8 )
  {
    v135 = v12;
    v71 = (__int64 *)v123;
    v72 = v122;
    do
    {
      v110 = (unsigned __int8 *)*v71;
      v76 = *(_QWORD *)*v71;
      if ( *(_BYTE *)(v76 + 8) == 11 && *(_DWORD *)(v76 + 8) <= 0x1FFFu )
      {
        v77 = sub_1643350(*(_QWORD **)v76);
        v110 = (unsigned __int8 *)*v71;
        v78 = (__int64 **)v77;
        if ( *(_QWORD *)*v71 != v77 )
        {
          v79 = sub_15FBEB0((_QWORD *)*v71, 0, v77, 0);
          v80 = a1[1];
          v156 = 257;
          v110 = sub_1708970(v80, v79, *v71, v78, (__int64 *)&v154);
        }
      }
      v73 = (unsigned int)v147;
      if ( (unsigned int)v147 >= HIDWORD(v147) )
      {
        v121 = v110;
        sub_16CD150((__int64)&v146, v148, 0, 8, (int)v110, v49);
        v73 = (unsigned int)v147;
        v110 = v121;
      }
      v146[v73] = (__int64)v110;
      LODWORD(v147) = v147 + 1;
      v74 = sub_1560230(&v142, v72);
      v75 = (unsigned int)v150;
      if ( (unsigned int)v150 >= HIDWORD(v150) )
      {
        v120 = v74;
        sub_16CD150((__int64)&v149, v151, 0, 8, v74, v49);
        v75 = (unsigned int)v150;
        v74 = v120;
      }
      ++v72;
      v71 += 3;
      *(_QWORD *)&v149[8 * v75] = v74;
      LODWORD(v150) = v150 + 1;
    }
    while ( v72 != v118 );
    v12 = v135;
  }
  v81 = sub_1560250(&v142);
  if ( !*(_BYTE *)(v137 + 8) )
  {
    v139 = v81;
    v156 = 257;
    sub_164B780(v15, (__int64 *)&v154);
    v81 = v139;
  }
  v138 = v81;
  v82 = (__int64 *)sub_15E0530((__int64)v12);
  v83 = (unsigned int)v150;
  v84 = v149;
  v85 = v82;
  v86 = sub_1560BF0(v82, &v152);
  v87 = sub_155FDB0(v85, v138, v86, v84, v83);
  v154 = (__int64 *)&v156;
  v155 = 0x100000000LL;
  sub_1740980(&v141, (__int64)&v154);
  v88 = *(_BYTE *)(v15 + 16) == 29;
  v145 = 257;
  v89 = a1[1];
  if ( v88 )
  {
    v90 = (__int64 ***)((unsigned __int64)sub_173EA70(
                                            v89,
                                            (__int64)v12,
                                            *(_QWORD *)(v15 - 48),
                                            *(_QWORD *)(v15 - 24),
                                            v146,
                                            (unsigned int)v147,
                                            v154,
                                            (unsigned int)v155,
                                            v144)
                      & 0xFFFFFFFFFFFFFFF8LL);
  }
  else
  {
    v90 = (__int64 ***)(sub_173EC90(v89, v12, v146, (unsigned int)v147, v154, (unsigned int)v155, v144, 0)
                      & 0xFFFFFFFFFFFFFFF8LL);
    *((_WORD *)v90 + 9) = *(_WORD *)(v15 + 18) & 3 | *((_WORD *)v90 + 9) & 0xFFFC;
  }
  sub_164B7C0((__int64)v90, v15);
  v91 = *((_WORD *)v90 + 9) & 0x8000;
  v92 = *(unsigned __int16 *)((v141 & 0xFFFFFFFFFFFFFFF8LL) + 18);
  v93 = *((_WORD *)v90 + 9) & 3;
  v90[7] = (__int64 **)v87;
  *((_WORD *)v90 + 9) = v91 | v93 | (4 * ((v92 >> 2) & 0xDFFF));
  if ( (unsigned __int8)sub_1625980(v15, &v143) )
    sub_15F3B70((__int64)v90, v143);
  if ( v126 == *v90 )
    goto LABEL_145;
  if ( !*(_QWORD *)(v15 + 8) )
    goto LABEL_168;
  if ( !*((_BYTE *)*v90 + 8) )
  {
    v90 = (__int64 ***)sub_1599EF0(*(__int64 ***)v15);
    goto LABEL_145;
  }
  v145 = 257;
  v94 = sub_15FE030((__int64)v90, (__int64)v126, (__int64)v144, 0);
  v95 = *(_QWORD *)(v15 + 48);
  v90 = (__int64 ***)v94;
  v144[0] = v95;
  if ( v95 )
  {
    v96 = (__int64 *)(v94 + 48);
    sub_1623A60((__int64)v144, v95, 2);
    if ( v96 == v144 )
    {
      if ( v144[0] )
        sub_161E7C0((__int64)v144, v144[0]);
      goto LABEL_140;
    }
    v111 = (__int64)v90[6];
    if ( !v111 )
    {
LABEL_176:
      v112 = (unsigned __int8 *)v144[0];
      v90[6] = (__int64 **)v144[0];
      if ( v112 )
        sub_1623210((__int64)v144, v112, (__int64)v96);
      goto LABEL_140;
    }
LABEL_175:
    sub_161E7C0((__int64)v96, v111);
    goto LABEL_176;
  }
  v96 = (__int64 *)(v94 + 48);
  if ( (__int64 *)(v94 + 48) != v144 )
  {
    v111 = *(_QWORD *)(v94 + 48);
    if ( v111 )
      goto LABEL_175;
  }
LABEL_140:
  v97 = v15;
  if ( *(_BYTE *)(v15 + 16) == 29 )
  {
    v98 = sub_157EE30(*(_QWORD *)(v15 - 48));
    v97 = v98;
    if ( v98 )
      v97 = v98 - 24;
  }
  sub_1740140(a1, (__int64)v90, v97);
  v99 = *(_QWORD *)(v15 + 8);
  v100 = *a1;
  if ( !v99 )
    goto LABEL_168;
  do
  {
    v101 = sub_1648700(v99);
    sub_170B990(v100, (__int64)v101);
    v99 = *(_QWORD *)(v99 + 8);
  }
  while ( v99 );
LABEL_145:
  v102 = *(_QWORD *)(v15 + 8);
  if ( !v102 )
  {
LABEL_168:
    if ( (*(_BYTE *)(v15 + 17) & 1) != 0 )
    {
      if ( v126 == *v90 )
        sub_164CC90(v15, (__int64)v90);
      else
        sub_164BAF0(v15);
    }
    goto LABEL_151;
  }
  v103 = *a1;
  do
  {
    v104 = sub_1648700(v102);
    sub_170B990(v103, (__int64)v104);
    v102 = *(_QWORD *)(v102 + 8);
  }
  while ( v102 );
  if ( v90 == (__int64 ***)v15 )
    v90 = (__int64 ***)sub_1599EF0(*v90);
  sub_164D160(v15, (__int64)v90, a3, a4, a5, a6, v105, v106, a9, a10);
LABEL_151:
  sub_170BC50((__int64)a1, v15);
  v107 = v154;
  v108 = &v154[7 * (unsigned int)v155];
  if ( v154 != v108 )
  {
    do
    {
      v109 = *(v108 - 3);
      v108 -= 7;
      if ( v109 )
        j_j___libc_free_0(v109, v108[6] - v109);
      if ( (__int64 *)*v108 != v108 + 2 )
        j_j___libc_free_0(*v108, v108[2] + 1);
    }
    while ( v107 != v108 );
    v107 = v154;
  }
  if ( v107 != (__int64 *)&v156 )
    _libc_free((unsigned __int64)v107);
  sub_173D510(v153);
  if ( v149 != v151 )
    _libc_free((unsigned __int64)v149);
  if ( v146 != (__int64 *)v148 )
    _libc_free((unsigned __int64)v146);
  return 1;
}
