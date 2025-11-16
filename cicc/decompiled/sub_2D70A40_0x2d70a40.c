// Function: sub_2D70A40
// Address: 0x2d70a40
//
__int64 __fastcall sub_2D70A40(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  unsigned int v6; // eax
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r8
  unsigned int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 *v21; // r13
  __int64 *i; // r15
  __int64 v23; // rbx
  int v24; // eax
  char v25; // al
  __int64 v27; // rax
  bool v28; // al
  const char *v29; // rbx
  __int64 v30; // rax
  __int64 j; // r13
  __int64 v32; // r15
  char v33; // al
  __int64 **v34; // rbx
  __int64 v35; // rax
  __int64 v36; // rbx
  __int64 v37; // rdi
  __int64 (__fastcall *v38)(__int64, __int64, __int64, __int64, unsigned int); // rax
  unsigned __int64 *v39; // rax
  __int64 v40; // rbx
  __int64 v41; // rax
  __int64 *v42; // rax
  const char *v43; // rdx
  _QWORD *v44; // rax
  _QWORD *v45; // r12
  __int64 v46; // r15
  __int64 v47; // r12
  __int64 v48; // r9
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rax
  char *v59; // rax
  char *v60; // rdx
  __int64 *v61; // rax
  __int64 v62; // r12
  const char *v63; // rdx
  int v64; // ebx
  __int64 v65; // rax
  int v66; // ebx
  __int64 v67; // r15
  char *v68; // rax
  char *v69; // rdx
  __int64 *v70; // rax
  __int64 v71; // r12
  __int64 v72; // r13
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 *v75; // r15
  unsigned int v76; // eax
  __int64 v77; // r12
  __int64 v78; // rcx
  __int64 v79; // rdx
  __int64 v80; // rbx
  const char *v81; // rax
  __int64 *v82; // rax
  __int64 *v83; // rax
  __int64 v84; // r11
  _QWORD *v85; // rax
  __int64 v86; // r15
  __int64 v87; // r12
  unsigned int v88; // r15d
  __int64 *v89; // rax
  __int64 v90; // r8
  __int64 v91; // r9
  char *v92; // rax
  char *v93; // rdx
  __int64 v94; // rbx
  __int64 *k; // rax
  __int64 v96; // [rsp+10h] [rbp-270h]
  unsigned __int8 v97; // [rsp+18h] [rbp-268h]
  char v98; // [rsp+20h] [rbp-260h]
  char v99; // [rsp+28h] [rbp-258h]
  unsigned __int8 v100; // [rsp+28h] [rbp-258h]
  unsigned __int8 v101; // [rsp+28h] [rbp-258h]
  __int64 v102; // [rsp+28h] [rbp-258h]
  unsigned __int8 v103; // [rsp+28h] [rbp-258h]
  __int64 v104; // [rsp+30h] [rbp-250h]
  __int64 v105; // [rsp+30h] [rbp-250h]
  __int64 v106; // [rsp+30h] [rbp-250h]
  __int64 v107; // [rsp+30h] [rbp-250h]
  __int64 v108; // [rsp+30h] [rbp-250h]
  __int64 v109; // [rsp+48h] [rbp-238h]
  __int64 v110; // [rsp+48h] [rbp-238h]
  __int64 v111; // [rsp+48h] [rbp-238h]
  __int64 v112; // [rsp+48h] [rbp-238h]
  __int64 **v113; // [rsp+58h] [rbp-228h]
  __int64 v116; // [rsp+78h] [rbp-208h]
  __int64 v117; // [rsp+88h] [rbp-1F8h] BYREF
  __int64 v118; // [rsp+90h] [rbp-1F0h] BYREF
  __int64 v119; // [rsp+98h] [rbp-1E8h]
  __int64 v120; // [rsp+A0h] [rbp-1E0h]
  unsigned int v121; // [rsp+A8h] [rbp-1D8h]
  unsigned __int64 v122; // [rsp+B0h] [rbp-1D0h] BYREF
  char *v123; // [rsp+B8h] [rbp-1C8h]
  __int64 *v124; // [rsp+C0h] [rbp-1C0h]
  __int64 v125; // [rsp+C8h] [rbp-1B8h]
  __int64 v126[4]; // [rsp+D0h] [rbp-1B0h] BYREF
  const char *v127; // [rsp+F0h] [rbp-190h] BYREF
  const char *v128; // [rsp+F8h] [rbp-188h]
  __int64 *v129; // [rsp+100h] [rbp-180h]
  __int64 v130; // [rsp+108h] [rbp-178h]
  __int16 v131; // [rsp+110h] [rbp-170h]
  _BYTE *v132; // [rsp+120h] [rbp-160h] BYREF
  __int64 v133; // [rsp+128h] [rbp-158h]
  _BYTE v134[32]; // [rsp+130h] [rbp-150h] BYREF
  __int64 v135; // [rsp+150h] [rbp-130h] BYREF
  char *v136; // [rsp+158h] [rbp-128h]
  __int64 v137; // [rsp+160h] [rbp-120h]
  int v138; // [rsp+168h] [rbp-118h]
  char v139; // [rsp+16Ch] [rbp-114h]
  char v140; // [rsp+170h] [rbp-110h] BYREF
  __int64 v141; // [rsp+190h] [rbp-F0h] BYREF
  char *v142; // [rsp+198h] [rbp-E8h]
  __int64 v143; // [rsp+1A0h] [rbp-E0h]
  int v144; // [rsp+1A8h] [rbp-D8h]
  char v145; // [rsp+1ACh] [rbp-D4h]
  char v146; // [rsp+1B0h] [rbp-D0h] BYREF
  __int64 v147; // [rsp+1D0h] [rbp-B0h] BYREF
  char *v148; // [rsp+1D8h] [rbp-A8h]
  __int64 v149; // [rsp+1E0h] [rbp-A0h]
  int v150; // [rsp+1E8h] [rbp-98h]
  char v151; // [rsp+1ECh] [rbp-94h]
  char v152; // [rsp+1F0h] [rbp-90h] BYREF
  __int64 v153; // [rsp+210h] [rbp-70h] BYREF
  char *v154; // [rsp+218h] [rbp-68h]
  __int64 v155; // [rsp+220h] [rbp-60h]
  int v156; // [rsp+228h] [rbp-58h]
  char v157; // [rsp+22Ch] [rbp-54h]
  char v158; // [rsp+230h] [rbp-50h] BYREF

  v4 = 0;
  v116 = a2[1];
  v6 = sub_B19060(a3, (__int64)a2, a3, a4);
  if ( (_BYTE)v6 )
    return v4;
  v4 = v6;
  v7 = *(unsigned __int8 *)(a2[1] + 8);
  if ( (unsigned __int8)v7 > 0xCu || (v8 = 4143, !_bittest64(&v8, v7)) )
  {
    if ( (v7 & 0xFD) != 4 )
      return v4;
  }
  v132 = v134;
  v133 = 0x400000000LL;
  sub_9C95B0((__int64)&v132, (__int64)a2);
  v136 = &v140;
  v142 = &v146;
  v135 = 0;
  v137 = 4;
  v138 = 0;
  v139 = 1;
  v141 = 0;
  v143 = 4;
  v144 = 0;
  v145 = 1;
  sub_2D61D00((__int64)&v153, (__int64)&v135, a2, v9, v10, v11);
  sub_2D61D00((__int64)&v153, a3, a2, v12, v13, v14);
  v147 = 0;
  v148 = &v152;
  v154 = &v158;
  v16 = v133;
  v149 = 4;
  v150 = 0;
  v151 = 1;
  v153 = 0;
  v155 = 4;
  v156 = 0;
  v157 = 1;
  if ( !(_DWORD)v133 )
    goto LABEL_14;
  v99 = 0;
  v113 = 0;
  v104 = a3;
  do
  {
    v17 = (__int64)v132;
    v18 = v16;
    v19 = *(_QWORD *)&v132[8 * v16 - 8];
    LODWORD(v133) = v16 - 1;
    if ( *(_BYTE *)v19 == 84 )
    {
      v20 = sub_986520(v19);
      v21 = (__int64 *)(v20 + 32LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF));
      if ( (__int64 *)v20 != v21 )
      {
        for ( i = (__int64 *)v20; v21 != i; i += 4 )
        {
          v23 = *i;
          v24 = *(unsigned __int8 *)*i;
          if ( (unsigned __int8)v24 > 0x1Cu )
          {
            if ( (_BYTE)v24 == 84 )
            {
              if ( !(unsigned __int8)sub_B19060((__int64)&v135, *i, v17, v18) )
              {
                sub_2D61D00((__int64)&v127, v104, (__int64 *)v23, v18, v15, v48);
                if ( !(_BYTE)v131 )
                {
LABEL_128:
                  v25 = v157;
                  v4 = 0;
                  goto LABEL_12;
                }
                sub_2D61D00((__int64)&v127, (__int64)&v135, (__int64 *)v23, v49, v50, v51);
                sub_9C95B0((__int64)&v132, v23);
              }
            }
            else
            {
              switch ( (_BYTE)v24 )
              {
                case '=':
                  if ( sub_B46500((unsigned __int8 *)*i) || (*(_BYTE *)(v23 + 2) & 1) != 0 )
                    goto LABEL_11;
                  break;
                case 'Z':
                  break;
                case 'N':
                  v27 = *(_QWORD *)(v23 - 32);
                  if ( v113 )
                  {
                    if ( v113 != *(__int64 ***)(v27 + 8) )
                      goto LABEL_11;
                  }
                  else
                  {
                    v113 = *(__int64 ***)(v27 + 8);
                  }
                  sub_BED950((__int64)&v127, (__int64)&v147, v23);
                  if ( (_BYTE)v131 )
                  {
                    v98 = v131;
                    sub_9C95B0((__int64)&v132, v23);
                    v17 = **(unsigned __int8 **)(v23 - 32);
                    if ( (unsigned __int8)v17 <= 0x1Cu )
                    {
                      v99 = v98;
                    }
                    else
                    {
                      v28 = (_BYTE)v17 == 61;
                      LOBYTE(v17) = (_BYTE)v17 == 90;
                      v99 |= (v17 | v28) ^ 1;
                    }
                  }
                  continue;
                default:
                  goto LABEL_11;
              }
              sub_BED950((__int64)&v127, (__int64)&v147, v23);
              if ( (_BYTE)v131 )
                sub_9C95B0((__int64)&v132, v23);
            }
          }
          else
          {
            if ( (unsigned int)(v24 - 12) > 9 )
              goto LABEL_11;
            v29 = (const char *)sub_AE6EC0((__int64)&v141, *i);
            v30 = sub_254BB00((__int64)&v141);
            v127 = v29;
            v128 = (const char *)v30;
            sub_254BBF0((__int64)&v127);
          }
        }
      }
    }
    for ( j = *(_QWORD *)(v19 + 16); j; j = *(_QWORD *)(j + 8) )
    {
      v32 = *(_QWORD *)(j + 24);
      v33 = *(_BYTE *)v32;
      if ( *(_BYTE *)v32 <= 0x1Cu )
        goto LABEL_11;
      switch ( v33 )
      {
        case 'T':
          if ( !(unsigned __int8)sub_B19060((__int64)&v135, *(_QWORD *)(j + 24), v17, v18) )
          {
            if ( (unsigned __int8)sub_B19060(v104, v32, v17, v18) )
              goto LABEL_128;
            sub_2D61D00((__int64)&v127, (__int64)&v135, (__int64 *)v32, v52, v53, v54);
            sub_2D61D00((__int64)&v127, v104, (__int64 *)v32, v55, v56, v57);
            sub_9C95B0((__int64)&v132, v32);
          }
          break;
        case '>':
          if ( sub_B46500(*(unsigned __int8 **)(j + 24)) )
            goto LABEL_11;
          if ( (*(_BYTE *)(v32 + 2) & 1) != 0 )
            goto LABEL_11;
          v58 = *(_QWORD *)(v32 - 64);
          if ( v19 != v58 || !v58 )
            goto LABEL_11;
          sub_BED950((__int64)&v127, (__int64)&v153, v32);
          break;
        case 'N':
          v34 = *(__int64 ***)(v32 + 8);
          if ( v113 )
          {
            if ( v113 != v34 )
              goto LABEL_11;
          }
          sub_BED950((__int64)&v127, (__int64)&v153, *(_QWORD *)(j + 24));
          v35 = *(_QWORD *)(v32 + 16);
          if ( v35 )
          {
            while ( 1 )
            {
              v17 = *(_QWORD *)(v35 + 24);
              if ( *(_BYTE *)v17 != 62 )
                break;
              v35 = *(_QWORD *)(v35 + 8);
              if ( !v35 )
                goto LABEL_43;
            }
            v99 = 1;
          }
LABEL_43:
          v113 = v34;
          break;
        default:
          goto LABEL_11;
      }
    }
    v16 = v133;
  }
  while ( (_DWORD)v133 );
  v36 = v104;
  if ( v113 && v99 == 1 )
  {
    v37 = *(_QWORD *)(a1 + 16);
    v38 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, unsigned int))(*(_QWORD *)v37 + 1352LL);
    v4 = v38 == sub_2D57C50
       ? sub_2D57C50(v37, v116, (__int64)v113, v18, v15)
       : ((__int64 (__fastcall *)(__int64, __int64, __int64 **))v38)(v37, v116, v113);
    if ( (_BYTE)v4 )
    {
      v118 = 0;
      v119 = 0;
      v120 = 0;
      v121 = 0;
      v126[1] = sub_254BB00((__int64)&v141);
      v126[0] = (__int64)v142;
      sub_254BBF0((__int64)v126);
      v126[2] = (__int64)&v141;
      v126[3] = v141;
      v127 = (const char *)sub_254BB00((__int64)&v141);
      v128 = v127;
      sub_254BBF0((__int64)&v127);
      v129 = &v141;
      v130 = v141;
      v39 = (unsigned __int64 *)v126[0];
      if ( (const char *)v126[0] != v127 )
      {
        do
        {
          v40 = *v39;
          v41 = sub_AD4C90(*v39, v113, 0);
          v122 = v40;
          *sub_2D707F0((__int64)&v118, (__int64 *)&v122) = v41;
          v126[0] += 8;
          sub_254BBF0((__int64)v126);
          v39 = (unsigned __int64 *)v126[0];
        }
        while ( (const char *)v126[0] != v127 );
        v4 = (unsigned __int8)v4;
        v36 = v104;
      }
      sub_2D64B10(&v122, &v147, v148);
      sub_2D611F0(v126, (__int64)&v147);
      v105 = v126[0];
      v42 = (__int64 *)v122;
      if ( v126[0] != v122 )
      {
        v100 = v4;
        do
        {
          v46 = *v42;
          if ( *(_BYTE *)*v42 == 78 )
          {
            v47 = *(_QWORD *)sub_986520(*v42);
            v127 = (const char *)v46;
            *sub_2D707F0((__int64)&v118, (__int64 *)&v127) = v47;
            sub_BED950((__int64)&v127, a4, v46);
          }
          else
          {
            v109 = *(_QWORD *)(v46 + 32);
            v127 = sub_BD5D20(*v42);
            v129 = (__int64 *)".bc";
            v131 = 773;
            v128 = v43;
            v44 = sub_BD2C40(72, 1u);
            v45 = v44;
            if ( v44 )
              sub_B51BF0((__int64)v44, v46, (__int64)v113, (__int64)&v127, v109, 0);
            v117 = v46;
            *sub_2D707F0((__int64)&v118, &v117) = v45;
          }
          v122 += 8LL;
          sub_254BBF0((__int64)&v122);
          v42 = (__int64 *)v122;
        }
        while ( v105 != v122 );
        v4 = v100;
      }
      v59 = v136;
      if ( v139 )
        v60 = &v136[8 * HIDWORD(v137)];
      else
        v60 = &v136[8 * (unsigned int)v137];
      v122 = (unsigned __int64)v136;
      v123 = v60;
      if ( v136 != v60 )
      {
        do
        {
          if ( *(_QWORD *)v59 < 0xFFFFFFFFFFFFFFFELL )
            break;
          v59 += 8;
          v122 = (unsigned __int64)v59;
        }
        while ( v59 != v60 );
      }
      v125 = v135;
      v124 = &v135;
      sub_2D61DF0(v126, (__int64)&v135);
      v61 = (__int64 *)v122;
      v106 = v126[0];
      if ( v126[0] != v122 )
      {
        v101 = v4;
        v96 = v36;
        do
        {
          v62 = *v61;
          v110 = *v61 + 24;
          v127 = sub_BD5D20(*v61);
          v131 = 773;
          v128 = v63;
          v129 = (__int64 *)&unk_444DE88;
          v64 = *(_DWORD *)(v62 + 4);
          v65 = sub_BD2DA0(80);
          v66 = v64 & 0x7FFFFFF;
          v67 = v65;
          if ( v65 )
          {
            sub_B44260(v65, (__int64)v113, 55, 0x8000000u, v110, 0);
            *(_DWORD *)(v67 + 72) = v66;
            sub_BD6B50((unsigned __int8 *)v67, &v127);
            sub_BD2A10(v67, *(_DWORD *)(v67 + 72), 1);
          }
          v117 = v62;
          *sub_2D707F0((__int64)&v118, &v117) = v67;
          v122 += 8LL;
          sub_254BBF0((__int64)&v122);
          v61 = (__int64 *)v122;
        }
        while ( v106 != v122 );
        v4 = v101;
        v36 = v96;
      }
      v68 = v136;
      if ( v139 )
        v69 = &v136[8 * HIDWORD(v137)];
      else
        v69 = &v136[8 * (unsigned int)v137];
      v122 = (unsigned __int64)v136;
      v123 = v69;
      if ( v136 != v69 )
      {
        do
        {
          if ( *(_QWORD *)v68 < 0xFFFFFFFFFFFFFFFELL )
            break;
          v68 += 8;
          v122 = (unsigned __int64)v68;
        }
        while ( v68 != v69 );
      }
      v125 = v135;
      v124 = &v135;
      sub_2D61DF0(v126, (__int64)&v135);
      v107 = v126[0];
      v70 = (__int64 *)v122;
      if ( v126[0] != v122 )
      {
        v97 = v4;
        v102 = v36;
        do
        {
          v71 = *v70;
          v127 = (const char *)v71;
          v72 = v71;
          v75 = (__int64 *)*sub_2D707F0((__int64)&v118, (__int64 *)&v127);
          v76 = *(_DWORD *)(v71 + 4);
          v77 = 0;
          v76 &= 0x7FFFFFFu;
          v78 = 8LL * v76;
          v111 = v78;
          if ( v76 )
          {
            do
            {
              v79 = *(_QWORD *)(v72 - 8);
              v80 = *(_QWORD *)(v79 + 32LL * *(unsigned int *)(v72 + 72) + v77);
              v81 = *(const char **)(v79 + 4 * v77);
              v77 += 8;
              v127 = v81;
              v82 = sub_2D707F0((__int64)&v118, (__int64 *)&v127);
              sub_F0A850((__int64)v75, *v82, v80);
            }
            while ( v111 != v77 );
          }
          sub_2D61D00((__int64)&v127, v102, v75, v78, v73, v74);
          v122 += 8LL;
          sub_254BBF0((__int64)&v122);
          v70 = (__int64 *)v122;
        }
        while ( v107 != v122 );
        v4 = v97;
      }
      sub_2D64B10(&v122, &v153, v154);
      sub_2D611F0(v126, (__int64)&v153);
      v108 = v126[0];
      v83 = (__int64 *)v122;
      if ( v126[0] != v122 )
      {
        v103 = v4;
        do
        {
          v87 = *v83;
          if ( *(_BYTE *)*v83 == 78 )
          {
            sub_BED950((__int64)&v127, a4, *v83);
            v88 = *(unsigned __int8 *)(a1 + 832);
            v127 = *(const char **)sub_986520(v87);
            v89 = sub_2D707F0((__int64)&v118, (__int64 *)&v127);
            sub_2D594F0(v87, *v89, (__int64 *)(a1 + 840), v88, v90, v91);
          }
          else
          {
            v117 = *(_QWORD *)sub_986520(*v83);
            v84 = *sub_2D707F0((__int64)&v118, &v117);
            v127 = "bc";
            v112 = v84;
            v131 = 259;
            v85 = sub_BD2C40(72, 1u);
            v86 = (__int64)v85;
            if ( v85 )
              sub_B51BF0((__int64)v85, v112, v116, (__int64)&v127, v87 + 24, 0);
            sub_11A11E0(v87, 0, v86);
          }
          v122 += 8LL;
          sub_254BBF0((__int64)&v122);
          v83 = (__int64 *)v122;
        }
        while ( v108 != v122 );
        v4 = v103;
      }
      v92 = v136;
      if ( v139 )
        v93 = &v136[8 * HIDWORD(v137)];
      else
        v93 = &v136[8 * (unsigned int)v137];
      v122 = (unsigned __int64)v136;
      v123 = v93;
      if ( v136 != v93 )
      {
        do
        {
          if ( *(_QWORD *)v92 < 0xFFFFFFFFFFFFFFFELL )
            break;
          v92 += 8;
          v122 = (unsigned __int64)v92;
        }
        while ( v92 != v93 );
      }
      v125 = v135;
      v124 = &v135;
      sub_2D61DF0(v126, (__int64)&v135);
      v94 = v126[0];
      for ( k = (__int64 *)v122; v122 != v94; k = (__int64 *)v122 )
      {
        sub_BED950((__int64)&v127, a4, *k);
        v122 += 8LL;
        sub_254BBF0((__int64)&v122);
      }
      sub_C7D6A0(v119, 16LL * v121, 8);
    }
  }
LABEL_11:
  v25 = v157;
LABEL_12:
  if ( !v25 )
    _libc_free((unsigned __int64)v154);
LABEL_14:
  if ( v151 )
  {
    if ( v145 )
      goto LABEL_16;
LABEL_67:
    _libc_free((unsigned __int64)v142);
    if ( !v139 )
      goto LABEL_68;
  }
  else
  {
    _libc_free((unsigned __int64)v148);
    if ( !v145 )
      goto LABEL_67;
LABEL_16:
    if ( !v139 )
LABEL_68:
      _libc_free((unsigned __int64)v136);
  }
  if ( v132 != v134 )
    _libc_free((unsigned __int64)v132);
  return v4;
}
