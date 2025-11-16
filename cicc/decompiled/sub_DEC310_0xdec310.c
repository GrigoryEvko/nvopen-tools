// Function: sub_DEC310
// Address: 0xdec310
//
__int64 __fastcall sub_DEC310(__int64 a1, __int64 *a2, __int64 a3, char **a4, __int64 a5, __int64 a6)
{
  bool v9; // zf
  __int64 v10; // rdi
  unsigned int v11; // r14d
  __int64 v12; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v19; // rax
  char v20; // r8
  char v21; // r9
  __int64 v22; // r11
  __int64 v23; // rax
  __int64 *v24; // r13
  char v25; // al
  __int64 v26; // r11
  char v27; // r8
  char v28; // r9
  __int64 v29; // rsi
  __int64 v30; // rdi
  int v31; // eax
  __int64 v32; // rax
  unsigned __int64 *v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rax
  _QWORD *v36; // rax
  _QWORD *v37; // rax
  __int64 v38; // r9
  _QWORD *v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  _QWORD *v42; // r14
  __int64 v43; // rax
  unsigned int v44; // r13d
  unsigned int v45; // eax
  __int64 v46; // rcx
  _QWORD *v47; // r13
  __int64 *v48; // r14
  __int64 v49; // r13
  __int64 v50; // rax
  _QWORD *v51; // r9
  __int64 v52; // rax
  const void **v53; // rsi
  bool v54; // al
  __int64 v55; // rcx
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 *v61; // rax
  __int64 v62; // r11
  __int64 v63; // r9
  __int64 v64; // r14
  unsigned int v65; // eax
  const void **v66; // rsi
  __int64 v67; // r11
  unsigned int v70; // eax
  unsigned int v71; // r8d
  __int64 v72; // rax
  _QWORD *v73; // rax
  __int64 v74; // rax
  _QWORD *v75; // r8
  __int64 v76; // r9
  __int64 v77; // rax
  const void *v78; // rdx
  __int64 v79; // r13
  __int64 v80; // rax
  const void **v81; // rsi
  bool v82; // al
  __int64 v83; // r9
  __int64 v84; // rcx
  __int64 *v85; // rax
  __int64 v86; // rax
  _QWORD *v87; // rax
  __int64 v88; // rax
  unsigned int v89; // r13d
  const void *v90; // rdx
  __int128 v91; // [rsp-10h] [rbp-1F0h]
  __int128 v92; // [rsp-10h] [rbp-1F0h]
  __int128 v93; // [rsp-10h] [rbp-1F0h]
  char v94; // [rsp+18h] [rbp-1C8h]
  char v95; // [rsp+27h] [rbp-1B9h]
  __int64 v96; // [rsp+28h] [rbp-1B8h]
  _BYTE *v97; // [rsp+28h] [rbp-1B8h]
  char v98; // [rsp+30h] [rbp-1B0h]
  char v99; // [rsp+30h] [rbp-1B0h]
  _BYTE *v100; // [rsp+30h] [rbp-1B0h]
  _QWORD *v101; // [rsp+30h] [rbp-1B0h]
  int v102; // [rsp+38h] [rbp-1A8h]
  char v103; // [rsp+38h] [rbp-1A8h]
  const void **v104; // [rsp+38h] [rbp-1A8h]
  __int64 *v105; // [rsp+40h] [rbp-1A0h]
  __int64 *v106; // [rsp+40h] [rbp-1A0h]
  __int64 v107; // [rsp+40h] [rbp-1A0h]
  __int64 v108; // [rsp+40h] [rbp-1A0h]
  __int64 v109; // [rsp+40h] [rbp-1A0h]
  __int64 v110; // [rsp+48h] [rbp-198h]
  __int64 v111; // [rsp+48h] [rbp-198h]
  __int64 v112; // [rsp+48h] [rbp-198h]
  __int64 v113; // [rsp+48h] [rbp-198h]
  char v114; // [rsp+50h] [rbp-190h]
  _QWORD *v115; // [rsp+50h] [rbp-190h]
  __int64 v116; // [rsp+50h] [rbp-190h]
  char v117; // [rsp+50h] [rbp-190h]
  unsigned int v118; // [rsp+50h] [rbp-190h]
  unsigned int v119; // [rsp+58h] [rbp-188h]
  char v120; // [rsp+58h] [rbp-188h]
  __int64 v121; // [rsp+58h] [rbp-188h]
  char v122; // [rsp+58h] [rbp-188h]
  unsigned int v123; // [rsp+58h] [rbp-188h]
  _QWORD *v124; // [rsp+58h] [rbp-188h]
  char v125; // [rsp+58h] [rbp-188h]
  unsigned int v126; // [rsp+58h] [rbp-188h]
  __int64 v127; // [rsp+58h] [rbp-188h]
  __int64 v128; // [rsp+58h] [rbp-188h]
  __int64 v129; // [rsp+58h] [rbp-188h]
  __int64 v130; // [rsp+58h] [rbp-188h]
  unsigned int v131; // [rsp+60h] [rbp-180h]
  __int64 v132; // [rsp+60h] [rbp-180h]
  __int128 v133; // [rsp+60h] [rbp-180h]
  __int64 v134; // [rsp+60h] [rbp-180h]
  char v135; // [rsp+60h] [rbp-180h]
  __int64 v136; // [rsp+60h] [rbp-180h]
  char v137; // [rsp+60h] [rbp-180h]
  __int64 v138; // [rsp+60h] [rbp-180h]
  __int64 v139; // [rsp+60h] [rbp-180h]
  __int64 v140; // [rsp+60h] [rbp-180h]
  const void *v141; // [rsp+70h] [rbp-170h] BYREF
  unsigned int v142; // [rsp+78h] [rbp-168h]
  const void *v143; // [rsp+80h] [rbp-160h] BYREF
  unsigned int v144; // [rsp+88h] [rbp-158h]
  char *v145; // [rsp+90h] [rbp-150h] BYREF
  unsigned int v146; // [rsp+98h] [rbp-148h]
  __int64 v147; // [rsp+A0h] [rbp-140h] BYREF
  unsigned int v148; // [rsp+A8h] [rbp-138h]
  const void *v149; // [rsp+B0h] [rbp-130h] BYREF
  unsigned int v150; // [rsp+B8h] [rbp-128h]
  const void *v151; // [rsp+C0h] [rbp-120h] BYREF
  unsigned int v152; // [rsp+C8h] [rbp-118h]
  char v153; // [rsp+D0h] [rbp-110h]
  const void *v154; // [rsp+E0h] [rbp-100h] BYREF
  unsigned int v155; // [rsp+E8h] [rbp-F8h]
  char v156; // [rsp+F0h] [rbp-F0h]
  unsigned __int64 v157; // [rsp+100h] [rbp-E0h] BYREF
  __int64 v158; // [rsp+108h] [rbp-D8h]
  __int64 *v159; // [rsp+110h] [rbp-D0h] BYREF
  _QWORD *v160; // [rsp+118h] [rbp-C8h]
  _BYTE *v161; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v162; // [rsp+128h] [rbp-B8h]
  _BYTE v163[48]; // [rsp+130h] [rbp-B0h] BYREF
  unsigned int v164; // [rsp+160h] [rbp-80h] BYREF
  __int64 v165; // [rsp+168h] [rbp-78h] BYREF
  unsigned int v166; // [rsp+170h] [rbp-70h]
  char *v167; // [rsp+178h] [rbp-68h] BYREF
  unsigned int v168; // [rsp+180h] [rbp-60h]
  const void *v169; // [rsp+188h] [rbp-58h] BYREF
  unsigned int v170; // [rsp+190h] [rbp-50h]
  const void *v171; // [rsp+198h] [rbp-48h] BYREF
  unsigned int v172; // [rsp+1A0h] [rbp-40h]
  char v173; // [rsp+1A8h] [rbp-38h]

  v9 = *(_WORD *)(a3 + 24) == 0;
  v161 = v163;
  v162 = 0x600000000LL;
  if ( v9 )
  {
    v10 = *(_QWORD *)(a3 + 32);
    v11 = *(_DWORD *)(v10 + 32);
    if ( v11 <= 0x40 )
    {
      if ( !*(_QWORD *)(v10 + 24) )
        goto LABEL_4;
    }
    else if ( v11 == (unsigned int)sub_C444A0(v10 + 24) )
    {
LABEL_4:
      v12 = a3;
      sub_D97F80(a1, a3, a3, (__int64)a4, a5, a6);
      goto LABEL_5;
    }
    goto LABEL_9;
  }
  v119 = a6;
  v131 = a5;
  v19 = sub_D98140((__int64)a2, a3);
  v20 = v131;
  v21 = v119;
  v22 = v19;
  if ( *(_WORD *)(v19 + 24) != 8 )
  {
    if ( !(_BYTE)v119 )
      goto LABEL_9;
    v40 = sub_DEAB30((__int64)a2, a3, (__int64)a4, (__int64)&v161, v131, v119);
    v20 = v131;
    v21 = v119;
    v22 = v40;
    if ( !v40 )
      goto LABEL_9;
  }
  if ( a4 != *(char ***)(v22 + 48) )
    goto LABEL_9;
  v23 = *(_QWORD *)(v22 + 40);
  if ( v23 != 3 )
  {
LABEL_13:
    if ( v23 == 2 )
    {
      v114 = v21;
      v120 = v20;
      v132 = v22;
      v105 = sub_DDF4E0((__int64)a2, **(__int64 ****)(v22 + 32), *a4);
      v24 = sub_DDF4E0((__int64)a2, *(__int64 ***)(*(_QWORD *)(v132 + 32) + 8LL), *a4);
      if ( sub_DADE90((__int64)a2, (__int64)v24, (__int64)a4) )
      {
        sub_DE4EA0((__int64)&v164, (__int64)a4, a2);
        v96 = sub_DE2740((__int64)a2, (__int64)v24, (__int64)&v164);
        v25 = sub_DBEC00((__int64)a2, v96);
        v26 = v132;
        v95 = v25;
        v27 = v120;
        v121 = (__int64)v105;
        v28 = v114;
        if ( !v25 )
        {
          v125 = v27;
          if ( !(unsigned __int8)sub_DBED40((__int64)a2, v96) )
            goto LABEL_72;
          v61 = sub_DCAF50(a2, (__int64)v105, 0);
          v26 = v132;
          v27 = v125;
          v121 = (__int64)v61;
          v28 = v114;
        }
        if ( *((_WORD *)v24 + 12) )
        {
          if ( !v27 || (*(_BYTE *)(v26 + 28) & 1) == 0 )
            goto LABEL_72;
        }
        else
        {
          v29 = v24[4];
          v30 = v29 + 24;
          if ( *(_DWORD *)(v29 + 32) <= 0x40u )
          {
            if ( *(_QWORD *)(v29 + 24) == 1 )
              goto LABEL_19;
          }
          else
          {
            v94 = v28;
            v98 = v27;
            v102 = *(_DWORD *)(v29 + 32);
            v110 = v26;
            v31 = sub_C444A0(v30);
            v30 = v29 + 24;
            v26 = v110;
            v27 = v98;
            v28 = v94;
            if ( v31 == v102 - 1 )
            {
LABEL_19:
              v32 = sub_DE2740((__int64)a2, v121, (__int64)&v164);
              sub_DBDFD0((__int64)&v149, (__int64)a2, v32);
              sub_DBDFD0((__int64)&v157, (__int64)a2, v121);
              v33 = &v157;
              if ( (int)sub_C49970((__int64)&v149, &v157) < 0 )
                v33 = (unsigned __int64 *)&v149;
              if ( v150 <= 0x40 && *((_DWORD *)v33 + 2) <= 0x40u )
              {
                v78 = (const void *)*v33;
                v150 = *((_DWORD *)v33 + 2);
                v149 = v78;
              }
              else
              {
                sub_C43990((__int64)&v149, (__int64)v33);
              }
              sub_969240((__int64 *)&v157);
              v34 = sub_D95540(v121);
              v115 = sub_DA2C50((__int64)a2, v34, 0, 0);
              v35 = sub_D95540(v121);
              v36 = sub_DA2C50((__int64)a2, v35, 1, 0);
              v37 = sub_DC7ED0(a2, v121, (__int64)v36, 0, 0);
              v38 = (__int64)v115;
              v116 = (__int64)v37;
              if ( (unsigned __int8)sub_DDD5B0(a2, (__int64)a4, 33, (__int64)v37, v38) )
              {
                v64 = sub_DBB9F0((__int64)a2, v116, 0, 0);
                LODWORD(v158) = *(_DWORD *)(v64 + 8);
                if ( (unsigned int)v158 > 0x40 )
                  sub_C43780((__int64)&v157, (const void **)v64);
                else
                  v157 = *(_QWORD *)v64;
                LODWORD(v160) = *(_DWORD *)(v64 + 24);
                if ( (unsigned int)v160 > 0x40 )
                  sub_C43780((__int64)&v159, (const void **)(v64 + 16));
                else
                  v159 = *(__int64 **)(v64 + 16);
                sub_AB0910((__int64)&v151, (__int64)&v157);
                sub_C46F20((__int64)&v151, 1u);
                v65 = v152;
                v152 = 0;
                v155 = v65;
                v154 = v151;
                v66 = &v154;
                if ( (int)sub_C49970((__int64)&v149, (unsigned __int64 *)&v154) < 0 )
                  v66 = &v149;
                if ( v150 <= 0x40 && *((_DWORD *)v66 + 2) <= 0x40u )
                {
                  v90 = *v66;
                  v150 = *((_DWORD *)v66 + 2);
                  v149 = v90;
                }
                else
                {
                  sub_C43990((__int64)&v149, (__int64)v66);
                }
                sub_969240((__int64 *)&v154);
                sub_969240((__int64 *)&v151);
                sub_969240((__int64 *)&v159);
                sub_969240((__int64 *)&v157);
              }
              *(_QWORD *)&v133 = v161;
              *((_QWORD *)&v133 + 1) = (unsigned int)v162;
              v39 = sub_DA26C0(a2, (__int64)&v149);
              sub_D97FA0(a1, v121, (__int64)v39, v121, 0, *((__int64 *)&v133 + 1), v133);
              sub_969240((__int64 *)&v149);
              goto LABEL_26;
            }
          }
          v99 = v28;
          v103 = v27;
          v111 = v26;
          if ( sub_986760(v30) )
            goto LABEL_19;
          v26 = v111;
          v28 = v99;
          if ( !v103 )
          {
LABEL_35:
            v135 = v28;
            if ( !sub_9867B0(v30) )
            {
              v42 = 0;
              if ( v135 )
                v42 = &v161;
              v106 = sub_DCAF50(a2, (__int64)v105, 0);
              v43 = v24[4];
              v44 = *(_DWORD *)(v43 + 32);
              v112 = v43;
              v104 = (const void **)(v43 + 24);
              if ( v44 <= 0x40 )
              {
                _RAX = *(_QWORD *)(v43 + 24);
                __asm { tzcnt   rdx, rax }
                v9 = _RAX == 0;
                v70 = 64;
                if ( !v9 )
                  v70 = _RDX;
                if ( v44 <= v70 )
                  v70 = v44;
                v123 = v70;
                v71 = sub_DB55F0((__int64)a2, (__int64)v106);
                v72 = 1LL << v123;
                if ( v123 <= v71 )
                  goto LABEL_40;
                LODWORD(v158) = v44;
                v157 = 0;
              }
              else
              {
                v123 = sub_C44590((__int64)v104);
                if ( v123 <= (unsigned int)sub_DB55F0((__int64)a2, (__int64)v106) )
                  goto LABEL_40;
                LODWORD(v158) = v44;
                sub_C43690((__int64)&v157, 0, 0);
                v72 = 1LL << v123;
                if ( (unsigned int)v158 > 0x40 )
                {
                  *(_QWORD *)(v157 + 8LL * (v123 >> 6)) |= 1LL << v123;
                  goto LABEL_162;
                }
              }
              v157 |= v72;
LABEL_162:
              v73 = sub_DA26C0(a2, (__int64)&v157);
              v100 = sub_DCFA50(a2, (__int64)v106, (__int64)v73);
              if ( (unsigned int)v158 > 0x40 && v157 )
                j_j___libc_free_0_0(v157);
              v74 = sub_D95540((__int64)v106);
              v97 = sub_DA2C50((__int64)a2, v74, 0, 0);
              if ( !(unsigned __int8)sub_DC3A60((__int64)a2, 32, v100, v97) )
              {
                if ( !v42 || (unsigned __int8)sub_DC3A60((__int64)a2, 33, v100, v97) )
                {
                  v49 = sub_D970F0((__int64)a2);
LABEL_64:
                  v50 = sub_D970F0((__int64)a2);
                  v51 = (_QWORD *)v49;
                  if ( v49 != v50 )
                  {
                    v52 = sub_DE2740((__int64)a2, v49, (__int64)&v164);
                    sub_DBDFD0((__int64)&v154, (__int64)a2, v52);
                    sub_DBDFD0((__int64)&v157, (__int64)a2, v49);
                    v53 = (const void **)&v157;
                    if ( (int)sub_C49970((__int64)&v154, &v157) < 0 )
                      v53 = &v154;
                    v124 = sub_DA26C0(a2, (__int64)v53);
                    sub_969240((__int64 *)&v157);
                    sub_969240((__int64 *)&v154);
                    v51 = v124;
                  }
                  v136 = (__int64)v51;
                  v54 = sub_D96A50(v49);
                  v55 = v49;
                  *((_QWORD *)&v91 + 1) = (unsigned int)v162;
                  *(_QWORD *)&v91 = v161;
                  if ( v54 )
                    v55 = v136;
                  sub_D97FA0(a1, v49, v136, v55, 0, v136, v91);
                  goto LABEL_26;
                }
                v75 = sub_DA4260((__int64)a2, (__int64)v100, (__int64)v97);
                v77 = *((unsigned int *)v42 + 2);
                if ( v77 + 1 > (unsigned __int64)*((unsigned int *)v42 + 3) )
                {
                  v101 = v75;
                  sub_C8D5F0((__int64)v42, v42 + 2, v77 + 1, 8u, (__int64)v75, v76);
                  v77 = *((unsigned int *)v42 + 2);
                  v75 = v101;
                }
                *(_QWORD *)(*v42 + 8 * v77) = v75;
                ++*((_DWORD *)v42 + 2);
              }
LABEL_40:
              v45 = *(_DWORD *)(v112 + 32);
              LODWORD(v158) = v45;
              if ( v45 > 0x40 )
              {
                sub_C43780((__int64)&v157, v104);
                v45 = v158;
                if ( (unsigned int)v158 > 0x40 )
                {
                  sub_C482E0((__int64)&v157, v123);
                  goto LABEL_44;
                }
              }
              else
              {
                v157 = *(_QWORD *)(v112 + 24);
              }
              if ( v45 == v123 )
                v157 = 0;
              else
                v157 >>= v123;
LABEL_44:
              sub_C44740((__int64)&v151, (char **)&v157, v44 - v123);
              if ( (unsigned int)v158 > 0x40 && v157 )
                j_j___libc_free_0_0(v157);
              sub_C473B0((__int64)&v157, (__int64)&v151);
              sub_C449B0((__int64)&v154, (const void **)&v157, v44);
              if ( (unsigned int)v158 > 0x40 && v157 )
                j_j___libc_free_0_0(v157);
              LODWORD(v158) = v44;
              v46 = 1LL << v123;
              if ( v44 > 0x40 )
              {
                sub_C43690((__int64)&v157, 0, 0);
                v46 = 1LL << v123;
                if ( (unsigned int)v158 > 0x40 )
                {
                  *(_QWORD *)(v157 + 8LL * (v123 >> 6)) |= 1LL << v123;
LABEL_53:
                  v47 = sub_DA26C0(a2, (__int64)&v157);
                  if ( (unsigned int)v158 > 0x40 && v157 )
                    j_j___libc_free_0_0(v157);
                  v160 = sub_DA26C0(a2, (__int64)&v154);
                  v159 = v106;
                  v157 = (unsigned __int64)&v159;
                  v158 = 0x200000002LL;
                  v48 = sub_DC8BD0(a2, (__int64)&v157, 0, 0);
                  if ( (__int64 **)v157 != &v159 )
                    _libc_free(v157, &v157);
                  v49 = sub_DCC290(a2, (__int64)v48, (__int64)v47);
                  if ( v155 > 0x40 && v154 )
                    j_j___libc_free_0_0(v154);
                  if ( v152 > 0x40 && v151 )
                    j_j___libc_free_0_0(v151);
                  goto LABEL_64;
                }
              }
              else
              {
                v157 = 0;
              }
              v157 |= v46;
              goto LABEL_53;
            }
            goto LABEL_72;
          }
          if ( (*(_BYTE *)(v111 + 28) & 1) == 0 )
          {
LABEL_80:
            v30 = v29 + 24;
            goto LABEL_35;
          }
        }
        v137 = v28;
        if ( (unsigned __int8)sub_DB5FD0((__int64)a2, *(_QWORD *)(v26 + 48)) )
        {
          if ( (unsigned __int8)sub_DB6630((__int64)a2, (__int64)a4) || (unsigned __int8)sub_DBE090((__int64)a2, v96) )
          {
            if ( v95 )
              v24 = sub_DCAF50(a2, (__int64)v24, 0);
            v79 = sub_DCB270((__int64)a2, v121, (__int64)v24);
            v129 = sub_D970F0((__int64)a2);
            if ( v79 != sub_D970F0((__int64)a2) )
            {
              v80 = sub_DE2740((__int64)a2, v79, (__int64)&v164);
              sub_DBDFD0((__int64)&v154, (__int64)a2, v80);
              sub_DBDFD0((__int64)&v157, (__int64)a2, v79);
              v81 = (const void **)&v157;
              if ( (int)sub_C49970((__int64)&v154, &v157) < 0 )
                v81 = &v154;
              v129 = (__int64)sub_DA26C0(a2, (__int64)v81);
              sub_969240((__int64 *)&v157);
              sub_969240((__int64 *)&v154);
            }
            v82 = sub_D96A50(v79);
            *((_QWORD *)&v93 + 1) = (unsigned int)v162;
            v84 = v129;
            *(_QWORD *)&v93 = v161;
            if ( !v82 )
              v84 = v79;
            sub_D97FA0(a1, v79, v129, v84, 0, v83, v93);
            goto LABEL_26;
          }
          goto LABEL_72;
        }
        if ( *((_WORD *)v24 + 12) )
        {
LABEL_72:
          v56 = sub_D970F0((__int64)a2);
          sub_D97F80(a1, v56, v57, v58, v59, v60);
LABEL_26:
          v12 = 16LL * (unsigned int)v167;
          sub_C7D6A0(v165, v12, 8);
          goto LABEL_5;
        }
        v29 = v24[4];
        v28 = v137;
        goto LABEL_80;
      }
    }
LABEL_9:
    v12 = sub_D970F0((__int64)a2);
    sub_D97F80(a1, v12, v14, v15, v16, v17);
    goto LABEL_5;
  }
  v134 = v22;
  v117 = v21;
  v122 = v20;
  v41 = sub_D95540(**(_QWORD **)(v22 + 32));
  v22 = v134;
  if ( *(_BYTE *)(v41 + 8) != 12 )
  {
    v23 = *(_QWORD *)(v134 + 40);
    v21 = v117;
    v20 = v122;
    goto LABEL_13;
  }
  v142 = 1;
  v141 = 0;
  v144 = 1;
  v143 = 0;
  v146 = 1;
  v145 = 0;
  v148 = 1;
  v147 = 0;
  sub_D92D30((__int64)&v164, v134);
  v62 = v134;
  if ( !v173 )
  {
    v153 = 0;
    goto LABEL_83;
  }
  if ( v172 <= 0x40 )
  {
    v142 = v172;
    v141 = v171;
  }
  else
  {
    sub_C43990((__int64)&v141, (__int64)&v171);
    v62 = v134;
  }
  if ( v144 <= 0x40 && v170 <= 0x40 )
  {
    v144 = v170;
    v143 = v169;
  }
  else
  {
    v138 = v62;
    sub_C43990((__int64)&v143, (__int64)&v169);
    v62 = v138;
  }
  if ( v146 <= 0x40 && v168 <= 0x40 )
  {
    v146 = v168;
    v145 = v167;
  }
  else
  {
    v139 = v62;
    sub_C43990((__int64)&v145, (__int64)&v167);
    v62 = v139;
  }
  if ( v148 <= 0x40 && v166 <= 0x40 )
  {
    v148 = v166;
    v147 = v165;
  }
  else
  {
    v140 = v62;
    sub_C43990((__int64)&v147, (__int64)&v165);
    v62 = v140;
  }
  v118 = v164;
  v126 = v164 + 1;
  LODWORD(v158) = v146;
  if ( v146 > 0x40 )
  {
    v113 = v62;
    sub_C43780((__int64)&v157, (const void **)&v145);
    v62 = v113;
  }
  else
  {
    v157 = (unsigned __int64)v145;
  }
  v152 = v144;
  if ( v144 > 0x40 )
  {
    v109 = v62;
    sub_C43780((__int64)&v151, &v143);
    v62 = v109;
  }
  else
  {
    v151 = v143;
  }
  v150 = v142;
  if ( v142 > 0x40 )
  {
    v108 = v62;
    sub_C43780((__int64)&v149, &v141);
    v62 = v108;
  }
  else
  {
    v149 = v141;
  }
  v107 = v62;
  sub_C4CD10((__int64)&v154, &v149, (__int64 *)&v151, &v157, v126);
  v67 = v107;
  if ( v150 > 0x40 && v149 )
  {
    j_j___libc_free_0_0(v149);
    v67 = v107;
  }
  if ( v152 > 0x40 && v151 )
  {
    v127 = v67;
    j_j___libc_free_0_0(v151);
    v67 = v127;
  }
  if ( (unsigned int)v158 > 0x40 && v157 )
  {
    v128 = v67;
    j_j___libc_free_0_0(v157);
    v67 = v128;
  }
  if ( v156 )
  {
    v130 = v67;
    v85 = (__int64 *)sub_B2BE50(*a2);
    v86 = sub_ACCFD0(v85, (__int64)&v154);
    v87 = sub_DA2570((__int64)a2, v86);
    v88 = sub_DD0540(v130, (__int64)v87, a2)[4];
    v89 = *(_DWORD *)(v88 + 32);
    if ( v89 <= 0x40 )
    {
      if ( *(_QWORD *)(v88 + 24) )
        goto LABEL_190;
    }
    else if ( v89 != (unsigned int)sub_C444A0(v88 + 24) )
    {
LABEL_190:
      v153 = 0;
LABEL_191:
      if ( v156 )
      {
        v156 = 0;
        if ( v155 > 0x40 )
        {
          if ( v154 )
            j_j___libc_free_0_0(v154);
        }
      }
      goto LABEL_136;
    }
    LOBYTE(v159) = 0;
    if ( v156 )
    {
      sub_9865C0((__int64)&v157, (__int64)&v154);
      LOBYTE(v159) = 1;
    }
    sub_D92430((__int64)&v151, (__int64)&v157, v118);
    if ( (_BYTE)v159 )
    {
      LOBYTE(v159) = 0;
      sub_969240((__int64 *)&v157);
    }
    goto LABEL_191;
  }
  v153 = 0;
LABEL_136:
  if ( v173 )
  {
    v173 = 0;
    if ( v172 > 0x40 && v171 )
      j_j___libc_free_0_0(v171);
    if ( v170 > 0x40 && v169 )
      j_j___libc_free_0_0(v169);
    if ( v168 > 0x40 && v167 )
      j_j___libc_free_0_0(v167);
    if ( v166 > 0x40 && v165 )
      j_j___libc_free_0_0(v165);
  }
LABEL_83:
  if ( v148 > 0x40 && v147 )
    j_j___libc_free_0_0(v147);
  if ( v146 > 0x40 && v145 )
    j_j___libc_free_0_0(v145);
  if ( v144 > 0x40 && v143 )
    j_j___libc_free_0_0(v143);
  if ( v142 > 0x40 && v141 )
    j_j___libc_free_0_0(v141);
  if ( !v153 )
    goto LABEL_9;
  v12 = (__int64)sub_DA26C0(a2, (__int64)&v151);
  *((_QWORD *)&v92 + 1) = (unsigned int)v162;
  *(_QWORD *)&v92 = v161;
  sub_D97FA0(a1, v12, v12, v12, 0, v63, v92);
  if ( v153 )
  {
    v153 = 0;
    sub_969240((__int64 *)&v151);
  }
LABEL_5:
  if ( v161 != v163 )
    _libc_free(v161, v12);
  return a1;
}
