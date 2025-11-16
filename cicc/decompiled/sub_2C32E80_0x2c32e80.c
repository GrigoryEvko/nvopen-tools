// Function: sub_2C32E80
// Address: 0x2c32e80
//
void __fastcall sub_2C32E80(__int64 *a1)
{
  __int64 v1; // rsi
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned __int64 v22; // r9
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // rax
  __int64 i; // r8
  __int64 v30; // rcx
  char v31; // di
  __int64 v32; // rdx
  __int64 v33; // r8
  __int64 v34; // r9
  _BYTE *v35; // rdi
  _QWORD *v36; // rcx
  _QWORD *v37; // rdi
  size_t v38; // rcx
  __int64 v39; // rdx
  char *v40; // rax
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rbx
  __int64 v45; // rax
  __int64 v46; // r9
  _QWORD *v47; // r13
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rax
  char v52; // bl
  __int64 v53; // rax
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // r13
  _QWORD *v58; // rbx
  __int64 v59; // r13
  __int64 v60; // r9
  __int64 v61; // r9
  __int64 v62; // r8
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // rdx
  unsigned __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // r12
  _BYTE *v71; // rsi
  size_t v72; // rdx
  _QWORD *v73; // rdi
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r8
  __int64 v79; // r9
  __int64 v80; // r13
  _BYTE *v81; // rsi
  __int64 v82; // rdx
  __int64 v83; // rdx
  __int64 v84; // r8
  __int64 v85; // r9
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // r8
  __int64 v93; // r9
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 v96; // r8
  __int64 v97; // r9
  __int64 v98; // rdx
  __int64 v99; // r8
  __int64 v100; // r9
  __int64 v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // r8
  __int64 v104; // r9
  __int64 v105; // rax
  _QWORD *v106; // rax
  __int64 v107; // rdx
  __int64 v108; // rcx
  __int64 v109; // r8
  __int64 v110; // r9
  __int64 v111; // r10
  __int64 v112; // r11
  __int64 v113; // rbx
  __int64 v114; // rdx
  __int64 v115; // rcx
  __int64 v116; // r8
  __int64 v117; // r9
  __int64 v118; // r12
  __int64 v119; // rdi
  __int64 v120; // rdx
  __int64 v121; // rcx
  __int64 v122; // r8
  __int64 v123; // r9
  size_t v124; // rdx
  __int32 v125; // [rsp+24h] [rbp-56Ch]
  _BYTE *v126; // [rsp+30h] [rbp-560h]
  unsigned __int8 *v127; // [rsp+38h] [rbp-558h]
  _QWORD *v128; // [rsp+40h] [rbp-550h]
  __int64 v129; // [rsp+48h] [rbp-548h]
  __int64 v131; // [rsp+60h] [rbp-530h]
  _BYTE *v132; // [rsp+78h] [rbp-518h]
  _QWORD *v133; // [rsp+88h] [rbp-508h]
  __int64 v134; // [rsp+88h] [rbp-508h]
  __int64 v135; // [rsp+90h] [rbp-500h]
  __int64 v136; // [rsp+90h] [rbp-500h]
  __int64 v137; // [rsp+98h] [rbp-4F8h]
  __int64 v138; // [rsp+98h] [rbp-4F8h]
  __int64 v139[2]; // [rsp+A0h] [rbp-4F0h] BYREF
  _QWORD v140[2]; // [rsp+B0h] [rbp-4E0h] BYREF
  _QWORD *v141; // [rsp+C0h] [rbp-4D0h] BYREF
  size_t n; // [rsp+C8h] [rbp-4C8h]
  _QWORD v143[2]; // [rsp+D0h] [rbp-4C0h] BYREF
  _BYTE *v144; // [rsp+E0h] [rbp-4B0h] BYREF
  __int64 v145; // [rsp+E8h] [rbp-4A8h]
  _BYTE v146[48]; // [rsp+F0h] [rbp-4A0h] BYREF
  __m128i v147; // [rsp+120h] [rbp-470h] BYREF
  char *v148; // [rsp+130h] [rbp-460h]
  __int16 v149; // [rsp+140h] [rbp-450h]
  __int64 v150; // [rsp+180h] [rbp-410h]
  __int64 v151; // [rsp+188h] [rbp-408h]
  __int16 v152; // [rsp+198h] [rbp-3F8h]
  _QWORD v153[12]; // [rsp+1A0h] [rbp-3F0h] BYREF
  __int64 v154; // [rsp+200h] [rbp-390h]
  __int64 v155; // [rsp+208h] [rbp-388h]
  __int16 v156; // [rsp+218h] [rbp-378h]
  __int16 v157; // [rsp+228h] [rbp-368h]
  __m128i v158[2]; // [rsp+230h] [rbp-360h] BYREF
  __int16 v159; // [rsp+250h] [rbp-340h]
  __int64 v160; // [rsp+290h] [rbp-300h]
  __int64 v161; // [rsp+298h] [rbp-2F8h]
  __int16 v162; // [rsp+2A8h] [rbp-2E8h] BYREF
  _QWORD v163[15]; // [rsp+2B0h] [rbp-2E0h] BYREF
  __int16 v164; // [rsp+328h] [rbp-268h]
  __int16 v165; // [rsp+338h] [rbp-258h]
  __m128i v166[2]; // [rsp+340h] [rbp-250h] BYREF
  __int16 v167; // [rsp+360h] [rbp-230h]
  __int16 v168; // [rsp+3B8h] [rbp-1D8h]
  _BYTE v169[120]; // [rsp+3C0h] [rbp-1D0h] BYREF
  __int16 v170; // [rsp+438h] [rbp-158h]
  __int16 v171; // [rsp+448h] [rbp-148h]
  _BYTE v172[120]; // [rsp+450h] [rbp-140h] BYREF
  __int16 v173; // [rsp+4C8h] [rbp-C8h]
  _BYTE v174[120]; // [rsp+4D0h] [rbp-C0h] BYREF
  __int16 v175; // [rsp+548h] [rbp-48h]
  __int16 v176; // [rsp+558h] [rbp-38h]

  v1 = *a1;
  v144 = v146;
  v145 = 0x600000000LL;
  sub_2C2F4B0(v158, v1);
  sub_2C31060((__int64)v166, (__int64)v158, v2, v3, v4, v5);
  sub_2AB1B50((__int64)&v162);
  sub_2AB1B50((__int64)v158);
  sub_2ABCC20(&v147, (__int64)v166, v6, v7, v8, v9);
  v152 = v168;
  sub_2ABCC20(v153, (__int64)v169, v10, v11, v12, v13);
  v156 = v170;
  v157 = v171;
  sub_2ABCC20(v158, (__int64)v172, v14, v15, v16, v17);
  v162 = v173;
  sub_2ABCC20(v163, (__int64)v174, v18, v19, v20, v21);
  v23 = v150;
  v164 = v175;
  v165 = v176;
  v24 = v151;
LABEL_2:
  v25 = v160;
  v26 = v161 - v160;
  if ( v24 - v23 != v161 - v160 )
  {
LABEL_3:
    v27 = *(_QWORD *)(v24 - 32);
    v28 = *(_QWORD *)(v27 + 120);
    for ( i = v27 + 112; i != v28; v28 = *(_QWORD *)(v28 + 8) )
    {
      while ( 1 )
      {
        if ( !v28 )
          BUG();
        if ( *(_BYTE *)(v28 - 16) == 9 && *(_BYTE *)(v28 + 137) )
          break;
        v28 = *(_QWORD *)(v28 + 8);
        if ( i == v28 )
          goto LABEL_12;
      }
      v23 = (unsigned int)v145;
      v22 = (unsigned int)v145 + 1LL;
      if ( v22 > HIDWORD(v145) )
      {
        v135 = v28;
        v137 = i;
        sub_C8D5F0((__int64)&v144, v146, (unsigned int)v145 + 1LL, 8u, i, v22);
        v23 = (unsigned int)v145;
        v28 = v135;
        i = v137;
      }
      v26 = (__int64)v144;
      v25 = v28 - 24;
      *(_QWORD *)&v144[8 * v23] = v28 - 24;
      LODWORD(v145) = v145 + 1;
    }
    while ( 1 )
    {
LABEL_12:
      sub_2AD7320((__int64)&v147, v25, v23, v26, i, v22);
      v24 = v151;
      v23 = v150;
      v25 = v154;
      if ( v151 - v150 == v155 - v154 )
      {
        if ( v150 == v151 )
          goto LABEL_2;
        v30 = v150;
        while ( *(_QWORD *)v30 == *(_QWORD *)v25 )
        {
          v31 = *(_BYTE *)(v30 + 24);
          if ( v31 != *(_BYTE *)(v25 + 24)
            || v31 && (*(_QWORD *)(v30 + 8) != *(_QWORD *)(v25 + 8) || *(_QWORD *)(v30 + 16) != *(_QWORD *)(v25 + 16)) )
          {
            break;
          }
          v30 += 32;
          v25 += 32;
          if ( v151 == v30 )
            goto LABEL_2;
        }
      }
      v26 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v151 - 32) + 8LL) - 1;
      if ( (unsigned int)v26 <= 1 )
        goto LABEL_2;
    }
  }
  while ( v23 != v24 )
  {
    v26 = *(_QWORD *)v25;
    if ( *(_QWORD *)v23 != *(_QWORD *)v25 )
      goto LABEL_3;
    v26 = *(unsigned __int8 *)(v23 + 24);
    if ( (_BYTE)v26 != *(_BYTE *)(v25 + 24) )
      goto LABEL_3;
    if ( (_BYTE)v26 )
    {
      v26 = *(_QWORD *)(v25 + 8);
      if ( *(_QWORD *)(v23 + 8) != v26 )
        goto LABEL_3;
      v26 = *(_QWORD *)(v25 + 16);
      if ( *(_QWORD *)(v23 + 16) != v26 )
        goto LABEL_3;
    }
    v23 += 32;
    v25 += 32;
  }
  sub_2AB1B50((__int64)v163);
  sub_2AB1B50((__int64)v158);
  sub_2AB1B50((__int64)v153);
  sub_2AB1B50((__int64)&v147);
  sub_2AB1B50((__int64)v174);
  sub_2AB1B50((__int64)v172);
  sub_2AB1B50((__int64)v169);
  sub_2AB1B50((__int64)v166);
  v35 = v144;
  v126 = &v144[8 * (unsigned int)v145];
  if ( v126 == v144 )
    goto LABEL_90;
  v132 = v144;
  v36 = v140;
  v129 = (__int64)(a1 + 74);
  v125 = 0;
  do
  {
    v118 = *(_QWORD *)v132;
    v131 = *(_QWORD *)(*(_QWORD *)v132 + 80LL);
    v138 = sub_2BFA260(v131, *(_QWORD *)v132 + 24LL, v32, (__int64)v36, v33, v34);
    v119 = *(_QWORD *)(*(_QWORD *)(v118 + 136) + 40LL);
    if ( (*(_BYTE *)(v119 + 7) & 0x10) != 0 )
    {
      v159 = 265;
      v158[0].m128i_i32[0] = v125;
      v147.m128i_i64[0] = (__int64)sub_BD5D20(v119);
      v147.m128i_i64[1] = v120;
      v149 = 773;
      v148 = ".";
      sub_9C6370(v166, &v147, v158, v121, v122, v123);
      ++v125;
    }
    else
    {
      v167 = 257;
    }
    sub_CA0F50((__int64 *)&v141, (void **)v166);
    v37 = *(_QWORD **)(v138 + 16);
    if ( v141 == v143 )
    {
      v124 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)v37 = v143[0];
        else
          memcpy(v37, v143, n);
        v124 = n;
        v37 = *(_QWORD **)(v138 + 16);
      }
      *(_QWORD *)(v138 + 24) = v124;
      *((_BYTE *)v37 + v124) = 0;
      v37 = v141;
    }
    else
    {
      v38 = n;
      if ( v37 == (_QWORD *)(v138 + 32) )
      {
        *(_QWORD *)(v138 + 16) = v141;
        *(_QWORD *)(v138 + 24) = v38;
        *(_QWORD *)(v138 + 32) = v143[0];
      }
      else
      {
        *(_QWORD *)(v138 + 16) = v141;
        v39 = *(_QWORD *)(v138 + 32);
        *(_QWORD *)(v138 + 24) = v38;
        *(_QWORD *)(v138 + 32) = v143[0];
        if ( v37 )
        {
          v141 = v37;
          v143[0] = v39;
          goto LABEL_40;
        }
      }
      v37 = v143;
      v141 = v143;
    }
LABEL_40:
    n = 0;
    *(_BYTE *)v37 = 0;
    if ( v141 != v143 )
      j_j___libc_free_0((unsigned __int64)v141);
    v40 = sub_B458E0((unsigned int)**(unsigned __int8 **)(v118 + 136) - 29);
    v159 = 257;
    if ( *v40 )
    {
      v158[0].m128i_i64[0] = (__int64)v40;
      LOBYTE(v159) = 3;
    }
    v147.m128i_i64[0] = (__int64)"pred.";
    v149 = 259;
    sub_9C6370(v166, &v147, v158, v41, v42, v43);
    sub_CA0F50(v139, (void **)v166);
    v44 = *(_QWORD *)(*(_QWORD *)(v118 + 48) + 8LL * (unsigned int)(*(_DWORD *)(v118 + 56) - 1));
    v45 = sub_2BF0490(v44);
    if ( v45 )
    {
      v147.m128i_i64[0] = *(_QWORD *)(v45 + 88);
      if ( v147.m128i_i64[0] )
        sub_2C25AB0(v147.m128i_i64);
    }
    else
    {
      v147.m128i_i64[0] = 0;
    }
    v47 = (_QWORD *)sub_22077B0(0x60u);
    if ( v47 )
    {
      v166[0].m128i_i64[0] = v147.m128i_i64[0];
      if ( v147.m128i_i64[0] )
        sub_2C25AB0(v166[0].m128i_i64);
      v158[0].m128i_i64[0] = v44;
      sub_2AAF310((__int64)v47, 0, v158[0].m128i_i64, 1, v166[0].m128i_i64, v46);
      sub_9C6650(v166);
      *v47 = &unk_4A245F8;
      v47[5] = &unk_4A24638;
    }
    sub_9C6650(&v147);
    v149 = 260;
    v158[0].m128i_i64[0] = (__int64)".entry";
    v147.m128i_i64[0] = (__int64)v139;
    v159 = 259;
    sub_9C6370(v166, &v147, v158, v48, v49, v50);
    v51 = sub_2C2B270((__int64)a1, (void **)v166, v47);
    v52 = *(_BYTE *)(v118 + 160);
    v136 = v51;
    v127 = *(unsigned __int8 **)(v118 + 136);
    v128 = *(_QWORD **)(v118 + 48);
    v133 = &v128[*(unsigned int *)(v118 + 56) - 1];
    v53 = sub_22077B0(0xA8u);
    v57 = v53;
    if ( v53 )
    {
      sub_2ABDBC0(v53, 9, v128, v133, v127, v56);
      *(_BYTE *)(v57 + 160) = v52;
      *(_BYTE *)(v57 + 161) = 0;
      *(_QWORD *)v57 = &unk_4A237B0;
      *(_QWORD *)(v57 + 40) = &unk_4A237F8;
      *(_QWORD *)(v57 + 96) = &unk_4A23830;
    }
    v58 = 0;
    v158[0].m128i_i64[0] = (__int64)".if";
    v149 = 260;
    v147.m128i_i64[0] = (__int64)v139;
    v159 = 259;
    sub_9C6370(v166, &v147, v158, v54, v55, v56);
    v134 = sub_2C2B270((__int64)a1, (void **)v166, (_QWORD *)v57);
    if ( *(_DWORD *)(v118 + 120) )
    {
      v141 = *(_QWORD **)(v57 + 88);
      if ( v141 )
        sub_2C25AB0((__int64 *)&v141);
      v59 = v57 + 96;
      v58 = (_QWORD *)sub_22077B0(0x98u);
      if ( v58 )
      {
        v147.m128i_i64[0] = (__int64)v141;
        if ( v141 )
        {
          sub_2C25AB0(v147.m128i_i64);
          v158[0].m128i_i64[0] = v59;
          v166[0].m128i_i64[0] = v147.m128i_i64[0];
          if ( v147.m128i_i64[0] )
            sub_2C25AB0(v166[0].m128i_i64);
        }
        else
        {
          v158[0].m128i_i64[0] = v59;
          v166[0].m128i_i64[0] = 0;
        }
        sub_2AAF310((__int64)v58, 28, v158[0].m128i_i64, 1, v166[0].m128i_i64, v60);
        sub_9C6650(v166);
        sub_2BF0340((__int64)(v58 + 12), 1, 0, (__int64)v58, (__int64)(v58 + 12), v61);
        *v58 = &unk_4A231C8;
        v58[5] = &unk_4A23200;
        v58[12] = &unk_4A23238;
        sub_9C6650(&v147);
        *v58 = &unk_4A24670;
        v58[5] = &unk_4A246B0;
        v58[12] = &unk_4A246E8;
        sub_9C6650(&v141);
        v62 = (__int64)(v58 + 12);
      }
      else
      {
        sub_9C6650(&v141);
        v62 = 0;
      }
      sub_2BF1250(v118 + 96, v62);
      sub_2AAED30((__int64)(v58 + 5), 0, v59);
    }
    sub_2C19E60((__int64 *)v118);
    v158[0].m128i_i64[0] = (__int64)".continue";
    v149 = 260;
    v147.m128i_i64[0] = (__int64)v139;
    v159 = 259;
    sub_9C6370(v166, &v147, v158, v63, v64, v65);
    v70 = sub_22077B0(0x80u);
    if ( v70 )
    {
      sub_CA0F50((__int64 *)&v141, (void **)v166);
      *(_BYTE *)(v70 + 8) = 1;
      v71 = v141;
      v72 = n;
      *(_QWORD *)v70 = &unk_4A23970;
      *(_QWORD *)(v70 + 16) = v70 + 32;
      sub_2C256A0((__int64 *)(v70 + 16), v71, (__int64)&v71[v72]);
      v67 = 0x100000000LL;
      *(_QWORD *)(v70 + 48) = 0;
      *(_QWORD *)(v70 + 56) = v70 + 72;
      v73 = v141;
      *(_QWORD *)(v70 + 64) = 0x100000000LL;
      *(_QWORD *)(v70 + 80) = v70 + 96;
      *(_QWORD *)(v70 + 88) = 0x100000000LL;
      *(_QWORD *)(v70 + 104) = 0;
      if ( v73 != v143 )
        j_j___libc_free_0((unsigned __int64)v73);
      *(_QWORD *)v70 = &unk_4A23A00;
      *(_QWORD *)(v70 + 120) = v70 + 112;
      v66 = (v70 + 112) | 4;
      *(_QWORD *)(v70 + 112) = v66;
      if ( v58 )
      {
        v58[4] = v70 + 112;
        v74 = v58[3];
        v67 = (v70 + 112) & 0xFFFFFFFFFFFFFFF8LL;
        v58[10] = v70;
        *(_QWORD *)(v67 + 8) = v58 + 3;
        v58[3] = v67 | v74 & 7;
        v66 = *(_QWORD *)(v70 + 112) & 7LL | (unsigned __int64)(v58 + 3);
        *(_QWORD *)(v70 + 112) = v66;
      }
    }
    sub_2AB9570(v129, v70, v66, v67, v68, v69);
    v75 = sub_22077B0(0x88u);
    v80 = v75;
    if ( v75 )
    {
      v81 = (_BYTE *)v139[0];
      *(_BYTE *)(v75 + 8) = 0;
      v82 = v139[1];
      *(_QWORD *)v75 = &unk_4A23970;
      *(_QWORD *)(v75 + 16) = v75 + 32;
      sub_2C256A0((__int64 *)(v75 + 16), v81, (__int64)&v81[v82]);
      *(_QWORD *)(v80 + 64) = 0x100000000LL;
      *(_QWORD *)(v80 + 56) = v80 + 72;
      *(_QWORD *)(v80 + 80) = v80 + 96;
      *(_QWORD *)(v80 + 48) = 0;
      *(_QWORD *)(v80 + 88) = 0x100000000LL;
      *(_QWORD *)v80 = &unk_4A23A38;
      *(_QWORD *)(v80 + 104) = 0;
      *(_QWORD *)(v80 + 112) = v136;
      *(_QWORD *)(v136 + 48) = v80;
      *(_QWORD *)(v80 + 120) = v70;
      *(_BYTE *)(v80 + 128) = 1;
      *(_QWORD *)(v70 + 48) = v80;
    }
    sub_2AB9570(v129, v80, v76, v77, v78, v79);
    sub_2AB9570(v136 + 80, v134, v83, v136, v84, v85);
    sub_2AB9570(v136 + 80, v70, v86, v87, v88, v89);
    sub_2AB9570(v134 + 56, v136, v90, v91, v92, v93);
    sub_2AB9570(v70 + 56, v136, v94, v95, v96, v97);
    *(_QWORD *)(v134 + 48) = *(_QWORD *)(v136 + 48);
    *(_QWORD *)(v70 + 48) = *(_QWORD *)(v136 + 48);
    sub_2AB9570(v134 + 80, v70, v98, v136, v99, v100);
    sub_2AB9570(v70 + 56, v134, v101, v102, v103, v104);
    if ( (_QWORD *)v139[0] != v140 )
      j_j___libc_free_0(v139[0]);
    v105 = *(_QWORD *)(v131 + 48);
    v158[0].m128i_i64[0] = v131;
    *(_QWORD *)(v80 + 48) = v105;
    v166[0].m128i_i64[0] = v138;
    sub_2C25750(*(_QWORD **)(v131 + 80), *(_QWORD *)(v131 + 80) + 8LL * *(unsigned int *)(v131 + 88), v166[0].m128i_i64);
    v106 = sub_2C25750(
             *(_QWORD **)(v138 + 56),
             *(_QWORD *)(v138 + 56) + 8LL * *(unsigned int *)(v138 + 64),
             v158[0].m128i_i64);
    v113 = ((__int64)v106 - v112) >> 3;
    if ( (_DWORD)v110 == -1 )
    {
      sub_2AB9570(v131 + 80, v80, v107, v108, v109, v110);
    }
    else
    {
      v110 = (unsigned int)v110;
      *(_QWORD *)(v111 + 8LL * (unsigned int)v110) = v80;
    }
    sub_2AB9570(v80 + 56, v131, v107, v108, v109, v110);
    sub_2AB9570(v80 + 80, v138, v114, v115, v116, v117);
    if ( (_DWORD)v113 == -1 )
      sub_2AB9570(v138 + 56, v80, v32, (__int64)v36, v33, v34);
    else
      *(_QWORD *)(*(_QWORD *)(v138 + 56) + 8LL * (unsigned int)v113) = v80;
    v132 += 8;
  }
  while ( v126 != v132 );
  v35 = v144;
LABEL_90:
  if ( v35 != v146 )
    _libc_free((unsigned __int64)v35);
}
