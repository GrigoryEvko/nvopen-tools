// Function: sub_2B43390
// Address: 0x2b43390
//
__int64 __fastcall sub_2B43390(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r9
  unsigned __int64 *v31; // r14
  __int64 v32; // r8
  unsigned __int64 *v33; // r15
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r9
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  unsigned __int64 *v41; // r15
  unsigned __int64 *v42; // r13
  __int64 v43; // rdi
  int v44; // ecx
  unsigned int v45; // eax
  char *v46; // rsi
  char *v47; // r14
  int v48; // ecx
  int v49; // r8d
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rdx
  __int64 *v53; // rax
  int v54; // esi
  unsigned int v55; // edx
  __int64 *v56; // rcx
  char *v57; // r8
  unsigned __int8 v58; // al
  __int64 v59; // rdi
  unsigned __int64 v60; // rax
  __int64 v61; // rdx
  unsigned __int64 *v62; // r14
  unsigned __int64 *v63; // r13
  unsigned __int64 *v64; // r14
  unsigned __int64 *v65; // r13
  unsigned __int64 *v66; // r14
  unsigned __int64 *v67; // r13
  unsigned __int64 *v68; // r14
  unsigned __int64 *v69; // r13
  unsigned __int64 *v70; // r14
  unsigned __int64 *v71; // r13
  unsigned __int64 *v72; // r14
  unsigned __int64 *v73; // r13
  unsigned int v74; // r13d
  unsigned __int64 *v75; // rbx
  unsigned __int64 *v76; // r12
  int v78; // eax
  unsigned __int64 v79; // rdx
  unsigned __int64 v80; // rax
  int v81; // ecx
  int v82; // r9d
  __int64 v84; // [rsp+30h] [rbp-EB0h]
  unsigned __int64 v85[2]; // [rsp+38h] [rbp-EA8h] BYREF
  char v86; // [rsp+48h] [rbp-E98h] BYREF
  __int64 v87; // [rsp+88h] [rbp-E58h] BYREF
  __int64 v88; // [rsp+98h] [rbp-E48h]
  char *v89; // [rsp+A8h] [rbp-E38h]
  char v90; // [rsp+B8h] [rbp-E28h] BYREF
  char *v91; // [rsp+C8h] [rbp-E18h]
  char v92; // [rsp+D8h] [rbp-E08h] BYREF
  char *v93; // [rsp+108h] [rbp-DD8h]
  char v94; // [rsp+118h] [rbp-DC8h] BYREF
  unsigned __int64 *v95; // [rsp+128h] [rbp-DB8h]
  unsigned int v96; // [rsp+130h] [rbp-DB0h]
  char v97; // [rsp+138h] [rbp-DA8h] BYREF
  unsigned int v98; // [rsp+1F0h] [rbp-CF0h]
  __int64 v99; // [rsp+200h] [rbp-CE0h]
  unsigned __int64 v100[2]; // [rsp+208h] [rbp-CD8h] BYREF
  char v101; // [rsp+218h] [rbp-CC8h] BYREF
  __int64 v102; // [rsp+258h] [rbp-C88h] BYREF
  __int64 v103; // [rsp+268h] [rbp-C78h]
  char *v104; // [rsp+278h] [rbp-C68h]
  char v105; // [rsp+288h] [rbp-C58h] BYREF
  char *v106; // [rsp+298h] [rbp-C48h]
  char v107; // [rsp+2A8h] [rbp-C38h] BYREF
  char *v108; // [rsp+2D8h] [rbp-C08h]
  char v109; // [rsp+2E8h] [rbp-BF8h] BYREF
  unsigned __int64 *v110; // [rsp+2F8h] [rbp-BE8h]
  unsigned int v111; // [rsp+300h] [rbp-BE0h]
  char v112; // [rsp+308h] [rbp-BD8h] BYREF
  unsigned int v113; // [rsp+3C0h] [rbp-B20h]
  __int64 v114; // [rsp+3D0h] [rbp-B10h]
  unsigned __int64 v115[2]; // [rsp+3D8h] [rbp-B08h] BYREF
  char v116; // [rsp+3E8h] [rbp-AF8h] BYREF
  __int64 v117; // [rsp+428h] [rbp-AB8h] BYREF
  __int64 v118; // [rsp+438h] [rbp-AA8h]
  char *v119; // [rsp+448h] [rbp-A98h]
  char v120; // [rsp+458h] [rbp-A88h] BYREF
  char *v121; // [rsp+468h] [rbp-A78h]
  char v122; // [rsp+478h] [rbp-A68h] BYREF
  char *v123; // [rsp+4A8h] [rbp-A38h]
  char v124; // [rsp+4B8h] [rbp-A28h] BYREF
  unsigned __int64 *v125; // [rsp+4C8h] [rbp-A18h]
  unsigned int v126; // [rsp+4D0h] [rbp-A10h]
  char v127; // [rsp+4D8h] [rbp-A08h] BYREF
  unsigned int v128; // [rsp+590h] [rbp-950h]
  __int64 v129; // [rsp+5A0h] [rbp-940h]
  unsigned __int64 v130[2]; // [rsp+5A8h] [rbp-938h] BYREF
  char v131; // [rsp+5B8h] [rbp-928h] BYREF
  __int64 v132; // [rsp+5F8h] [rbp-8E8h] BYREF
  __int64 v133; // [rsp+608h] [rbp-8D8h]
  char *v134; // [rsp+618h] [rbp-8C8h]
  char v135; // [rsp+628h] [rbp-8B8h] BYREF
  char *v136; // [rsp+638h] [rbp-8A8h]
  char v137; // [rsp+648h] [rbp-898h] BYREF
  char *v138; // [rsp+678h] [rbp-868h]
  char v139; // [rsp+688h] [rbp-858h] BYREF
  unsigned __int64 *v140; // [rsp+698h] [rbp-848h]
  unsigned int v141; // [rsp+6A0h] [rbp-840h]
  char v142; // [rsp+6A8h] [rbp-838h] BYREF
  unsigned int v143; // [rsp+760h] [rbp-780h]
  __int64 v144; // [rsp+770h] [rbp-770h]
  unsigned __int64 v145[2]; // [rsp+778h] [rbp-768h] BYREF
  char v146; // [rsp+788h] [rbp-758h] BYREF
  __int64 v147; // [rsp+7C8h] [rbp-718h] BYREF
  __int64 v148; // [rsp+7D8h] [rbp-708h]
  char *v149; // [rsp+7E8h] [rbp-6F8h]
  char v150; // [rsp+7F8h] [rbp-6E8h] BYREF
  char *v151; // [rsp+808h] [rbp-6D8h]
  char v152; // [rsp+818h] [rbp-6C8h] BYREF
  char *v153; // [rsp+848h] [rbp-698h]
  char v154; // [rsp+858h] [rbp-688h] BYREF
  unsigned __int64 *v155; // [rsp+868h] [rbp-678h]
  unsigned int v156; // [rsp+870h] [rbp-670h]
  char v157; // [rsp+878h] [rbp-668h] BYREF
  unsigned int v158; // [rsp+930h] [rbp-5B0h]
  __int64 v159; // [rsp+940h] [rbp-5A0h]
  unsigned __int64 v160[2]; // [rsp+948h] [rbp-598h] BYREF
  char v161; // [rsp+958h] [rbp-588h] BYREF
  __int64 v162; // [rsp+998h] [rbp-548h] BYREF
  __int64 v163; // [rsp+9A8h] [rbp-538h]
  char *v164; // [rsp+9B8h] [rbp-528h]
  char v165; // [rsp+9C8h] [rbp-518h] BYREF
  char *v166; // [rsp+9D8h] [rbp-508h]
  char v167; // [rsp+9E8h] [rbp-4F8h] BYREF
  char *v168; // [rsp+A18h] [rbp-4C8h]
  char v169; // [rsp+A28h] [rbp-4B8h] BYREF
  unsigned __int64 *v170; // [rsp+A38h] [rbp-4A8h]
  unsigned int v171; // [rsp+A40h] [rbp-4A0h]
  char v172; // [rsp+A48h] [rbp-498h] BYREF
  unsigned int v173; // [rsp+B00h] [rbp-3E0h]
  __int64 v174; // [rsp+B10h] [rbp-3D0h]
  char *v175; // [rsp+B18h] [rbp-3C8h] BYREF
  char v176; // [rsp+B28h] [rbp-3B8h] BYREF
  __int64 v177; // [rsp+B68h] [rbp-378h] BYREF
  __int64 v178; // [rsp+B78h] [rbp-368h]
  char *v179; // [rsp+B88h] [rbp-358h]
  char v180; // [rsp+B98h] [rbp-348h] BYREF
  char *v181; // [rsp+BA8h] [rbp-338h]
  char v182; // [rsp+BB8h] [rbp-328h] BYREF
  int v183; // [rsp+BE0h] [rbp-300h]
  char *v184; // [rsp+BE8h] [rbp-2F8h]
  char v185; // [rsp+BF8h] [rbp-2E8h] BYREF
  unsigned __int64 *v186; // [rsp+C08h] [rbp-2D8h]
  unsigned int v187; // [rsp+C10h] [rbp-2D0h]
  char v188; // [rsp+C18h] [rbp-2C8h] BYREF
  unsigned int v189; // [rsp+CD0h] [rbp-210h]
  __int64 v190; // [rsp+CE0h] [rbp-200h] BYREF
  unsigned __int64 v191[2]; // [rsp+CE8h] [rbp-1F8h] BYREF
  _BYTE v192[64]; // [rsp+CF8h] [rbp-1E8h] BYREF
  _QWORD v193[2]; // [rsp+D38h] [rbp-1A8h] BYREF
  __int64 v194; // [rsp+D48h] [rbp-198h]
  _BYTE *v195; // [rsp+D58h] [rbp-188h]
  _BYTE v196[16]; // [rsp+D68h] [rbp-178h] BYREF
  _BYTE *v197; // [rsp+D78h] [rbp-168h]
  _BYTE v198[48]; // [rsp+D88h] [rbp-158h] BYREF
  _BYTE *v199; // [rsp+DB8h] [rbp-128h]
  _BYTE v200[16]; // [rsp+DC8h] [rbp-118h] BYREF
  unsigned __int64 *v201; // [rsp+DD8h] [rbp-108h]
  unsigned int v202; // [rsp+DE0h] [rbp-100h]
  _BYTE v203[184]; // [rsp+DE8h] [rbp-F8h] BYREF
  unsigned int v204; // [rsp+EA0h] [rbp-40h]

  v84 = *a1;
  sub_2B43020((__int64)v85, a1[1], a3, a4, a5, a6);
  v7 = *(_QWORD *)(a2 + 16);
  v98 = *(_DWORD *)a1[2];
  v99 = v84;
  sub_2B43020((__int64)v100, (__int64)v85, v8, v9, v10, v11);
  v113 = v98;
  v114 = v99;
  sub_2B43020((__int64)v115, (__int64)v100, v12, v13, v14, v15);
  v128 = v113;
  v129 = v114;
  sub_2B43020((__int64)v130, (__int64)v115, v16, v17, v18, v19);
  v143 = v128;
  v190 = v129;
  sub_2B43020((__int64)v191, (__int64)v130, v20, v21, v22, v23);
  v204 = v143;
  v144 = v190;
  sub_2B43020((__int64)v145, (__int64)v191, v24, v25, v26, v27);
  v31 = v201;
  v158 = v204;
  v32 = 10LL * v202;
  v33 = &v201[v32];
  if ( v201 != &v201[v32] )
  {
    do
    {
      v33 -= 10;
      if ( (unsigned __int64 *)*v33 != v33 + 2 )
        _libc_free(*v33);
    }
    while ( v31 != v33 );
    v33 = v201;
  }
  if ( v33 != (unsigned __int64 *)v203 )
    _libc_free((unsigned __int64)v33);
  if ( v199 != v200 )
    _libc_free((unsigned __int64)v199);
  if ( v197 != v198 )
    _libc_free((unsigned __int64)v197);
  if ( v195 != v196 )
    _libc_free((unsigned __int64)v195);
  LOBYTE(v29) = v194 != 0;
  LOBYTE(v28) = v194 != -4096;
  if ( ((unsigned __int8)v28 & (v194 != 0)) != 0 && v194 != -8192 )
    sub_BD60C0(v193);
  if ( (_BYTE *)v191[0] != v192 )
    _libc_free(v191[0]);
  v159 = v144;
  sub_2B43020((__int64)v160, (__int64)v145, v28, v29, (__int64)v160, v30);
  v173 = v158;
  v190 = v159;
  sub_2B43020((__int64)v191, (__int64)v160, v34, v35, (__int64)v160, v36);
  v204 = v173;
  v174 = v190;
  sub_2B43020((__int64)&v175, (__int64)v191, v37, v38, v39, v40);
  v41 = v201;
  v189 = v204;
  v42 = &v201[10 * v202];
  if ( v201 != v42 )
  {
    do
    {
      v42 -= 10;
      if ( (unsigned __int64 *)*v42 != v42 + 2 )
        _libc_free(*v42);
    }
    while ( v41 != v42 );
    v42 = v201;
  }
  if ( v42 != (unsigned __int64 *)v203 )
    _libc_free((unsigned __int64)v42);
  if ( v199 != v200 )
    _libc_free((unsigned __int64)v199);
  if ( v197 != v198 )
    _libc_free((unsigned __int64)v197);
  if ( v195 != v196 )
    _libc_free((unsigned __int64)v195);
  if ( v194 != 0 && v194 != -4096 && v194 != -8192 )
    sub_BD60C0(v193);
  if ( (_BYTE *)v191[0] != v192 )
    _libc_free(v191[0]);
  while ( v7 )
  {
    v47 = *(char **)(v7 + 24);
    if ( (*(_BYTE *)(v174 + 88) & 1) != 0 )
    {
      v43 = v174 + 96;
      v44 = 3;
    }
    else
    {
      v48 = *(_DWORD *)(v174 + 104);
      v43 = *(_QWORD *)(v174 + 96);
      if ( !v48 )
        goto LABEL_47;
      v44 = v48 - 1;
    }
    v45 = v44 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
    v46 = *(char **)(v43 + 72LL * v45);
    if ( v47 == v46 )
      goto LABEL_41;
    v49 = 1;
    while ( v46 != (char *)-4096LL )
    {
      v45 = v44 & (v49 + v45);
      v46 = *(char **)(v43 + 72LL * v45);
      if ( v47 == v46 )
        goto LABEL_41;
      ++v49;
    }
LABEL_47:
    if ( !v183 )
    {
      v50 = *(_QWORD *)(v174 + 3272);
      if ( v50 )
      {
        if ( (*(_BYTE *)(v50 + 8) & 1) != 0 )
        {
          v51 = v50 + 16;
          v54 = 3;
          v53 = (__int64 *)(v50 + 48);
LABEL_52:
          v55 = v54 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
          v56 = (__int64 *)(v51 + 8LL * v55);
          v57 = (char *)*v56;
          if ( v47 == (char *)*v56 )
          {
LABEL_53:
            if ( v56 != v53 )
              goto LABEL_41;
          }
          else
          {
            v81 = 1;
            while ( v57 != (char *)-4096LL )
            {
              v82 = v81 + 1;
              v55 = v54 & (v81 + v55);
              v56 = (__int64 *)(v51 + 8LL * v55);
              v57 = (char *)*v56;
              if ( v47 == (char *)*v56 )
                goto LABEL_53;
              v81 = v82;
            }
          }
          goto LABEL_54;
        }
        v51 = *(_QWORD *)(v50 + 16);
        v52 = *(unsigned int *)(v50 + 24);
        v53 = (__int64 *)(v51 + 8 * v52);
        if ( (_DWORD)v52 )
        {
          v54 = v52 - 1;
          goto LABEL_52;
        }
      }
    }
LABEL_54:
    v58 = *v47;
    if ( (unsigned __int8)*v47 > 0x1Cu && (v58 == 82 || v58 == 83) )
      break;
    v59 = *((_QWORD *)v47 + 1);
    v60 = *(unsigned __int8 *)(v59 + 8);
    if ( (unsigned __int8)v60 > 0xCu || (v61 = 4143, !_bittest64(&v61, v60)) )
    {
      if ( (v60 & 0xFB) != 0xA && (v60 & 0xFD) != 4 )
      {
        if ( (unsigned __int8)(v60 - 15) > 3u && (_BYTE)v60 != 20 || !(unsigned __int8)sub_BCEBA0(v59, 0) )
          break;
        v59 = *((_QWORD *)v47 + 1);
      }
    }
    if ( sub_BCEA30(v59) )
      break;
    v190 = sub_9208B0(*(_QWORD *)(v174 + 3344), *((_QWORD *)v47 + 1));
    v191[0] = v79;
    v80 = sub_CA1930(&v190);
    if ( v80 > v189 )
      break;
LABEL_41:
    v7 = *(_QWORD *)(v7 + 8);
  }
  v62 = v186;
  v63 = &v186[10 * v187];
  if ( v186 != v63 )
  {
    do
    {
      v63 -= 10;
      if ( (unsigned __int64 *)*v63 != v63 + 2 )
        _libc_free(*v63);
    }
    while ( v62 != v63 );
    v63 = v186;
  }
  if ( v63 != (unsigned __int64 *)&v188 )
    _libc_free((unsigned __int64)v63);
  if ( v184 != &v185 )
    _libc_free((unsigned __int64)v184);
  if ( v181 != &v182 )
    _libc_free((unsigned __int64)v181);
  if ( v179 != &v180 )
    _libc_free((unsigned __int64)v179);
  if ( v178 != 0 && v178 != -4096 && v178 != -8192 )
    sub_BD60C0(&v177);
  if ( v175 != &v176 )
    _libc_free((unsigned __int64)v175);
  v64 = v170;
  v65 = &v170[10 * v171];
  if ( v170 != v65 )
  {
    do
    {
      v65 -= 10;
      if ( (unsigned __int64 *)*v65 != v65 + 2 )
        _libc_free(*v65);
    }
    while ( v64 != v65 );
    v65 = v170;
  }
  if ( v65 != (unsigned __int64 *)&v172 )
    _libc_free((unsigned __int64)v65);
  if ( v168 != &v169 )
    _libc_free((unsigned __int64)v168);
  if ( v166 != &v167 )
    _libc_free((unsigned __int64)v166);
  if ( v164 != &v165 )
    _libc_free((unsigned __int64)v164);
  if ( v163 != 0 && v163 != -4096 && v163 != -8192 )
    sub_BD60C0(&v162);
  if ( (char *)v160[0] != &v161 )
    _libc_free(v160[0]);
  v66 = v155;
  v67 = &v155[10 * v156];
  if ( v155 != v67 )
  {
    do
    {
      v67 -= 10;
      if ( (unsigned __int64 *)*v67 != v67 + 2 )
        _libc_free(*v67);
    }
    while ( v66 != v67 );
    v67 = v155;
  }
  if ( v67 != (unsigned __int64 *)&v157 )
    _libc_free((unsigned __int64)v67);
  if ( v153 != &v154 )
    _libc_free((unsigned __int64)v153);
  if ( v151 != &v152 )
    _libc_free((unsigned __int64)v151);
  if ( v149 != &v150 )
    _libc_free((unsigned __int64)v149);
  if ( v148 != 0 && v148 != -4096 && v148 != -8192 )
    sub_BD60C0(&v147);
  if ( (char *)v145[0] != &v146 )
    _libc_free(v145[0]);
  v68 = v140;
  v69 = &v140[10 * v141];
  if ( v140 != v69 )
  {
    do
    {
      v69 -= 10;
      if ( (unsigned __int64 *)*v69 != v69 + 2 )
        _libc_free(*v69);
    }
    while ( v68 != v69 );
    v69 = v140;
  }
  if ( v69 != (unsigned __int64 *)&v142 )
    _libc_free((unsigned __int64)v69);
  if ( v138 != &v139 )
    _libc_free((unsigned __int64)v138);
  if ( v136 != &v137 )
    _libc_free((unsigned __int64)v136);
  if ( v134 != &v135 )
    _libc_free((unsigned __int64)v134);
  if ( v133 != -4096 && v133 != 0 && v133 != -8192 )
    sub_BD60C0(&v132);
  if ( (char *)v130[0] != &v131 )
    _libc_free(v130[0]);
  v70 = v125;
  v71 = &v125[10 * v126];
  if ( v125 != v71 )
  {
    do
    {
      v71 -= 10;
      if ( (unsigned __int64 *)*v71 != v71 + 2 )
        _libc_free(*v71);
    }
    while ( v70 != v71 );
    v71 = v125;
  }
  if ( v71 != (unsigned __int64 *)&v127 )
    _libc_free((unsigned __int64)v71);
  if ( v123 != &v124 )
    _libc_free((unsigned __int64)v123);
  if ( v121 != &v122 )
    _libc_free((unsigned __int64)v121);
  if ( v119 != &v120 )
    _libc_free((unsigned __int64)v119);
  if ( v118 != -4096 && v118 != 0 && v118 != -8192 )
    sub_BD60C0(&v117);
  if ( (char *)v115[0] != &v116 )
    _libc_free(v115[0]);
  v72 = v110;
  v73 = &v110[10 * v111];
  if ( v110 != v73 )
  {
    do
    {
      v73 -= 10;
      if ( (unsigned __int64 *)*v73 != v73 + 2 )
        _libc_free(*v73);
    }
    while ( v72 != v73 );
    v73 = v110;
  }
  if ( v73 != (unsigned __int64 *)&v112 )
    _libc_free((unsigned __int64)v73);
  if ( v108 != &v109 )
    _libc_free((unsigned __int64)v108);
  if ( v106 != &v107 )
    _libc_free((unsigned __int64)v106);
  if ( v104 != &v105 )
    _libc_free((unsigned __int64)v104);
  if ( v103 != 0 && v103 != -4096 && v103 != -8192 )
    sub_BD60C0(&v102);
  if ( (char *)v100[0] != &v101 )
    _libc_free(v100[0]);
  v74 = 0;
  if ( v7 )
  {
    LOBYTE(v78) = sub_2B1EE10((__int64 *)a1[3], (_BYTE *)a2, (unsigned int *)a1[2]);
    v74 = v78 ^ 1;
  }
  v75 = v95;
  v76 = &v95[10 * v96];
  if ( v95 != v76 )
  {
    do
    {
      v76 -= 10;
      if ( (unsigned __int64 *)*v76 != v76 + 2 )
        _libc_free(*v76);
    }
    while ( v75 != v76 );
    v76 = v95;
  }
  if ( v76 != (unsigned __int64 *)&v97 )
    _libc_free((unsigned __int64)v76);
  if ( v93 != &v94 )
    _libc_free((unsigned __int64)v93);
  if ( v91 != &v92 )
    _libc_free((unsigned __int64)v91);
  if ( v89 != &v90 )
    _libc_free((unsigned __int64)v89);
  if ( v88 != 0 && v88 != -4096 && v88 != -8192 )
    sub_BD60C0(&v87);
  if ( (char *)v85[0] != &v86 )
    _libc_free(v85[0]);
  return v74;
}
