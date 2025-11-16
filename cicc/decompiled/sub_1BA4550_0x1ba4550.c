// Function: sub_1BA4550
// Address: 0x1ba4550
//
__int64 __fastcall sub_1BA4550(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 *v5; // rax
  __int64 v6; // r15
  _QWORD *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r13
  char v11; // al
  unsigned __int64 v12; // rbx
  _QWORD *v13; // rdx
  __int64 *v14; // rbx
  unsigned __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rax
  size_t v18; // rdi
  unsigned int v19; // esi
  __int64 v20; // rax
  __int64 *v21; // rsi
  __int64 v22; // r9
  char v23; // r8
  unsigned int v24; // edi
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 *v30; // r14
  __int64 v31; // rax
  unsigned int v32; // ebx
  __int64 *v33; // r15
  __int64 v34; // r14
  __int64 v35; // r12
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rsi
  __int64 v41; // r13
  __int64 v42; // rax
  __int64 v43; // rdi
  int v44; // edx
  __int64 v45; // rdi
  __int64 v46; // r8
  int v47; // edx
  unsigned int v48; // esi
  __int64 *v49; // rax
  __int64 v50; // r10
  __int64 v51; // rax
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // rsi
  unsigned int v56; // esi
  _QWORD *v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rdi
  int v60; // eax
  __int64 v61; // rcx
  __int64 v62; // rsi
  int v63; // edi
  unsigned int v64; // edx
  __int64 *v65; // rax
  __int64 v66; // r8
  __int64 v67; // rax
  unsigned __int64 v68; // rax
  int v69; // r8d
  int v70; // r9d
  unsigned __int64 v71; // rbx
  char *v72; // rdi
  __int64 v73; // rbx
  __int64 v74; // rax
  __int64 v75; // r14
  __int64 v76; // r13
  __int64 i; // r15
  _QWORD *v78; // rax
  __int64 v79; // rsi
  unsigned int v80; // r15d
  __int64 v81; // rbx
  __int64 *v82; // r13
  __int64 v83; // r13
  __int64 *v84; // r12
  unsigned int v85; // r14d
  unsigned int v86; // ebx
  __int64 *v87; // r15
  __int64 v88; // rax
  __int64 v89; // rax
  unsigned __int64 *v90; // r8
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 *v93; // rbx
  unsigned __int64 v94; // r13
  __int64 v95; // r14
  __int64 v96; // rax
  __int64 v97; // rcx
  __int64 v98; // r8
  __int64 v99; // r9
  __int64 v100; // rbx
  __int64 v101; // r14
  __int64 v102; // rax
  __int64 v103; // r13
  __int64 v104; // rdx
  __int64 v105; // rax
  __int64 v106; // rcx
  __int64 v107; // r8
  __int64 v108; // r9
  __int64 v109; // rdx
  __int64 v110; // r14
  __int64 v111; // r13
  __int64 v112; // rax
  __int64 v113; // r12
  __int64 v114; // rdi
  unsigned int v115; // esi
  __int64 v116; // rdx
  __int64 v117; // rax
  __int64 v118; // rcx
  __int64 v119; // r12
  __int64 v120; // r13
  __int64 *v121; // rax
  __int64 v122; // rcx
  unsigned __int64 v123; // rdx
  __int64 v124; // rdx
  __int64 *v125; // rax
  __int64 v126; // rcx
  unsigned __int64 v127; // rdx
  __int64 v128; // rdx
  int v130; // eax
  unsigned int v131; // esi
  int v132; // eax
  int v133; // eax
  __int64 *v134; // rax
  __int64 *v135; // rcx
  int v136; // ecx
  int v137; // eax
  int v138; // r9d
  __int64 v139; // [rsp+10h] [rbp-1A0h]
  __int64 v140; // [rsp+18h] [rbp-198h]
  __int64 v141; // [rsp+28h] [rbp-188h]
  int v142; // [rsp+30h] [rbp-180h]
  int v143; // [rsp+34h] [rbp-17Ch]
  _QWORD *v144; // [rsp+40h] [rbp-170h]
  __int64 v145; // [rsp+48h] [rbp-168h]
  _QWORD *v146; // [rsp+48h] [rbp-168h]
  unsigned int v147; // [rsp+48h] [rbp-168h]
  __int64 *v148; // [rsp+50h] [rbp-160h]
  __int64 v149; // [rsp+58h] [rbp-158h]
  _QWORD *v150; // [rsp+60h] [rbp-150h]
  _QWORD *v151; // [rsp+60h] [rbp-150h]
  __int64 v152; // [rsp+60h] [rbp-150h]
  __int64 v153; // [rsp+60h] [rbp-150h]
  __int64 v154; // [rsp+68h] [rbp-148h] BYREF
  unsigned __int64 v155[2]; // [rsp+70h] [rbp-140h] BYREF
  _QWORD *v156; // [rsp+80h] [rbp-130h]
  unsigned __int64 v157[2]; // [rsp+90h] [rbp-120h] BYREF
  __int16 v158; // [rsp+A0h] [rbp-110h]
  const char *v159; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v160; // [rsp+B8h] [rbp-F8h]
  _WORD s[8]; // [rsp+C0h] [rbp-F0h] BYREF
  unsigned __int64 v162[2]; // [rsp+D0h] [rbp-E0h] BYREF
  _QWORD *v163; // [rsp+E0h] [rbp-D0h]
  __int64 *v164; // [rsp+E8h] [rbp-C8h]
  __int64 v165; // [rsp+F0h] [rbp-C0h]
  __int64 v166; // [rsp+F8h] [rbp-B8h]
  __int64 *v167; // [rsp+100h] [rbp-B0h]
  char v168; // [rsp+108h] [rbp-A8h]
  _QWORD v169[2]; // [rsp+110h] [rbp-A0h] BYREF
  unsigned __int64 v170; // [rsp+120h] [rbp-90h]
  char v171[120]; // [rsp+138h] [rbp-78h] BYREF

  v5 = (__int64 *)(a1 + 96);
  v6 = a1;
  v7 = *(_QWORD **)(a1 + 120);
  v154 = a2;
  v148 = v5;
  v8 = sub_1643350(v7);
  v9 = sub_159C470(v8, 0, 0);
  v10 = *(_QWORD *)(v6 + 448);
  v145 = v9;
  v11 = sub_1BA0DE0(v10 + 72, &v154, v162);
  v12 = v162[0];
  if ( v11 )
    goto LABEL_2;
  v131 = *(_DWORD *)(v10 + 96);
  v132 = *(_DWORD *)(v10 + 88);
  ++*(_QWORD *)(v10 + 72);
  v133 = v132 + 1;
  if ( 4 * v133 >= 3 * v131 )
  {
    v131 *= 2;
    goto LABEL_128;
  }
  if ( v131 - *(_DWORD *)(v10 + 92) - v133 <= v131 >> 3 )
  {
LABEL_128:
    sub_1BA42A0(v10 + 72, v131);
    sub_1BA0DE0(v10 + 72, &v154, v162);
    v12 = v162[0];
    v133 = *(_DWORD *)(v10 + 88) + 1;
  }
  *(_DWORD *)(v10 + 88) = v133;
  if ( *(_QWORD *)v12 != -8 )
    --*(_DWORD *)(v10 + 92);
  *(_QWORD *)v12 = v154;
  memset((void *)(v12 + 8), 0, 0xA8u);
  *(_QWORD *)(v12 + 8) = 6;
  *(_QWORD *)(v12 + 80) = v12 + 112;
  *(_QWORD *)(v12 + 88) = v12 + 112;
  *(_QWORD *)(v12 + 96) = 8;
LABEL_2:
  v162[0] = 6;
  v162[1] = 0;
  v163 = *(_QWORD **)(v12 + 24);
  if ( v163 != 0 && v163 + 1 != 0 && v163 != (_QWORD *)-16LL )
    sub_1649AC0(v162, *(_QWORD *)(v12 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  v164 = *(__int64 **)(v12 + 32);
  v165 = *(_QWORD *)(v12 + 40);
  v166 = *(_QWORD *)(v12 + 48);
  v167 = *(__int64 **)(v12 + 56);
  v168 = *(_BYTE *)(v12 + 64);
  sub_16CCCB0(v169, (__int64)v171, v12 + 72);
  v13 = v163;
  v155[0] = 6;
  v155[1] = 0;
  v142 = v165;
  v156 = v163;
  if ( v163 + 1 != 0 && v163 != 0 && v163 != (_QWORD *)-16LL )
  {
    sub_1649AC0(v155, v162[0] & 0xFFFFFFFFFFFFFFF8LL);
    v13 = v156;
  }
  v14 = v164;
  v149 = (__int64)v164;
  v143 = HIDWORD(v165);
  sub_1B91520(v6, v148, (__int64)v13);
  v15 = sub_157EBA0(*(_QWORD *)(v6 + 168));
  sub_17050D0(v148, v15);
  v141 = *(_QWORD *)sub_1B9C240((unsigned int *)v6, v14, 0);
  if ( v142 == 6 || v142 == 9 )
  {
    v19 = *(_DWORD *)(v6 + 88);
    v150 = v156;
    if ( v19 == 1 )
    {
      v146 = v156;
    }
    else
    {
      v159 = "minmax.ident";
      s[0] = 259;
      v150 = (_QWORD *)sub_156DA60(v148, v19, v156, (__int64 *)&v159);
      v146 = v150;
    }
  }
  else
  {
    v16 = v141;
    if ( *(_BYTE *)(v141 + 8) == 16 )
      v16 = **(_QWORD **)(v141 + 16);
    v17 = sub_1B16200(v142, v16);
    v18 = *(unsigned int *)(v6 + 88);
    v150 = (_QWORD *)v17;
    if ( (_DWORD)v18 == 1 )
    {
      v146 = v156;
    }
    else
    {
      v150 = (_QWORD *)sub_15A0390(v18, v17);
      s[0] = 257;
      v146 = (_QWORD *)sub_156D8B0(v148, (__int64)v150, (__int64)v156, v145, (__int64)&v159);
    }
  }
  v20 = sub_13FCB50(*(_QWORD *)(v6 + 8));
  v21 = (__int64 *)v154;
  v22 = v20;
  v23 = *(_BYTE *)(v154 + 23) & 0x40;
  v24 = *(_DWORD *)(v154 + 20) & 0xFFFFFFF;
  if ( v24 )
  {
    v25 = 24LL * *(unsigned int *)(v154 + 56) + 8;
    v26 = 0;
    while ( 1 )
    {
      v27 = v154 - 24LL * v24;
      if ( v23 )
        v27 = *(_QWORD *)(v154 - 8);
      if ( v22 == *(_QWORD *)(v27 + v25) )
        break;
      ++v26;
      v25 += 8;
      if ( v24 == (_DWORD)v26 )
        goto LABEL_116;
    }
    v28 = 24 * v26;
    if ( v23 )
      goto LABEL_23;
  }
  else
  {
LABEL_116:
    v28 = 0x17FFFFFFE8LL;
    if ( v23 )
    {
LABEL_23:
      v29 = *(_QWORD *)(v154 - 8);
      goto LABEL_24;
    }
  }
  v29 = v154 - 24LL * v24;
LABEL_24:
  v30 = *(__int64 **)(v29 + v28);
  if ( *(_DWORD *)(v6 + 92) )
  {
    v31 = v6;
    v32 = 0;
    v33 = v30;
    v34 = v31;
    while ( 1 )
    {
      v35 = sub_1B9C240((unsigned int *)v34, v21, v32);
      v36 = sub_1B9C240((unsigned int *)v34, v33, v32);
      v40 = (__int64)v150;
      if ( !v32 )
        v40 = (__int64)v146;
      v41 = v36;
      sub_1704F80(v35, v40, *(_QWORD *)(v34 + 168), v37, v38, v39);
      v42 = *(_QWORD *)(v34 + 24);
      v43 = 0;
      v44 = *(_DWORD *)(v42 + 24);
      if ( v44 )
      {
        v45 = *(_QWORD *)(v34 + 200);
        v46 = *(_QWORD *)(v42 + 8);
        v47 = v44 - 1;
        v48 = v47 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
        v49 = (__int64 *)(v46 + 16LL * v48);
        v50 = *v49;
        if ( v45 == *v49 )
        {
LABEL_31:
          v43 = v49[1];
        }
        else
        {
          v130 = 1;
          while ( v50 != -8 )
          {
            v136 = v130 + 1;
            v48 = v47 & (v130 + v48);
            v49 = (__int64 *)(v46 + 16LL * v48);
            v50 = *v49;
            if ( v45 == *v49 )
              goto LABEL_31;
            v130 = v136;
          }
          v43 = 0;
        }
      }
      v51 = sub_13FCB50(v43);
      ++v32;
      sub_1704F80(v35, v41, v51, v52, v53, v54);
      if ( *(_DWORD *)(v34 + 92) <= v32 )
        break;
      v21 = (__int64 *)v154;
    }
    v6 = v34;
  }
  v55 = sub_157EE30(*(_QWORD *)(v6 + 184));
  if ( v55 )
    v55 -= 24;
  sub_17050D0(v148, v55);
  sub_1B91520(v6, v148, v149);
  v56 = *(_DWORD *)(v6 + 88);
  v144 = (_QWORD *)(v6 + 288);
  if ( v56 > 1 && v167 != *(__int64 **)v154 )
  {
    v57 = sub_16463B0(v167, v56);
    v58 = *(_QWORD *)(v6 + 24);
    v59 = 0;
    v140 = (__int64)v57;
    v60 = *(_DWORD *)(v58 + 24);
    if ( v60 )
    {
      v61 = *(_QWORD *)(v6 + 200);
      v62 = *(_QWORD *)(v58 + 8);
      v63 = v60 - 1;
      v64 = (v60 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
      v65 = (__int64 *)(v62 + 16LL * v64);
      v66 = *v65;
      if ( v61 == *v65 )
      {
LABEL_40:
        v59 = v65[1];
      }
      else
      {
        v137 = 1;
        while ( v66 != -8 )
        {
          v138 = v137 + 1;
          v64 = v63 & (v137 + v64);
          v65 = (__int64 *)(v62 + 16LL * v64);
          v66 = *v65;
          if ( v61 == *v65 )
            goto LABEL_40;
          v137 = v138;
        }
        v59 = 0;
      }
    }
    v67 = sub_13FCB50(v59);
    v68 = sub_157EBA0(v67);
    sub_17050D0(v148, v68);
    v71 = *(unsigned int *)(v6 + 92);
    v72 = (char *)s;
    v159 = (const char *)s;
    v160 = 0x200000000LL;
    if ( (unsigned int)v71 > 2 )
    {
      sub_16CD150((__int64)&v159, s, v71, 8, v69, v70);
      v72 = (char *)v159;
    }
    LODWORD(v160) = v71;
    if ( 8 * v71 )
      memset(v72, 0, 8 * v71);
    if ( *(_DWORD *)(v6 + 92) )
    {
      v139 = v6;
      v73 = (__int64)v159;
      v147 = 0;
      do
      {
        v157[0] = v149;
        *(_QWORD *)(8LL * v147 + v73) = *(_QWORD *)(*sub_1B99AC0(v144, v157) + 8LL * v147);
        v158 = 257;
        v74 = sub_12AA3B0(v148, 0x24u, *(_QWORD *)&v159[8 * v147], v140, (__int64)v157);
        v151 = (_QWORD *)v74;
        v158 = 257;
        if ( v168 )
          v75 = sub_12AA3B0(v148, 0x26u, v74, v141, (__int64)v157);
        else
          v75 = sub_12AA3B0(v148, 0x25u, v74, v141, (__int64)v157);
        v73 = (__int64)v159;
        v76 = *(_QWORD *)&v159[8 * v147];
        for ( i = *(_QWORD *)(v76 + 8); i; v76 = *(_QWORD *)(v73 + 8LL * v147) )
        {
          v78 = sub_1648700(i);
          i = *(_QWORD *)(i + 8);
          if ( v151 != v78 )
          {
            sub_1648780((__int64)v78, v76, v75);
            *(_QWORD *)&v159[8 * v147] = v75;
            v73 = (__int64)v159;
          }
        }
        ++v147;
      }
      while ( *(_DWORD *)(v139 + 92) > v147 );
      v6 = v139;
    }
    v79 = sub_157EE30(*(_QWORD *)(v6 + 184));
    if ( v79 )
      v79 -= 24;
    sub_17050D0(v148, v79);
    if ( *(_DWORD *)(v6 + 92) )
    {
      v152 = v6;
      v80 = 0;
      do
      {
        v81 = v80++;
        v82 = (__int64 *)&v159[8 * v81];
        v158 = 257;
        *v82 = sub_12AA3B0(v148, 0x24u, *v82, v140, (__int64)v157);
        v83 = *(_QWORD *)&v159[8 * v81];
        v157[0] = v149;
        *(_QWORD *)(*sub_1B99AC0(v144, v157) + 8 * v81) = v83;
      }
      while ( *(_DWORD *)(v152 + 92) > v80 );
      v6 = v152;
    }
    if ( v159 != (const char *)s )
      _libc_free((unsigned __int64)v159);
  }
  v159 = (const char *)v149;
  v84 = *(__int64 **)*sub_1B99AC0(v144, (unsigned __int64 *)&v159);
  v85 = sub_1B16280(v142);
  sub_1B91520(v6, v148, (__int64)v84);
  if ( *(_DWORD *)(v6 + 92) > 1u )
  {
    v86 = 1;
    v153 = v6;
    v87 = v84;
    do
    {
      while ( 1 )
      {
        v159 = (const char *)v149;
        v90 = sub_1B99AC0(v144, (unsigned __int64 *)&v159);
        v91 = *(_QWORD *)(*v90 + 8LL * v86);
        if ( v85 - 51 <= 1 )
          break;
        ++v86;
        v159 = "bin.rdx";
        s[0] = 259;
        v88 = sub_1904E90(
                (__int64)v148,
                v85,
                v91,
                (__int64)v87,
                (__int64 *)&v159,
                0,
                *(double *)a3.m128i_i64,
                *(double *)a4.m128i_i64,
                *(double *)a5.m128i_i64);
        v87 = (__int64 *)sub_1B8ED40(v88);
        v89 = v153;
        if ( *(_DWORD *)(v153 + 92) <= v86 )
          goto LABEL_68;
      }
      v92 = *(_QWORD *)(*v90 + 8LL * v86++);
      v87 = sub_1B16290((__int64)v148, v143, v87, v92);
      v89 = v153;
    }
    while ( *(_DWORD *)(v153 + 92) > v86 );
LABEL_68:
    v84 = v87;
    v6 = v89;
  }
  if ( *(_DWORD *)(v6 + 88) <= 1u )
    goto LABEL_70;
  v134 = sub_1B1A980(
           (__int64)v148,
           *(__int64 **)(v6 + 56),
           (__int64)v162,
           v84,
           *(_BYTE *)(*(_QWORD *)(v6 + 448) + 448LL),
           a3,
           a4,
           a5);
  v93 = (__int64 *)v154;
  v84 = v134;
  if ( v167 != *(__int64 **)v154 )
  {
    s[0] = 257;
    v135 = *(__int64 **)v154;
    if ( v168 )
      v84 = (__int64 *)sub_12AA3B0(v148, 0x26u, (__int64)v134, (__int64)v135, (__int64)&v159);
    else
      v84 = (__int64 *)sub_12AA3B0(v148, 0x25u, (__int64)v134, (__int64)v135, (__int64)&v159);
LABEL_70:
    v93 = (__int64 *)v154;
  }
  v94 = sub_157EBA0(*(_QWORD *)(v6 + 176));
  s[0] = 259;
  v159 = "bc.merge.rdx";
  v95 = *v93;
  v96 = sub_1648B60(64);
  v100 = v96;
  if ( v96 )
  {
    sub_15F1EA0(v96, v95, 53, 0, 0, v94);
    *(_DWORD *)(v100 + 56) = 2;
    sub_164B780(v100, (__int64 *)&v159);
    sub_1648880(v100, *(_DWORD *)(v100 + 56), 1);
  }
  v101 = 0;
  v102 = *(unsigned int *)(v6 + 224);
  v103 = 8 * v102;
  if ( (_DWORD)v102 )
  {
    do
    {
      v104 = *(_QWORD *)(*(_QWORD *)(v6 + 216) + v101);
      v101 += 8;
      sub_1704F80(v100, (__int64)v156, v104, v97, v98, v99);
    }
    while ( v101 != v103 );
  }
  sub_1704F80(v100, (__int64)v84, *(_QWORD *)(v6 + 184), v97, v98, v99);
  v105 = sub_157F280(*(_QWORD *)(v6 + 192));
  v110 = v109;
  v111 = v105;
  while ( v110 != v111 )
  {
    if ( (*(_BYTE *)(v111 + 23) & 0x40) != 0 )
    {
      if ( v149 == **(_QWORD **)(v111 - 8) )
        goto LABEL_84;
    }
    else if ( v149 == *(_QWORD *)(v111 - 24LL * (*(_DWORD *)(v111 + 20) & 0xFFFFFFF)) )
    {
LABEL_84:
      sub_1704F80(v111, (__int64)v84, *(_QWORD *)(v6 + 184), v106, v107, v108);
    }
    v112 = *(_QWORD *)(v111 + 32);
    if ( !v112 )
      BUG();
    v111 = 0;
    if ( *(_BYTE *)(v112 - 8) == 77 )
      v111 = v112 - 24;
  }
  v113 = v154;
  v114 = sub_13FCB50(*(_QWORD *)(v6 + 8));
  v115 = *(_DWORD *)(v113 + 20) & 0xFFFFFFF;
  if ( v115 )
  {
    v116 = 24LL * *(unsigned int *)(v113 + 56) + 8;
    v117 = 0;
    while ( 1 )
    {
      v118 = v113 - 24LL * v115;
      if ( (*(_BYTE *)(v113 + 23) & 0x40) != 0 )
        v118 = *(_QWORD *)(v113 - 8);
      if ( v114 == *(_QWORD *)(v118 + v116) )
        break;
      ++v117;
      v116 += 8;
      if ( v115 == (_DWORD)v117 )
        goto LABEL_115;
    }
    v119 = 24 * v117;
    v120 = 24LL * ((_DWORD)v117 == 0);
  }
  else
  {
LABEL_115:
    v119 = 0x17FFFFFFE8LL;
    v120 = 0;
  }
  v121 = (__int64 *)(v120 + sub_13CF970(v154));
  if ( *v121 )
  {
    v122 = v121[1];
    v123 = v121[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v123 = v122;
    if ( v122 )
      *(_QWORD *)(v122 + 16) = *(_QWORD *)(v122 + 16) & 3LL | v123;
  }
  *v121 = v100;
  if ( v100 )
  {
    v124 = *(_QWORD *)(v100 + 8);
    v121[1] = v124;
    if ( v124 )
      *(_QWORD *)(v124 + 16) = (unsigned __int64)(v121 + 1) | *(_QWORD *)(v124 + 16) & 3LL;
    v121[2] = (v100 + 8) | v121[2] & 3;
    *(_QWORD *)(v100 + 8) = v121;
  }
  v125 = (__int64 *)(v119 + sub_13CF970(v154));
  if ( *v125 )
  {
    v126 = v125[1];
    v127 = v125[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v127 = v126;
    if ( v126 )
      *(_QWORD *)(v126 + 16) = *(_QWORD *)(v126 + 16) & 3LL | v127;
  }
  *v125 = v149;
  if ( v149 )
  {
    v128 = *(_QWORD *)(v149 + 8);
    v125[1] = v128;
    if ( v128 )
      *(_QWORD *)(v128 + 16) = (unsigned __int64)(v125 + 1) | *(_QWORD *)(v128 + 16) & 3LL;
    v125[2] = (v149 + 8) | v125[2] & 3;
    *(_QWORD *)(v149 + 8) = v125;
  }
  sub_1455FA0((__int64)v155);
  if ( v170 != v169[1] )
    _libc_free(v170);
  return sub_1455FA0((__int64)v162);
}
