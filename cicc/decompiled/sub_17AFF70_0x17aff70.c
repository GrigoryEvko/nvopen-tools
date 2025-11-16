// Function: sub_17AFF70
// Address: 0x17aff70
//
__int64 __fastcall sub_17AFF70(
        _BYTE *a1,
        __int64 a2,
        __int64 ***a3,
        __int64 *a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v13; // r12
  __int64 v14; // rbx
  __int64 v15; // r13
  char v16; // al
  __int64 v17; // r15
  _QWORD *v18; // rax
  __int64 v19; // rax
  __int64 v20; // r8
  int v21; // r9d
  __int64 v22; // rax
  _QWORD *v24; // rax
  __int64 **v25; // rax
  __int64 v26; // rax
  int v27; // r8d
  int v28; // r9d
  __int64 v29; // rdx
  __int64 v30; // r15
  __int64 *v31; // rax
  __int64 *i; // rdx
  _QWORD *v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  int v36; // r8d
  int v37; // r9d
  __int64 v38; // r14
  __int64 *v39; // rax
  __int64 *j; // rdx
  __int64 v41; // r11
  __int64 v42; // rax
  __int64 v43; // rdx
  unsigned int v44; // r15d
  _QWORD *v45; // rax
  __int64 ***v46; // r8
  __int64 ***v47; // r9
  __int64 **v48; // rax
  _QWORD *v49; // r14
  __int64 v50; // r11
  __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r15
  __int64 v55; // r14
  _QWORD *v56; // rax
  __int64 v57; // rax
  __int64 v58; // rsi
  __int64 *v59; // r13
  int v60; // r13d
  __int64 v61; // r14
  __int64 v62; // rsi
  _QWORD *v63; // rax
  __int64 v64; // rax
  int v65; // r8d
  __int64 v66; // r9
  __int64 v67; // rax
  __int64 v68; // r13
  _QWORD *v69; // rax
  __int64 v70; // rax
  __int64 *v71; // rbx
  _QWORD *v72; // rax
  __int64 v73; // rax
  __int64 v74; // r11
  __int64 **v75; // r15
  __int64 *v76; // r10
  __int64 v77; // rbx
  __int64 *v78; // r14
  int v79; // r8d
  __int64 v80; // r9
  __int64 v81; // rax
  __int64 v82; // r8
  int v83; // r9d
  __int64 v84; // rax
  __int64 v85; // r11
  __int64 *v86; // r10
  __int64 v87; // rcx
  unsigned __int8 v88; // al
  __int64 v89; // rax
  __int64 v90; // r15
  __int64 *v91; // rdi
  __int64 v92; // rdi
  _QWORD *v93; // rax
  _QWORD *v94; // rax
  __int64 v95; // r11
  __int64 *v96; // r10
  _QWORD *v97; // r14
  __int64 *v98; // r10
  __int64 k; // rbx
  __int64 ***v100; // r12
  __int64 v101; // r13
  _QWORD *v102; // rax
  __int64 v103; // r15
  __int64 v104; // rdx
  __int64 ***v105; // r15
  __int64 v106; // r12
  __int64 v107; // rbx
  __int64 v108; // r13
  _QWORD *v109; // rax
  double v110; // xmm4_8
  double v111; // xmm5_8
  __int64 ***v112; // r12
  __int64 v113; // r15
  __int64 v114; // rax
  __int64 *v115; // r10
  __int64 *v116; // r15
  __int64 v117; // rsi
  __int64 v118; // rsi
  __int64 v119; // rdx
  unsigned __int8 *v120; // rsi
  __int64 v121; // rsi
  __int64 v122; // rax
  __int64 v123; // [rsp+8h] [rbp-128h]
  __int64 *v124; // [rsp+10h] [rbp-120h]
  __int64 v125; // [rsp+10h] [rbp-120h]
  __int64 v126; // [rsp+18h] [rbp-118h]
  __int64 v127; // [rsp+18h] [rbp-118h]
  __int64 *v128; // [rsp+18h] [rbp-118h]
  __int64 ***v129; // [rsp+18h] [rbp-118h]
  __int64 v130; // [rsp+20h] [rbp-110h]
  __int64 *v131; // [rsp+20h] [rbp-110h]
  __int64 v132; // [rsp+20h] [rbp-110h]
  __int64 *v133; // [rsp+20h] [rbp-110h]
  int v135; // [rsp+28h] [rbp-108h]
  __int64 v136; // [rsp+28h] [rbp-108h]
  _QWORD *v137; // [rsp+28h] [rbp-108h]
  __int64 v138; // [rsp+28h] [rbp-108h]
  __int64 v139; // [rsp+30h] [rbp-100h]
  unsigned int v140; // [rsp+30h] [rbp-100h]
  const void *v141; // [rsp+30h] [rbp-100h]
  __int64 v142; // [rsp+30h] [rbp-100h]
  __int64 v143; // [rsp+30h] [rbp-100h]
  __int64 *v144; // [rsp+30h] [rbp-100h]
  __int64 v145; // [rsp+38h] [rbp-F8h]
  int v146; // [rsp+38h] [rbp-F8h]
  __int64 *v147; // [rsp+38h] [rbp-F8h]
  int v148; // [rsp+38h] [rbp-F8h]
  __int64 *v149; // [rsp+38h] [rbp-F8h]
  __int64 v150; // [rsp+38h] [rbp-F8h]
  __int64 *v151; // [rsp+38h] [rbp-F8h]
  __int64 *v152; // [rsp+38h] [rbp-F8h]
  unsigned int v153; // [rsp+40h] [rbp-F0h]
  unsigned int v154; // [rsp+40h] [rbp-F0h]
  _QWORD *v155; // [rsp+40h] [rbp-F0h]
  __int64 v156; // [rsp+48h] [rbp-E8h]
  __int64 v157; // [rsp+48h] [rbp-E8h]
  __int64 v158; // [rsp+48h] [rbp-E8h]
  _QWORD v159[2]; // [rsp+50h] [rbp-E0h] BYREF
  __int16 v160; // [rsp+60h] [rbp-D0h]
  __int64 *v161; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v162; // [rsp+78h] [rbp-B8h]
  _BYTE v163[176]; // [rsp+80h] [rbp-B0h] BYREF

  v13 = (__int64)a1;
  v14 = a2;
  v15 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  v16 = a1[16];
  if ( v16 == 9 )
  {
    v24 = (_QWORD *)sub_16498A0((__int64)a1);
    v25 = (__int64 **)sub_1643350(v24);
    v26 = sub_1599EF0(v25);
    *(_DWORD *)(a2 + 8) = 0;
    v29 = (unsigned int)v15;
    v30 = v26;
    if ( *(_DWORD *)(a2 + 12) < (unsigned int)v15 )
    {
      sub_16CD150(a2, (const void *)(a2 + 16), (unsigned int)v15, 8, v27, v28);
      v29 = (unsigned int)v15;
    }
    v31 = *(__int64 **)a2;
    *(_DWORD *)(a2 + 8) = v15;
    for ( i = &v31[v29]; i != v31; ++v31 )
      *v31 = v30;
    if ( a3 )
      return sub_1599EF0(*a3);
    return v13;
  }
  if ( v16 == 10 )
  {
    v33 = (_QWORD *)sub_16498A0((__int64)a1);
    v34 = sub_1643350(v33);
    v35 = sub_159C470(v34, 0, 0);
    *(_DWORD *)(a2 + 8) = 0;
    v38 = v35;
    if ( *(_DWORD *)(a2 + 12) < (unsigned int)v15 )
      sub_16CD150(a2, (const void *)(a2 + 16), (unsigned int)v15, 8, v36, v37);
    v39 = *(__int64 **)a2;
    *(_DWORD *)(a2 + 8) = v15;
    for ( j = &v39[(unsigned int)v15]; j != v39; ++v39 )
      *v39 = v38;
    return v13;
  }
  if ( v16 != 84 )
    goto LABEL_4;
  v41 = *((_QWORD *)a1 - 6);
  if ( *(_BYTE *)(v41 + 16) != 83 )
    goto LABEL_4;
  v42 = *(_QWORD *)(v41 - 24);
  if ( *(_BYTE *)(v42 + 16) != 13 )
    goto LABEL_4;
  v43 = *((_QWORD *)a1 - 3);
  if ( *(_BYTE *)(v43 + 16) != 13 )
    goto LABEL_4;
  v44 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( *(_DWORD *)(v42 + 32) <= 0x40u )
    v157 = *(_QWORD *)(v42 + 24);
  else
    v157 = **(_QWORD **)(v42 + 24);
  v45 = *(_QWORD **)(v43 + 24);
  if ( *(_DWORD *)(v43 + 32) > 0x40u )
    v45 = (_QWORD *)*v45;
  v46 = *(__int64 ****)(v41 - 48);
  v47 = (__int64 ***)*((_QWORD *)a1 - 9);
  v153 = (unsigned int)v45;
  if ( a3 == v46 || !a3 )
  {
    v145 = *(_QWORD *)(v41 - 48);
    v139 = *((_QWORD *)a1 - 6);
    v49 = (_QWORD *)sub_17AFF70(v47, a2, v46, a4);
    if ( *v49 == *(_QWORD *)v145 )
    {
      v68 = (unsigned int)(*(_DWORD *)(*(_QWORD *)v145 + 32LL) + v157);
      v69 = (_QWORD *)sub_16498A0((__int64)a1);
      v70 = sub_1643350(v69);
      v71 = (__int64 *)(*(_QWORD *)a2 + 8LL * (v153 % v44));
      *v71 = sub_159C470(v70, v68, 0);
      return (__int64)v49;
    }
    v50 = v139;
    v51 = **(_QWORD **)(v139 - 48);
    v52 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    v53 = *(_QWORD *)(v51 + 32);
    v140 = v52;
    v154 = v53;
    if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) != *(_QWORD *)(v51 + 24) || (unsigned int)v52 <= (unsigned int)v53 )
    {
LABEL_37:
      v54 = (unsigned int)v15;
      v55 = 0;
      if ( (_DWORD)v15 )
      {
        do
        {
          v56 = (_QWORD *)sub_16498A0(v13);
          v57 = sub_1643350(v56);
          v58 = v55;
          v59 = (__int64 *)(*(_QWORD *)v14 + 8 * v55++);
          *v59 = sub_159C470(v57, v58, 0);
        }
        while ( v55 != v54 );
      }
      return v13;
    }
    v124 = a4;
    v161 = (__int64 *)v163;
    v135 = v53;
    v126 = v50;
    v162 = 0x1000000000LL;
    v72 = (_QWORD *)sub_16498A0((__int64)a1);
    v73 = sub_1643350(v72);
    v74 = v126;
    v75 = (__int64 **)v73;
    v76 = v124;
    if ( v135 )
    {
      v125 = v14;
      v77 = 0;
      v78 = v76;
      do
      {
        v80 = sub_159C470((__int64)v75, v77, 0);
        v81 = (unsigned int)v162;
        if ( (unsigned int)v162 >= HIDWORD(v162) )
        {
          v123 = v80;
          sub_16CD150((__int64)&v161, v163, 0, 8, v79, v80);
          v81 = (unsigned int)v162;
          v80 = v123;
        }
        ++v77;
        v161[v81] = v80;
        LODWORD(v162) = v162 + 1;
      }
      while ( v135 != v77 );
      v74 = v126;
      v14 = v125;
      v76 = v78;
    }
    v136 = v74;
    v131 = v76;
    do
    {
      v82 = sub_1599EF0(v75);
      v84 = (unsigned int)v162;
      if ( (unsigned int)v162 >= HIDWORD(v162) )
      {
        v127 = v82;
        sub_16CD150((__int64)&v161, v163, 0, 8, v82, v83);
        v84 = (unsigned int)v162;
        v82 = v127;
      }
      ++v154;
      v161[v84] = v82;
      LODWORD(v162) = v162 + 1;
    }
    while ( v140 > v154 );
    v85 = v136;
    v86 = v131;
    v87 = *(_QWORD *)(v136 - 48);
    v88 = *(_BYTE *)(v87 + 16);
    v155 = (_QWORD *)v87;
    if ( v88 <= 0x17u )
    {
      v90 = 0;
    }
    else
    {
      if ( v88 != 77 )
      {
        v89 = *(_QWORD *)(v87 + 40);
        v90 = *(_QWORD *)(v136 - 48);
LABEL_64:
        if ( *((_QWORD *)a1 + 5) != v89
          || (v92 = *((_QWORD *)a1 + 1)) != 0
          && !*(_QWORD *)(v92 + 8)
          && (v93 = sub_1648700(v92), v85 = v136, v86 = v131, *((_BYTE *)v93 + 16) == 84) )
        {
          v91 = v161;
          if ( v161 == (__int64 *)v163 )
            goto LABEL_37;
          goto LABEL_66;
        }
        v128 = v86;
        v142 = v85;
        v132 = sub_1599EF0((__int64 **)v51);
        v160 = 257;
        v137 = (_QWORD *)sub_15A01B0(v161, (unsigned int)v162);
        v94 = sub_1648A60(56, 3u);
        v95 = v142;
        v96 = v128;
        v97 = v94;
        if ( v94 )
        {
          sub_15FA660((__int64)v94, v155, v132, v137, (__int64)v159, 0);
          v96 = v128;
          v95 = v142;
        }
        if ( v90 && *(_BYTE *)(v90 + 16) != 77 )
        {
          v147 = v96;
          sub_15F2180((__int64)v97, v90);
          v98 = v147;
LABEL_75:
          v138 = v14;
          v148 = v15;
          v143 = v13;
          v133 = v98;
          for ( k = v155[1]; k; k = *(_QWORD *)(k + 8) )
          {
            v100 = (__int64 ***)sub_1648700(k);
            if ( *((_BYTE *)v100 + 16) == 83 && v100[5] == (__int64 **)v97[5] )
            {
              v160 = 257;
              v101 = (__int64)*(v100 - 3);
              v102 = sub_1648A60(56, 2u);
              v103 = (__int64)v102;
              if ( v102 )
                sub_15FA320((__int64)v102, v97, v101, (__int64)v159, 0);
              sub_15F2180(v103, (__int64)v100);
              v104 = (__int64)v100[1];
              if ( v104 )
              {
                v129 = (__int64 ***)v103;
                v105 = v100;
                v106 = k;
                v107 = v104;
                v108 = *v133;
                do
                {
                  v109 = sub_1648700(v107);
                  sub_170B990(v108, (__int64)v109);
                  v107 = *(_QWORD *)(v107 + 8);
                }
                while ( v107 );
                k = v106;
                v112 = v105;
                v113 = (__int64)v129;
                if ( v112 == v129 )
                  v113 = sub_1599EF0(*v112);
                sub_164D160((__int64)v112, v113, a5, a6, a7, a8, v110, v111, a11, a12);
              }
            }
          }
          LODWORD(v15) = v148;
          v13 = v143;
          v14 = v138;
          v91 = v161;
          if ( v161 == (__int64 *)v163 )
            goto LABEL_37;
LABEL_66:
          _libc_free((unsigned __int64)v91);
          goto LABEL_37;
        }
        v149 = v96;
        v114 = sub_157EE30(*(_QWORD *)(v95 + 40));
        v115 = v149;
        v116 = (__int64 *)v114;
        if ( !v114 )
          BUG();
        v117 = *(_QWORD *)(v114 + 24);
        v159[0] = v117;
        if ( v117 )
        {
          sub_1623A60((__int64)v159, v117, 2);
          v118 = v97[6];
          v115 = v149;
          v119 = (__int64)(v97 + 6);
          if ( !v118 )
            goto LABEL_97;
        }
        else
        {
          v118 = v97[6];
          v119 = (__int64)(v97 + 6);
          if ( !v118 )
          {
LABEL_99:
            v152 = v115;
            sub_157E9D0(v116[2] + 40, (__int64)v97);
            v121 = *v116;
            v122 = v97[3];
            v97[4] = v116;
            v121 &= 0xFFFFFFFFFFFFFFF8LL;
            v97[3] = v121 | v122 & 7;
            *(_QWORD *)(v121 + 8) = v97 + 3;
            *v116 = *v116 & 7 | (unsigned __int64)(v97 + 3);
            sub_170B990(*v152, (__int64)v97);
            v98 = v152;
            goto LABEL_75;
          }
        }
        v144 = v115;
        v150 = v119;
        sub_161E7C0(v119, v118);
        v115 = v144;
        v119 = v150;
LABEL_97:
        v120 = (unsigned __int8 *)v159[0];
        v97[6] = v159[0];
        if ( v120 )
        {
          v151 = v115;
          sub_1623210((__int64)v159, v120, v119);
          v115 = v151;
        }
        goto LABEL_99;
      }
      v90 = *(_QWORD *)(v136 - 48);
    }
    v89 = *(_QWORD *)(v136 + 40);
    goto LABEL_64;
  }
  v48 = *v46;
  if ( a3 != v47 )
  {
    if ( *a3 == v48 )
    {
      v158 = *((_QWORD *)a1 - 6);
      if ( (unsigned __int8)sub_17AE390(a1, v46, a3, a2) )
        return *(_QWORD *)(v158 - 48);
    }
LABEL_4:
    if ( (_DWORD)v15 )
    {
      v17 = 0;
      do
      {
        v18 = (_QWORD *)sub_16498A0((__int64)a1);
        v19 = sub_1643350(v18);
        v20 = sub_159C470(v19, v17, 0);
        v22 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v22 >= *(_DWORD *)(a2 + 12) )
        {
          v156 = v20;
          sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v20, v21);
          v22 = *(unsigned int *)(a2 + 8);
          v20 = v156;
        }
        ++v17;
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v22) = v20;
        ++*(_DWORD *)(a2 + 8);
      }
      while ( (unsigned int)v15 != v17 );
    }
    return v13;
  }
  v146 = *((_DWORD *)v48 + 8);
  v141 = (const void *)(a2 + 16);
  if ( (_DWORD)v15 )
  {
    v60 = 0;
    v61 = *((_QWORD *)a1 - 6);
    do
    {
      v62 = (unsigned int)(v146 + v60);
      if ( v153 == v60 )
        v62 = (unsigned int)v157;
      v63 = (_QWORD *)sub_16498A0((__int64)a1);
      v64 = sub_1643350(v63);
      v66 = sub_159C470(v64, v62, 0);
      v67 = *(unsigned int *)(v14 + 8);
      if ( (unsigned int)v67 >= *(_DWORD *)(v14 + 12) )
      {
        v130 = v66;
        sub_16CD150(v14, v141, 0, 8, v65, v66);
        v67 = *(unsigned int *)(v14 + 8);
        v66 = v130;
      }
      ++v60;
      *(_QWORD *)(*(_QWORD *)v14 + 8 * v67) = v66;
      ++*(_DWORD *)(v14 + 8);
    }
    while ( v44 != v60 );
    return *(_QWORD *)(v61 - 48);
  }
  return (__int64)v46;
}
