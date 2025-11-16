// Function: sub_DC1A10
// Address: 0xdc1a10
//
_QWORD *__fastcall sub_DC1A10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int16 v7; // ax
  __int64 v8; // r13
  unsigned int v9; // eax
  _QWORD *result; // rax
  unsigned int v11; // r13d
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 *v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // rbx
  __int16 v21; // ax
  __int64 v22; // rdx
  unsigned int v23; // ebx
  void *v24; // r15
  __int64 v25; // rdx
  __int64 v26; // rbx
  __int64 *v27; // rax
  __int64 v28; // rbx
  __int16 v29; // ax
  unsigned int v30; // r15d
  __int64 v31; // r13
  unsigned int v32; // r15d
  __int64 v33; // r13
  unsigned int v34; // r13d
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 *v37; // rbx
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rdi
  unsigned int v45; // r13d
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 *v48; // rbx
  __int64 v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // rax
  __int64 v56; // rsi
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 *v62; // r9
  int v63; // eax
  _QWORD *v64; // rax
  __int64 v65; // rcx
  unsigned int v66; // r13d
  int v67; // eax
  _QWORD *v68; // rax
  __int64 v69; // rax
  __int64 v70; // r8
  __int64 v71; // rdx
  unsigned int v72; // r15d
  __int64 v73; // r13
  __int64 v74; // rax
  __int64 v75; // rax
  _QWORD *v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // r8
  _QWORD *v80; // rax
  __int64 v81; // r8
  unsigned int v82; // r13d
  __int64 v83; // r15
  __int64 v84; // rax
  __int64 *v85; // r15
  __int64 *v86; // rbx
  __int64 v87; // rsi
  __int64 v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  __int64 *v93; // r8
  __int64 *v94; // r13
  __int64 *v95; // rbx
  __int64 v96; // rsi
  __int64 v97; // rax
  __int64 v98; // rdx
  __int64 v99; // rcx
  __int64 v100; // r8
  __int64 v101; // r9
  _QWORD *v102; // rbx
  __int64 v103; // rax
  __int64 v104; // rax
  __int64 v105; // rbx
  __int64 v106; // r15
  _QWORD *v107; // rax
  __int64 v108; // rax
  __int64 v109; // rax
  __int64 v110; // r15
  __int64 v111; // rax
  __int64 v112; // rax
  unsigned int v113; // r15d
  _QWORD *v114; // rax
  __int64 v115; // rax
  __int64 v116; // rax
  __int64 v117; // rax
  __int64 v118; // rax
  __int64 v119; // rax
  __int64 v120; // rax
  __int64 v121; // [rsp+0h] [rbp-1D0h]
  __int64 v122; // [rsp+8h] [rbp-1C8h]
  __int64 v123; // [rsp+10h] [rbp-1C0h]
  __int64 v124; // [rsp+10h] [rbp-1C0h]
  __int64 v125; // [rsp+18h] [rbp-1B8h]
  __int64 v126; // [rsp+18h] [rbp-1B8h]
  unsigned int v127; // [rsp+20h] [rbp-1B0h]
  __int64 v128; // [rsp+28h] [rbp-1A8h]
  __int64 v129; // [rsp+28h] [rbp-1A8h]
  __int64 v130; // [rsp+30h] [rbp-1A0h]
  unsigned int v131; // [rsp+38h] [rbp-198h]
  __int64 v132; // [rsp+38h] [rbp-198h]
  __int64 v133; // [rsp+38h] [rbp-198h]
  __int64 v134; // [rsp+40h] [rbp-190h]
  __int64 v135; // [rsp+40h] [rbp-190h]
  _QWORD *v136; // [rsp+40h] [rbp-190h]
  unsigned int v137; // [rsp+40h] [rbp-190h]
  __int64 v138; // [rsp+40h] [rbp-190h]
  __int64 v139; // [rsp+48h] [rbp-188h]
  __int64 v140; // [rsp+48h] [rbp-188h]
  __int64 *v141; // [rsp+50h] [rbp-180h]
  __int64 v142; // [rsp+50h] [rbp-180h]
  __int64 *v143; // [rsp+58h] [rbp-178h]
  __int64 v144; // [rsp+58h] [rbp-178h]
  __int64 *v145; // [rsp+58h] [rbp-178h]
  _QWORD *v146; // [rsp+58h] [rbp-178h]
  __int64 *v147; // [rsp+58h] [rbp-178h]
  unsigned int v148; // [rsp+58h] [rbp-178h]
  _QWORD *v149; // [rsp+58h] [rbp-178h]
  __int64 v150; // [rsp+58h] [rbp-178h]
  __int64 v151; // [rsp+58h] [rbp-178h]
  int v152; // [rsp+58h] [rbp-178h]
  _QWORD *v153; // [rsp+60h] [rbp-170h]
  _QWORD *v154; // [rsp+60h] [rbp-170h]
  __int64 v155; // [rsp+68h] [rbp-168h] BYREF
  __int64 *v156; // [rsp+78h] [rbp-158h] BYREF
  const void *v157; // [rsp+80h] [rbp-150h] BYREF
  unsigned int v158; // [rsp+88h] [rbp-148h]
  __int64 v159; // [rsp+90h] [rbp-140h] BYREF
  unsigned int v160; // [rsp+98h] [rbp-138h]
  __int64 v161[2]; // [rsp+A0h] [rbp-130h] BYREF
  __int64 v162; // [rsp+B0h] [rbp-120h] BYREF
  __int64 *v163; // [rsp+C0h] [rbp-110h] BYREF
  int v164; // [rsp+C8h] [rbp-108h]
  __int64 v165; // [rsp+D0h] [rbp-100h] BYREF
  __int64 *v166; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v167; // [rsp+E8h] [rbp-E8h]
  __int64 v168[4]; // [rsp+F0h] [rbp-E0h] BYREF
  _DWORD *v169; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v170; // [rsp+118h] [rbp-B8h]
  _DWORD v171[44]; // [rsp+120h] [rbp-B0h] BYREF

  v7 = *(_WORD *)(a2 + 24);
  v155 = a2;
  if ( !v7 )
  {
    v8 = *(_QWORD *)(a2 + 32);
    v9 = sub_D97050(a1, a3);
    sub_C449B0((__int64)&v169, (const void **)(v8 + 24), v9);
    result = sub_DA26C0((__int64 *)a1, (__int64)&v169);
    if ( (unsigned int)v170 > 0x40 )
    {
      if ( v169 )
      {
        v153 = result;
        j_j___libc_free_0_0(v169);
        return v153;
      }
    }
    return result;
  }
  v11 = a4;
  if ( v7 == 3 )
    return (_QWORD *)sub_DC2B70(a1, *(_QWORD *)(a2 + 32), a3, (unsigned int)(a4 + 1));
  v171[0] = 3;
  v169 = v171;
  v170 = 0x2000000001LL;
  sub_D953B0((__int64)&v169, a2, a3, a4, a5, a6);
  sub_D953B0((__int64)&v169, a3, v12, v13, v14, v15);
  v16 = (__int64 *)&v169;
  v156 = 0;
  v143 = (__int64 *)(a1 + 1032);
  result = sub_C65B40(a1 + 1032, (__int64)&v169, (__int64 *)&v156, (__int64)off_49DEA80);
  if ( !result )
  {
    if ( v11 > dword_4F89148 )
      goto LABEL_21;
    v20 = v155;
    v21 = *(_WORD *)(v155 + 24);
    if ( v21 == 2 )
    {
      v125 = *(_QWORD *)(v155 + 32);
      v22 = sub_DBB9F0(a1, v125, 0, 0);
      v158 = *(_DWORD *)(v22 + 8);
      if ( v158 > 0x40 )
      {
        v134 = v22;
        sub_C43780((__int64)&v157, (const void **)v22);
        v22 = v134;
      }
      else
      {
        v157 = *(const void **)v22;
      }
      v160 = *(_DWORD *)(v22 + 24);
      if ( v160 > 0x40 )
        sub_C43780((__int64)&v159, (const void **)(v22 + 16));
      else
        v159 = *(_QWORD *)(v22 + 16);
      v131 = sub_D97050(a1, *(_QWORD *)(v155 + 40));
      v23 = sub_D97050(a1, a3);
      sub_AB4490((__int64)v161, (__int64)&v157, v131);
      sub_AB3F90((__int64)&v163, (__int64)v161, v23);
      sub_AB4D50((__int64)&v166, (__int64)&v157, v23);
      LOBYTE(v23) = sub_AB1BB0((__int64)&v163, (__int64)&v166);
      sub_969240(v168);
      sub_969240((__int64 *)&v166);
      sub_969240(&v165);
      sub_969240((__int64 *)&v163);
      sub_969240(&v162);
      sub_969240(v161);
      if ( (_BYTE)v23 )
      {
        v16 = (__int64 *)v125;
        v144 = sub_DC5760(a1, v125, a3, v11);
        sub_969240(&v159);
        sub_969240((__int64 *)&v157);
        result = (_QWORD *)v144;
        goto LABEL_9;
      }
      sub_969240(&v159);
      sub_969240((__int64 *)&v157);
      v20 = v155;
      v21 = *(_WORD *)(v155 + 24);
    }
    if ( v21 != 8 || *(_QWORD *)(v20 + 40) != 2 )
    {
LABEL_27:
      if ( (unsigned __int8)sub_DCFD50(a1, v20, &v163, &v166) )
      {
        v32 = v11 + 1;
        v33 = sub_DC2B70(a1, v166, a3, v11 + 1);
        v16 = (__int64 *)sub_DC2B70(a1, v163, a3, v32);
        result = (_QWORD *)sub_DCFA50(a1, v16, v33);
        goto LABEL_9;
      }
      v28 = v155;
      v29 = *(_WORD *)(v155 + 24);
      if ( v29 == 7 )
      {
        v30 = v11 + 1;
        v31 = sub_DC2B70(a1, *(_QWORD *)(v155 + 40), a3, v11 + 1);
        v16 = (__int64 *)sub_DC2B70(a1, *(_QWORD *)(v155 + 32), a3, v30);
        result = (_QWORD *)sub_DCB270(a1, v16, v31);
        goto LABEL_9;
      }
      if ( v29 == 5 )
      {
        if ( (*(_BYTE *)(v155 + 28) & 2) != 0 )
        {
          v34 = v11 + 1;
          v166 = v168;
          v167 = 0x400000000LL;
          v35 = *(_QWORD *)(v155 + 32);
          v36 = *(_QWORD *)(v155 + 40);
          if ( v35 + 8 * v36 != v35 )
          {
            v145 = (__int64 *)(v35 + 8 * v36);
            v37 = *(__int64 **)(v155 + 32);
            do
            {
              v38 = *v37++;
              v39 = sub_DC2B70(a1, v38, a3, v34);
              sub_D9B3A0((__int64)&v166, v39, v40, v41, v42, v43);
            }
            while ( v145 != v37 );
          }
          v16 = (__int64 *)&v166;
          result = (_QWORD *)sub_DC7EB0(a1, &v166, 2, v34);
          v44 = (__int64)v166;
          if ( v166 == v168 )
            goto LABEL_9;
          goto LABEL_37;
        }
        v75 = **(_QWORD **)(v155 + 32);
        if ( *(_WORD *)(v75 + 24) )
          goto LABEL_20;
        sub_DB5770((__int64)v161, a1, *(_QWORD *)(v75 + 32), v155);
        if ( !sub_D94970((__int64)v161, 0) )
        {
          v76 = sub_DA26C0((__int64 *)a1, (__int64)v161);
          v133 = sub_DC2B70(a1, v76, a3, v11);
          sub_9865C0((__int64)&v163, (__int64)v161);
          sub_AADAA0((__int64)&v166, (__int64)&v163, v77, v78, v79);
          v80 = sub_DA26C0((__int64 *)a1, (__int64)&v166);
          v81 = v11;
          v82 = v11 + 1;
          v83 = sub_DC7ED0(a1, v80, v155, 0, v81);
          sub_969240((__int64 *)&v166);
          sub_969240((__int64 *)&v163);
          v84 = sub_DC2B70(a1, v83, a3, v82);
          v70 = v82;
          v71 = v84;
          v16 = (__int64 *)v133;
          goto LABEL_58;
        }
        sub_969240(v161);
        v28 = v155;
        v29 = *(_WORD *)(v155 + 24);
      }
      if ( v29 == 6 )
      {
        if ( (*(_BYTE *)(v28 + 28) & 2) != 0 )
        {
          v45 = v11 + 1;
          v166 = v168;
          v167 = 0x400000000LL;
          v46 = *(_QWORD *)(v28 + 32);
          v47 = *(_QWORD *)(v28 + 40);
          if ( v46 + 8 * v47 != v46 )
          {
            v147 = (__int64 *)(v46 + 8 * v47);
            v48 = *(__int64 **)(v28 + 32);
            do
            {
              v49 = *v48++;
              v50 = sub_DC2B70(a1, v49, a3, v45);
              sub_D9B3A0((__int64)&v166, v50, v51, v52, v53, v54);
            }
            while ( v147 != v48 );
          }
          v16 = (__int64 *)&v166;
          result = (_QWORD *)sub_DC8BD0(a1, &v166, 2, v45);
          v44 = (__int64)v166;
          if ( v166 == v168 )
            goto LABEL_9;
          goto LABEL_37;
        }
        if ( *(_QWORD *)(v28 + 40) != 2 )
          goto LABEL_20;
        v102 = *(_QWORD **)(v28 + 32);
        v140 = *v102;
        if ( *(_WORD *)(*v102 + 24LL) )
          goto LABEL_20;
        v103 = *(_QWORD *)(*v102 + 32LL);
        if ( *(_DWORD *)(v103 + 32) > 0x40u )
        {
          if ( (unsigned int)sub_C44630(v103 + 24) == 1 )
            goto LABEL_84;
        }
        else
        {
          v104 = *(_QWORD *)(v103 + 24);
          if ( v104 && (v104 & (v104 - 1)) == 0 )
          {
LABEL_84:
            v105 = v102[1];
            if ( *(_WORD *)(v105 + 24) == 2 )
            {
              v152 = sub_D97050(a1, *(_QWORD *)(v105 + 40));
              v106 = *(_QWORD *)(v140 + 32);
              LODWORD(v106) = sub_9871A0(v106 + 24) - *(_DWORD *)(v106 + 32) + v152 + 1;
              v107 = (_QWORD *)sub_B2BE50(*(_QWORD *)a1);
              v108 = sub_BCCE00(v107, v106);
              v109 = sub_DC5200(a1, *(_QWORD *)(v105 + 32), v108, 0);
              v110 = sub_DC2B70(a1, v109, a3, 0);
              v16 = (__int64 *)sub_DC2B70(a1, v140, a3, 0);
              result = (_QWORD *)sub_DCA690(a1, v16, v110, 2, v11 + 1);
              goto LABEL_9;
            }
          }
        }
LABEL_20:
        v16 = (__int64 *)&v169;
        result = sub_C65B40((__int64)v143, (__int64)&v169, (__int64 *)&v156, (__int64)off_49DEA80);
        if ( result )
          goto LABEL_9;
LABEL_21:
        v24 = sub_C65D30((__int64)&v169, (unsigned __int64 *)(a1 + 1064));
        v26 = v25;
        v27 = (__int64 *)sub_A777F0(0x30u, (__int64 *)(a1 + 1064));
        if ( v27 )
        {
          v141 = v27;
          sub_D96B80((__int64)v27, (__int64)v24, v26, v155, a3);
          v27 = v141;
        }
        v142 = (__int64)v27;
        sub_C657C0(v143, v27, v156, (__int64)off_49DEA80);
        v16 = (__int64 *)v142;
        sub_DAEE00(a1, v142, &v155, 1);
        result = (_QWORD *)v142;
        goto LABEL_9;
      }
      if ( (v29 & 0xFFFD) == 9 )
      {
        v166 = v168;
        v167 = 0x400000000LL;
        v93 = *(__int64 **)(v28 + 32);
        v94 = &v93[*(_QWORD *)(v28 + 40)];
        if ( v94 != v93 )
        {
          v151 = v28;
          v95 = *(__int64 **)(v28 + 32);
          do
          {
            v96 = *v95++;
            v97 = sub_DC2B70(a1, v96, a3, 0);
            sub_D9B3A0((__int64)&v166, v97, v98, v99, v100, v101);
          }
          while ( v94 != v95 );
          v28 = v151;
        }
        v16 = (__int64 *)&v166;
        if ( *(_WORD *)(v28 + 24) == 11 )
          result = (_QWORD *)sub_DCEE50(a1, &v166, 0);
        else
          result = (_QWORD *)sub_DCE040(a1, &v166);
      }
      else
      {
        if ( v29 != 13 )
          goto LABEL_20;
        v166 = v168;
        v167 = 0x400000000LL;
        v85 = *(__int64 **)(v28 + 32);
        v86 = &v85[*(_QWORD *)(v28 + 40)];
        while ( v86 != v85 )
        {
          v87 = *v85++;
          v88 = sub_DC2B70(a1, v87, a3, 0);
          sub_D9B3A0((__int64)&v166, v88, v89, v90, v91, v92);
        }
        v16 = (__int64 *)&v166;
        result = (_QWORD *)sub_DCEE50(a1, &v166, 1);
      }
      v44 = (__int64)v166;
      if ( v166 == v168 )
        goto LABEL_9;
LABEL_37:
      v146 = result;
      _libc_free(v44, &v166);
      result = v146;
      goto LABEL_9;
    }
    v130 = **(_QWORD **)(v20 + 32);
    v139 = sub_D33D80((_QWORD *)v20, a1, v17, v18, v19);
    v55 = sub_D95540(**(_QWORD **)(v20 + 32));
    v127 = sub_D97050(a1, v55);
    v132 = *(_QWORD *)(v20 + 48);
    if ( (*(_BYTE *)(v20 + 28) & 2) != 0 )
      goto LABEL_60;
    v56 = *(_QWORD *)(v20 + 48);
    v126 = sub_DCF3A0(a1, v132, 1);
    if ( !sub_D96A50(v126) )
    {
      v57 = sub_D95540(v130);
      v123 = sub_DC5760(a1, v126, v57, v11);
      v58 = sub_D95540(v126);
      v56 = v123;
      if ( v126 == sub_DC5760(a1, v123, v58, v11) )
      {
        v114 = (_QWORD *)sub_B2BE50(*(_QWORD *)a1);
        v137 = v11 + 1;
        v129 = sub_BCCE00(v114, 2 * v127);
        v115 = sub_DCA690(a1, v123, v139, 0, v11 + 1);
        v116 = sub_DC7ED0(a1, v130, v115, 0, v11 + 1);
        v122 = sub_DC2B70(a1, v116, v129, v11 + 1);
        v121 = sub_DC2B70(a1, v130, v129, v11 + 1);
        v124 = sub_DC2B70(a1, v123, v129, v11 + 1);
        v117 = sub_DC2B70(a1, v139, v129, v11 + 1);
        v118 = sub_DCA690(a1, v124, v117, 0, v11 + 1);
        if ( v122 == sub_DC7ED0(a1, v121, v118, 0, v11 + 1) )
        {
          sub_D97270(a1, v20, 2);
          v72 = v11 + 1;
          goto LABEL_61;
        }
        v119 = sub_DC5000(a1, v139, v129, v137);
        v120 = sub_DCA690(a1, v124, v119, 0, v137);
        v56 = v121;
        if ( v122 == sub_DC7ED0(a1, v121, v120, 0, v137) )
        {
          sub_D97270(a1, v20, 1);
          v113 = v11 + 1;
          goto LABEL_97;
        }
      }
    }
    if ( !sub_D96A50(v126) || *(_BYTE *)(a1 + 16) )
      goto LABEL_53;
    v111 = *(_QWORD *)(a1 + 32);
    if ( !*(_BYTE *)(v111 + 192) )
    {
      v138 = *(_QWORD *)(a1 + 32);
      sub_CFDFC0(v138, v56, v59, v60, v61, v62);
      v111 = v138;
    }
    if ( *(_DWORD *)(v111 + 24) )
    {
LABEL_53:
      v63 = sub_DDDF30(a1, v20);
      sub_D97270(a1, v20, v63);
      if ( (*(_BYTE *)(v20 + 28) & 2) != 0 )
      {
LABEL_60:
        v72 = v11 + 1;
LABEL_61:
        v73 = sub_DE8860(v20, a3, a1, v72);
        v74 = sub_DC2B70(a1, v139, a3, v72);
LABEL_62:
        v16 = (__int64 *)v73;
        result = sub_DC1960(a1, v73, v74, v132, *(_WORD *)(v20 + 28) & 7);
        goto LABEL_9;
      }
      if ( (unsigned __int8)sub_DBEC00(a1, v139) )
      {
        v112 = sub_DBB9F0(a1, v139, 1u, 0);
        sub_AB14C0((__int64)&v163, v112);
        sub_9691E0((__int64)v161, v127, -1, 1u, 0);
        sub_D949F0((__int64)&v166, v161, (__int64)&v163);
        v136 = sub_DA26C0((__int64 *)a1, (__int64)&v166);
        sub_969240((__int64 *)&v166);
        sub_969240(v161);
        sub_969240((__int64 *)&v163);
        if ( (unsigned __int8)sub_DDDA00(a1, v132, 34, v20, v136) || (unsigned __int8)sub_DDDEB0(a1, 34, v20, v136) )
        {
          v113 = v11 + 1;
          sub_D97270(a1, v20, 1);
LABEL_97:
          v73 = sub_DE8860(v20, a3, a1, v113);
          v74 = sub_DC5000(a1, v139, a3, v113);
          goto LABEL_62;
        }
      }
    }
    if ( !*(_WORD *)(v130 + 24) )
    {
      v128 = *(_QWORD *)(v130 + 32) + 24LL;
      sub_DB56A0((__int64)v161, a1, v128, v139);
      if ( !sub_D94970((__int64)v161, 0) )
      {
        v64 = sub_DA26C0((__int64 *)a1, (__int64)v161);
        v65 = v11;
        v66 = v11 + 1;
        v135 = sub_DC2B70(a1, v64, a3, v65);
        v148 = *(_WORD *)(v20 + 28) & 7;
        sub_9865C0((__int64)&v163, v128);
        sub_C46B40((__int64)&v163, v161);
        v67 = v164;
        v164 = 0;
        LODWORD(v167) = v67;
        v166 = v163;
        v68 = sub_DA26C0((__int64 *)a1, (__int64)&v166);
        v149 = sub_DC1960(a1, (__int64)v68, v139, v132, v148);
        sub_969240((__int64 *)&v166);
        sub_969240((__int64 *)&v163);
        v69 = sub_DC2B70(a1, v149, a3, v66);
        v70 = v66;
        v71 = v69;
        v16 = (__int64 *)v135;
LABEL_58:
        v150 = sub_DC7ED0(a1, v16, v71, 6, v70);
        sub_969240(v161);
        result = (_QWORD *)v150;
        goto LABEL_9;
      }
      sub_969240(v161);
    }
    if ( !(unsigned __int8)sub_DC3C40(a1, v130, v139, v132) )
    {
      v20 = v155;
      goto LABEL_27;
    }
    sub_D97270(a1, v20, 2);
    goto LABEL_60;
  }
LABEL_9:
  if ( v169 != v171 )
  {
    v154 = result;
    _libc_free(v169, v16);
    return v154;
  }
  return result;
}
