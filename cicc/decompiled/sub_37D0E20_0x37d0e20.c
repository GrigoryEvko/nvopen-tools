// Function: sub_37D0E20
// Address: 0x37d0e20
//
__int64 __fastcall sub_37D0E20(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r15
  __int64 v3; // rbx
  __int64 *v4; // rdi
  __int64 v5; // rax
  __int64 (*v6)(); // rcx
  __int64 (*v7)(); // rax
  __int64 v9; // rax
  __int64 *v10; // rdi
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 (*v13)(); // rcx
  __int64 (*v14)(); // rax
  __int64 v15; // rdx
  unsigned int v16; // eax
  __int64 v18; // r15
  unsigned int v19; // ebx
  int v20; // r14d
  __int64 v21; // rcx
  int v22; // esi
  __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  __int64 v25; // rdi
  __int16 *v26; // rax
  __int64 v27; // rdi
  int v28; // edx
  signed int v29; // esi
  __int16 *v30; // rax
  __int16 *v31; // r8
  int v32; // eax
  __int64 v33; // r14
  __int64 v34; // rdi
  __int64 v35; // rcx
  _DWORD *v36; // r12
  int v37; // eax
  __int64 v38; // r12
  int v39; // r13d
  int v40; // r13d
  char v41; // al
  __int64 v42; // rdi
  __int64 v43; // rcx
  bool v44; // zf
  __int16 *v45; // rax
  int v46; // esi
  int v47; // edx
  unsigned int v48; // esi
  __int16 v49; // dx
  __int64 v50; // r13
  __int64 v51; // r12
  __int64 v52; // rdx
  __int64 v53; // rsi
  unsigned int *v54; // rcx
  unsigned int v55; // eax
  __int64 v56; // rdx
  __int64 v57; // rdi
  __int16 *v58; // rax
  __int16 *v59; // rdx
  unsigned __int16 v60; // ax
  __int64 v61; // r13
  int v62; // r12d
  __int16 *v63; // rax
  int v64; // r12d
  __int64 v65; // r13
  __int64 v66; // r14
  __int64 v67; // r12
  __int64 v68; // rdx
  __int64 v69; // rsi
  unsigned int v70; // eax
  __int64 v71; // rdx
  __int64 v72; // rdi
  int v73; // eax
  __int16 *i; // rdx
  __int64 v75; // rsi
  unsigned __int64 v76; // rdi
  __int64 v77; // r15
  int v78; // r12d
  int v79; // r14d
  __int64 v80; // rax
  __int64 v81; // rsi
  unsigned int v82; // edx
  __int16 v83; // dx
  _QWORD *v84; // rdi
  unsigned int v85; // esi
  __int16 *v86; // rax
  __int16 *v87; // rcx
  __int16 *v88; // rax
  __int16 *v89; // rdx
  unsigned __int16 v90; // ax
  __int64 v91; // r13
  int v92; // r12d
  __int16 *v93; // rax
  int v94; // esi
  int v95; // edx
  unsigned int v96; // esi
  __int16 v97; // dx
  __int64 v98; // rbx
  __int64 v99; // r12
  __int64 v100; // r15
  __int16 *v101; // r14
  int v102; // eax
  __int64 v103; // r12
  int v104; // r13d
  int v105; // r13d
  char v106; // al
  __int64 v107; // rdi
  unsigned int v108; // r8d
  __int16 *v109; // rax
  int v110; // esi
  int v111; // edx
  unsigned int v112; // esi
  __int16 v113; // dx
  __int64 v114; // r13
  __int64 v115; // r12
  __int64 v116; // rsi
  __int64 v117; // rdx
  unsigned int *v118; // r15
  __int64 v119; // r13
  unsigned int v120; // eax
  unsigned __int8 v121; // [rsp+1Bh] [rbp-95h]
  int v122; // [rsp+1Ch] [rbp-94h]
  int v123; // [rsp+20h] [rbp-90h]
  int v124; // [rsp+28h] [rbp-88h]
  __int64 v125; // [rsp+38h] [rbp-78h]
  int v126; // [rsp+38h] [rbp-78h]
  __int16 v127; // [rsp+40h] [rbp-70h]
  unsigned int *v128; // [rsp+40h] [rbp-70h]
  unsigned int v129; // [rsp+40h] [rbp-70h]
  __int64 v130; // [rsp+48h] [rbp-68h]
  __int16 *v131; // [rsp+48h] [rbp-68h]
  unsigned int *v132; // [rsp+48h] [rbp-68h]
  unsigned __int16 v133; // [rsp+48h] [rbp-68h]
  unsigned int *v134; // [rsp+48h] [rbp-68h]
  int v135; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v136; // [rsp+54h] [rbp-5Ch] BYREF
  _BYTE v137[4]; // [rsp+58h] [rbp-58h] BYREF
  unsigned __int16 v138; // [rsp+5Ch] [rbp-54h] BYREF
  __int16 v139; // [rsp+5Eh] [rbp-52h]
  __int16 *v140; // [rsp+60h] [rbp-50h] BYREF
  __int16 *v141; // [rsp+68h] [rbp-48h] BYREF
  __int16 *v142; // [rsp+70h] [rbp-40h] BYREF
  __int16 *v143; // [rsp+78h] [rbp-38h]

  v2 = a2;
  v3 = a1;
  v4 = *(__int64 **)(a1 + 32);
  v135 = -1;
  v5 = *v4;
  v6 = *(__int64 (**)())(*v4 + 136);
  if ( v6 != sub_2E85450 )
  {
    if ( ((unsigned int (__fastcall *)(__int64 *, unsigned __int64, int *))v6)(v4, a2, &v135) )
      goto LABEL_6;
    v4 = *(__int64 **)(v3 + 32);
    v5 = *v4;
  }
  v7 = *(__int64 (**)())(v5 + 104);
  if ( v7 == sub_2E85440 || !((unsigned int (__fastcall *)(__int64 *, unsigned __int64, int *))v7)(v4, a2, &v135) )
    return 0;
LABEL_6:
  v9 = sub_2E88D60(a2);
  v10 = *(__int64 **)(v3 + 32);
  v11 = v9;
  v12 = *v10;
  v13 = *(__int64 (**)())(*v10 + 136);
  if ( v13 == sub_2E85450 )
  {
LABEL_7:
    v14 = *(__int64 (**)())(v12 + 104);
    if ( v14 != sub_2E85440 && ((unsigned int (__fastcall *)(__int64 *, unsigned __int64, _BYTE *))v14)(v10, a2, v137) )
      goto LABEL_9;
    return 0;
  }
  if ( !((unsigned int (__fastcall *)(__int64 *, unsigned __int64, _BYTE *))v13)(v10, a2, v137) )
  {
    v10 = *(__int64 **)(v3 + 32);
    v12 = *v10;
    goto LABEL_7;
  }
LABEL_9:
  v141 = (__int16 *)sub_37C7360(v3, a2);
  if ( BYTE4(v141) )
  {
    v15 = *(_QWORD *)(v3 + 408);
    v16 = *(_DWORD *)(v15 + 288);
    if ( v16 )
    {
      v130 = v11;
      v18 = v3;
      v19 = 0;
      v20 = (_DWORD)v141 - 1;
      do
      {
        v21 = *(unsigned int *)(*(_QWORD *)(v15 + 64) + 4LL * (*(_DWORD *)(v15 + 284) + v19 + v20 * v16));
        LODWORD(v142) = v21;
        v22 = (_DWORD)v21 << 8;
        v23 = *(_QWORD *)(v15 + 32) + 8 * v21;
        v24 = *(_QWORD *)v23 & 0xFFFFFF0000000000LL
            | *(_DWORD *)(v18 + 416) & 0xFFFFF
            | ((unsigned __int64)(*(_DWORD *)(v18 + 420) & 0xFFFFF) << 20);
        *(_QWORD *)v23 = v24;
        *(_DWORD *)(v23 + 4) = v22 | BYTE4(v24);
        v25 = *(_QWORD *)(v18 + 432);
        if ( v25 )
          sub_37D0D70(v25, (unsigned int)v142, a2, 1u);
        v15 = *(_QWORD *)(v18 + 408);
        ++v19;
        v16 = *(_DWORD *)(v15 + 288);
      }
      while ( v16 > v19 );
      v3 = v18;
      v2 = a2;
      v11 = v130;
    }
  }
  v121 = sub_37C7470(v3, v2, v11, &v136);
  if ( v121 )
  {
    v26 = (__int16 *)sub_37C70E0(v3, v2);
    v27 = *(_QWORD *)(v3 + 16);
    v142 = v26;
    v28 = (int)v26;
    v29 = v136;
    v30 = (__int16 *)(*(_QWORD *)(v27 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v27 + 8) + 24LL * v136 + 4));
    v31 = v30 + 1;
    v32 = *v30;
    v124 = v136 + v32;
    if ( !(_WORD)v32 )
    {
      v122 = v28 - 1;
LABEL_34:
      v58 = (__int16 *)sub_2FF6F50(v27, v29, *(_QWORD *)(v3 + 24));
      v143 = v59;
      v142 = v58;
      v60 = sub_CA1930(&v142);
      v61 = *(_QWORD *)(v3 + 408);
      LODWORD(v141) = v60;
      v62 = *(_DWORD *)(v61 + 288) * v122;
      if ( (unsigned __int8)sub_37BD660(v61 + 824, (unsigned __int16 *)&v141, &v142) )
      {
        v63 = v142 + 2;
      }
      else
      {
        v63 = sub_37C5CE0(v61 + 824, (unsigned __int16 *)&v141, v142) + 2;
        *(v63 - 2) = (__int16)v141;
        v83 = WORD1(v141);
        *(_DWORD *)v63 = 0;
        *(v63 - 1) = v83;
      }
      v64 = *(_DWORD *)(v61 + 284) + v62;
      v65 = *(_QWORD *)(v3 + 408);
      v66 = v136;
      v67 = (unsigned int)(*(_DWORD *)v63 + v64);
      v68 = *(_QWORD *)(v65 + 64);
      v69 = v65;
      v70 = *(_DWORD *)(v68 + 4LL * v136);
      if ( v70 == -1 )
      {
        v134 = (unsigned int *)(v68 + 4LL * v136);
        v70 = sub_37BA230(*(_QWORD *)(v3 + 408), v136);
        *v134 = v70;
        v69 = *(_QWORD *)(v3 + 408);
        v68 = *(_QWORD *)(v69 + 64);
      }
      v71 = *(unsigned int *)(v68 + 4 * v67);
      *(_QWORD *)(*(_QWORD *)(v69 + 32) + 8 * v71) = *(_QWORD *)(*(_QWORD *)(v65 + 32) + 8LL * v70);
      v72 = *(_QWORD *)(v3 + 432);
      if ( v72 )
        sub_37CEB50(v72, *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v3 + 408) + 64LL) + 4 * v66), v71, v2);
      return v121;
    }
    v131 = v31;
    v33 = (unsigned __int16)(v136 + v32);
    v122 = v28 - 1;
    while ( 1 )
    {
      v34 = *(_QWORD *)(v3 + 408);
      v35 = 4 * v33;
      v36 = (_DWORD *)(4 * v33 + *(_QWORD *)(v34 + 64));
      if ( *v36 == -1 )
      {
        v73 = sub_37BA230(v34, (unsigned __int16)v33);
        v35 = 4 * v33;
        *v36 = v73;
      }
      v125 = v35;
      v37 = sub_E91E30(*(_QWORD **)(v3 + 16), v136, (unsigned __int16)v33);
      v38 = *(_QWORD *)(v3 + 408);
      v39 = v37;
      v127 = sub_2FF7530(*(_QWORD *)(v38 + 16), v37);
      WORD1(v140) = sub_2FF7550(*(_QWORD *)(v38 + 16), v39);
      LOWORD(v140) = v127;
      v40 = *(_DWORD *)(v38 + 288) * v122;
      v41 = sub_37BD660(v38 + 824, (unsigned __int16 *)&v140, &v141);
      v42 = v38 + 824;
      v43 = v125;
      v44 = v41 == 0;
      v45 = v141;
      if ( v44 )
        break;
LABEL_27:
      v50 = (unsigned int)(*((_DWORD *)v45 + 1) + *(_DWORD *)(v38 + 284) + v40);
      v51 = *(_QWORD *)(v3 + 408);
      v52 = *(_QWORD *)(v51 + 64);
      v53 = v51;
      v54 = (unsigned int *)(v52 + v43);
      v55 = *v54;
      if ( *v54 == -1 )
      {
        v128 = v54;
        v55 = sub_37BA230(*(_QWORD *)(v3 + 408), (unsigned __int16)v33);
        *v128 = v55;
        v53 = *(_QWORD *)(v3 + 408);
        v52 = *(_QWORD *)(v53 + 64);
      }
      v56 = *(unsigned int *)(v52 + 4 * v50);
      *(_QWORD *)(*(_QWORD *)(v53 + 32) + 8 * v56) = *(_QWORD *)(*(_QWORD *)(v51 + 32) + 8LL * v55);
      v57 = *(_QWORD *)(v3 + 432);
      if ( v57 )
        sub_37CEB50(v57, *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v3 + 408) + 64LL) + 4 * v33), v56, v2);
      if ( !*v131++ )
      {
        v27 = *(_QWORD *)(v3 + 16);
        v29 = v136;
        goto LABEL_34;
      }
      v124 += *(v131 - 1);
      v33 = (unsigned __int16)v124;
    }
    v142 = v141;
    v46 = *(_DWORD *)(v38 + 840);
    ++*(_QWORD *)(v38 + 824);
    v47 = v46 + 1;
    v48 = *(_DWORD *)(v38 + 848);
    if ( 4 * v47 >= 3 * v48 )
    {
      v48 *= 2;
    }
    else if ( v48 - *(_DWORD *)(v38 + 844) - v47 > v48 >> 3 )
    {
      goto LABEL_24;
    }
    sub_37C5A70(v42, v48);
    sub_37BD660(v42, (unsigned __int16 *)&v140, &v142);
    v43 = v125;
    v47 = *(_DWORD *)(v38 + 840) + 1;
    v45 = v142;
LABEL_24:
    *(_DWORD *)(v38 + 840) = v47;
    if ( *v45 != -1 || v45[1] != -1 )
      --*(_DWORD *)(v38 + 844);
    *v45 = (__int16)v140;
    v49 = WORD1(v140);
    *((_DWORD *)v45 + 1) = 0;
    v45[1] = v49;
    goto LABEL_27;
  }
  v141 = (__int16 *)sub_37C74E0(v3, v2, v11, &v136);
  v121 = BYTE4(v141);
  if ( !BYTE4(v141) )
    return v121;
  sub_37B9A30((char **)&v142, v136, *(_QWORD **)(v3 + 16), 1);
  for ( i = v142; v143 != i; v142 = i )
  {
    v77 = *(_QWORD *)(v3 + 408);
    v78 = *(_DWORD *)(v3 + 420);
    v79 = *(_DWORD *)(v3 + 416);
    v80 = *(_QWORD *)(v77 + 64);
    v81 = (unsigned __int16)*i;
    v82 = *(_DWORD *)(v80 + 4 * v81);
    if ( v82 == -1 )
    {
      v132 = (unsigned int *)(v80 + 4 * v81);
      v82 = sub_37BA230(*(_QWORD *)(v3 + 408), v81);
      *v132 = v82;
    }
    v75 = *(_QWORD *)(v77 + 32) + 8LL * v82;
    v76 = *(_QWORD *)v75 & 0xFFFFFF0000000000LL | v79 & 0xFFFFF | ((unsigned __int64)(v78 & 0xFFFFF) << 20);
    *(_QWORD *)v75 = v76;
    *(_DWORD *)(v75 + 4) = BYTE4(v76) | (v82 << 8);
    i = v142 + 1;
  }
  v84 = *(_QWORD **)(v3 + 16);
  v85 = v136;
  v86 = (__int16 *)(v84[7] + 2LL * *(unsigned int *)(v84[1] + 24LL * v136 + 4));
  v87 = v86 + 1;
  LODWORD(v86) = *v86;
  v126 = v136 + (_DWORD)v86;
  if ( !(_WORD)v86 )
  {
    v123 = (_DWORD)v141 - 1;
    goto LABEL_55;
  }
  v100 = (unsigned __int16)v126;
  v101 = v87;
  v123 = (_DWORD)v141 - 1;
  while ( 1 )
  {
    v102 = sub_E91E30(v84, v85, (unsigned __int16)v100);
    v103 = *(_QWORD *)(v3 + 408);
    v104 = v102;
    v133 = sub_2FF7530(*(_QWORD *)(v103 + 16), v102);
    v139 = sub_2FF7550(*(_QWORD *)(v103 + 16), v104);
    v138 = v133;
    v105 = *(_DWORD *)(v103 + 288) * v123;
    v106 = sub_37BD660(v103 + 824, &v138, &v140);
    v107 = v103 + 824;
    v108 = (unsigned __int16)v100;
    v44 = v106 == 0;
    v109 = v140;
    if ( !v44 )
      goto LABEL_69;
    v142 = v140;
    v110 = *(_DWORD *)(v103 + 840);
    ++*(_QWORD *)(v103 + 824);
    v111 = v110 + 1;
    v112 = *(_DWORD *)(v103 + 848);
    if ( 4 * v111 >= 3 * v112 )
    {
      v129 = (unsigned __int16)v100;
      v112 *= 2;
    }
    else
    {
      if ( v112 - *(_DWORD *)(v103 + 844) - v111 > v112 >> 3 )
        goto LABEL_66;
      v129 = (unsigned __int16)v100;
    }
    sub_37C5A70(v107, v112);
    sub_37BD660(v107, &v138, &v142);
    v108 = v129;
    v111 = *(_DWORD *)(v103 + 840) + 1;
    v109 = v142;
LABEL_66:
    *(_DWORD *)(v103 + 840) = v111;
    if ( *v109 != -1 || v109[1] != -1 )
      --*(_DWORD *)(v103 + 844);
    *v109 = v138;
    v113 = v139;
    *((_DWORD *)v109 + 1) = 0;
    v109[1] = v113;
LABEL_69:
    v114 = (unsigned int)(*((_DWORD *)v109 + 1) + *(_DWORD *)(v103 + 284) + v105);
    v115 = *(_QWORD *)(v3 + 408);
    v116 = *(_QWORD *)(v115 + 64);
    v117 = *(_QWORD *)(v115 + 32);
    v118 = (unsigned int *)(v116 + 4 * v100);
    v119 = *(_QWORD *)(v117 + 8LL * *(unsigned int *)(v116 + 4 * v114));
    v120 = *v118;
    if ( *v118 == -1 )
    {
      v120 = sub_37BA230(*(_QWORD *)(v3 + 408), v108);
      *v118 = v120;
      v117 = *(_QWORD *)(v115 + 32);
    }
    ++v101;
    *(_QWORD *)(v117 + 8LL * v120) = v119;
    if ( !*(v101 - 1) )
      break;
    v126 += *(v101 - 1);
    v84 = *(_QWORD **)(v3 + 16);
    v100 = (unsigned __int16)v126;
    v85 = v136;
  }
  v84 = *(_QWORD **)(v3 + 16);
  v85 = v136;
LABEL_55:
  v88 = (__int16 *)sub_2FF6F50((__int64)v84, v85, *(_QWORD *)(v3 + 24));
  v143 = v89;
  v142 = v88;
  v90 = sub_CA1930(&v142);
  v91 = *(_QWORD *)(v3 + 408);
  v138 = v90;
  v139 = 0;
  v92 = *(_DWORD *)(v91 + 288) * v123;
  v44 = (unsigned __int8)sub_37BD660(v91 + 824, &v138, &v140) == 0;
  v93 = v140;
  if ( v44 )
  {
    v142 = v140;
    v94 = *(_DWORD *)(v91 + 840);
    ++*(_QWORD *)(v91 + 824);
    v95 = v94 + 1;
    v96 = *(_DWORD *)(v91 + 848);
    if ( 4 * v95 >= 3 * v96 )
    {
      v96 *= 2;
    }
    else if ( v96 - *(_DWORD *)(v91 + 844) - v95 > v96 >> 3 )
    {
LABEL_58:
      *(_DWORD *)(v91 + 840) = v95;
      if ( *v93 != -1 || v93[1] != -1 )
        --*(_DWORD *)(v91 + 844);
      *v93 = v138;
      v97 = v139;
      *((_DWORD *)v93 + 1) = 0;
      v93[1] = v97;
      goto LABEL_61;
    }
    sub_37C5A70(v91 + 824, v96);
    sub_37BD660(v91 + 824, &v138, &v142);
    v95 = *(_DWORD *)(v91 + 840) + 1;
    v93 = v142;
    goto LABEL_58;
  }
LABEL_61:
  v98 = *(_QWORD *)(v3 + 408);
  v99 = *(_QWORD *)(*(_QWORD *)(v98 + 32)
                  + 8LL
                  * *(unsigned int *)(*(_QWORD *)(v98 + 64)
                                    + 4LL * (unsigned int)(*((_DWORD *)v93 + 1) + *(_DWORD *)(v91 + 284) + v92)));
  *(_QWORD *)(*(_QWORD *)(v98 + 32) + 8LL * (unsigned int)sub_37BA440(v98, v136)) = v99;
  return v121;
}
