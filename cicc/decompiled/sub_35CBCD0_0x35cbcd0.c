// Function: sub_35CBCD0
// Address: 0x35cbcd0
//
void __fastcall sub_35CBCD0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // edx
  unsigned int v10; // edi
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 *v17; // rcx
  __int64 v18; // rcx
  __int64 v19; // rdx
  unsigned int v20; // edi
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r10
  __int64 v29; // r9
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rsi
  __int64 *v33; // rax
  unsigned int v34; // edi
  __int64 v35; // rdx
  __int64 v36; // r9
  unsigned int v37; // ecx
  __int64 *v38; // rdx
  __int64 *v39; // rcx
  __int64 v40; // rcx
  char v41; // al
  __int64 v42; // rax
  int v43; // edi
  __int64 v44; // r10
  __int64 v45; // r9
  int v46; // r11d
  __int64 v47; // r14
  __int64 *v48; // rsi
  __int64 v49; // rbx
  __int64 *v50; // rdx
  _QWORD **v51; // r13
  __int64 j; // rdx
  __int64 *v53; // rax
  __int64 v54; // r12
  _QWORD *v55; // rsi
  unsigned int i; // edi
  _QWORD **v57; // rax
  _QWORD *v58; // rax
  __int64 v59; // rcx
  __int64 *v60; // rdi
  __int64 *v61; // rdx
  __int64 *n; // r8
  __int64 v63; // rsi
  __int64 v64; // rax
  int v65; // edx
  unsigned int v66; // r9d
  __int64 v67; // rax
  unsigned int v68; // edx
  __int64 v69; // r11
  unsigned int v70; // esi
  __int64 v71; // rdx
  __int64 v72; // rsi
  __int64 v73; // rax
  __int64 v74; // rdi
  __int64 v75; // rsi
  unsigned int v76; // edx
  unsigned int v77; // r8d
  __int64 v78; // rax
  __int64 v79; // r9
  unsigned int v80; // esi
  __int64 v81; // rdx
  __int64 v82; // rsi
  __int64 v83; // r12
  int v84; // eax
  __int64 *v85; // r13
  int v86; // r13d
  int v87; // eax
  __int64 v88; // rcx
  int v89; // esi
  __int64 *v90; // rax
  __int64 v91; // rdi
  __int64 v92; // rdi
  _BYTE *v93; // r12
  __int64 v94; // rbx
  _BYTE *v95; // r9
  __int64 v96; // rcx
  __int64 *v97; // rdi
  __int64 v98; // r9
  __int64 v99; // r10
  _BYTE *v100; // r11
  __int64 *v101; // rdx
  __int64 *v102; // r14
  unsigned int v103; // r13d
  __int64 v104; // rsi
  __int64 v105; // r8
  unsigned int v106; // edx
  __int64 v107; // rax
  __int64 v108; // r8
  unsigned int v109; // esi
  __int64 v110; // rdx
  __int64 v111; // rsi
  __int64 v112; // rdx
  unsigned int v113; // eax
  __int64 v114; // rax
  int v115; // ecx
  __int64 v116; // rsi
  int v117; // ecx
  unsigned int v118; // edx
  __int64 *v119; // rax
  __int64 v120; // rdi
  _QWORD **v121; // rax
  _QWORD *v122; // rax
  unsigned int k; // edx
  unsigned int v124; // edi
  __int64 *v125; // rax
  __int64 v126; // r9
  _QWORD **v127; // rax
  _QWORD *v128; // rax
  unsigned int m; // ecx
  int v130; // eax
  int v131; // eax
  __int64 v132; // r12
  unsigned int v133; // r8d
  int v134; // edx
  unsigned int v135; // esi
  __int64 *v136; // rdx
  __int64 v137; // rdi
  int v138; // r13d
  int v139; // eax
  unsigned __int64 v140; // r13
  __int64 v141; // rax
  int v142; // r13d
  __int64 *v143; // rax
  int v144; // esi
  int v145; // eax
  int v146; // edx
  int v147; // r9d
  int v148; // eax
  __int64 v149; // r13
  int v150; // r10d
  int v151; // r9d
  int v152; // r13d
  int v153; // [rsp+14h] [rbp-6Ch]
  __int64 v154; // [rsp+18h] [rbp-68h]
  _BYTE *v155; // [rsp+20h] [rbp-60h] BYREF
  __int64 v156; // [rsp+28h] [rbp-58h]
  _BYTE v157[80]; // [rsp+30h] [rbp-50h] BYREF

  v6 = a1[67];
  if ( !v6 )
  {
    v18 = a1[68];
    a1[67] = a2;
    if ( v18 )
      goto LABEL_14;
LABEL_172:
    a1[68] = a2;
LABEL_173:
    v140 = sub_2E313E0(a2);
    if ( v140 == a2 + 48 )
    {
LABEL_196:
      v27 = a1[68];
      goto LABEL_22;
    }
    while ( !(unsigned __int8)sub_35CB4A0((__int64)a1, v140, a3, 1) )
    {
      if ( !v140 )
        BUG();
      if ( (*(_BYTE *)v140 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v140 + 44) & 8) != 0 )
          v140 = *(_QWORD *)(v140 + 8);
      }
      v140 = *(_QWORD *)(v140 + 8);
      if ( v140 == a2 + 48 )
        goto LABEL_196;
    }
    if ( *(_DWORD *)(a2 + 120) )
    {
      v141 = sub_35C9ED0(
               a1[68],
               *(_QWORD *)(a1[68] + 112LL),
               *(_QWORD *)(a1[68] + 112LL) + 8LL * *(unsigned int *)(a1[68] + 120LL));
      a1[68] = v141;
      v27 = v141;
      goto LABEL_22;
    }
LABEL_42:
    a1[68] = 0;
    return;
  }
  v7 = *(_QWORD *)(*(_QWORD *)(v6 + 32) + 328LL);
  if ( v6 != v7 && a2 != v7 )
  {
    v8 = a1[65];
    v9 = *(_DWORD *)(a2 + 24);
    v10 = *(_DWORD *)(v8 + 32);
    v11 = (unsigned int)(*(_DWORD *)(v6 + 24) + 1);
    if ( (unsigned int)v11 >= v10 )
    {
      v16 = (unsigned int)(v9 + 1);
      if ( v10 <= (unsigned int)v16 )
        BUG();
      v12 = *(_QWORD *)(v8 + 24);
      v15 = 0;
    }
    else
    {
      v12 = *(_QWORD *)(v8 + 24);
      v13 = v9 + 1;
      v14 = 0;
      v15 = *(__int64 **)(v12 + 8 * v11);
      if ( v10 <= v13 )
      {
LABEL_11:
        while ( v15 != v14 )
        {
          if ( *((_DWORD *)v15 + 4) < *((_DWORD *)v14 + 4) )
          {
            v17 = v15;
            v15 = v14;
            v14 = v17;
          }
          v15 = (__int64 *)v15[1];
        }
        v7 = *v14;
        goto LABEL_13;
      }
      v16 = v13;
    }
    v14 = *(__int64 **)(v12 + 8 * v16);
    goto LABEL_11;
  }
LABEL_13:
  v18 = a1[68];
  a1[67] = v7;
  if ( !v18 )
    goto LABEL_172;
LABEL_14:
  v19 = a1[66];
  v20 = *(_DWORD *)(v19 + 56);
  v21 = (unsigned int)(*(_DWORD *)(a2 + 24) + 1);
  if ( v20 <= (unsigned int)v21 )
    goto LABEL_42;
  v22 = *(_QWORD *)(v19 + 48);
  v23 = *(_QWORD *)(v22 + 8 * v21);
  if ( !v23 )
    goto LABEL_42;
  v24 = (unsigned int)(*(_DWORD *)(v18 + 24) + 1);
  if ( v20 <= (unsigned int)v24 )
  {
    v25 = 0;
  }
  else
  {
    v25 = *(_QWORD *)(v22 + 8 * v24);
    if ( v23 == v25 )
      goto LABEL_21;
  }
  do
  {
    if ( *(_DWORD *)(v25 + 16) < *(_DWORD *)(v23 + 16) )
    {
      v26 = v25;
      v25 = v23;
      v23 = v26;
    }
    v25 = *(_QWORD *)(v25 + 8);
  }
  while ( v25 != v23 );
LABEL_21:
  v27 = *(_QWORD *)v23;
  a1[68] = *(_QWORD *)v23;
  if ( a2 == v27 )
    goto LABEL_173;
LABEL_22:
  if ( !v27 )
    return;
  v28 = a1[65];
  v29 = a1[67];
  while ( 1 )
  {
    while ( 1 )
    {
      if ( !(unsigned __int8)sub_2E6D360(v28, v29, v27) )
      {
        v30 = a1[67];
        v27 = a1[68];
        v31 = *(_QWORD *)(*(_QWORD *)(v30 + 32) + 328LL);
        if ( v30 != v31 && v27 != v31 )
        {
          v32 = a1[65];
          v33 = 0;
          v34 = *(_DWORD *)(v32 + 32);
          v35 = (unsigned int)(*(_DWORD *)(v30 + 24) + 1);
          if ( (unsigned int)v35 < v34 )
            v33 = *(__int64 **)(*(_QWORD *)(v32 + 24) + 8 * v35);
          if ( v27 )
          {
            v36 = (unsigned int)(*(_DWORD *)(v27 + 24) + 1);
            v37 = *(_DWORD *)(v27 + 24) + 1;
          }
          else
          {
            v36 = 0;
            v37 = 0;
          }
          v38 = 0;
          if ( v34 > v37 )
            v38 = *(__int64 **)(*(_QWORD *)(v32 + 24) + 8 * v36);
          for ( ; v33 != v38; v33 = (__int64 *)v33[1] )
          {
            if ( *((_DWORD *)v33 + 4) < *((_DWORD *)v38 + 4) )
            {
              v39 = v33;
              v33 = v38;
              v38 = v39;
            }
          }
          v31 = *v38;
        }
        a1[67] = v31;
LABEL_39:
        if ( !v27 )
          return;
        v40 = a1[67];
        goto LABEL_41;
      }
      sub_2EB3EB0(a1[66], a1[68], a1[67]);
      if ( v41 )
        break;
      v73 = a1[68];
      v74 = a1[66];
      v75 = 0;
      v76 = 0;
      v40 = a1[67];
      if ( v73 )
      {
        v75 = (unsigned int)(*(_DWORD *)(v73 + 24) + 1);
        v76 = *(_DWORD *)(v73 + 24) + 1;
      }
      v77 = *(_DWORD *)(v74 + 56);
      v78 = 0;
      if ( v76 < v77 )
        v78 = *(_QWORD *)(*(_QWORD *)(v74 + 48) + 8 * v75);
      if ( v40 )
      {
        v79 = (unsigned int)(*(_DWORD *)(v40 + 24) + 1);
        v80 = *(_DWORD *)(v40 + 24) + 1;
      }
      else
      {
        v79 = 0;
        v80 = 0;
      }
      v81 = 0;
      if ( v77 > v80 )
        v81 = *(_QWORD *)(*(_QWORD *)(v74 + 48) + 8 * v79);
      while ( v78 != v81 )
      {
        if ( *(_DWORD *)(v78 + 16) < *(_DWORD *)(v81 + 16) )
        {
          v82 = v78;
          v78 = v81;
          v81 = v82;
        }
        v78 = *(_QWORD *)(v78 + 8);
      }
      v27 = *(_QWORD *)v81;
      a1[68] = *(_QWORD *)v81;
      if ( !v27 )
        return;
      v45 = a1[70];
      v43 = *(_DWORD *)(v45 + 24);
      if ( v43 )
        goto LABEL_92;
LABEL_41:
      v28 = a1[65];
      v29 = v40;
    }
    v42 = a1[70];
    v40 = a1[67];
    v43 = *(_DWORD *)(v42 + 24);
    v44 = *(_QWORD *)(v42 + 8);
    v45 = v42;
    if ( !v43 )
      return;
    v46 = v43 - 1;
    v47 = (v43 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
    v48 = (__int64 *)(v44 + 16 * v47);
    v49 = *v48;
    v50 = v48;
    if ( v40 != *v48 )
      break;
LABEL_47:
    v27 = a1[68];
    if ( !v50[1] )
      goto LABEL_157;
    if ( !v27 )
      return;
    if ( v40 == v49 )
    {
LABEL_50:
      v51 = (_QWORD **)v48[1];
      LODWORD(j) = v46 & (((unsigned int)v27 >> 4) ^ ((unsigned int)v27 >> 9));
      v53 = (__int64 *)(v44 + 16LL * (unsigned int)j);
      v54 = *v53;
      if ( v51 )
        goto LABEL_51;
      goto LABEL_95;
    }
LABEL_93:
    v83 = v49;
    LODWORD(j) = v47;
    v84 = 1;
    while ( v83 != -4096 )
    {
      v142 = v84 + 1;
      j = v46 & (unsigned int)(v84 + j);
      v143 = (__int64 *)(v44 + 16LL * (unsigned int)j);
      v83 = *v143;
      if ( v40 == *v143 )
      {
        v86 = 1;
        if ( !v143[1] )
          break;
LABEL_98:
        while ( v49 != -4096 )
        {
          LODWORD(v47) = v46 & (v86 + v47);
          v48 = (__int64 *)(v44 + 16LL * (unsigned int)v47);
          v49 = *v48;
          if ( v40 == *v48 )
          {
            LODWORD(j) = v46 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
            v53 = (__int64 *)(v44 + 16LL * (unsigned int)j);
            v54 = *v53;
            goto LABEL_166;
          }
          ++v86;
        }
        if ( !v43 )
          goto LABEL_100;
        i = 0;
        LODWORD(j) = v46 & (((unsigned int)v27 >> 4) ^ ((unsigned int)v27 >> 9));
        v53 = (__int64 *)(v44 + 16LL * (unsigned int)j);
        v54 = *v53;
        goto LABEL_53;
      }
      v84 = v142;
    }
LABEL_95:
    j = v46 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
    v53 = (__int64 *)(v44 + 16 * j);
    v54 = *v53;
    v85 = v53;
    if ( *v53 != v27 )
    {
      v154 = *v53;
      v138 = 1;
      v153 = j;
      while ( v154 != -4096 )
      {
        v148 = v138 + 1;
        v149 = v46 & (unsigned int)(v153 + v138);
        v153 = v149;
        v85 = (__int64 *)(v44 + 16 * v149);
        v154 = *v85;
        if ( *v85 == v27 )
        {
          v53 = (__int64 *)(v44 + 16LL * (v46 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4))));
          goto LABEL_96;
        }
        v138 = v148;
      }
LABEL_163:
      v28 = a1[65];
      v29 = v40;
      goto LABEL_75;
    }
LABEL_96:
    if ( !v85[1] )
      goto LABEL_163;
    v86 = 1;
    if ( v40 != v49 )
      goto LABEL_98;
LABEL_166:
    v51 = (_QWORD **)v48[1];
    if ( !v51 )
    {
      i = 0;
      goto LABEL_53;
    }
LABEL_51:
    v55 = *v51;
    for ( i = 1; v55; ++i )
      v55 = (_QWORD *)*v55;
LABEL_53:
    if ( v54 != v27 )
    {
      v130 = 1;
      while ( v54 != -4096 )
      {
        v144 = v130 + 1;
        LODWORD(j) = v46 & (v130 + j);
        v53 = (__int64 *)(v44 + 16LL * (unsigned int)j);
        v54 = *v53;
        if ( *v53 == v27 )
          goto LABEL_54;
        v130 = v144;
      }
LABEL_148:
      j = 0;
      goto LABEL_57;
    }
LABEL_54:
    v57 = (_QWORD **)v53[1];
    if ( !v57 )
      goto LABEL_148;
    v58 = *v57;
    for ( j = 1; v58; j = (unsigned int)(j + 1) )
      v58 = (_QWORD *)*v58;
LABEL_57:
    if ( (unsigned int)j >= i )
    {
LABEL_100:
      v155 = v157;
      v156 = 0x400000000LL;
      v87 = *(_DWORD *)(v45 + 24);
      v88 = *(_QWORD *)(v45 + 8);
      if ( v87 )
      {
        v89 = v87 - 1;
        j = (v87 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v90 = (__int64 *)(v88 + 16 * j);
        v91 = *v90;
        if ( *v90 == v27 )
        {
LABEL_102:
          v92 = v90[1];
          goto LABEL_103;
        }
        v131 = 1;
        while ( v91 != -4096 )
        {
          v45 = (unsigned int)(v131 + 1);
          j = v89 & (unsigned int)(v131 + j);
          v90 = (__int64 *)(v88 + 16LL * (unsigned int)j);
          v91 = *v90;
          if ( *v90 == v27 )
            goto LABEL_102;
          v131 = v45;
        }
      }
      v92 = 0;
LABEL_103:
      sub_2EA42C0(v92, (__int64)&v155, j, v88, v27, v45);
      v93 = v155;
      v94 = a1[68];
      if ( &v155[8 * (unsigned int)v156] != v155 )
      {
        v95 = v155;
        while ( 1 )
        {
          v97 = (__int64 *)sub_35C9B40(
                             *(_QWORD *)(*(_QWORD *)v95 + 112LL),
                             *(_QWORD *)(*(_QWORD *)v95 + 112LL) + 8LL * *(unsigned int *)(*(_QWORD *)v95 + 120LL),
                             1);
          v102 = v101;
          if ( v97 != v101 )
            break;
LABEL_124:
          if ( v99 == v27 || !v27 )
            goto LABEL_184;
          v95 = (_BYTE *)(v98 + 8);
          if ( v100 == v95 )
            goto LABEL_133;
        }
        v103 = *(_DWORD *)(v96 + 56);
        while ( 1 )
        {
          v104 = *v97;
          if ( v27 )
          {
            v105 = (unsigned int)(*(_DWORD *)(v27 + 24) + 1);
            v106 = v105;
          }
          else
          {
            v105 = 0;
            v106 = 0;
          }
          v107 = 0;
          if ( v106 < v103 )
            v107 = *(_QWORD *)(*(_QWORD *)(v96 + 48) + 8 * v105);
          if ( v104 )
          {
            v108 = (unsigned int)(*(_DWORD *)(v104 + 24) + 1);
            v109 = *(_DWORD *)(v104 + 24) + 1;
          }
          else
          {
            v108 = 0;
            v109 = 0;
          }
          v110 = 0;
          if ( v103 > v109 )
            v110 = *(_QWORD *)(*(_QWORD *)(v96 + 48) + 8 * v108);
          for ( ; v107 != v110; v107 = *(_QWORD *)(v107 + 8) )
          {
            if ( *(_DWORD *)(v107 + 16) < *(_DWORD *)(v110 + 16) )
            {
              v111 = v107;
              v107 = v110;
              v110 = v111;
            }
          }
          v27 = *(_QWORD *)v110;
          if ( *(_QWORD *)v110 )
          {
            v112 = (unsigned int)(*(_DWORD *)(v27 + 24) + 1);
            v113 = *(_DWORD *)(v27 + 24) + 1;
          }
          else
          {
            v112 = 0;
            v113 = 0;
          }
          if ( v113 >= v103 )
            BUG();
          if ( !**(_QWORD **)(*(_QWORD *)(v96 + 48) + 8 * v112) )
            goto LABEL_184;
          if ( v102 == ++v97 )
            goto LABEL_124;
        }
      }
      if ( !v94 )
        goto LABEL_184;
      v27 = a1[68];
LABEL_133:
      v114 = a1[70];
      v115 = *(_DWORD *)(v114 + 24);
      v116 = *(_QWORD *)(v114 + 8);
      if ( !v115 )
        goto LABEL_184;
      v117 = v115 - 1;
      v118 = v117 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v119 = (__int64 *)(v116 + 16LL * v118);
      v120 = *v119;
      if ( v27 == *v119 )
      {
LABEL_135:
        v121 = (_QWORD **)v119[1];
        if ( v121 )
        {
          v122 = *v121;
          for ( k = 1; v122; ++k )
            v122 = (_QWORD *)*v122;
          goto LABEL_138;
        }
      }
      else
      {
        v139 = 1;
        while ( v120 != -4096 )
        {
          v151 = v139 + 1;
          v118 = v117 & (v139 + v118);
          v119 = (__int64 *)(v116 + 16LL * v118);
          v120 = *v119;
          if ( v27 == *v119 )
            goto LABEL_135;
          v139 = v151;
        }
      }
      k = 0;
LABEL_138:
      v124 = v117 & (((unsigned int)v94 >> 9) ^ ((unsigned int)v94 >> 4));
      v125 = (__int64 *)(v116 + 16LL * v124);
      v126 = *v125;
      if ( v94 == *v125 )
      {
LABEL_139:
        v127 = (_QWORD **)v125[1];
        if ( !v127 )
          goto LABEL_184;
        v128 = *v127;
        for ( m = 1; v128; ++m )
          v128 = (_QWORD *)*v128;
        if ( m <= k )
          goto LABEL_184;
        a1[68] = v27;
        if ( v93 != v157 )
        {
          _libc_free((unsigned __int64)v93);
          v27 = a1[68];
        }
        goto LABEL_39;
      }
      v145 = 1;
      while ( v126 != -4096 )
      {
        v150 = v145 + 1;
        v124 = v117 & (v145 + v124);
        v125 = (__int64 *)(v116 + 16LL * v124);
        v126 = *v125;
        if ( v94 == *v125 )
          goto LABEL_139;
        v145 = v150;
      }
LABEL_184:
      a1[68] = 0;
      if ( v93 != v157 )
        _libc_free((unsigned __int64)v93);
      return;
    }
    v60 = (__int64 *)sub_35C9B40(*(_QWORD *)(v40 + 64), *(_QWORD *)(v40 + 64) + 8LL * *(unsigned int *)(v40 + 72), 1);
    for ( n = v61; v60 != n; ++v60 )
    {
      v63 = *v60;
      v64 = *(_QWORD *)(*(_QWORD *)(v29 + 32) + 328LL);
      if ( v64 == v29 || v63 == v64 )
      {
        v29 = *(_QWORD *)(*(_QWORD *)(v29 + 32) + 328LL);
      }
      else
      {
        v65 = *(_DWORD *)(v29 + 24);
        v66 = *(_DWORD *)(v28 + 32);
        v67 = 0;
        v68 = v65 + 1;
        if ( v68 < v66 )
          v67 = *(_QWORD *)(*(_QWORD *)(v28 + 24) + 8LL * v68);
        if ( v63 )
        {
          v69 = (unsigned int)(*(_DWORD *)(v63 + 24) + 1);
          v70 = *(_DWORD *)(v63 + 24) + 1;
        }
        else
        {
          v69 = 0;
          v70 = 0;
        }
        v71 = 0;
        if ( v66 > v70 )
          v71 = *(_QWORD *)(*(_QWORD *)(v28 + 24) + 8 * v69);
        for ( ; v67 != v71; v67 = *(_QWORD *)(v67 + 8) )
        {
          if ( *(_DWORD *)(v67 + 16) < *(_DWORD *)(v71 + 16) )
          {
            v72 = v67;
            v67 = v71;
            v71 = v72;
          }
        }
        v29 = *(_QWORD *)v71;
      }
    }
    if ( v59 == v29 )
    {
      a1[67] = 0;
      return;
    }
    a1[67] = v29;
    if ( !v29 )
      return;
LABEL_75:
    v27 = a1[68];
  }
  v132 = *v48;
  v133 = (v43 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
  v134 = 1;
  while ( v132 != -4096 )
  {
    v152 = v134 + 1;
    v133 = v46 & (v134 + v133);
    v50 = (__int64 *)(v44 + 16LL * v133);
    v132 = *v50;
    if ( v40 == *v50 )
      goto LABEL_47;
    v134 = v152;
  }
  v27 = a1[68];
LABEL_157:
  v135 = v46 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
  v136 = (__int64 *)(v44 + 16LL * v135);
  v137 = *v136;
  if ( *v136 == v27 )
  {
LABEL_158:
    if ( !v136[1] || !v27 )
      return;
    v43 = *(_DWORD *)(v42 + 24);
    v45 = a1[70];
LABEL_92:
    v46 = v43 - 1;
    v44 = *(_QWORD *)(v45 + 8);
    v47 = (v43 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
    v48 = (__int64 *)(v44 + 16 * v47);
    v49 = *v48;
    if ( v40 == *v48 )
      goto LABEL_50;
    goto LABEL_93;
  }
  v146 = 1;
  while ( v137 != -4096 )
  {
    v147 = v146 + 1;
    v135 = v46 & (v146 + v135);
    v136 = (__int64 *)(v44 + 16LL * v135);
    v137 = *v136;
    if ( *v136 == v27 )
      goto LABEL_158;
    v146 = v147;
  }
}
