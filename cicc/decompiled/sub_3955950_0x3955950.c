// Function: sub_3955950
// Address: 0x3955950
//
__int64 __fastcall sub_3955950(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r15
  _QWORD *v5; // r12
  unsigned __int8 v6; // al
  _QWORD *v7; // rcx
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  _QWORD *v12; // r8
  __int64 v13; // rdi
  __int64 v14; // r15
  __int64 v15; // r14
  unsigned int i; // eax
  _QWORD *v17; // rdx
  __int64 v18; // rbx
  unsigned int v19; // r9d
  __int64 *v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 *v23; // r13
  int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rsi
  unsigned int v27; // ecx
  __int64 *v28; // rax
  __int64 v29; // r9
  unsigned int v30; // esi
  __int64 v31; // r12
  unsigned int v32; // r8d
  __int64 v33; // rdi
  unsigned int v34; // ecx
  __int64 v35; // rax
  __int64 v36; // rdx
  unsigned int v37; // eax
  unsigned __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // r9
  unsigned int v42; // r11d
  int j; // r10d
  int v44; // eax
  int v45; // eax
  __int64 v46; // rsi
  int v47; // edi
  unsigned int v48; // ecx
  __int64 v49; // rdx
  __int64 *v50; // rax
  unsigned __int64 v51; // rdx
  __int64 v52; // rax
  __int64 *v53; // rax
  _QWORD *v54; // rax
  int v55; // edi
  unsigned int v56; // eax
  __int64 *v57; // r8
  __int64 v58; // rsi
  __int64 v59; // r13
  __int64 v60; // r8
  unsigned int v61; // esi
  __int64 v62; // rdx
  unsigned int v63; // r9d
  unsigned int v64; // edi
  __int64 v65; // rax
  __int64 v66; // rcx
  unsigned int v67; // eax
  unsigned __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rax
  unsigned int v71; // esi
  int v72; // edi
  unsigned int v73; // ecx
  __int64 v74; // rax
  __int64 *v75; // rax
  int v76; // r8d
  int v77; // r9d
  unsigned __int64 v78; // rdx
  __int64 v79; // rax
  __int64 *v80; // rax
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // r12
  __int64 v84; // rcx
  int v85; // eax
  __int64 v86; // rsi
  __int64 v87; // rdx
  __int64 *v88; // rax
  __int64 v89; // rdx
  int v90; // eax
  __int64 v91; // r10
  int m; // r11d
  int v93; // r11d
  __int64 v94; // r10
  int v95; // eax
  int v96; // edx
  __int64 v97; // rcx
  __int64 v98; // r8
  unsigned int v99; // r10d
  int k; // r9d
  int v101; // r8d
  unsigned int v102; // edx
  __int64 v103; // rsi
  int v104; // r9d
  __int64 *v105; // r8
  int v106; // r8d
  __int64 *v107; // rdi
  unsigned int v108; // r12d
  __int64 v109; // rcx
  unsigned int v110; // edx
  __int64 v111; // rcx
  unsigned int v112; // esi
  __int64 *v113; // rax
  __int64 v114; // r8
  __int64 v115; // r12
  __int64 *v116; // rax
  int v117; // r8d
  int v118; // r11d
  __int64 v119; // r9
  int v120; // edx
  __int64 v121; // rcx
  int v122; // r9d
  int v123; // r11d
  __int64 *v124; // r9
  int v125; // eax
  int v126; // r9d
  int v127; // r10d
  __int64 *v128; // r11
  unsigned int v129; // eax
  __int64 *v130; // r10
  unsigned int v131; // eax
  unsigned int v132; // [rsp+8h] [rbp-E8h]
  __int64 v133; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v134; // [rsp+10h] [rbp-E0h]
  __int64 v135; // [rsp+20h] [rbp-D0h]
  int v136; // [rsp+28h] [rbp-C8h]
  __int64 v137; // [rsp+28h] [rbp-C8h]
  __int64 v138; // [rsp+28h] [rbp-C8h]
  int v139; // [rsp+30h] [rbp-C0h]
  unsigned int v140; // [rsp+34h] [rbp-BCh]
  __int64 v141; // [rsp+38h] [rbp-B8h]
  __int64 v142; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v143; // [rsp+48h] [rbp-A8h] BYREF
  __int64 v144; // [rsp+50h] [rbp-A0h] BYREF
  unsigned __int64 v145; // [rsp+58h] [rbp-98h]
  __int64 v146; // [rsp+60h] [rbp-90h]
  __int64 v147; // [rsp+68h] [rbp-88h]
  _QWORD *v148; // [rsp+70h] [rbp-80h] BYREF
  __int64 v149; // [rsp+78h] [rbp-78h]
  _QWORD v150[14]; // [rsp+80h] [rbp-70h] BYREF

  result = (unsigned int)a2 >> 9;
  v140 = result ^ ((unsigned int)a2 >> 4);
  v141 = *(_QWORD *)(a2 + 8);
  if ( !v141 )
    return result;
  v4 = a2;
  do
  {
    v5 = sub_1648700(v141);
    v6 = *((_BYTE *)v5 + 16);
    if ( v6 <= 0x17u )
      goto LABEL_20;
    if ( v6 == 78 )
    {
      v89 = *(v5 - 3);
      if ( !*(_BYTE *)(v89 + 16) && (*(_BYTE *)(v89 + 33) & 0x20) != 0 )
      {
        if ( (unsigned int)(*(_DWORD *)(v89 + 36) - 35) <= 3 )
          goto LABEL_20;
        if ( (*(_BYTE *)(v89 + 33) & 0x20) != 0 )
        {
          v90 = *(_DWORD *)(v89 + 36);
          if ( v90 == 4 || (unsigned int)(v90 - 116) <= 1 )
            goto LABEL_20;
        }
      }
LABEL_83:
      v8 = v5[5];
      goto LABEL_9;
    }
    if ( v6 != 77 )
      goto LABEL_83;
    if ( (*((_BYTE *)v5 + 23) & 0x40) != 0 )
      v7 = (_QWORD *)*(v5 - 1);
    else
      v7 = &v5[-3 * (*((_DWORD *)v5 + 5) & 0xFFFFFFF)];
    v8 = v7[3 * *((unsigned int *)v5 + 14) + 1 + -1431655765 * (unsigned int)((v141 - (__int64)v7) >> 3)];
LABEL_9:
    v9 = *(_QWORD *)a1;
    if ( *(_BYTE *)(v4 + 16) <= 0x17u )
    {
      v86 = *(_QWORD *)(v9 + 80);
      v87 = v86 - 24;
      if ( !v86 )
        v87 = 0;
      v135 = v87;
    }
    else
    {
      v135 = *(_QWORD *)(v4 + 40);
    }
    v10 = sub_1632FA0(*(_QWORD *)(v9 + 40));
    v11 = sub_3952EB0(v4, v10);
    v139 = v11;
    v134 = HIDWORD(v11);
    if ( *((_BYTE *)v5 + 16) != 77 )
      goto LABEL_12;
    v110 = *(_DWORD *)(a1 + 136);
    v111 = *(_QWORD *)(a1 + 120);
    if ( v110 )
    {
      v112 = (v110 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v113 = (__int64 *)(v111 + 16LL * v112);
      v114 = *v113;
      if ( v8 == *v113 )
        goto LABEL_127;
      v125 = 1;
      while ( v114 != -8 )
      {
        v126 = v125 + 1;
        v112 = (v110 - 1) & (v125 + v112);
        v113 = (__int64 *)(v111 + 16LL * v112);
        v114 = *v113;
        if ( v8 == *v113 )
          goto LABEL_127;
        v125 = v126;
      }
    }
    v113 = (__int64 *)(v111 + 16LL * v110);
LABEL_127:
    v115 = v113[1];
    v148 = (_QWORD *)v4;
    if ( !(unsigned __int8)sub_39555F0(a1, v4, v115) )
    {
      v116 = sub_3953D00(a1 + 56, (__int64 *)&v148);
      *(_QWORD *)(*(_QWORD *)(v115 + 48) + 8LL * (*((_DWORD *)v116 + 2) >> 6)) |= 1LL << *((_DWORD *)v116 + 2);
    }
LABEL_12:
    v12 = v150;
    v150[0] = v8;
    v13 = v4;
    v144 = 0;
    v14 = a1;
    v15 = v13;
    v145 = 0;
    v146 = 0;
    v147 = 0;
    v148 = v150;
    v149 = 0x800000001LL;
    for ( i = 1; ; i = v149 )
    {
      v17 = &v12[i];
      if ( !i )
        break;
      while ( 1 )
      {
        --i;
        v18 = *(v17 - 1);
        LODWORD(v149) = i;
        if ( !(_DWORD)v147 )
        {
          ++v144;
          goto LABEL_108;
        }
        v19 = (v147 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v20 = (__int64 *)(v145 + 8LL * v19);
        v21 = *v20;
        if ( v18 != *v20 )
          break;
LABEL_16:
        --v17;
        if ( !i )
          goto LABEL_17;
      }
      v136 = 1;
      v23 = 0;
      while ( v21 != -8 )
      {
        if ( v21 != -16 || v23 )
          v20 = v23;
        v19 = (v147 - 1) & (v136 + v19);
        v21 = *(_QWORD *)(v145 + 8LL * v19);
        if ( v18 == v21 )
          goto LABEL_16;
        ++v136;
        v23 = v20;
        v20 = (__int64 *)(v145 + 8LL * v19);
      }
      if ( !v23 )
        v23 = v20;
      ++v144;
      v24 = v146 + 1;
      if ( 4 * ((int)v146 + 1) < (unsigned int)(3 * v147) )
      {
        if ( (int)v147 - (v24 + HIDWORD(v146)) > (unsigned int)v147 >> 3 )
          goto LABEL_28;
        sub_13B3D40((__int64)&v144, v147);
        if ( (_DWORD)v147 )
        {
          v106 = 1;
          v107 = 0;
          v108 = (v147 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v23 = (__int64 *)(v145 + 8LL * v108);
          v109 = *v23;
          v24 = v146 + 1;
          if ( v18 != *v23 )
          {
            while ( v109 != -8 )
            {
              if ( v109 == -16 && !v107 )
                v107 = v23;
              v108 = (v147 - 1) & (v106 + v108);
              v23 = (__int64 *)(v145 + 8LL * v108);
              v109 = *v23;
              if ( v18 == *v23 )
                goto LABEL_28;
              ++v106;
            }
            if ( v107 )
              v23 = v107;
          }
          goto LABEL_28;
        }
LABEL_185:
        LODWORD(v146) = v146 + 1;
        BUG();
      }
LABEL_108:
      sub_13B3D40((__int64)&v144, 2 * v147);
      if ( !(_DWORD)v147 )
        goto LABEL_185;
      v102 = (v147 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v23 = (__int64 *)(v145 + 8LL * v102);
      v103 = *v23;
      v24 = v146 + 1;
      if ( v18 != *v23 )
      {
        v104 = 1;
        v105 = 0;
        while ( v103 != -8 )
        {
          if ( v103 == -16 && !v105 )
            v105 = v23;
          v102 = (v147 - 1) & (v104 + v102);
          v23 = (__int64 *)(v145 + 8LL * v102);
          v103 = *v23;
          if ( v18 == *v23 )
            goto LABEL_28;
          ++v104;
        }
        if ( v105 )
          v23 = v105;
      }
LABEL_28:
      LODWORD(v146) = v24;
      if ( *v23 != -8 )
        --HIDWORD(v146);
      *v23 = v18;
      v25 = *(unsigned int *)(v14 + 136);
      v26 = *(_QWORD *)(v14 + 120);
      if ( (_DWORD)v25 )
      {
        v27 = (v25 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v28 = (__int64 *)(v26 + 16LL * v27);
        v29 = *v28;
        if ( v18 == *v28 )
          goto LABEL_32;
        v85 = 1;
        while ( v29 != -8 )
        {
          v117 = v85 + 1;
          v27 = (v25 - 1) & (v85 + v27);
          v28 = (__int64 *)(v26 + 16LL * v27);
          v29 = *v28;
          if ( v18 == *v28 )
            goto LABEL_32;
          v85 = v117;
        }
      }
      v28 = (__int64 *)(v26 + 16 * v25);
LABEL_32:
      v30 = *(_DWORD *)(v14 + 80);
      v31 = v28[1];
      v142 = v15;
      if ( !v30 )
        goto LABEL_39;
      v32 = v30 - 1;
      v33 = *(_QWORD *)(v14 + 64);
      v34 = (v30 - 1) & v140;
      v35 = v33 + 16LL * v34;
      v36 = *(_QWORD *)v35;
      if ( v15 == *(_QWORD *)v35 )
      {
        v37 = *(_DWORD *)(v35 + 8);
        v38 = v37 & 0x3F;
        v39 = 8LL * (v37 >> 6);
        goto LABEL_35;
      }
      v41 = *(_QWORD *)v35;
      v42 = (v30 - 1) & v140;
      for ( j = 1; ; ++j )
      {
        if ( v41 == -8 )
          goto LABEL_39;
        v42 = v32 & (j + v42);
        v41 = *(_QWORD *)(v33 + 16LL * v42);
        if ( v15 == v41 )
          break;
      }
      v118 = 1;
      v119 = 0;
      while ( v36 != -8 )
      {
        if ( v36 != -16 || v119 )
          v35 = v119;
        v34 = v32 & (v118 + v34);
        v130 = (__int64 *)(v33 + 16LL * v34);
        v36 = *v130;
        if ( v15 == *v130 )
        {
          v131 = *((_DWORD *)v130 + 2);
          v38 = v131 & 0x3F;
          v39 = 8LL * (v131 >> 6);
          goto LABEL_35;
        }
        ++v118;
        v119 = v35;
        v35 = v33 + 16LL * v34;
      }
      if ( !v119 )
        v119 = v35;
      ++*(_QWORD *)(v14 + 56);
      v138 = v14 + 56;
      v120 = *(_DWORD *)(v14 + 72) + 1;
      if ( 4 * v120 >= 3 * v30 )
      {
        v30 *= 2;
LABEL_143:
        sub_1BFE340(v138, v30);
        sub_1BFD9C0(v138, &v142, &v143);
        v119 = v143;
        v121 = v142;
        v120 = *(_DWORD *)(v14 + 72) + 1;
        goto LABEL_139;
      }
      v121 = v15;
      if ( v30 - *(_DWORD *)(v14 + 76) - v120 <= v30 >> 3 )
        goto LABEL_143;
LABEL_139:
      *(_DWORD *)(v14 + 72) = v120;
      if ( *(_QWORD *)v119 != -8 )
        --*(_DWORD *)(v14 + 76);
      *(_QWORD *)v119 = v121;
      v38 = 0;
      v39 = 0;
      *(_DWORD *)(v119 + 8) = 0;
LABEL_35:
      v40 = *(_QWORD *)(*(_QWORD *)(v31 + 24) + v39);
      if ( _bittest64(&v40, v38) )
        goto LABEL_36;
LABEL_39:
      if ( *(_BYTE *)(v15 + 16) == 77 )
      {
        if ( v135 == v18 )
        {
          v143 = v15;
          if ( !(unsigned __int8)sub_39557A0(v14, v15, v31) )
          {
            v88 = sub_3953D00(v14 + 56, &v143);
            *(_QWORD *)(*(_QWORD *)(v31 + 24) + 8LL * (*((_DWORD *)v88 + 2) >> 6)) |= 1LL << *((_DWORD *)v88 + 2);
            *(_DWORD *)(v31 + 12) += v134;
            *(_DWORD *)(v31 + 8) += v139;
            *(_DWORD *)v31 += v139;
            *(_DWORD *)(v31 + 4) += v134;
          }
          goto LABEL_36;
        }
      }
      else if ( v135 == v18 )
      {
        goto LABEL_36;
      }
      v44 = *(_DWORD *)(v14 + 80);
      v142 = v15;
      v143 = v15;
      if ( v44 )
      {
        v45 = v44 - 1;
        v46 = *(_QWORD *)(v14 + 64);
        v47 = 1;
        v48 = v45 & v140;
        v49 = *(_QWORD *)(v46 + 16LL * (v45 & v140));
        if ( v15 == v49 )
        {
LABEL_43:
          v137 = v14 + 56;
          v50 = sub_3953D00(v14 + 56, &v143);
          v51 = *((unsigned int *)v50 + 2);
          v52 = *(_QWORD *)(*(_QWORD *)(v31 + 24) + 8LL * (*((_DWORD *)v50 + 2) >> 6));
          if ( _bittest64(&v52, v51) )
            goto LABEL_46;
          v53 = sub_3953D00(v137, &v142);
          goto LABEL_45;
        }
        while ( v49 != -8 )
        {
          v48 = v45 & (v47 + v48);
          v49 = *(_QWORD *)(v46 + 16LL * v48);
          if ( v15 == v49 )
            goto LABEL_43;
          ++v47;
        }
      }
      v137 = v14 + 56;
      v53 = sub_3953D00(v14 + 56, &v142);
LABEL_45:
      *(_QWORD *)(*(_QWORD *)(v31 + 24) + 8LL * (*((_DWORD *)v53 + 2) >> 6)) |= 1LL << *((_DWORD *)v53 + 2);
      *(_DWORD *)(v31 + 8) += v139;
      *(_DWORD *)(v31 + 12) += v134;
      *(_DWORD *)v31 += v139;
      *(_DWORD *)(v31 + 4) += v134;
      do
      {
LABEL_46:
        v18 = *(_QWORD *)(v18 + 8);
        if ( !v18 )
          goto LABEL_36;
        v54 = sub_1648700(v18);
      }
      while ( (unsigned __int8)(*((_BYTE *)v54 + 16) - 25) > 9u );
LABEL_66:
      v82 = *(unsigned int *)(v14 + 136);
      v83 = v54[5];
      v84 = *(_QWORD *)(v14 + 120);
      if ( (_DWORD)v82 )
      {
        v55 = v82 - 1;
        v56 = (v82 - 1) & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
        v57 = (__int64 *)(v84 + 16LL * v56);
        v58 = *v57;
        if ( v83 == *v57 )
        {
          v59 = v57[1];
LABEL_51:
          v60 = v57[1];
        }
        else
        {
          v98 = *v57;
          v99 = (v82 - 1) & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
          for ( k = 1; ; k = v123 )
          {
            if ( v98 == -8 )
            {
              v59 = *(_QWORD *)(v84 + 16LL * (unsigned int)v82 + 8);
              goto LABEL_99;
            }
            v123 = k + 1;
            v99 = v55 & (k + v99);
            v124 = (__int64 *)(v84 + 16LL * v99);
            v98 = *v124;
            if ( v83 == *v124 )
              break;
          }
          v59 = v124[1];
LABEL_99:
          v101 = 1;
          while ( v58 != -8 )
          {
            v122 = v101 + 1;
            v56 = v55 & (v101 + v56);
            v57 = (__int64 *)(v84 + 16LL * v56);
            v58 = *v57;
            if ( v83 == *v57 )
              goto LABEL_51;
            v101 = v122;
          }
          v60 = *(_QWORD *)(v84 + 16 * v82 + 8);
        }
      }
      else
      {
        v59 = *(_QWORD *)(v84 + 8);
        v60 = v59;
      }
      v61 = *(_DWORD *)(v14 + 80);
      v62 = *(_QWORD *)(v14 + 64);
      v142 = v15;
      if ( v61 )
      {
        v63 = v61 - 1;
        v64 = (v61 - 1) & v140;
        v65 = v62 + 16LL * v64;
        v66 = *(_QWORD *)v65;
        if ( v15 != *(_QWORD *)v65 )
        {
          v132 = (v61 - 1) & v140;
          v91 = *(_QWORD *)v65;
          for ( m = 1; ; ++m )
          {
            if ( v91 == -8 )
              goto LABEL_57;
            v132 = v63 & (v132 + m);
            v91 = *(_QWORD *)(v62 + 16LL * v132);
            if ( v15 == v91 )
              break;
          }
          v93 = 1;
          v94 = 0;
          while ( v66 != -8 )
          {
            if ( v94 || v66 != -16 )
              v65 = v94;
            v127 = v93 + 1;
            v64 = v63 & (v93 + v64);
            v128 = (__int64 *)(v62 + 16LL * v64);
            v66 = *v128;
            if ( v15 == *v128 )
            {
              v129 = *((_DWORD *)v128 + 2);
              v68 = v129 & 0x3F;
              v69 = 8LL * (v129 >> 6);
              goto LABEL_55;
            }
            v93 = v127;
            v94 = v65;
            v65 = v62 + 16LL * v64;
          }
          if ( !v94 )
            v94 = v65;
          v95 = *(_DWORD *)(v14 + 72);
          ++*(_QWORD *)(v14 + 56);
          v96 = v95 + 1;
          if ( 4 * (v95 + 1) >= 3 * v61 )
          {
            v133 = v60;
            v61 *= 2;
          }
          else
          {
            v97 = v15;
            if ( v61 - *(_DWORD *)(v14 + 76) - v96 > v61 >> 3 )
            {
LABEL_93:
              *(_DWORD *)(v14 + 72) = v96;
              if ( *(_QWORD *)v94 != -8 )
                --*(_DWORD *)(v14 + 76);
              *(_QWORD *)v94 = v97;
              v68 = 0;
              v69 = 0;
              *(_DWORD *)(v94 + 8) = 0;
              goto LABEL_55;
            }
            v133 = v60;
          }
          sub_1BFE340(v137, v61);
          sub_1BFD9C0(v137, &v142, &v143);
          v94 = v143;
          v97 = v142;
          v60 = v133;
          v96 = *(_DWORD *)(v14 + 72) + 1;
          goto LABEL_93;
        }
        v67 = *(_DWORD *)(v65 + 8);
        v68 = v67 & 0x3F;
        v69 = 8LL * (v67 >> 6);
LABEL_55:
        v70 = *(_QWORD *)(*(_QWORD *)(v60 + 48) + v69);
        if ( _bittest64(&v70, v68) )
          goto LABEL_64;
        v62 = *(_QWORD *)(v14 + 64);
        v61 = *(_DWORD *)(v14 + 80);
      }
LABEL_57:
      v142 = v15;
      v143 = v15;
      if ( v61 )
      {
        v71 = v61 - 1;
        v72 = 1;
        v73 = v71 & v140;
        v74 = *(_QWORD *)(v62 + 16LL * (v71 & v140));
        if ( v15 == v74 )
        {
LABEL_59:
          v75 = sub_3953D00(v137, &v143);
          v78 = *((unsigned int *)v75 + 2);
          v79 = *(_QWORD *)(*(_QWORD *)(v59 + 48) + 8LL * (*((_DWORD *)v75 + 2) >> 6));
          if ( _bittest64(&v79, v78) )
            goto LABEL_61;
        }
        else
        {
          while ( v74 != -8 )
          {
            v73 = v71 & (v72 + v73);
            v74 = *(_QWORD *)(v62 + 16LL * v73);
            if ( v15 == v74 )
              goto LABEL_59;
            ++v72;
          }
        }
      }
      v80 = sub_3953D00(v137, &v142);
      *(_QWORD *)(*(_QWORD *)(v59 + 48) + 8LL * (*((_DWORD *)v80 + 2) >> 6)) |= 1LL << *((_DWORD *)v80 + 2);
LABEL_61:
      v81 = (unsigned int)v149;
      if ( (unsigned int)v149 >= HIDWORD(v149) )
      {
        sub_16CD150((__int64)&v148, v150, 0, 8, v76, v77);
        v81 = (unsigned int)v149;
      }
      v148[v81] = v83;
      LODWORD(v149) = v149 + 1;
LABEL_64:
      while ( 1 )
      {
        v18 = *(_QWORD *)(v18 + 8);
        if ( !v18 )
          break;
        v54 = sub_1648700(v18);
        if ( (unsigned __int8)(*((_BYTE *)v54 + 16) - 25) <= 9u )
          goto LABEL_66;
      }
LABEL_36:
      v12 = v148;
    }
LABEL_17:
    v22 = v15;
    a1 = v14;
    v4 = v22;
    if ( v12 != v150 )
      _libc_free((unsigned __int64)v12);
    j___libc_free_0(v145);
LABEL_20:
    result = *(_QWORD *)(v141 + 8);
    v141 = result;
  }
  while ( result );
  return result;
}
