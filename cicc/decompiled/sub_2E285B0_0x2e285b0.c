// Function: sub_2E285B0
// Address: 0x2e285b0
//
__int64 __fastcall sub_2E285B0(_QWORD *a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r15
  unsigned int v5; // eax
  __int64 v6; // r9
  __int64 v7; // rax
  __int16 *v8; // rax
  unsigned __int64 v9; // rdx
  unsigned __int16 v10; // cx
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // rbx
  __int64 v14; // r8
  unsigned int v16; // esi
  int v17; // r11d
  __int64 v18; // rcx
  unsigned int v19; // edi
  __int64 v20; // rdx
  __int64 v21; // r8
  unsigned int v22; // edx
  __int16 *v23; // rbx
  unsigned int v24; // r15d
  char *v25; // r13
  char *v26; // rax
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rdx
  char v30; // r13
  __int64 v31; // rax
  unsigned int v32; // esi
  __int64 v33; // rdi
  unsigned int v34; // edx
  __int64 v35; // rax
  __int64 v36; // rcx
  unsigned int v37; // edx
  unsigned int *v38; // rbx
  char v39; // r15
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  _QWORD *v43; // r13
  __int64 v44; // rax
  __int64 v45; // rdx
  _QWORD *v46; // r15
  char v47; // r13
  __int64 v48; // rax
  __int64 v49; // r8
  unsigned __int64 v50; // r9
  __int64 v51; // rcx
  __int16 *v52; // rax
  __int16 *v53; // r14
  unsigned __int16 v54; // dx
  __int64 v55; // rbx
  unsigned int v56; // r13d
  char *v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rdi
  __int64 v61; // rdi
  __int16 *v62; // rax
  int v63; // edx
  __int16 *v64; // rbx
  unsigned int v65; // eax
  char *v66; // rdi
  int v67; // ecx
  char *v68; // rdx
  int v69; // eax
  unsigned __int64 v70; // rax
  __int64 *v71; // rsi
  unsigned __int64 v72; // r13
  __int64 *v73; // r14
  unsigned __int64 v74; // rdx
  bool v75; // al
  __int64 v76; // rsi
  __int64 v77; // rdi
  __int64 v78; // rcx
  __int64 v79; // rcx
  int *v80; // rdi
  int *v81; // rax
  unsigned int v82; // eax
  int v83; // eax
  int v84; // eax
  int v85; // r11d
  int v86; // ecx
  int v87; // ecx
  int v88; // r11d
  int v89; // r11d
  unsigned int v90; // edx
  __int64 v91; // r8
  int v92; // edi
  __int64 v93; // rsi
  unsigned int i; // esi
  int v95; // r10d
  int v96; // r10d
  __int64 v97; // r8
  __int64 v98; // rdx
  unsigned int v99; // r13d
  int v100; // esi
  __int64 v101; // rdi
  int v102; // r8d
  int v103; // r8d
  unsigned int v104; // edx
  __int64 v105; // r11
  int v106; // edi
  __int64 v107; // rsi
  unsigned int v108; // eax
  __int64 v109; // rax
  __int64 v110; // rdx
  __int64 v111; // rbx
  unsigned int v112; // eax
  __int64 v113; // rax
  int v114; // r8d
  int v115; // r8d
  __int64 v116; // rdx
  unsigned int v117; // ebx
  int v118; // esi
  __int64 v119; // rdi
  __int64 v120; // [rsp+8h] [rbp-168h]
  __int64 v121; // [rsp+10h] [rbp-160h]
  __int64 v123; // [rsp+28h] [rbp-148h]
  __int64 v124; // [rsp+30h] [rbp-140h]
  __int64 v125; // [rsp+38h] [rbp-138h]
  __int16 *v127; // [rsp+50h] [rbp-120h]
  unsigned int v128; // [rsp+58h] [rbp-118h]
  unsigned int v129; // [rsp+5Ch] [rbp-114h]
  __int64 v130; // [rsp+60h] [rbp-110h]
  __int64 v131; // [rsp+70h] [rbp-100h]
  __int64 v132; // [rsp+78h] [rbp-F8h]
  unsigned int v133; // [rsp+80h] [rbp-F0h]
  _QWORD *v134; // [rsp+88h] [rbp-E8h]
  unsigned int *v135; // [rsp+88h] [rbp-E8h]
  unsigned int v136; // [rsp+88h] [rbp-E8h]
  int v137; // [rsp+90h] [rbp-E0h]
  __int16 *v138; // [rsp+90h] [rbp-E0h]
  __int16 *v139; // [rsp+98h] [rbp-D8h]
  __int64 v140; // [rsp+98h] [rbp-D8h]
  int v141; // [rsp+98h] [rbp-D8h]
  __int64 v142; // [rsp+98h] [rbp-D8h]
  __int64 v143; // [rsp+98h] [rbp-D8h]
  __int64 v144; // [rsp+A8h] [rbp-C8h] BYREF
  __int64 v145; // [rsp+B0h] [rbp-C0h] BYREF
  unsigned int v146; // [rsp+B8h] [rbp-B8h]
  __int64 v147; // [rsp+C0h] [rbp-B0h]
  __int64 v148; // [rsp+C8h] [rbp-A8h]
  __int64 v149; // [rsp+D0h] [rbp-A0h]
  char *v150; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v151; // [rsp+E8h] [rbp-88h]
  _BYTE v152[32]; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v153; // [rsp+110h] [rbp-60h] BYREF
  __int64 v154; // [rsp+118h] [rbp-58h] BYREF
  unsigned __int64 v155; // [rsp+120h] [rbp-50h]
  __int64 *v156; // [rsp+128h] [rbp-48h]
  __int64 *v157; // [rsp+130h] [rbp-40h]
  __int64 v158; // [rsp+138h] [rbp-38h]

  v123 = 8LL * a2;
  v125 = a2;
  v3 = *(_QWORD *)(a1[16] + v123);
  v130 = *(_QWORD *)(a1[13] + v123);
  if ( !(v3 | v130) )
    return 0;
  v4 = (__int64)a1;
  if ( !v3 )
    v3 = *(_QWORD *)(a1[13] + 8LL * a2);
  v144 = v3;
  v121 = (__int64)(a1 + 22);
  v5 = *(_DWORD *)sub_2E263C0((__int64)(a1 + 22), &v144);
  v156 = &v154;
  LODWORD(v154) = 0;
  v128 = v5;
  v150 = v152;
  v151 = 0x800000000LL;
  v7 = a1[12];
  v155 = 0;
  v157 = &v154;
  v158 = 0;
  v120 = 24LL * a2;
  v8 = (__int16 *)(*(_QWORD *)(v7 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v7 + 8) + v120 + 4));
  v9 = (unsigned __int64)(v8 + 1);
  LODWORD(v8) = *v8;
  v10 = a2 + (_WORD)v8;
  v133 = a2 + (_DWORD)v8;
  if ( (_WORD)v8 )
  {
    v11 = a1[13];
    v139 = (__int16 *)v9;
    v12 = v10;
    v129 = 0;
    v124 = 0;
    while ( 1 )
    {
      v13 = *(_QWORD *)(v11 + 8LL * (unsigned __int16)v12);
      if ( v13 && v13 != v130 )
      {
        v16 = *(_DWORD *)(v4 + 200);
        if ( v16 )
        {
          v6 = *(_QWORD *)(v4 + 184);
          v17 = 1;
          v18 = 0;
          v19 = (v16 - 1) & (((unsigned int)v13 >> 4) ^ ((unsigned int)v13 >> 9));
          v20 = v6 + 16LL * v19;
          v21 = *(_QWORD *)v20;
          if ( v13 == *(_QWORD *)v20 )
          {
LABEL_20:
            v22 = *(_DWORD *)(v20 + 8);
            if ( v129 < v22 )
            {
              v129 = v22;
              v124 = v13;
            }
            goto LABEL_9;
          }
          while ( v21 != -4096 )
          {
            if ( !v18 && v21 == -8192 )
              v18 = v20;
            v19 = (v16 - 1) & (v17 + v19);
            v20 = v6 + 16LL * v19;
            v21 = *(_QWORD *)v20;
            if ( v13 == *(_QWORD *)v20 )
              goto LABEL_20;
            ++v17;
          }
          v83 = *(_DWORD *)(v4 + 192);
          if ( !v18 )
            v18 = v20;
          ++*(_QWORD *)(v4 + 176);
          v84 = v83 + 1;
          if ( 4 * v84 < 3 * v16 )
          {
            if ( v16 - *(_DWORD *)(v4 + 196) - v84 <= v16 >> 3 )
            {
              sub_2E261E0(v121, v16);
              v95 = *(_DWORD *)(v4 + 200);
              if ( !v95 )
              {
LABEL_217:
                ++*(_DWORD *)(v4 + 192);
                BUG();
              }
              v96 = v95 - 1;
              v97 = *(_QWORD *)(v4 + 184);
              v98 = 0;
              v99 = v96 & (((unsigned int)v13 >> 4) ^ ((unsigned int)v13 >> 9));
              v100 = 1;
              v84 = *(_DWORD *)(v4 + 192) + 1;
              v18 = v97 + 16LL * v99;
              v101 = *(_QWORD *)v18;
              if ( v13 != *(_QWORD *)v18 )
              {
                while ( v101 != -4096 )
                {
                  if ( v101 == -8192 && !v98 )
                    v98 = v18;
                  v6 = (unsigned int)(v100 + 1);
                  v99 = v96 & (v100 + v99);
                  v18 = v97 + 16LL * v99;
                  v101 = *(_QWORD *)v18;
                  if ( v13 == *(_QWORD *)v18 )
                    goto LABEL_134;
                  ++v100;
                }
                if ( v98 )
                  v18 = v98;
              }
            }
            goto LABEL_134;
          }
        }
        else
        {
          ++*(_QWORD *)(v4 + 176);
        }
        sub_2E261E0(v121, 2 * v16);
        v88 = *(_DWORD *)(v4 + 200);
        if ( !v88 )
          goto LABEL_217;
        v89 = v88 - 1;
        v6 = *(_QWORD *)(v4 + 184);
        v90 = v89 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v84 = *(_DWORD *)(v4 + 192) + 1;
        v18 = v6 + 16LL * v90;
        v91 = *(_QWORD *)v18;
        if ( v13 != *(_QWORD *)v18 )
        {
          v92 = 1;
          v93 = 0;
          while ( v91 != -4096 )
          {
            if ( v91 == -8192 && !v93 )
              v93 = v18;
            v90 = v89 & (v92 + v90);
            v18 = v6 + 16LL * v90;
            v91 = *(_QWORD *)v18;
            if ( v13 == *(_QWORD *)v18 )
              goto LABEL_134;
            ++v92;
          }
          if ( v93 )
            v18 = v93;
        }
LABEL_134:
        *(_DWORD *)(v4 + 192) = v84;
        if ( *(_QWORD *)v18 != -4096 )
          --*(_DWORD *)(v4 + 196);
        *(_QWORD *)v18 = v13;
        *(_DWORD *)(v18 + 8) = 0;
        v11 = *(_QWORD *)(v4 + 104);
        goto LABEL_9;
      }
      v132 = *(_QWORD *)(*(_QWORD *)(v4 + 128) + 8LL * (unsigned __int16)v12);
      if ( !v132 )
        goto LABEL_9;
      v137 = v12;
      v23 = (__int16 *)(*(_QWORD *)(*(_QWORD *)(v4 + 96) + 56LL)
                      + 2LL
                      * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v4 + 96) + 8LL) + 24LL * (unsigned __int16)v12 + 4));
      if ( !v23 )
        goto LABEL_37;
      v131 = v4;
      v24 = v12;
      LODWORD(v145) = v12;
      if ( !v158 )
      {
LABEL_24:
        v25 = &v150[4 * (unsigned int)v151];
        if ( v150 == v25 )
        {
          if ( (unsigned int)v151 <= 7uLL )
          {
LABEL_42:
            if ( (unsigned __int64)(unsigned int)v151 + 1 > HIDWORD(v151) )
            {
              sub_C8D5F0((__int64)&v150, v152, (unsigned int)v151 + 1LL, 4u, v12, v6);
              v25 = &v150[4 * (unsigned int)v151];
            }
            *(_DWORD *)v25 = v24;
            LODWORD(v151) = v151 + 1;
            goto LABEL_29;
          }
        }
        else
        {
          v26 = v150;
          while ( v24 != *(_DWORD *)v26 )
          {
            v26 += 4;
            if ( v25 == v26 )
              goto LABEL_41;
          }
          if ( v25 != v26 )
            goto LABEL_29;
LABEL_41:
          if ( (unsigned int)v151 <= 7uLL )
            goto LABEL_42;
          v127 = v23;
          v38 = (unsigned int *)v150;
          v135 = (unsigned int *)&v150[4 * (unsigned int)v151];
          do
          {
            v41 = sub_B9AB10(&v153, (__int64)&v154, v38);
            v43 = (_QWORD *)v42;
            if ( v42 )
            {
              v39 = v41 || (__int64 *)v42 == &v154 || *v38 < *(_DWORD *)(v42 + 32);
              v40 = sub_22077B0(0x28u);
              *(_DWORD *)(v40 + 32) = *v38;
              sub_220F040(v39, v40, v43, &v154);
              ++v158;
            }
            ++v38;
          }
          while ( v135 != v38 );
          v23 = v127;
        }
        LODWORD(v151) = 0;
        v44 = sub_B996D0((__int64)&v153, (unsigned int *)&v145);
        v46 = (_QWORD *)v45;
        if ( v45 )
        {
          v47 = v44 || (__int64 *)v45 == &v154 || (unsigned int)v145 < *(_DWORD *)(v45 + 32);
          v48 = sub_22077B0(0x28u);
          *(_DWORD *)(v48 + 32) = v145;
          sub_220F040(v47, v48, v46, &v154);
          ++v158;
        }
        goto LABEL_29;
      }
      while ( 1 )
      {
        v28 = sub_B996D0((__int64)&v153, (unsigned int *)&v145);
        if ( !v29 )
        {
LABEL_29:
          v27 = *v23++;
          if ( !(_WORD)v27 )
            break;
          goto LABEL_30;
        }
        v30 = v28 || (__int64 *)v29 == &v154 || v24 < *(_DWORD *)(v29 + 32);
        v134 = (_QWORD *)v29;
        ++v23;
        v31 = sub_22077B0(0x28u);
        *(_DWORD *)(v31 + 32) = v145;
        sub_220F040(v30, v31, v134, &v154);
        ++v158;
        v27 = *(v23 - 1);
        if ( !*(v23 - 1) )
          break;
LABEL_30:
        v137 += v27;
        v24 = (unsigned __int16)v137;
        LODWORD(v145) = (unsigned __int16)v137;
        if ( !v158 )
          goto LABEL_24;
      }
      v4 = v131;
LABEL_37:
      v32 = *(_DWORD *)(v4 + 200);
      if ( !v32 )
      {
        ++*(_QWORD *)(v4 + 176);
LABEL_167:
        sub_2E261E0(v121, 2 * v32);
        v102 = *(_DWORD *)(v4 + 200);
        if ( !v102 )
          goto LABEL_216;
        v103 = v102 - 1;
        v6 = *(_QWORD *)(v4 + 184);
        v87 = *(_DWORD *)(v4 + 192) + 1;
        v104 = v103 & (((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4));
        v35 = v6 + 16LL * v104;
        v105 = *(_QWORD *)v35;
        if ( v132 != *(_QWORD *)v35 )
        {
          v106 = 1;
          v107 = 0;
          while ( v105 != -4096 )
          {
            if ( v105 == -8192 && !v107 )
              v107 = v35;
            v104 = v103 & (v106 + v104);
            v35 = v6 + 16LL * v104;
            v105 = *(_QWORD *)v35;
            if ( v132 == *(_QWORD *)v35 )
              goto LABEL_143;
            ++v106;
          }
          if ( v107 )
            v35 = v107;
        }
        goto LABEL_143;
      }
      v33 = *(_QWORD *)(v4 + 184);
      v34 = (v32 - 1) & (((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4));
      v35 = v33 + 16LL * v34;
      v36 = *(_QWORD *)v35;
      if ( v132 == *(_QWORD *)v35 )
      {
LABEL_39:
        v37 = *(_DWORD *)(v35 + 8);
        v11 = *(_QWORD *)(v4 + 104);
        if ( v128 < v37 )
        {
          v128 = v37;
          v144 = v132;
        }
        goto LABEL_9;
      }
      v85 = 1;
      v6 = 0;
      while ( v36 != -4096 )
      {
        if ( v36 == -8192 && !v6 )
          v6 = v35;
        v34 = (v32 - 1) & (v85 + v34);
        v35 = v33 + 16LL * v34;
        v36 = *(_QWORD *)v35;
        if ( v132 == *(_QWORD *)v35 )
          goto LABEL_39;
        ++v85;
      }
      v86 = *(_DWORD *)(v4 + 192);
      if ( v6 )
        v35 = v6;
      ++*(_QWORD *)(v4 + 176);
      v87 = v86 + 1;
      if ( 4 * v87 >= 3 * v32 )
        goto LABEL_167;
      if ( v32 - *(_DWORD *)(v4 + 196) - v87 <= v32 >> 3 )
      {
        sub_2E261E0(v121, v32);
        v114 = *(_DWORD *)(v4 + 200);
        if ( !v114 )
        {
LABEL_216:
          ++*(_DWORD *)(v4 + 192);
          BUG();
        }
        v115 = v114 - 1;
        v6 = *(_QWORD *)(v4 + 184);
        v116 = 0;
        v117 = v115 & (((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4));
        v118 = 1;
        v87 = *(_DWORD *)(v4 + 192) + 1;
        v35 = v6 + 16LL * v117;
        v119 = *(_QWORD *)v35;
        if ( v132 != *(_QWORD *)v35 )
        {
          while ( v119 != -4096 )
          {
            if ( !v116 && v119 == -8192 )
              v116 = v35;
            v117 = v115 & (v118 + v117);
            v35 = v6 + 16LL * v117;
            v119 = *(_QWORD *)v35;
            if ( v132 == *(_QWORD *)v35 )
              goto LABEL_143;
            ++v118;
          }
          if ( v116 )
            v35 = v116;
        }
      }
LABEL_143:
      *(_DWORD *)(v4 + 192) = v87;
      if ( *(_QWORD *)v35 != -4096 )
        --*(_DWORD *)(v4 + 196);
      *(_DWORD *)(v35 + 8) = 0;
      *(_QWORD *)v35 = v132;
      v11 = *(_QWORD *)(v4 + 104);
LABEL_9:
      v9 = (unsigned int)*v139++;
      if ( !*(v139 - 1) )
        goto LABEL_10;
      v133 += v9;
      v12 = (unsigned __int16)v133;
    }
  }
  v124 = 0;
  v11 = a1[13];
LABEL_10:
  v14 = *(_QWORD *)(v11 + 8 * v125);
  if ( *(_QWORD *)(*(_QWORD *)(v4 + 128) + 8 * v125) )
  {
    if ( v144 != a3 && v14 == v144 )
    {
      if ( v124 )
      {
        v145 = 1610612736;
        v147 = 0;
        v146 = a2;
        v148 = 0;
        v149 = 0;
        sub_2E8F270(v124, &v145, v9, a2, v14, v6);
      }
      else
      {
        v143 = v144;
        v108 = sub_2E8E710(v144, a2, *(_QWORD *)(v4 + 96), 0, 0);
        if ( v108 == -1 )
          BUG();
        v109 = *(_QWORD *)(v143 + 32) + 40LL * v108;
        v110 = *(_QWORD *)(v4 + 96);
        if ( (*(_BYTE *)(v109 + 4) & 4) == 0 || *(_DWORD *)(v109 + 8) == a2 )
        {
          sub_2E8F690(v144, a2, v110, 1);
        }
        else
        {
          sub_2E8F690(v144, a2, v110, 1);
          v111 = v144;
          v112 = sub_2E8E710(v144, a2, 0, 0, 0);
          if ( v112 != -1 )
          {
            v113 = *(_QWORD *)(v111 + 32) + 40LL * v112;
            if ( v113 )
              *(_BYTE *)(v113 + 4) |= 4u;
          }
        }
      }
    }
    else
    {
      sub_2E8F280(v144, a2, *(_QWORD *)(v4 + 96), 1);
    }
    goto LABEL_14;
  }
  sub_2E8F690(v14, a2, *(_QWORD *)(v4 + 96), 1);
  v51 = v120;
  v52 = (__int16 *)(*(_QWORD *)(*(_QWORD *)(v4 + 96) + 56LL)
                  + 2LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v4 + 96) + 8LL) + v120 + 4));
  v53 = v52 + 1;
  LODWORD(v52) = *v52;
  v54 = a2 + (_WORD)v52;
  v136 = a2 + (_DWORD)v52;
  if ( (_WORD)v52 )
  {
    v55 = v54;
    v138 = v53;
    v56 = v54;
    if ( v158 )
      goto LABEL_87;
    while ( 1 )
    {
      v57 = v150;
      v58 = (__int64)&v150[4 * (unsigned int)v151];
      if ( v150 != (char *)v58 )
      {
        while ( v56 != *(_DWORD *)v57 )
        {
          v57 += 4;
          if ( (char *)v58 == v57 )
            goto LABEL_85;
        }
        if ( v57 != (char *)v58 )
          break;
      }
      while ( 1 )
      {
LABEL_85:
        if ( !*v138++ )
          goto LABEL_14;
        v136 += *(v138 - 1);
        v51 = v136;
        v55 = (unsigned __int16)v136;
        v56 = (unsigned __int16)v136;
        if ( !v158 )
          break;
LABEL_87:
        v70 = v155;
        if ( v155 )
        {
          v71 = &v154;
          do
          {
            while ( 1 )
            {
              v51 = *(_QWORD *)(v70 + 16);
              v58 = *(_QWORD *)(v70 + 24);
              if ( v56 <= *(_DWORD *)(v70 + 32) )
                break;
              v70 = *(_QWORD *)(v70 + 24);
              if ( !v58 )
                goto LABEL_92;
            }
            v71 = (__int64 *)v70;
            v70 = *(_QWORD *)(v70 + 16);
          }
          while ( v51 );
LABEL_92:
          if ( v71 != &v154 && v56 >= *((_DWORD *)v71 + 8) )
            goto LABEL_70;
        }
      }
    }
LABEL_70:
    v59 = *(_QWORD *)(v4 + 104);
    v60 = *(_QWORD *)(v59 + v123);
    if ( v60 == *(_QWORD *)(v59 + 8 * v55) )
    {
      v142 = *(_QWORD *)(v59 + v123);
      v82 = sub_2E8E710(v60, v56, 0, 0, 0);
      if ( v82 != -1 )
      {
        v58 = 5LL * v82;
        if ( *(_QWORD *)(v142 + 32) + 40LL * v82 )
        {
LABEL_72:
          v61 = sub_2E26610(v4, v56);
          v140 = 24 * v55;
          if ( v61 )
          {
            sub_2E8F280(v61, v56, *(_QWORD *)(v4 + 96), 1);
            v49 = 24 * v55;
            v62 = (__int16 *)(*(_QWORD *)(*(_QWORD *)(v4 + 96) + 56LL)
                            + 2LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v4 + 96) + 8LL) + v140 + 4));
          }
          else
          {
            sub_2E8F280(v144, v56, *(_QWORD *)(v4 + 96), 1);
            v49 = 24 * v55;
            v62 = (__int16 *)(*(_QWORD *)(*(_QWORD *)(v4 + 96) + 56LL)
                            + 2LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v4 + 96) + 8LL) + v140 + 4));
            if ( v62 )
            {
              for ( i = v56; ; v55 = (unsigned __int16)i )
              {
                ++v62;
                *(_QWORD *)(*(_QWORD *)(v4 + 128) + 8 * v55) = v144;
                if ( !*(v62 - 1) )
                  break;
                i += *(v62 - 1);
              }
              v62 = (__int16 *)(*(_QWORD *)(*(_QWORD *)(v4 + 96) + 56LL)
                              + 2LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v4 + 96) + 8LL) + v140 + 4));
            }
          }
          v63 = *v62;
          v64 = v62 + 1;
          v141 = v63 + v56;
          if ( *v62 )
          {
            v65 = (unsigned __int16)(v63 + v56);
            if ( !v158 )
            {
LABEL_76:
              v66 = v150;
              v67 = v151;
              v68 = &v150[4 * (unsigned int)v151];
              if ( v150 != v68 )
              {
                while ( v65 != *(_DWORD *)v66 )
                {
                  v66 += 4;
                  if ( v68 == v66 )
                    goto LABEL_84;
                }
                if ( v68 != v66 )
                {
                  if ( v68 != v66 + 4 )
                  {
                    memmove(v66, v66 + 4, v68 - (v66 + 4));
                    v67 = v151;
                  }
                  LODWORD(v151) = v67 - 1;
                }
              }
              goto LABEL_84;
            }
            while ( 2 )
            {
              v50 = v155;
              if ( v155 )
              {
                v72 = v155;
                v73 = &v154;
                while ( 1 )
                {
                  while ( v65 > *(_DWORD *)(v72 + 32) )
                  {
                    v72 = *(_QWORD *)(v72 + 24);
                    if ( !v72 )
                      goto LABEL_102;
                  }
                  v74 = *(_QWORD *)(v72 + 16);
                  if ( v65 >= *(_DWORD *)(v72 + 32) )
                    break;
                  v73 = (__int64 *)v72;
                  v72 = *(_QWORD *)(v72 + 16);
                  if ( !v74 )
                  {
LABEL_102:
                    v75 = v73 == &v154;
                    goto LABEL_103;
                  }
                }
                v76 = *(_QWORD *)(v72 + 24);
                if ( v76 )
                {
                  do
                  {
                    while ( 1 )
                    {
                      v77 = *(_QWORD *)(v76 + 16);
                      v78 = *(_QWORD *)(v76 + 24);
                      if ( v65 < *(_DWORD *)(v76 + 32) )
                        break;
                      v76 = *(_QWORD *)(v76 + 24);
                      if ( !v78 )
                        goto LABEL_111;
                    }
                    v73 = (__int64 *)v76;
                    v76 = *(_QWORD *)(v76 + 16);
                  }
                  while ( v77 );
                }
LABEL_111:
                while ( v74 )
                {
                  while ( 1 )
                  {
                    v79 = *(_QWORD *)(v74 + 24);
                    if ( v65 <= *(_DWORD *)(v74 + 32) )
                      break;
                    v74 = *(_QWORD *)(v74 + 24);
                    if ( !v79 )
                      goto LABEL_114;
                  }
                  v72 = v74;
                  v74 = *(_QWORD *)(v74 + 16);
                }
LABEL_114:
                if ( v156 != (__int64 *)v72 || v73 != &v154 )
                {
                  for ( ; v73 != (__int64 *)v72; --v158 )
                  {
                    v80 = (int *)v72;
                    v72 = sub_220EF30(v72);
                    v81 = sub_220F330(v80, &v154);
                    j_j___libc_free_0((unsigned __int64)v81);
                  }
                  goto LABEL_84;
                }
              }
              else
              {
                v75 = 1;
                v73 = &v154;
LABEL_103:
                if ( v156 != v73 || !v75 )
                {
LABEL_84:
                  v69 = *v64++;
                  if ( !(_WORD)v69 )
                    goto LABEL_85;
                  v141 += v69;
                  v65 = (unsigned __int16)v141;
                  if ( !v158 )
                    goto LABEL_76;
                  continue;
                }
              }
              break;
            }
            sub_2E24A60(v155);
            v155 = 0;
            v156 = &v154;
            v157 = &v154;
            v158 = 0;
            goto LABEL_84;
          }
          goto LABEL_85;
        }
      }
      v51 = v123;
      v60 = *(_QWORD *)(*(_QWORD *)(v4 + 104) + v123);
    }
    v146 = v56;
    v145 = 805306368;
    v147 = 0;
    v148 = 0;
    v149 = 0;
    sub_2E8F270(v60, &v145, v58, v51, v49, v50);
    goto LABEL_72;
  }
LABEL_14:
  sub_2E24A60(v155);
  if ( v150 != v152 )
    _libc_free((unsigned __int64)v150);
  return 1;
}
