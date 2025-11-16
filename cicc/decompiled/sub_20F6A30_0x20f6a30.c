// Function: sub_20F6A30
// Address: 0x20f6a30
//
void __fastcall sub_20F6A30(__int64 a1)
{
  __int64 v2; // rcx
  __int64 v3; // rsi
  __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // ecx
  __int64 v7; // rdi
  __int64 (*v8)(); // rsi
  __int64 v9; // rax
  __int64 v10; // rcx
  int v11; // r15d
  int v12; // r12d
  int *v13; // r9
  int v14; // esi
  __int64 v15; // r11
  unsigned int *v16; // rax
  unsigned int v17; // edi
  __int64 v18; // rax
  unsigned int v19; // esi
  int v20; // r13d
  __int64 v21; // r9
  unsigned int v22; // edi
  int *v23; // rbx
  int v24; // ecx
  unsigned int v25; // r13d
  char v26; // cl
  unsigned int v27; // esi
  __int64 v28; // rax
  unsigned int v29; // edi
  unsigned int v30; // esi
  __int64 v31; // r9
  int v32; // ebx
  unsigned int v33; // ecx
  unsigned int *v34; // rax
  unsigned int v35; // edx
  __int64 v36; // r12
  __int64 *v37; // rdx
  __int64 v38; // r8
  _QWORD *v39; // r9
  int v40; // eax
  __int64 v41; // rcx
  __int64 *v42; // rax
  __int64 *v43; // r12
  unsigned __int64 v44; // r13
  __int64 v45; // rcx
  _BYTE *v46; // r15
  __int64 v47; // r14
  int v48; // ebx
  unsigned int v49; // ebx
  __int64 v50; // r12
  _BYTE *v51; // rax
  unsigned int v52; // eax
  __int64 v53; // rdx
  __int64 *v54; // r14
  __int64 i; // r15
  unsigned int v56; // r13d
  _QWORD *v57; // rbx
  __int64 v58; // r13
  unsigned __int64 v59; // rdx
  unsigned __int64 v60; // r15
  __int64 j; // r15
  unsigned int v62; // edx
  int v63; // edi
  unsigned int v64; // ecx
  unsigned int *v65; // r9
  int v66; // r8d
  __int64 v67; // rax
  int v68; // r8d
  unsigned int *v69; // rdi
  int v70; // edx
  int v71; // r8d
  int v72; // r8d
  __int64 v73; // r9
  unsigned int *v74; // rcx
  unsigned int v75; // ebx
  int v76; // esi
  unsigned int v77; // edi
  int v78; // r8d
  int v79; // r8d
  __int64 v80; // r9
  unsigned int v81; // ecx
  unsigned int v82; // r11d
  int v83; // edi
  unsigned int *v84; // rsi
  int v85; // edx
  unsigned int *v86; // r8
  int v87; // edx
  int *v88; // rax
  int v89; // eax
  int v90; // eax
  int *v91; // rax
  int v92; // edi
  int v93; // edi
  __int64 v94; // r9
  __int64 v95; // rdx
  int v96; // r8d
  int v97; // esi
  int *v98; // rcx
  int v99; // edi
  int v100; // edi
  int v101; // ecx
  __int64 v102; // r9
  __int64 v103; // r8
  int *v104; // rdx
  int v105; // esi
  unsigned int v106; // edx
  unsigned int v107; // edi
  int *v108; // rdi
  unsigned int v109; // edx
  unsigned int v110; // esi
  int v111; // ecx
  int v112; // eax
  int v113; // eax
  int v114; // esi
  unsigned int *v115; // rcx
  __int64 v116; // [rsp+28h] [rbp-228h]
  __int64 v117; // [rsp+30h] [rbp-220h]
  __int64 v118; // [rsp+38h] [rbp-218h]
  __int64 v119; // [rsp+40h] [rbp-210h]
  __int64 v120; // [rsp+48h] [rbp-208h]
  unsigned int v121; // [rsp+50h] [rbp-200h]
  __int64 v122; // [rsp+50h] [rbp-200h]
  int v123; // [rsp+58h] [rbp-1F8h]
  __int64 v124; // [rsp+58h] [rbp-1F8h]
  unsigned int v125; // [rsp+6Ch] [rbp-1E4h] BYREF
  _BYTE *v126; // [rsp+70h] [rbp-1E0h] BYREF
  __int64 v127; // [rsp+78h] [rbp-1D8h]
  _BYTE v128[16]; // [rsp+80h] [rbp-1D0h] BYREF
  __int64 v129; // [rsp+90h] [rbp-1C0h] BYREF
  __int64 *v130; // [rsp+98h] [rbp-1B8h]
  __int64 v131; // [rsp+A0h] [rbp-1B0h]
  unsigned int v132; // [rsp+A8h] [rbp-1A8h]
  _BYTE *v133; // [rsp+B0h] [rbp-1A0h] BYREF
  __int64 v134; // [rsp+B8h] [rbp-198h]
  _BYTE v135[128]; // [rsp+C0h] [rbp-190h] BYREF
  _QWORD v136[3]; // [rsp+140h] [rbp-110h] BYREF
  __int64 v137; // [rsp+158h] [rbp-F8h]
  __int64 v138; // [rsp+160h] [rbp-F0h]
  __int64 v139; // [rsp+168h] [rbp-E8h]
  __int64 v140; // [rsp+170h] [rbp-E0h]
  __int64 v141; // [rsp+178h] [rbp-D8h]
  int v142; // [rsp+180h] [rbp-D0h]
  char v143; // [rsp+184h] [rbp-CCh]
  __int64 v144; // [rsp+188h] [rbp-C8h]
  __int64 v145; // [rsp+190h] [rbp-C0h]
  _BYTE *v146; // [rsp+198h] [rbp-B8h]
  _BYTE *v147; // [rsp+1A0h] [rbp-B0h]
  __int64 v148; // [rsp+1A8h] [rbp-A8h]
  int v149; // [rsp+1B0h] [rbp-A0h]
  _BYTE v150[32]; // [rsp+1B8h] [rbp-98h] BYREF
  __int64 v151; // [rsp+1D8h] [rbp-78h]
  _BYTE *v152; // [rsp+1E0h] [rbp-70h]
  _BYTE *v153; // [rsp+1E8h] [rbp-68h]
  __int64 v154; // [rsp+1F0h] [rbp-60h]
  int v155; // [rsp+1F8h] [rbp-58h]
  _BYTE v156[80]; // [rsp+200h] [rbp-50h] BYREF

  v2 = *(_QWORD *)(a1 + 56);
  v3 = *(_QWORD *)(a1 + 16);
  v126 = v128;
  v127 = 0x400000000LL;
  v4 = *(_QWORD *)(a1 + 8);
  v136[2] = &v126;
  v5 = *(_QWORD *)(v4 + 40);
  v138 = v3;
  v139 = v2;
  v6 = 0;
  v136[0] = &unk_4A00C10;
  v137 = v5;
  v7 = *(_QWORD *)(v4 + 16);
  v136[1] = 0;
  v8 = *(__int64 (**)())(*(_QWORD *)v7 + 40LL);
  v9 = 0;
  if ( v8 != sub_1D00B00 )
  {
    v9 = ((__int64 (__fastcall *)(__int64, __int64 (*)(), __int64, _QWORD))v8)(v7, v8, v5, 0);
    v6 = v127;
    v5 = v137;
  }
  v140 = v9;
  v146 = v150;
  v147 = v150;
  v142 = v6;
  v141 = a1;
  v143 = 0;
  v144 = 0;
  v145 = 0;
  v148 = 4;
  v149 = 0;
  v151 = 0;
  v152 = v156;
  v153 = v156;
  v154 = 4;
  v155 = 0;
  *(_QWORD *)(v5 + 8) = v136;
  v10 = *(_QWORD *)(a1 + 64);
  v11 = *(_DWORD *)(v10 + 32);
  if ( v11 )
  {
    v12 = 0;
    while ( 1 )
    {
      v18 = *(_QWORD *)(*(_QWORD *)(v10 + 24) + 16LL * (v12 & 0x7FFFFFFF) + 8);
      if ( !v18 )
        goto LABEL_7;
      if ( (*(_BYTE *)(v18 + 3) & 0x10) != 0 )
        break;
      v28 = *(_QWORD *)(v18 + 32);
      if ( !v28 )
        goto LABEL_7;
      if ( (*(_BYTE *)(v28 + 3) & 0x10) != 0 )
        break;
      if ( ++v12 == v11 )
        goto LABEL_19;
LABEL_8:
      v10 = *(_QWORD *)(a1 + 64);
    }
    v19 = *(_DWORD *)(a1 + 360);
    v20 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 312LL) + 4LL * (v12 & 0x7FFFFFFF));
    if ( v19 )
    {
      v21 = *(_QWORD *)(a1 + 344);
      v123 = 37 * v20;
      v22 = (v19 - 1) & (37 * v20);
      v23 = (int *)(v21 + 168LL * v22);
      v24 = *v23;
      if ( v20 == *v23 )
        goto LABEL_13;
      v87 = 1;
      v88 = 0;
      while ( v24 != -1 )
      {
        if ( !v88 && v24 == -2 )
          v88 = v23;
        v22 = (v19 - 1) & (v22 + v87);
        v23 = (int *)(v21 + 168LL * v22);
        v24 = *v23;
        if ( v20 == *v23 )
          goto LABEL_13;
        ++v87;
      }
      if ( v88 )
        v23 = v88;
      v89 = *(_DWORD *)(a1 + 352);
      ++*(_QWORD *)(a1 + 336);
      v90 = v89 + 1;
      if ( 4 * v90 < 3 * v19 )
      {
        if ( v19 - *(_DWORD *)(a1 + 356) - v90 <= v19 >> 3 )
        {
          sub_20EBDE0(a1 + 336, v19);
          v99 = *(_DWORD *)(a1 + 360);
          if ( !v99 )
          {
LABEL_210:
            ++*(_DWORD *)(a1 + 352);
            BUG();
          }
          v100 = v99 - 1;
          v101 = 1;
          v102 = *(_QWORD *)(a1 + 344);
          LODWORD(v103) = v100 & v123;
          v104 = 0;
          v23 = (int *)(v102 + 168LL * (v100 & (unsigned int)v123));
          v105 = *v23;
          v90 = *(_DWORD *)(a1 + 352) + 1;
          if ( v20 != *v23 )
          {
            while ( v105 != -1 )
            {
              if ( v105 == -2 && !v104 )
                v104 = v23;
              v103 = v100 & (unsigned int)(v103 + v101);
              v23 = (int *)(v102 + 168 * v103);
              v105 = *v23;
              if ( v20 == *v23 )
                goto LABEL_120;
              ++v101;
            }
            if ( v104 )
              v23 = v104;
          }
        }
        goto LABEL_120;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 336);
    }
    sub_20EBDE0(a1 + 336, 2 * v19);
    v92 = *(_DWORD *)(a1 + 360);
    if ( !v92 )
      goto LABEL_210;
    v93 = v92 - 1;
    v94 = *(_QWORD *)(a1 + 344);
    LODWORD(v95) = v93 & (37 * v20);
    v23 = (int *)(v94 + 168LL * (unsigned int)v95);
    v96 = *v23;
    v90 = *(_DWORD *)(a1 + 352) + 1;
    if ( v20 != *v23 )
    {
      v97 = 1;
      v98 = 0;
      while ( v96 != -1 )
      {
        if ( !v98 && v96 == -2 )
          v98 = v23;
        v95 = v93 & (unsigned int)(v95 + v97);
        v23 = (int *)(v94 + 168 * v95);
        v96 = *v23;
        if ( v20 == *v23 )
          goto LABEL_120;
        ++v97;
      }
      if ( v98 )
        v23 = v98;
    }
LABEL_120:
    *(_DWORD *)(a1 + 352) = v90;
    if ( *v23 != -1 )
      --*(_DWORD *)(a1 + 356);
    *v23 = v20;
    memset(v23 + 2, 0, 0xA0u);
    *((_BYTE *)v23 + 16) = 1;
    v91 = v23 + 6;
    do
    {
      if ( v91 )
        *v91 = -1;
      ++v91;
    }
    while ( v91 != v23 + 22 );
    *((_QWORD *)v23 + 11) = v23 + 26;
    *((_QWORD *)v23 + 12) = 0x1000000000LL;
LABEL_13:
    v25 = v12 | 0x80000000;
    v26 = v23[4] & 1;
    if ( v26 )
    {
      v13 = v23 + 6;
      v14 = 15;
    }
    else
    {
      v27 = v23[8];
      v13 = (int *)*((_QWORD *)v23 + 3);
      if ( !v27 )
      {
        v62 = v23[4];
        ++*((_QWORD *)v23 + 1);
        v16 = 0;
        v63 = (v62 >> 1) + 1;
        goto LABEL_77;
      }
      v14 = v27 - 1;
    }
    v15 = v14 & (37 * v25);
    v16 = (unsigned int *)&v13[v15];
    v17 = *v16;
    if ( v25 == *v16 )
      goto LABEL_7;
    v85 = 1;
    v86 = 0;
    while ( v17 != -1 )
    {
      if ( !v86 && v17 == -2 )
        v86 = v16;
      v15 = v14 & (unsigned int)(v15 + v85);
      v16 = (unsigned int *)&v13[v15];
      v17 = *v16;
      if ( v25 == *v16 )
        goto LABEL_7;
      ++v85;
    }
    v62 = v23[4];
    if ( v86 )
      v16 = v86;
    ++*((_QWORD *)v23 + 1);
    v63 = (v62 >> 1) + 1;
    if ( v26 )
    {
      v64 = 48;
      v27 = 16;
LABEL_78:
      LODWORD(v65) = 4 * v63;
      v66 = (_DWORD)v23 + 8;
      if ( 4 * v63 >= v64 )
      {
        sub_1F0D2C0((__int64)(v23 + 2), 2 * v27);
        if ( (v23[4] & 1) != 0 )
        {
          v65 = (unsigned int *)(v23 + 6);
          v66 = 15;
        }
        else
        {
          v113 = v23[8];
          v65 = (unsigned int *)*((_QWORD *)v23 + 3);
          if ( !v113 )
          {
LABEL_208:
            v23[4] = (2 * ((unsigned int)v23[4] >> 1) + 2) | v23[4] & 1;
            BUG();
          }
          v66 = v113 - 1;
        }
        v106 = v66 & (37 * v25);
        v16 = &v65[v106];
        v107 = *v16;
        if ( v25 != *v16 )
        {
          v114 = 1;
          v115 = 0;
          while ( v107 != -1 )
          {
            if ( v107 == -2 && !v115 )
              v115 = v16;
            v106 = v66 & (v114 + v106);
            v16 = &v65[v106];
            v107 = *v16;
            if ( v25 == *v16 )
              goto LABEL_144;
            ++v114;
          }
          if ( v115 )
          {
            v62 = v23[4];
            v16 = v115;
            goto LABEL_80;
          }
        }
      }
      else
      {
        if ( v27 - v23[5] - v63 > v27 >> 3 )
        {
LABEL_80:
          v23[4] = (2 * (v62 >> 1) + 2) | v62 & 1;
          if ( *v16 != -1 )
            --v23[5];
          *v16 = v25;
          v67 = (unsigned int)v23[24];
          if ( (unsigned int)v67 >= v23[25] )
          {
            sub_16CD150((__int64)(v23 + 22), v23 + 26, 0, 4, v66, (int)v65);
            v67 = (unsigned int)v23[24];
          }
          *(_DWORD *)(*((_QWORD *)v23 + 11) + 4 * v67) = v25;
          ++v23[24];
LABEL_7:
          if ( ++v12 == v11 )
            goto LABEL_19;
          goto LABEL_8;
        }
        sub_1F0D2C0((__int64)(v23 + 2), v27);
        if ( (v23[4] & 1) != 0 )
        {
          v108 = v23 + 6;
          v66 = 15;
        }
        else
        {
          v112 = v23[8];
          v108 = (int *)*((_QWORD *)v23 + 3);
          if ( !v112 )
            goto LABEL_208;
          v66 = v112 - 1;
        }
        v109 = v66 & (37 * v25);
        v16 = (unsigned int *)&v108[v109];
        v110 = *v16;
        if ( v25 != *v16 )
        {
          v111 = 1;
          v65 = 0;
          while ( v110 != -1 )
          {
            if ( !v65 && v110 == -2 )
              v65 = v16;
            v109 = v66 & (v111 + v109);
            v16 = (unsigned int *)&v108[v109];
            v110 = *v16;
            if ( v25 == *v16 )
              goto LABEL_144;
            ++v111;
          }
          if ( v65 )
            v16 = v65;
        }
      }
LABEL_144:
      v62 = v23[4];
      goto LABEL_80;
    }
    v27 = v23[8];
LABEL_77:
    v64 = 3 * v27;
    goto LABEL_78;
  }
LABEL_19:
  v116 = a1 + 248;
  v117 = *(_QWORD *)(a1 + 320);
  v120 = *(_QWORD *)(a1 + 312);
  if ( v120 != v117 )
  {
    v124 = a1;
    while ( 1 )
    {
      v29 = *(_DWORD *)v120;
      v30 = *(_DWORD *)(v124 + 272);
      v121 = *(_DWORD *)v120;
      if ( !v30 )
        break;
      v31 = *(_QWORD *)(v124 + 256);
      v32 = 37 * v29;
      v33 = (v30 - 1) & (37 * v29);
      v34 = (unsigned int *)(v31 + 16LL * v33);
      v35 = *v34;
      if ( v29 != *v34 )
      {
        v68 = 1;
        v69 = 0;
        while ( v35 != 0x7FFFFFFF )
        {
          if ( !v69 && v35 == 0x80000000 )
            v69 = v34;
          v33 = (v30 - 1) & (v68 + v33);
          v34 = (unsigned int *)(v31 + 16LL * v33);
          v35 = *v34;
          if ( v121 == *v34 )
            goto LABEL_24;
          ++v68;
        }
        if ( v69 )
          v34 = v69;
        ++*(_QWORD *)(v124 + 248);
        v70 = *(_DWORD *)(v124 + 264) + 1;
        if ( 4 * v70 < 3 * v30 )
        {
          if ( v30 - *(_DWORD *)(v124 + 268) - v70 <= v30 >> 3 )
          {
            sub_20EBB90(v116, v30);
            v71 = *(_DWORD *)(v124 + 272);
            if ( !v71 )
              goto LABEL_209;
            v72 = v71 - 1;
            v73 = *(_QWORD *)(v124 + 256);
            v74 = 0;
            v75 = v72 & v32;
            v70 = *(_DWORD *)(v124 + 264) + 1;
            v76 = 1;
            v34 = (unsigned int *)(v73 + 16LL * v75);
            v77 = *v34;
            if ( v121 != *v34 )
            {
              while ( v77 != 0x7FFFFFFF )
              {
                if ( v77 == 0x80000000 && !v74 )
                  v74 = v34;
                v75 = v72 & (v76 + v75);
                v34 = (unsigned int *)(v73 + 16LL * v75);
                v77 = *v34;
                if ( v121 == *v34 )
                  goto LABEL_91;
                ++v76;
              }
              if ( v74 )
                v34 = v74;
            }
          }
LABEL_91:
          *(_DWORD *)(v124 + 264) = v70;
          if ( *v34 != 0x7FFFFFFF )
            --*(_DWORD *)(v124 + 268);
          *((_QWORD *)v34 + 1) = 0;
          v119 = 0;
          *v34 = v121;
          goto LABEL_25;
        }
LABEL_101:
        sub_20EBB90(v116, 2 * v30);
        v78 = *(_DWORD *)(v124 + 272);
        if ( !v78 )
        {
LABEL_209:
          ++*(_DWORD *)(v124 + 264);
          BUG();
        }
        v79 = v78 - 1;
        v80 = *(_QWORD *)(v124 + 256);
        v70 = *(_DWORD *)(v124 + 264) + 1;
        v81 = v79 & (37 * v121);
        v34 = (unsigned int *)(v80 + 16LL * v81);
        v82 = *v34;
        if ( v121 != *v34 )
        {
          v83 = 1;
          v84 = 0;
          while ( v82 != 0x7FFFFFFF )
          {
            if ( v82 == 0x80000000 && !v84 )
              v84 = v34;
            v81 = v79 & (v83 + v81);
            v34 = (unsigned int *)(v80 + 16LL * v81);
            v82 = *v34;
            if ( v121 == *v34 )
              goto LABEL_91;
            ++v83;
          }
          if ( v84 )
            v34 = v84;
        }
        goto LABEL_91;
      }
LABEL_24:
      v119 = *((_QWORD *)v34 + 1);
LABEL_25:
      if ( *(_DWORD *)(v120 + 44) != *(_DWORD *)(v120 + 48) )
      {
        v36 = *(_QWORD *)(v120 + 8);
        v133 = v135;
        v134 = 0x1000000000LL;
        v129 = 0;
        v130 = 0;
        v131 = 0;
        v132 = 0;
        sub_20EEBC0(v124, v119, v36, v120 + 16, (__int64)&v133, (__int64)&v129);
        v118 = *(_QWORD *)(v124 + 24);
        v125 = v121;
        v37 = sub_20EB660((_QWORD *)(v118 + 344), (int *)&v125);
        v40 = v134 | v131;
        if ( (unsigned int)v134 | (unsigned int)v131 )
        {
          sub_1DB9460((__int64)(v37 + 2), v119, v36, *(_QWORD *)v37[10], v38, v39);
          v41 = (unsigned int)v131;
          if ( (_DWORD)v131 )
          {
            v42 = v130;
            v43 = &v130[2 * v132];
            if ( v130 != v43 )
            {
              while ( *v42 == -16 || *v42 == -8 )
              {
                v42 += 2;
                if ( v43 == v42 )
                  goto LABEL_32;
              }
              if ( v42 != v43 )
              {
                v54 = v42;
                for ( i = *v42; ; i = *v54 )
                {
                  v56 = *((_DWORD *)v54 + 2);
                  v57 = (_QWORD *)sub_1F14200((_QWORD *)(v124 + 96), v119, i, v41, v38, (int)v39);
                  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD, _QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(v124 + 72) + 408LL))(
                    *(_QWORD *)(v124 + 72),
                    i,
                    v57,
                    v56,
                    0,
                    v121,
                    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v124 + 64) + 24LL) + 16LL * (v56 & 0x7FFFFFFF))
                  & 0xFFFFFFFFFFFFFFF8LL,
                    *(_QWORD *)(v124 + 80));
                  v41 = v124;
                  v58 = *(_QWORD *)(v124 + 16);
                  v59 = *v57 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( !v59 )
                    BUG();
                  v60 = *v57 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( (*(_QWORD *)v59 & 4) == 0 && (*(_BYTE *)(v59 + 46) & 4) != 0 )
                  {
                    for ( j = *(_QWORD *)v59; ; j = *(_QWORD *)v60 )
                    {
                      v60 = j & 0xFFFFFFFFFFFFFFF8LL;
                      if ( (*(_BYTE *)(v60 + 46) & 4) == 0 )
                        break;
                    }
                  }
                  for ( ; (_QWORD *)v60 != v57; v60 = *(_QWORD *)(v60 + 8) )
                  {
                    while ( 1 )
                    {
                      sub_1DC1550(*(_QWORD *)(v58 + 272), v60, 0);
                      if ( !v60 )
                        BUG();
                      if ( (*(_BYTE *)v60 & 4) == 0 )
                        break;
                      v60 = *(_QWORD *)(v60 + 8);
                      if ( (_QWORD *)v60 == v57 )
                        goto LABEL_59;
                    }
                    while ( (*(_BYTE *)(v60 + 46) & 8) != 0 )
                      v60 = *(_QWORD *)(v60 + 8);
                  }
LABEL_59:
                  v54 += 2;
                  if ( v54 == v43 )
                    break;
                  while ( *v54 == -8 || *v54 == -16 )
                  {
                    v54 += 2;
                    if ( v43 == v54 )
                      goto LABEL_32;
                  }
                  if ( v54 == v43 )
                    break;
                }
              }
            }
          }
LABEL_32:
          v40 = v134;
        }
        v44 = (unsigned __int64)v133;
        v45 = v124;
        v46 = &v133[8 * v40];
        if ( v133 != v46 )
        {
          do
          {
            v47 = *(_QWORD *)v44;
            v48 = *(_DWORD *)(*(_QWORD *)v44 + 40LL);
            *(_QWORD *)(*(_QWORD *)v44 + 16LL) = *(_QWORD *)(*(_QWORD *)(v45 + 72) + 8LL) + 384LL;
            if ( v48 )
            {
              v49 = v48 - 1;
              v50 = 40LL * v49;
              while ( 1 )
              {
                v51 = (_BYTE *)(v50 + *(_QWORD *)(v47 + 32));
                if ( *v51
                  || (v52 = (unsigned __int8)v51[3], (v52 & 0x30) != 0x30)
                  || (v53 = v52, (((v52 & 0x10) != 0) & ((unsigned __int8)v52 >> 6)) != 0) )
                {
                  v50 -= 40;
                  if ( !v49 )
                    break;
                }
                else
                {
                  v122 = v45;
                  v50 -= 40;
                  LOBYTE(v53) = (unsigned __int8)v52 >> 6;
                  sub_1E16C90(v47, v49, v53, v45, v38, v39);
                  v45 = v122;
                  if ( !v49 )
                    break;
                }
                --v49;
              }
            }
            v44 += 8LL;
          }
          while ( v46 != (_BYTE *)v44 );
        }
        sub_21020C0(v136, &v133, 0, 0, *(_QWORD *)(v124 + 32));
        j___libc_free_0(v130);
        if ( v133 != v135 )
          _libc_free((unsigned __int64)v133);
      }
      v120 += 184;
      if ( v117 == v120 )
        goto LABEL_69;
    }
    ++*(_QWORD *)(v124 + 248);
    goto LABEL_101;
  }
LABEL_69:
  *(_QWORD *)(v137 + 8) = 0;
  if ( v153 != v152 )
    _libc_free((unsigned __int64)v153);
  if ( v147 != v146 )
    _libc_free((unsigned __int64)v147);
  if ( v126 != v128 )
    _libc_free((unsigned __int64)v126);
}
