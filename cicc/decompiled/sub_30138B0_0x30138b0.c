// Function: sub_30138B0
// Address: 0x30138b0
//
__int64 __fastcall sub_30138B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  int v4; // r11d
  _QWORD *v5; // rcx
  unsigned int v6; // r8d
  _QWORD *v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rax
  _QWORD **v10; // rdi
  __int64 v11; // rax
  int v12; // ecx
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  unsigned int v18; // esi
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // r8
  int v22; // r10d
  __int64 *v23; // rdx
  unsigned int v24; // edi
  __int64 *v25; // rax
  __int64 v26; // rcx
  unsigned int v27; // esi
  int v28; // r15d
  __int64 v29; // r8
  int v30; // r10d
  unsigned __int64 *v31; // rdx
  unsigned int v32; // edi
  _QWORD *v33; // rax
  unsigned __int64 v34; // rcx
  _DWORD *v35; // rax
  __int64 v36; // r15
  unsigned __int64 v37; // r12
  unsigned __int64 v38; // r14
  int v39; // edx
  unsigned int v40; // edx
  int v41; // eax
  __int64 v42; // rdi
  __int64 v43; // rax
  __int64 v44; // r8
  unsigned int v45; // ecx
  __int64 *v46; // rdx
  __int64 v47; // r10
  int v48; // r12d
  unsigned int v49; // esi
  __int64 v50; // r8
  int v51; // r11d
  unsigned __int64 *v52; // rdx
  unsigned int v53; // edi
  _QWORD *v54; // rax
  unsigned __int64 v55; // rcx
  _DWORD *v56; // rax
  __int64 v57; // rsi
  _QWORD *v58; // rbx
  _QWORD *v59; // r12
  __int64 v60; // rax
  unsigned __int64 *v61; // rax
  unsigned __int64 v62; // r13
  int v64; // r9d
  _QWORD *v65; // r11
  unsigned int v66; // edx
  __int64 v67; // rdi
  int v68; // eax
  int v69; // ecx
  int v70; // eax
  int v71; // edi
  __int64 v72; // rsi
  __int64 v73; // rax
  int v74; // ecx
  unsigned __int64 v75; // r8
  int v76; // r10d
  unsigned __int64 *v77; // r9
  int v78; // eax
  int v79; // eax
  int v80; // edi
  __int64 v81; // rsi
  __int64 v82; // rax
  __int64 v83; // r8
  int v84; // r10d
  __int64 *v85; // r9
  int v86; // eax
  int v87; // eax
  __int64 v88; // rdi
  unsigned __int64 *v89; // r8
  __int64 v90; // r12
  int v91; // r9d
  unsigned __int64 v92; // rsi
  int v93; // eax
  int v94; // eax
  __int64 v95; // rdi
  __int64 *v96; // r8
  __int64 v97; // r15
  int v98; // r9d
  __int64 v99; // rsi
  int v100; // edx
  int v101; // r11d
  int v102; // edi
  int v103; // edi
  __int64 v104; // r8
  __int64 v105; // rax
  int v106; // ecx
  unsigned __int64 v107; // rsi
  int v108; // r10d
  _QWORD *v109; // r9
  int v110; // eax
  int v111; // eax
  int v112; // eax
  __int64 v113; // r9
  int v114; // edi
  __int64 v115; // r15
  unsigned __int64 *v116; // rsi
  unsigned __int64 v117; // r8
  int v118; // r10d
  unsigned __int64 *v119; // r9
  __int64 v120; // [rsp+8h] [rbp-68h]
  __int64 v121; // [rsp+18h] [rbp-58h]
  __int64 v122; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v123; // [rsp+28h] [rbp-48h]
  int v124; // [rsp+30h] [rbp-40h]
  int v125; // [rsp+34h] [rbp-3Ch]
  unsigned int v126; // [rsp+38h] [rbp-38h]

  sub_B2AF20((__int64)&v122, a1);
  v3 = *(_QWORD *)(a1 + 80);
  v121 = a1 + 72;
  v120 = a2 + 64;
  if ( v3 != a1 + 72 )
  {
    while ( 1 )
    {
      if ( !v3 )
        BUG();
      v36 = v3 - 24;
      v37 = *(_QWORD *)(v3 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v37 == v3 + 24 )
        goto LABEL_55;
      if ( !v37 )
LABEL_205:
        BUG();
      v38 = v37 - 24;
      v39 = *(unsigned __int8 *)(v37 - 24);
      if ( (unsigned int)(v39 - 30) > 0xA )
LABEL_55:
        BUG();
      if ( (_BYTE)v39 != 34 )
        goto LABEL_20;
      if ( v126 )
      {
        v4 = 1;
        v5 = 0;
        v6 = (v126 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
        v7 = &v123[2 * v6];
        v8 = *v7;
        if ( v36 == *v7 )
        {
LABEL_4:
          v9 = v7[1];
          v10 = (_QWORD **)(v9 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (v9 & 4) != 0 )
            v10 = (_QWORD **)**v10;
          goto LABEL_6;
        }
        while ( v8 != -4096 )
        {
          if ( v8 == -8192 && !v5 )
            v5 = v7;
          v6 = (v126 - 1) & (v4 + v6);
          v7 = &v123[2 * v6];
          v8 = *v7;
          if ( v36 == *v7 )
            goto LABEL_4;
          ++v4;
        }
        if ( !v5 )
          v5 = v7;
        ++v122;
        v41 = v124 + 1;
        if ( 4 * (v124 + 1) < 3 * v126 )
        {
          if ( v126 - v125 - v41 <= v126 >> 3 )
          {
            sub_B2ACE0((__int64)&v122, v126);
            if ( !v126 )
            {
LABEL_210:
              ++v124;
              BUG();
            }
            v64 = 1;
            v65 = 0;
            v66 = (v126 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
            v41 = v124 + 1;
            v5 = &v123[2 * v66];
            v67 = *v5;
            if ( v36 != *v5 )
            {
              while ( v67 != -4096 )
              {
                if ( !v65 && v67 == -8192 )
                  v65 = v5;
                v66 = (v126 - 1) & (v64 + v66);
                v5 = &v123[2 * v66];
                v67 = *v5;
                if ( v36 == *v5 )
                  goto LABEL_30;
                ++v64;
              }
              if ( v65 )
                v5 = v65;
            }
          }
          goto LABEL_30;
        }
      }
      else
      {
        ++v122;
      }
      sub_B2ACE0((__int64)&v122, 2 * v126);
      if ( !v126 )
        goto LABEL_210;
      v40 = (v126 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
      v41 = v124 + 1;
      v5 = &v123[2 * v40];
      v42 = *v5;
      if ( v36 != *v5 )
      {
        v108 = 1;
        v109 = 0;
        while ( v42 != -4096 )
        {
          if ( v109 || v42 != -8192 )
            v5 = v109;
          v40 = (v126 - 1) & (v108 + v40);
          v42 = v123[2 * v40];
          if ( v36 == v42 )
          {
            v5 = &v123[2 * v40];
            goto LABEL_30;
          }
          ++v108;
          v109 = v5;
          v5 = &v123[2 * v40];
        }
        if ( v109 )
          v5 = v109;
      }
LABEL_30:
      v124 = v41;
      if ( *v5 != -4096 )
        --v125;
      *v5 = v36;
      v10 = 0;
      v5[1] = 0;
LABEL_6:
      v11 = sub_AA4FF0((__int64)v10);
      if ( !v11 )
        goto LABEL_205;
      v12 = *(unsigned __int8 *)(v11 - 24);
      if ( (unsigned int)(v12 - 80) > 1 )
      {
        v16 = *(_QWORD *)(v37 - 88);
        v13 = 0;
        if ( v16 )
          goto LABEL_12;
LABEL_34:
        v43 = *(unsigned int *)(a2 + 56);
        v44 = *(_QWORD *)(a2 + 40);
        if ( !(_DWORD)v43 )
          goto LABEL_12;
        v45 = (v43 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v46 = (__int64 *)(v44 + 16LL * v45);
        v47 = *v46;
        if ( v13 != *v46 )
        {
          v100 = 1;
          while ( v47 != -4096 )
          {
            v101 = v100 + 1;
            v45 = (v43 - 1) & (v100 + v45);
            v46 = (__int64 *)(v44 + 16LL * v45);
            v47 = *v46;
            if ( v13 == *v46 )
              goto LABEL_36;
            v100 = v101;
          }
          goto LABEL_12;
        }
LABEL_36:
        if ( v46 == (__int64 *)(v44 + 16 * v43) )
          goto LABEL_12;
        v48 = *((_DWORD *)v46 + 2);
        if ( v48 == -1 )
          goto LABEL_12;
        v49 = *(_DWORD *)(a2 + 88);
        if ( !v49 )
        {
          ++*(_QWORD *)(a2 + 64);
          goto LABEL_133;
        }
        v50 = *(_QWORD *)(a2 + 72);
        v51 = 1;
        v52 = 0;
        v53 = (v49 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
        v54 = (_QWORD *)(v50 + 16LL * v53);
        v55 = *v54;
        if ( v38 != *v54 )
        {
          while ( v55 != -4096 )
          {
            if ( !v52 && v55 == -8192 )
              v52 = v54;
            v53 = (v49 - 1) & (v51 + v53);
            v54 = (_QWORD *)(v50 + 16LL * v53);
            v55 = *v54;
            if ( v38 == *v54 )
              goto LABEL_40;
            ++v51;
          }
          if ( !v52 )
            v52 = v54;
          v110 = *(_DWORD *)(a2 + 80);
          ++*(_QWORD *)(a2 + 64);
          v106 = v110 + 1;
          if ( 4 * (v110 + 1) >= 3 * v49 )
          {
LABEL_133:
            sub_30132A0(v120, 2 * v49);
            v102 = *(_DWORD *)(a2 + 88);
            if ( !v102 )
              goto LABEL_208;
            v103 = v102 - 1;
            v104 = *(_QWORD *)(a2 + 72);
            LODWORD(v105) = v103 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
            v106 = *(_DWORD *)(a2 + 80) + 1;
            v52 = (unsigned __int64 *)(v104 + 16LL * (unsigned int)v105);
            v107 = *v52;
            if ( v38 != *v52 )
            {
              v118 = 1;
              v119 = 0;
              while ( v107 != -4096 )
              {
                if ( !v119 && v107 == -8192 )
                  v119 = v52;
                v105 = v103 & (unsigned int)(v105 + v118);
                v52 = (unsigned __int64 *)(v104 + 16 * v105);
                v107 = *v52;
                if ( v38 == *v52 )
                  goto LABEL_135;
                ++v118;
              }
              if ( v119 )
                v52 = v119;
            }
          }
          else if ( v49 - *(_DWORD *)(a2 + 84) - v106 <= v49 >> 3 )
          {
            sub_30132A0(v120, v49);
            v111 = *(_DWORD *)(a2 + 88);
            if ( !v111 )
            {
LABEL_208:
              ++*(_DWORD *)(a2 + 80);
LABEL_209:
              BUG();
            }
            v112 = v111 - 1;
            v113 = *(_QWORD *)(a2 + 72);
            v114 = 1;
            LODWORD(v115) = v112 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
            v116 = 0;
            v106 = *(_DWORD *)(a2 + 80) + 1;
            v52 = (unsigned __int64 *)(v113 + 16LL * (unsigned int)v115);
            v117 = *v52;
            if ( v38 != *v52 )
            {
              while ( v117 != -4096 )
              {
                if ( !v116 && v117 == -8192 )
                  v116 = v52;
                v115 = v112 & (unsigned int)(v115 + v114);
                v52 = (unsigned __int64 *)(v113 + 16 * v115);
                v117 = *v52;
                if ( v38 == *v52 )
                  goto LABEL_135;
                ++v114;
              }
              if ( v116 )
                v52 = v116;
            }
          }
LABEL_135:
          *(_DWORD *)(a2 + 80) = v106;
          if ( *v52 != -4096 )
            --*(_DWORD *)(a2 + 84);
          *v52 = v38;
          v56 = v52 + 1;
          *((_DWORD *)v52 + 2) = 0;
          goto LABEL_41;
        }
LABEL_40:
        v56 = v54 + 1;
LABEL_41:
        *v56 = v48;
        v3 = *(_QWORD *)(v3 + 8);
        if ( v121 == v3 )
          break;
      }
      else
      {
        v13 = v11 - 24;
        if ( (_BYTE)v12 == 81 )
        {
          v14 = *(_QWORD *)(v11 - 56);
          v15 = 0;
          if ( (*(_BYTE *)(v14 + 2) & 1) != 0 )
            v15 = *(_QWORD *)(*(_QWORD *)(v14 - 8) + 32LL);
        }
        else
        {
          if ( (_BYTE)v12 != 80 )
            goto LABEL_209;
          v15 = sub_3011DA0(v11 - 24);
        }
        v16 = *(_QWORD *)(v37 - 88);
        if ( v15 == v16 )
          goto LABEL_34;
LABEL_12:
        v17 = sub_AA4FF0(v16);
        v18 = *(_DWORD *)(a2 + 24);
        v19 = v17;
        v20 = v17 - 24;
        if ( v19 )
          v19 = v20;
        if ( !v18 )
        {
          ++*(_QWORD *)a2;
LABEL_109:
          sub_30136D0(a2, 2 * v18);
          v79 = *(_DWORD *)(a2 + 24);
          if ( !v79 )
            goto LABEL_206;
          v80 = v79 - 1;
          v81 = *(_QWORD *)(a2 + 8);
          LODWORD(v82) = (v79 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          v69 = *(_DWORD *)(a2 + 16) + 1;
          v23 = (__int64 *)(v81 + 16LL * (unsigned int)v82);
          v83 = *v23;
          if ( v19 != *v23 )
          {
            v84 = 1;
            v85 = 0;
            while ( v83 != -4096 )
            {
              if ( !v85 && v83 == -8192 )
                v85 = v23;
              v82 = v80 & (unsigned int)(v82 + v84);
              v23 = (__int64 *)(v81 + 16 * v82);
              v83 = *v23;
              if ( v19 == *v23 )
                goto LABEL_84;
              ++v84;
            }
            if ( v85 )
              v23 = v85;
          }
          goto LABEL_84;
        }
        v21 = *(_QWORD *)(a2 + 8);
        v22 = 1;
        v23 = 0;
        v24 = (v18 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v25 = (__int64 *)(v21 + 16LL * v24);
        v26 = *v25;
        if ( v19 == *v25 )
        {
LABEL_16:
          v27 = *(_DWORD *)(a2 + 88);
          v28 = *((_DWORD *)v25 + 2);
          if ( !v27 )
            goto LABEL_87;
          goto LABEL_17;
        }
        while ( v26 != -4096 )
        {
          if ( !v23 && v26 == -8192 )
            v23 = v25;
          v24 = (v18 - 1) & (v22 + v24);
          v25 = (__int64 *)(v21 + 16LL * v24);
          v26 = *v25;
          if ( v19 == *v25 )
            goto LABEL_16;
          ++v22;
        }
        if ( !v23 )
          v23 = v25;
        v68 = *(_DWORD *)(a2 + 16);
        ++*(_QWORD *)a2;
        v69 = v68 + 1;
        if ( 4 * (v68 + 1) >= 3 * v18 )
          goto LABEL_109;
        if ( v18 - *(_DWORD *)(a2 + 20) - v69 <= v18 >> 3 )
        {
          sub_30136D0(a2, v18);
          v93 = *(_DWORD *)(a2 + 24);
          if ( !v93 )
          {
LABEL_206:
            ++*(_DWORD *)(a2 + 16);
            BUG();
          }
          v94 = v93 - 1;
          v95 = *(_QWORD *)(a2 + 8);
          v96 = 0;
          LODWORD(v97) = v94 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          v98 = 1;
          v69 = *(_DWORD *)(a2 + 16) + 1;
          v23 = (__int64 *)(v95 + 16LL * (unsigned int)v97);
          v99 = *v23;
          if ( v19 != *v23 )
          {
            while ( v99 != -4096 )
            {
              if ( !v96 && v99 == -8192 )
                v96 = v23;
              v97 = v94 & (unsigned int)(v97 + v98);
              v23 = (__int64 *)(v95 + 16 * v97);
              v99 = *v23;
              if ( v19 == *v23 )
                goto LABEL_84;
              ++v98;
            }
            if ( v96 )
              v23 = v96;
          }
        }
LABEL_84:
        *(_DWORD *)(a2 + 16) = v69;
        if ( *v23 != -4096 )
          --*(_DWORD *)(a2 + 20);
        *v23 = v19;
        v28 = 0;
        *((_DWORD *)v23 + 2) = 0;
        v27 = *(_DWORD *)(a2 + 88);
        if ( !v27 )
        {
LABEL_87:
          ++*(_QWORD *)(a2 + 64);
          goto LABEL_88;
        }
LABEL_17:
        v29 = *(_QWORD *)(a2 + 72);
        v30 = 1;
        v31 = 0;
        v32 = (v27 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
        v33 = (_QWORD *)(v29 + 16LL * v32);
        v34 = *v33;
        if ( v38 != *v33 )
        {
          while ( v34 != -4096 )
          {
            if ( !v31 && v34 == -8192 )
              v31 = v33;
            v32 = (v27 - 1) & (v30 + v32);
            v33 = (_QWORD *)(v29 + 16LL * v32);
            v34 = *v33;
            if ( v38 == *v33 )
              goto LABEL_18;
            ++v30;
          }
          if ( !v31 )
            v31 = v33;
          v78 = *(_DWORD *)(a2 + 80);
          ++*(_QWORD *)(a2 + 64);
          v74 = v78 + 1;
          if ( 4 * (v78 + 1) >= 3 * v27 )
          {
LABEL_88:
            sub_30132A0(v120, 2 * v27);
            v70 = *(_DWORD *)(a2 + 88);
            if ( !v70 )
              goto LABEL_208;
            v71 = v70 - 1;
            v72 = *(_QWORD *)(a2 + 72);
            LODWORD(v73) = (v70 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
            v74 = *(_DWORD *)(a2 + 80) + 1;
            v31 = (unsigned __int64 *)(v72 + 16LL * (unsigned int)v73);
            v75 = *v31;
            if ( v38 != *v31 )
            {
              v76 = 1;
              v77 = 0;
              while ( v75 != -4096 )
              {
                if ( !v77 && v75 == -8192 )
                  v77 = v31;
                v73 = v71 & (unsigned int)(v73 + v76);
                v31 = (unsigned __int64 *)(v72 + 16 * v73);
                v75 = *v31;
                if ( v38 == *v31 )
                  goto LABEL_105;
                ++v76;
              }
              if ( v77 )
                v31 = v77;
            }
          }
          else if ( v27 - *(_DWORD *)(a2 + 84) - v74 <= v27 >> 3 )
          {
            sub_30132A0(v120, v27);
            v86 = *(_DWORD *)(a2 + 88);
            if ( !v86 )
              goto LABEL_208;
            v87 = v86 - 1;
            v88 = *(_QWORD *)(a2 + 72);
            v89 = 0;
            LODWORD(v90) = v87 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
            v91 = 1;
            v74 = *(_DWORD *)(a2 + 80) + 1;
            v31 = (unsigned __int64 *)(v88 + 16LL * (unsigned int)v90);
            v92 = *v31;
            if ( v38 != *v31 )
            {
              while ( v92 != -4096 )
              {
                if ( !v89 && v92 == -8192 )
                  v89 = v31;
                v90 = v87 & (unsigned int)(v90 + v91);
                v31 = (unsigned __int64 *)(v88 + 16 * v90);
                v92 = *v31;
                if ( v38 == *v31 )
                  goto LABEL_105;
                ++v91;
              }
              if ( v89 )
                v31 = v89;
            }
          }
LABEL_105:
          *(_DWORD *)(a2 + 80) = v74;
          if ( *v31 != -4096 )
            --*(_DWORD *)(a2 + 84);
          *v31 = v38;
          v35 = v31 + 1;
          *((_DWORD *)v31 + 2) = 0;
          goto LABEL_19;
        }
LABEL_18:
        v35 = v33 + 1;
LABEL_19:
        *v35 = v28;
LABEL_20:
        v3 = *(_QWORD *)(v3 + 8);
        if ( v121 == v3 )
          break;
      }
    }
  }
  v57 = v126;
  if ( v126 )
  {
    v58 = v123;
    v59 = &v123[2 * v126];
    do
    {
      if ( *v58 != -4096 && *v58 != -8192 )
      {
        v60 = v58[1];
        if ( v60 )
        {
          if ( (v60 & 4) != 0 )
          {
            v61 = (unsigned __int64 *)(v60 & 0xFFFFFFFFFFFFFFF8LL);
            v62 = (unsigned __int64)v61;
            if ( v61 )
            {
              if ( (unsigned __int64 *)*v61 != v61 + 2 )
                _libc_free(*v61);
              j_j___libc_free_0(v62);
            }
          }
        }
      }
      v58 += 2;
    }
    while ( v59 != v58 );
    v57 = v126;
  }
  return sub_C7D6A0((__int64)v123, 16 * v57, 8);
}
