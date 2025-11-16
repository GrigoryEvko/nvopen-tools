// Function: sub_2158E80
// Address: 0x2158e80
//
void __fastcall sub_2158E80(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // rdx
  int v7; // r12d
  __int64 v8; // r8
  unsigned int v9; // ecx
  unsigned __int64 *v10; // rbx
  unsigned __int64 v11; // rdx
  __int64 v12; // r9
  unsigned int v13; // esi
  int v14; // eax
  int v15; // r11d
  unsigned int v16; // edi
  unsigned int *v17; // rcx
  unsigned int v18; // edx
  unsigned int v19; // esi
  unsigned int v20; // r13d
  unsigned __int64 v21; // r14
  int v22; // eax
  int v23; // eax
  __int64 v24; // rdi
  __int64 v25; // rcx
  unsigned __int64 v26; // rsi
  int v27; // edx
  int v28; // edx
  int v29; // edx
  unsigned __int64 v30; // rdi
  __int64 v31; // rcx
  unsigned int *v32; // r10
  unsigned int v33; // esi
  int v34; // r14d
  unsigned int *v35; // r8
  unsigned int v36; // ebx
  __int64 v37; // rdx
  __int64 v38; // rdi
  unsigned int v39; // r12d
  unsigned int v40; // ecx
  __int64 *v41; // rax
  __int64 v42; // rdx
  int v43; // ecx
  unsigned int v44; // esi
  __int64 v45; // r8
  int v46; // ecx
  int v47; // ecx
  __int64 v48; // r9
  int v49; // edx
  unsigned int v50; // esi
  __int64 v51; // rdi
  _DWORD *v52; // rdx
  _QWORD *v53; // rdi
  __int64 v54; // rax
  int v55; // ecx
  _WORD *v56; // rdx
  __int64 v57; // rdi
  __int64 v58; // rax
  int v59; // ecx
  __int64 v60; // rdi
  _BYTE *v61; // rax
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rdi
  size_t v65; // rdx
  char *v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  int v70; // r11d
  __int64 *v71; // r10
  int v72; // ecx
  int v73; // ecx
  int v74; // ecx
  int v75; // r10d
  __int64 *v76; // r9
  __int64 v77; // rdi
  unsigned int v78; // r12d
  __int64 v79; // rsi
  int v80; // r11d
  unsigned __int64 *v81; // r10
  int v82; // ecx
  int v83; // ecx
  int v84; // ecx
  __int64 v85; // rdi
  unsigned __int64 *v86; // r8
  int v87; // r9d
  __int64 v88; // rax
  unsigned __int64 v89; // rsi
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // r13
  unsigned int v94; // eax
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rax
  int v98; // edx
  int v99; // edx
  unsigned __int64 v100; // rsi
  unsigned int *v101; // rdi
  unsigned int v102; // r14d
  int v103; // r9d
  unsigned int v104; // ecx
  __int64 *v105; // rdi
  int v106; // r11d
  __int64 *v107; // r10
  int v108; // r9d
  int v109; // r8d
  unsigned int v110; // r9d
  int v111; // [rsp+Ch] [rbp-164h]
  int v112; // [rsp+Ch] [rbp-164h]
  int v113; // [rsp+Ch] [rbp-164h]
  __int64 v114; // [rsp+10h] [rbp-160h]
  __int64 v115; // [rsp+10h] [rbp-160h]
  __int64 v116; // [rsp+20h] [rbp-150h]
  __int64 v117; // [rsp+30h] [rbp-140h]
  __int64 *v118; // [rsp+30h] [rbp-140h]
  unsigned int v119; // [rsp+30h] [rbp-140h]
  int v120; // [rsp+38h] [rbp-138h]
  __int64 v121; // [rsp+38h] [rbp-138h]
  __int64 v122; // [rsp+38h] [rbp-138h]
  int v123; // [rsp+38h] [rbp-138h]
  int v124; // [rsp+38h] [rbp-138h]
  int v125; // [rsp+38h] [rbp-138h]
  __int64 v126; // [rsp+38h] [rbp-138h]
  char *v127; // [rsp+40h] [rbp-130h] BYREF
  size_t v128; // [rsp+48h] [rbp-128h]
  __int64 v129; // [rsp+50h] [rbp-120h] BYREF
  char *v130[2]; // [rsp+60h] [rbp-110h] BYREF
  __int64 v131; // [rsp+70h] [rbp-100h] BYREF
  _QWORD v132[2]; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v133; // [rsp+90h] [rbp-E0h]
  _DWORD *v134; // [rsp+98h] [rbp-D8h]
  int v135; // [rsp+A0h] [rbp-D0h]
  unsigned __int64 *v136; // [rsp+A8h] [rbp-C8h]
  unsigned __int64 v137[2]; // [rsp+B0h] [rbp-C0h] BYREF
  _BYTE v138[176]; // [rsp+C0h] [rbp-B0h] BYREF

  v137[0] = (unsigned __int64)v138;
  v136 = v137;
  v137[1] = 0x8000000000LL;
  v132[0] = &unk_49EFC48;
  v135 = 1;
  v134 = 0;
  v133 = 0;
  v132[1] = 0;
  sub_16E7A40((__int64)v132, 0, 0, 0);
  v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 112LL))(*(_QWORD *)(a2 + 16));
  v4 = *(_QWORD *)(a2 + 56);
  v116 = v3;
  v5 = *(_QWORD *)(v4 + 48);
  if ( v5 )
  {
    v90 = sub_1263B40((__int64)v132, "\t.local .align ");
    v91 = sub_16E7A90(v90, *(unsigned int *)(v4 + 60));
    v92 = sub_1263B40(v91, " .b8 \t");
    v93 = sub_1263B40(v92, "__local_depot");
    v94 = sub_396DD70(a1);
    v95 = sub_16E7A90(v93, v94);
    v96 = sub_1263B40(v95, "[");
    v97 = sub_16E7A90(v96, v5);
    sub_1263B40(v97, "];\n");
    if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 936LL) )
    {
      sub_1263B40((__int64)v132, "\t.reg .b64 \t%SP;\n");
      sub_1263B40((__int64)v132, "\t.reg .b64 \t%SPL;\n");
    }
    else
    {
      sub_1263B40((__int64)v132, "\t.reg .b32 \t%SP;\n");
      sub_1263B40((__int64)v132, "\t.reg .b32 \t%SPL;\n");
    }
  }
  v6 = *(_QWORD *)(a1 + 800);
  v114 = a1 + 808;
  v7 = 0;
  v120 = *(_DWORD *)(v6 + 32);
  if ( !v120 )
    goto LABEL_25;
  while ( 2 )
  {
    v19 = *(_DWORD *)(a1 + 832);
    v20 = v7 | 0x80000000;
    v21 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * (v7 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v19 )
    {
      ++*(_QWORD *)(a1 + 808);
      goto LABEL_12;
    }
    v8 = *(_QWORD *)(a1 + 816);
    v9 = (v19 - 1) & (((unsigned int)v21 >> 4) ^ ((unsigned int)v21 >> 9));
    v10 = (unsigned __int64 *)(v8 + 40LL * v9);
    v11 = *v10;
    if ( v21 == *v10 )
    {
      v12 = v10[2];
      v13 = *((_DWORD *)v10 + 8);
      v14 = *((_DWORD *)v10 + 6) + 1;
      v15 = v14;
      goto LABEL_6;
    }
    v80 = 1;
    v81 = 0;
    while ( 1 )
    {
      if ( v11 == -8 )
      {
        v82 = *(_DWORD *)(a1 + 824);
        if ( v81 )
          v10 = v81;
        ++*(_QWORD *)(a1 + 808);
        v27 = v82 + 1;
        if ( 4 * (v82 + 1) < 3 * v19 )
        {
          if ( v19 - *(_DWORD *)(a1 + 828) - v27 > v19 >> 3 )
            goto LABEL_14;
          v119 = ((unsigned int)v21 >> 4) ^ ((unsigned int)v21 >> 9);
          sub_2158190(v114, v19);
          v83 = *(_DWORD *)(a1 + 832);
          if ( v83 )
          {
            v84 = v83 - 1;
            v85 = *(_QWORD *)(a1 + 816);
            v86 = 0;
            v87 = 1;
            LODWORD(v88) = v84 & v119;
            v10 = (unsigned __int64 *)(v85 + 40LL * (v84 & v119));
            v89 = *v10;
            v27 = *(_DWORD *)(a1 + 824) + 1;
            if ( v21 == *v10 )
              goto LABEL_14;
            while ( v89 != -8 )
            {
              if ( !v86 && v89 == -16 )
                v86 = v10;
              v88 = v84 & (unsigned int)(v88 + v87);
              v10 = (unsigned __int64 *)(v85 + 40 * v88);
              v89 = *v10;
              if ( v21 == *v10 )
                goto LABEL_14;
              ++v87;
            }
            goto LABEL_86;
          }
LABEL_151:
          ++*(_DWORD *)(a1 + 824);
          BUG();
        }
LABEL_12:
        sub_2158190(v114, 2 * v19);
        v22 = *(_DWORD *)(a1 + 832);
        if ( v22 )
        {
          v23 = v22 - 1;
          v24 = *(_QWORD *)(a1 + 816);
          LODWORD(v25) = v23 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v10 = (unsigned __int64 *)(v24 + 40LL * (unsigned int)v25);
          v26 = *v10;
          v27 = *(_DWORD *)(a1 + 824) + 1;
          if ( v21 == *v10 )
          {
LABEL_14:
            *(_DWORD *)(a1 + 824) = v27;
            if ( *v10 != -8 )
              --*(_DWORD *)(a1 + 828);
            *v10 = v21;
            v15 = 1;
            v10[1] = 0;
            v10[2] = 0;
            v10[3] = 0;
            *((_DWORD *)v10 + 8) = 0;
            v117 = (__int64)(v10 + 1);
LABEL_17:
            ++v10[1];
            v13 = 0;
            goto LABEL_18;
          }
          v108 = 1;
          v86 = 0;
          while ( v26 != -8 )
          {
            if ( !v86 && v26 == -16 )
              v86 = v10;
            v25 = v23 & (unsigned int)(v25 + v108);
            v10 = (unsigned __int64 *)(v24 + 40 * v25);
            v26 = *v10;
            if ( v21 == *v10 )
              goto LABEL_14;
            ++v108;
          }
LABEL_86:
          if ( v86 )
            v10 = v86;
          goto LABEL_14;
        }
        goto LABEL_151;
      }
      if ( v81 || v11 != -16 )
        v10 = v81;
      v9 = (v19 - 1) & (v80 + v9);
      v105 = (__int64 *)(v8 + 40LL * v9);
      v11 = *v105;
      if ( v21 == *v105 )
        break;
      ++v80;
      v81 = v10;
      v10 = (unsigned __int64 *)(v8 + 40LL * v9);
    }
    v12 = v105[2];
    v10 = (unsigned __int64 *)(v8 + 40LL * v9);
    v13 = *((_DWORD *)v105 + 8);
    v14 = *((_DWORD *)v105 + 6) + 1;
    v15 = v14;
LABEL_6:
    v117 = (__int64)(v10 + 1);
    if ( !v13 )
      goto LABEL_17;
    v16 = (v13 - 1) & (37 * v20);
    v17 = (unsigned int *)(v12 + 8LL * v16);
    v18 = *v17;
    if ( v20 == *v17 )
      goto LABEL_8;
    v112 = 1;
    v32 = 0;
    while ( v18 != -1 )
    {
      if ( v32 || v18 != -2 )
        v17 = v32;
      v16 = (v13 - 1) & (v112 + v16);
      v18 = *(_DWORD *)(v12 + 8LL * v16);
      if ( v20 == v18 )
        goto LABEL_8;
      ++v112;
      v32 = v17;
      v17 = (unsigned int *)(v12 + 8LL * v16);
    }
    if ( !v32 )
      v32 = v17;
    ++v10[1];
    if ( 4 * v14 >= 3 * v13 )
    {
LABEL_18:
      v111 = v15;
      sub_1392B70(v117, 2 * v13);
      v28 = *((_DWORD *)v10 + 8);
      if ( v28 )
      {
        v29 = v28 - 1;
        v30 = v10[2];
        v15 = v111;
        LODWORD(v31) = v29 & (37 * v20);
        v32 = (unsigned int *)(v30 + 8LL * (unsigned int)v31);
        v33 = *v32;
        v14 = *((_DWORD *)v10 + 6) + 1;
        if ( v20 != *v32 )
        {
          v34 = 1;
          v35 = 0;
          while ( v33 != -1 )
          {
            if ( v33 == -2 && !v35 )
              v35 = v32;
            v31 = v29 & (unsigned int)(v31 + v34);
            v32 = (unsigned int *)(v30 + 8 * v31);
            v33 = *v32;
            if ( v20 == *v32 )
              goto LABEL_74;
            ++v34;
          }
          if ( v35 )
            v32 = v35;
        }
        goto LABEL_74;
      }
LABEL_153:
      ++*((_DWORD *)v10 + 6);
      BUG();
    }
    if ( v13 - (v14 + *((_DWORD *)v10 + 7)) <= v13 >> 3 )
    {
      v113 = v15;
      sub_1392B70(v117, v13);
      v98 = *((_DWORD *)v10 + 8);
      if ( v98 )
      {
        v99 = v98 - 1;
        v100 = v10[2];
        v101 = 0;
        v15 = v113;
        v102 = v99 & (37 * v20);
        v103 = 1;
        v32 = (unsigned int *)(v100 + 8LL * v102);
        v104 = *v32;
        v14 = *((_DWORD *)v10 + 6) + 1;
        if ( v20 != *v32 )
        {
          while ( v104 != -1 )
          {
            if ( !v101 && v104 == -2 )
              v101 = v32;
            v109 = v103 + 1;
            v110 = v99 & (v102 + v103);
            v32 = (unsigned int *)(v100 + 8LL * v110);
            v102 = v110;
            v104 = *v32;
            if ( v20 == *v32 )
              goto LABEL_74;
            v103 = v109;
          }
          if ( v101 )
            v32 = v101;
        }
        goto LABEL_74;
      }
      goto LABEL_153;
    }
LABEL_74:
    *((_DWORD *)v10 + 6) = v14;
    if ( *v32 != -1 )
      --*((_DWORD *)v10 + 7);
    *v32 = v20;
    v32[1] = v15;
LABEL_8:
    if ( v120 != ++v7 )
    {
      v6 = *(_QWORD *)(a1 + 800);
      continue;
    }
    break;
  }
LABEL_25:
  v115 = a1 + 808;
  v36 = 0;
  v37 = *(_QWORD *)(v116 + 256);
  if ( (unsigned int)((*(_QWORD *)(v116 + 264) - v37) >> 3) )
  {
    while ( 1 )
    {
      v44 = *(_DWORD *)(a1 + 832);
      v45 = *(_QWORD *)(v37 + 8LL * v36);
      if ( !v44 )
        break;
      v38 = *(_QWORD *)(a1 + 816);
      v39 = ((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4);
      v40 = (v44 - 1) & v39;
      v41 = (__int64 *)(v38 + 40LL * v40);
      v42 = *v41;
      if ( v45 != *v41 )
      {
        v70 = 1;
        v71 = 0;
        while ( v42 != -8 )
        {
          if ( v42 == -16 && !v71 )
            v71 = v41;
          v40 = (v44 - 1) & (v70 + v40);
          v41 = (__int64 *)(v38 + 40LL * v40);
          v42 = *v41;
          if ( v45 == *v41 )
            goto LABEL_28;
          ++v70;
        }
        v72 = *(_DWORD *)(a1 + 824);
        if ( v71 )
          v41 = v71;
        ++*(_QWORD *)(a1 + 808);
        v49 = v72 + 1;
        if ( 4 * (v72 + 1) < 3 * v44 )
        {
          if ( v44 - *(_DWORD *)(a1 + 828) - v49 <= v44 >> 3 )
          {
            v126 = v45;
            sub_2158190(v115, v44);
            v73 = *(_DWORD *)(a1 + 832);
            if ( !v73 )
            {
LABEL_152:
              ++*(_DWORD *)(a1 + 824);
              BUG();
            }
            v74 = v73 - 1;
            v75 = 1;
            v76 = 0;
            v77 = *(_QWORD *)(a1 + 816);
            v78 = v74 & v39;
            v45 = v126;
            v49 = *(_DWORD *)(a1 + 824) + 1;
            v41 = (__int64 *)(v77 + 40LL * v78);
            v79 = *v41;
            if ( v126 != *v41 )
            {
              while ( v79 != -8 )
              {
                if ( !v76 && v79 == -16 )
                  v76 = v41;
                v78 = v74 & (v78 + v75);
                v41 = (__int64 *)(v77 + 40LL * v78);
                v79 = *v41;
                if ( v126 == *v41 )
                  goto LABEL_38;
                ++v75;
              }
              if ( v76 )
                v41 = v76;
            }
          }
          goto LABEL_38;
        }
LABEL_36:
        v122 = v45;
        sub_2158190(v115, 2 * v44);
        v46 = *(_DWORD *)(a1 + 832);
        if ( !v46 )
          goto LABEL_152;
        v45 = v122;
        v47 = v46 - 1;
        v48 = *(_QWORD *)(a1 + 816);
        v49 = *(_DWORD *)(a1 + 824) + 1;
        v50 = v47 & (((unsigned int)v122 >> 9) ^ ((unsigned int)v122 >> 4));
        v41 = (__int64 *)(v48 + 40LL * v50);
        v51 = *v41;
        if ( v122 != *v41 )
        {
          v106 = 1;
          v107 = 0;
          while ( v51 != -8 )
          {
            if ( v51 == -16 && !v107 )
              v107 = v41;
            v50 = v47 & (v50 + v106);
            v41 = (__int64 *)(v48 + 40LL * v50);
            v51 = *v41;
            if ( v122 == *v41 )
              goto LABEL_38;
            ++v106;
          }
          if ( v107 )
            v41 = v107;
        }
LABEL_38:
        *(_DWORD *)(a1 + 824) = v49;
        if ( *v41 != -8 )
          --*(_DWORD *)(a1 + 828);
        *v41 = v45;
        v41[1] = 0;
        v41[2] = 0;
        v41[3] = 0;
        *((_DWORD *)v41 + 8) = 0;
      }
LABEL_28:
      v118 = v41;
      v121 = v45;
      sub_2163730(&v127, v45);
      sub_21638D0(v130, v121);
      v43 = *((_DWORD *)v118 + 6);
      if ( v43 )
      {
        v52 = v134;
        if ( (unsigned __int64)(v133 - (_QWORD)v134) <= 5 )
        {
          v125 = *((_DWORD *)v118 + 6);
          v69 = sub_16E7EE0((__int64)v132, "\t.reg ", 6u);
          v43 = v125;
          v53 = (_QWORD *)v69;
        }
        else
        {
          *v134 = 1701981705;
          *((_WORD *)v52 + 2) = 8295;
          v53 = v132;
          v134 = (_DWORD *)((char *)v134 + 6);
        }
        v123 = v43;
        v54 = sub_16E7EE0((__int64)v53, v127, v128);
        v55 = v123;
        v56 = *(_WORD **)(v54 + 24);
        v57 = v54;
        if ( *(_QWORD *)(v54 + 16) - (_QWORD)v56 <= 1u )
        {
          v68 = sub_16E7EE0(v54, " \t", 2u);
          v55 = v123;
          v57 = v68;
        }
        else
        {
          *v56 = 2336;
          *(_QWORD *)(v54 + 24) += 2LL;
        }
        v124 = v55;
        v58 = sub_16E7EE0(v57, v130[0], (size_t)v130[1]);
        v59 = v124;
        v60 = v58;
        v61 = *(_BYTE **)(v58 + 24);
        if ( *(_BYTE **)(v60 + 16) == v61 )
        {
          v67 = sub_16E7EE0(v60, "<", 1u);
          v59 = v124;
          v60 = v67;
        }
        else
        {
          *v61 = 60;
          ++*(_QWORD *)(v60 + 24);
        }
        v62 = sub_16E7AB0(v60, v59 + 1);
        v63 = *(_QWORD *)(v62 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(v62 + 16) - v63) <= 2 )
        {
          sub_16E7EE0(v62, ">;\n", 3u);
        }
        else
        {
          *(_BYTE *)(v63 + 2) = 10;
          *(_WORD *)v63 = 15166;
          *(_QWORD *)(v62 + 24) += 3LL;
        }
      }
      if ( (__int64 *)v130[0] != &v131 )
        j_j___libc_free_0(v130[0], v131 + 1);
      if ( v127 != (char *)&v129 )
        j_j___libc_free_0(v127, v129 + 1);
      v37 = *(_QWORD *)(v116 + 256);
      if ( ++v36 >= (unsigned int)((*(_QWORD *)(v116 + 264) - v37) >> 3) )
        goto LABEL_49;
    }
    ++*(_QWORD *)(a1 + 808);
    goto LABEL_36;
  }
LABEL_49:
  v64 = *(_QWORD *)(a1 + 256);
  v65 = *((unsigned int *)v136 + 2);
  v66 = (char *)*v136;
  LOWORD(v131) = 261;
  v130[0] = (char *)&v127;
  v127 = v66;
  v128 = v65;
  sub_38DD5A0(v64, v130);
  v132[0] = &unk_49EFD28;
  sub_16E7960((__int64)v132);
  if ( (_BYTE *)v137[0] != v138 )
    _libc_free(v137[0]);
}
