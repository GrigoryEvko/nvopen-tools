// Function: sub_2523ED0
// Address: 0x2523ed0
//
__int64 __fastcall sub_2523ED0(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 *v4; // rax
  __int64 *v5; // r15
  __int64 v6; // r14
  __int64 *v7; // r15
  __int64 v8; // rax
  __int64 v9; // rdx
  int v10; // eax
  int v11; // eax
  __int64 v12; // rsi
  __int64 v13; // rcx
  _QWORD *v14; // rdi
  __int64 v15; // rdx
  int v16; // eax
  int v17; // eax
  int v18; // edi
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 *v24; // r14
  __int64 v25; // r15
  __int64 v26; // rdx
  __int64 v27; // r13
  __int64 v28; // rcx
  __int64 v29; // rbx
  size_t v30; // r10
  __int64 v31; // rbx
  __int64 v32; // rax
  int v33; // r9d
  int v34; // edx
  _QWORD *v35; // rax
  __int64 *v36; // r13
  __int64 *v37; // r12
  __int64 *i; // rbx
  unsigned __int64 v39; // rax
  __int64 *v40; // rdx
  unsigned __int64 v41; // rax
  __int64 v42; // r13
  __int64 v43; // rax
  __int64 v44; // rbx
  __int64 v45; // rbx
  __int64 v46; // rcx
  unsigned __int8 *v47; // rdi
  __int64 v48; // r13
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rbx
  unsigned __int64 v52; // r13
  unsigned __int64 v53; // rax
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // rsi
  __int64 v56; // rdi
  int v57; // eax
  int v58; // r13d
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rbx
  __int64 v62; // rax
  _BYTE *v63; // r12
  __int64 *v64; // rbx
  __int64 *v65; // r13
  __int64 v66; // r12
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // rbx
  __int64 v71; // r13
  int v72; // r12d
  __int64 v73; // rax
  unsigned __int8 *v74; // rbx
  __int64 *v75; // r14
  unsigned __int8 *v76; // r15
  __int64 v77; // rbx
  __int64 v78; // rax
  __int64 v79; // rsi
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rbx
  __int64 v83; // rbx
  __int64 v84; // r9
  __int64 v85; // rax
  unsigned __int64 v86; // rdx
  __int64 v87; // rsi
  __int64 v88; // r12
  unsigned __int64 v89; // r14
  _QWORD *v90; // r15
  __int64 v91; // r12
  _QWORD *v92; // rdi
  __int64 v93; // rsi
  _QWORD *v94; // rax
  int v95; // r8d
  _QWORD *v96; // r9
  int v97; // edi
  __int64 v98; // rdx
  __int64 v99; // rcx
  __int64 *v100; // rbx
  int *v101; // rdx
  __int64 v102; // rax
  __int64 *v103; // rdi
  __int64 v104; // rsi
  __int64 v105; // rax
  __int64 *v106; // r12
  int j; // eax
  __int64 v108; // rcx
  int v109; // edx
  __int64 v110; // rcx
  int v111; // edx
  __int64 v112; // rcx
  int v113; // edx
  __int64 v114; // rcx
  int v115; // edx
  __int64 v116; // rdi
  int v117; // eax
  int v118; // edx
  unsigned int v119; // esi
  __int64 *v120; // rax
  __int64 v121; // rcx
  _QWORD *v122; // rdi
  int v123; // eax
  int v124; // r8d
  __int64 v125; // [rsp-10h] [rbp-310h]
  _QWORD *v126; // [rsp+8h] [rbp-2F8h]
  __int64 *v127; // [rsp+10h] [rbp-2F0h]
  __int64 *v128; // [rsp+18h] [rbp-2E8h]
  __int64 v129; // [rsp+20h] [rbp-2E0h]
  unsigned __int64 v130; // [rsp+20h] [rbp-2E0h]
  unsigned int v131; // [rsp+30h] [rbp-2D0h]
  __int64 v132; // [rsp+30h] [rbp-2D0h]
  __int64 v133; // [rsp+38h] [rbp-2C8h]
  char v134; // [rsp+38h] [rbp-2C8h]
  _QWORD *v135; // [rsp+38h] [rbp-2C8h]
  __int64 v136; // [rsp+38h] [rbp-2C8h]
  __int64 v137; // [rsp+38h] [rbp-2C8h]
  int v138; // [rsp+38h] [rbp-2C8h]
  unsigned int v140; // [rsp+54h] [rbp-2ACh]
  int *v141; // [rsp+58h] [rbp-2A8h]
  __int64 *v142; // [rsp+60h] [rbp-2A0h]
  char v144; // [rsp+73h] [rbp-28Dh] BYREF
  int v145; // [rsp+74h] [rbp-28Ch] BYREF
  __int64 v146; // [rsp+78h] [rbp-288h] BYREF
  __int64 v147; // [rsp+80h] [rbp-280h] BYREF
  unsigned __int64 v148; // [rsp+88h] [rbp-278h] BYREF
  __int64 v149; // [rsp+90h] [rbp-270h] BYREF
  __int64 v150; // [rsp+98h] [rbp-268h] BYREF
  _QWORD v151[6]; // [rsp+A0h] [rbp-260h] BYREF
  __int64 *v152; // [rsp+D0h] [rbp-230h] BYREF
  __int64 v153; // [rsp+D8h] [rbp-228h]
  _BYTE v154[64]; // [rsp+E0h] [rbp-220h] BYREF
  __int64 *v155; // [rsp+120h] [rbp-1E0h] BYREF
  __int64 v156; // [rsp+128h] [rbp-1D8h]
  _BYTE v157[128]; // [rsp+130h] [rbp-1D0h] BYREF
  _BYTE *v158; // [rsp+1B0h] [rbp-150h] BYREF
  __int64 v159; // [rsp+1B8h] [rbp-148h]
  _BYTE v160[128]; // [rsp+1C0h] [rbp-140h] BYREF
  int *v161; // [rsp+240h] [rbp-C0h] BYREF
  __int64 *v162; // [rsp+248h] [rbp-B8h]
  _BYTE v163[16]; // [rsp+250h] [rbp-B0h] BYREF
  __int16 v164; // [rsp+260h] [rbp-A0h]

  if ( *(_DWORD *)(a1 + 184) )
  {
    v4 = *(__int64 **)(a1 + 176);
    v5 = &v4[11 * *(unsigned int *)(a1 + 192)];
    if ( v4 != v5 )
    {
      while ( 1 )
      {
        v6 = *v4;
        if ( *v4 != -4096 && v6 != -8192 )
          break;
        v4 += 11;
        if ( v5 == v4 )
          goto LABEL_2;
      }
      if ( v5 != v4 )
      {
        v142 = v5;
        v7 = v4;
        v2 = 1;
        while ( 1 )
        {
          v146 = v6;
          v8 = *(_QWORD *)(a1 + 200);
          v9 = *(_QWORD *)(v8 + 8);
          v10 = *(_DWORD *)(v8 + 24);
          if ( !v10 )
            goto LABEL_15;
          v11 = v10 - 1;
          v140 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
          v12 = v11 & v140;
          v13 = *(_QWORD *)(v9 + 8 * v12);
          if ( v13 != v6 )
          {
            v97 = 1;
            while ( v13 != -4096 )
            {
              v12 = v11 & (unsigned int)(v97 + v12);
              v13 = *(_QWORD *)(v9 + 8LL * (unsigned int)v12);
              if ( v13 == v6 )
                goto LABEL_13;
              ++v97;
            }
            goto LABEL_15;
          }
LABEL_13:
          if ( *(_DWORD *)(a1 + 3672) )
          {
            v15 = *(_QWORD *)(a1 + 3664);
            v16 = *(_DWORD *)(a1 + 3680);
            if ( v16 )
            {
              v17 = v16 - 1;
              v12 = v17 & v140;
              v13 = *(_QWORD *)(v15 + 8 * v12);
              if ( v13 == v6 )
                goto LABEL_15;
              v18 = 1;
              while ( v13 != -4096 )
              {
                v12 = v17 & (unsigned int)(v18 + v12);
                v13 = *(_QWORD *)(v15 + 8LL * (unsigned int)v12);
                if ( v13 == v6 )
                  goto LABEL_15;
                ++v18;
              }
            }
          }
          else
          {
            v14 = *(_QWORD **)(a1 + 3688);
            v12 = (__int64)&v14[*(unsigned int *)(a1 + 3696)];
            if ( (_QWORD *)v12 != sub_2506780(v14, v12, &v146) )
              goto LABEL_15;
          }
          v127 = v7 + 1;
          v155 = (__int64 *)v157;
          v156 = 0x1000000000LL;
          v159 = 0x1000000000LL;
          v19 = *(_QWORD *)(v6 + 120);
          v158 = v160;
          v147 = v19;
          if ( (*(_BYTE *)(v6 + 2) & 1) != 0 )
          {
            sub_B2C6D0(v6, v12, v15, v13);
            v20 = *(_QWORD *)(v6 + 96);
            v21 = v20 + 40LL * *(_QWORD *)(v6 + 104);
            if ( (*(_BYTE *)(v6 + 2) & 1) != 0 )
            {
              sub_B2C6D0(v6, v12, v98, v99);
              v20 = *(_QWORD *)(v6 + 96);
            }
            v22 = (unsigned int)v156;
            v133 = (unsigned int)v156;
            if ( v20 == v21 )
              goto LABEL_45;
          }
          else
          {
            v20 = *(_QWORD *)(v6 + 96);
            v21 = v20 + 40LL * *(_QWORD *)(v6 + 104);
            if ( v21 == v20 )
            {
              v148 = 0;
              v41 = sub_BCF480(
                      **(__int64 ***)(*(_QWORD *)(v6 + 24) + 16LL),
                      v157,
                      0,
                      *(_DWORD *)(*(_QWORD *)(v6 + 24) + 8LL) >> 8 != 0);
              goto LABEL_52;
            }
            v22 = 0;
          }
          v129 = v6;
          v23 = v21;
          v24 = v7;
          v25 = v20;
          v26 = v22;
          do
          {
            v27 = v24[1] + 8LL * *(unsigned int *)(v25 + 32);
            v28 = *(_QWORD *)v27;
            if ( *(_QWORD *)v27 )
            {
              v29 = *(unsigned int *)(v28 + 32);
              v2 = *(_QWORD *)(v28 + 24);
              v30 = 8 * v29;
              if ( v29 + v26 > (unsigned __int64)HIDWORD(v156) )
              {
                v132 = *(_QWORD *)(v28 + 24);
                sub_C8D5F0((__int64)&v155, v157, v29 + v26, 8u, v2, v29 + v26);
                v26 = (unsigned int)v156;
                v2 = v132;
                v30 = 8 * v29;
              }
              if ( v30 )
              {
                memcpy(&v155[v26], (const void *)v2, v30);
                LODWORD(v26) = v156;
              }
              LODWORD(v156) = v29 + v26;
              v31 = *(unsigned int *)(*(_QWORD *)v27 + 32LL);
              v32 = (unsigned int)v159;
              v33 = v31;
              v34 = v159;
              if ( v31 + (unsigned __int64)(unsigned int)v159 > HIDWORD(v159) )
              {
                v138 = *(_DWORD *)(*(_QWORD *)v27 + 32LL);
                sub_C8D5F0((__int64)&v158, v160, v31 + (unsigned int)v159, 8u, v2, v31);
                v32 = (unsigned int)v159;
                v33 = v138;
                v34 = v159;
              }
              v35 = &v158[8 * v32];
              if ( v31 )
              {
                do
                {
                  if ( v35 )
                    *v35 = 0;
                  ++v35;
                  --v31;
                }
                while ( v31 );
                v34 = v159;
              }
              LODWORD(v159) = v34 + v33;
            }
            else
            {
              v82 = *(_QWORD *)(v25 + 8);
              if ( v26 + 1 > (unsigned __int64)HIDWORD(v156) )
              {
                sub_C8D5F0((__int64)&v155, v157, v26 + 1, 8u, v2, v26 + 1);
                v26 = (unsigned int)v156;
              }
              v155[v26] = v82;
              LODWORD(v156) = v156 + 1;
              v83 = sub_A744E0(&v147, *(_DWORD *)(v25 + 32));
              v85 = (unsigned int)v159;
              v86 = (unsigned int)v159 + 1LL;
              if ( v86 > HIDWORD(v159) )
              {
                sub_C8D5F0((__int64)&v158, v160, v86, 8u, v2, v84);
                v85 = (unsigned int)v159;
              }
              *(_QWORD *)&v158[8 * v85] = v83;
              LODWORD(v159) = v159 + 1;
            }
            v25 += 40;
            v26 = (unsigned int)v156;
          }
          while ( v23 != v25 );
          v7 = v24;
          v6 = v129;
          v133 = (unsigned int)v156;
LABEL_45:
          v36 = v155;
          v148 = 0;
          v37 = &v155[v133];
          for ( i = v155; v37 != i; ++i )
          {
            if ( (unsigned int)*(unsigned __int8 *)(*i + 8) - 17 <= 1 )
            {
              v39 = sub_BCAE30(*i);
              v162 = v40;
              v161 = (int *)v39;
              if ( v39 < v148 )
                v39 = v148;
              v148 = v39;
            }
          }
          v41 = sub_BCF480(
                  **(__int64 ***)(*(_QWORD *)(v6 + 24) + 16LL),
                  v36,
                  v133,
                  *(_DWORD *)(*(_QWORD *)(v6 + 24) + 8LL) >> 8 != 0);
LABEL_52:
          v164 = 257;
          v42 = v41;
          v134 = *(_BYTE *)(v6 + 32) & 0xF;
          v131 = *(_DWORD *)(*(_QWORD *)(v6 + 8) + 8LL) >> 8;
          v43 = sub_BD2DA0(136);
          v44 = v43;
          if ( v43 )
            sub_B2C3B0(v43, v42, v134, v131, (__int64)&v161, 0);
          v149 = v44;
          sub_2519280(*(_QWORD *)(a1 + 200), &v149);
          v45 = v149;
          sub_BA8540(*(_QWORD *)(v6 + 40) + 24LL, v149);
          v46 = *(_QWORD *)(v6 + 56);
          *(_QWORD *)(v45 + 64) = v6 + 56;
          v46 &= 0xFFFFFFFFFFFFFFF8LL;
          v47 = (unsigned __int8 *)v149;
          *(_QWORD *)(v45 + 56) = v46 | *(_QWORD *)(v45 + 56) & 7LL;
          *(_QWORD *)(v46 + 8) = v45 + 56;
          *(_QWORD *)(v6 + 56) = *(_QWORD *)(v6 + 56) & 7LL | (v45 + 56);
          sub_BD6B90(v47, (unsigned __int8 *)v6);
          sub_B2EC90(v149, v6);
          v48 = v149;
          *(_BYTE *)(v149 + 128) = *(_BYTE *)(v6 + 128);
          v49 = sub_B92180(v6);
          sub_B994C0(v48, v49);
          sub_B994C0(v6, 0);
          v50 = sub_B2BE50(v6);
          v51 = v149;
          v135 = (_QWORD *)v50;
          v126 = v158;
          v130 = (unsigned int)v159;
          v52 = sub_A74610(&v147);
          v53 = sub_A74680(&v147);
          v54 = sub_A78180(v135, v53, v52, v126, v130);
          v55 = v148;
          v56 = v149;
          *(_QWORD *)(v51 + 120) = v54;
          sub_A75730(v56, v55);
          v57 = sub_B2DC70(v149);
          v145 = -1;
          v58 = v57;
          if ( (v57 & 3) != 0 )
          {
            v100 = v155;
            v101 = &v145;
            v161 = &v145;
            v102 = 8LL * (unsigned int)v156;
            v103 = &v155[(unsigned __int64)v102 / 8];
            v162 = &v149;
            v104 = v102 >> 3;
            v105 = v102 >> 5;
            if ( v105 )
            {
              v106 = &v155[4 * v105];
              for ( j = -1; ; j = *v161 )
              {
                v114 = *v100;
                *v101 = j + 1;
                v115 = *(unsigned __int8 *)(v114 + 8);
                if ( (unsigned int)(v115 - 17) <= 1 )
                  LOBYTE(v115) = *(_BYTE *)(**(_QWORD **)(v114 + 16) + 8LL);
                if ( (_BYTE)v115 == 14 && !(unsigned __int8)sub_B2D640(*v162, *v161, 50) )
                  break;
                v108 = v100[1];
                ++*v161;
                v109 = *(unsigned __int8 *)(v108 + 8);
                if ( (unsigned int)(v109 - 17) <= 1 )
                  LOBYTE(v109) = *(_BYTE *)(**(_QWORD **)(v108 + 16) + 8LL);
                if ( (_BYTE)v109 == 14 && !(unsigned __int8)sub_B2D640(*v162, *v161, 50) )
                {
                  ++v100;
                  goto LABEL_127;
                }
                v110 = v100[2];
                ++*v161;
                v111 = *(unsigned __int8 *)(v110 + 8);
                if ( (unsigned int)(v111 - 17) <= 1 )
                  LOBYTE(v111) = *(_BYTE *)(**(_QWORD **)(v110 + 16) + 8LL);
                if ( (_BYTE)v111 == 14 && !(unsigned __int8)sub_B2D640(*v162, *v161, 50) )
                {
                  v100 += 2;
                  goto LABEL_127;
                }
                v112 = v100[3];
                ++*v161;
                v113 = *(unsigned __int8 *)(v112 + 8);
                if ( (unsigned int)(v113 - 17) <= 1 )
                  LOBYTE(v113) = *(_BYTE *)(**(_QWORD **)(v112 + 16) + 8LL);
                if ( (_BYTE)v113 == 14 && !(unsigned __int8)sub_B2D640(*v162, *v161, 50) )
                {
                  v100 += 3;
                  goto LABEL_127;
                }
                v100 += 4;
                if ( v106 == v100 )
                {
                  v104 = v103 - v100;
                  goto LABEL_141;
                }
                v101 = v161;
              }
              goto LABEL_127;
            }
LABEL_141:
            switch ( v104 )
            {
              case 2LL:
LABEL_152:
                if ( (unsigned __int8)sub_2506070((__int64)&v161, *v100) )
                {
                  ++v100;
LABEL_144:
                  if ( (unsigned __int8)sub_2506070((__int64)&v161, *v100) )
                  {
LABEL_128:
                    sub_B2DC90(v149, v58 & 0xFFFFFFFC);
                    goto LABEL_55;
                  }
                }
                break;
              case 3LL:
                if ( (unsigned __int8)sub_2506070((__int64)&v161, *v100) )
                {
                  ++v100;
                  goto LABEL_152;
                }
                break;
              case 1LL:
                goto LABEL_144;
              default:
                goto LABEL_128;
            }
LABEL_127:
            if ( v103 != v100 )
              goto LABEL_55;
            goto LABEL_128;
          }
LABEL_55:
          sub_B2C300(v149, *(__int64 **)(v149 + 80), v6, *(unsigned __int64 **)(v6 + 80), (unsigned __int64 *)(v6 + 72));
          v61 = *(_QWORD *)(v6 + 16);
          v152 = (__int64 *)v154;
          v153 = 0x800000000LL;
          v62 = 0;
          if ( v61 )
          {
            do
            {
              v63 = *(_BYTE **)(v61 + 24);
              if ( *v63 == 4 )
              {
                if ( v62 + 1 > (unsigned __int64)HIDWORD(v153) )
                {
                  sub_C8D5F0((__int64)&v152, v154, v62 + 1, 8u, v59, v60);
                  v62 = (unsigned int)v153;
                }
                v152[v62] = (__int64)v63;
                v62 = (unsigned int)(v153 + 1);
                LODWORD(v153) = v153 + 1;
              }
              v61 = *(_QWORD *)(v61 + 8);
            }
            while ( v61 );
            v64 = &v152[v62];
            if ( v64 != v152 )
            {
              v65 = v152;
              do
              {
                v66 = *v65++;
                v67 = sub_ACC1C0(v149, *(_QWORD *)(v66 - 32));
                sub_BD84D0(v66, v67);
              }
              while ( v64 != v65 );
            }
          }
          v161 = (int *)v163;
          v162 = (__int64 *)0x800000000LL;
          v144 = 0;
          v151[0] = v127;
          v151[1] = &v149;
          v151[2] = v135;
          v151[3] = &v148;
          v151[4] = &v161;
          sub_25230B0(a1, (__int64 (__fastcall *)(__int64, __int64 *))sub_2508180, (__int64)v151, v6, 1, 0, &v144, 1);
          if ( (*(_BYTE *)(v6 + 2) & 1) != 0 )
            sub_B2C6D0(v6, v125, v68, v69);
          v70 = v149;
          v71 = *(_QWORD *)(v6 + 96);
          if ( (*(_BYTE *)(v149 + 2) & 1) != 0 )
            sub_B2C6D0(v149, v125, v68, v69);
          v72 = 0;
          v73 = 0;
          v74 = *(unsigned __int8 **)(v70 + 96);
          if ( *((_DWORD *)v7 + 4) )
          {
            v136 = v6;
            v75 = v7;
            v76 = v74;
            do
            {
              v77 = v75[1] + 8 * v73;
              v78 = *(_QWORD *)v77;
              if ( *(_QWORD *)v77 )
              {
                if ( *(_QWORD *)(v78 + 120) )
                {
                  v79 = *(_QWORD *)v77;
                  v150 = (__int64)v76;
                  (*(void (__fastcall **)(__int64, __int64, __int64, __int64 *))(v78 + 128))(
                    v78 + 104,
                    v79,
                    v149,
                    &v150);
                  v78 = *(_QWORD *)v77;
                }
                v80 = *(unsigned int *)(v78 + 32);
                if ( !(_DWORD)v80 )
                {
                  v81 = sub_ACADE0(*(__int64 ***)(v71 + 8));
                  sub_BD84D0(v71, v81);
                  v80 = *(unsigned int *)(*(_QWORD *)v77 + 32LL);
                }
                v76 += 40 * v80;
              }
              else
              {
                sub_BD6B90(v76, (unsigned __int8 *)v71);
                v87 = (__int64)v76;
                v76 += 40;
                sub_BD84D0(v71, v87);
              }
              v73 = (unsigned int)(v72 + 1);
              v71 += 40;
              v72 = v73;
            }
            while ( (unsigned int)v73 < *((_DWORD *)v75 + 4) );
            v7 = v75;
            v6 = v136;
          }
          v88 = 4LL * (unsigned int)v162;
          v141 = &v161[v88];
          if ( &v161[v88] != v161 )
          {
            v128 = v7;
            v137 = v6;
            v89 = (unsigned __int64)v161;
            do
            {
              v90 = *(_QWORD **)v89;
              v91 = *(_QWORD *)(v89 + 8);
              v89 += 16LL;
              v150 = sub_B43CB0((__int64)v90);
              sub_2518560(a2, &v150);
              sub_BD84D0((__int64)v90, v91);
              sub_B43D60(v90);
            }
            while ( v141 != (int *)v89 );
            v6 = v137;
            v7 = v128;
          }
          sub_29A24B0(*(_QWORD *)(a1 + 4368), v6, v149);
          if ( *(_DWORD *)(a2 + 16) )
          {
            v116 = *(_QWORD *)(a2 + 8);
            v117 = *(_DWORD *)(a2 + 24);
            if ( !v117 )
              goto LABEL_93;
            v118 = v117 - 1;
            v119 = (v117 - 1) & v140;
            v120 = (__int64 *)(v116 + 8LL * v119);
            v121 = *v120;
            if ( *v120 != v6 )
            {
              v123 = 1;
              while ( v121 != -4096 )
              {
                v124 = v123 + 1;
                v119 = v118 & (v123 + v119);
                v120 = (__int64 *)(v116 + 8LL * v119);
                v121 = *v120;
                if ( v6 == *v120 )
                  goto LABEL_137;
                v123 = v124;
              }
              goto LABEL_93;
            }
LABEL_137:
            *v120 = -8192;
            --*(_DWORD *)(a2 + 16);
            v122 = *(_QWORD **)(a2 + 32);
            ++*(_DWORD *)(a2 + 20);
            v93 = (__int64)&v122[*(unsigned int *)(a2 + 40)];
            v94 = sub_2506840(v122, v93, &v146);
            v96 = v94 + 1;
            if ( v94 + 1 != (_QWORD *)v93 )
            {
LABEL_91:
              memmove(v94, v94 + 1, v93 - (_QWORD)v96);
              v95 = *(_DWORD *)(a2 + 40);
            }
LABEL_92:
            *(_DWORD *)(a2 + 40) = v95 - 1;
            sub_2518560(a2, &v149);
            goto LABEL_93;
          }
          v92 = *(_QWORD **)(a2 + 32);
          v93 = (__int64)&v92[*(unsigned int *)(a2 + 40)];
          v94 = sub_2506840(v92, v93, &v146);
          if ( (_QWORD *)v93 != v94 )
          {
            v96 = v94 + 1;
            if ( (_QWORD *)v93 != v94 + 1 )
              goto LABEL_91;
            goto LABEL_92;
          }
LABEL_93:
          if ( v161 != (int *)v163 )
            _libc_free((unsigned __int64)v161);
          if ( v152 != (__int64 *)v154 )
            _libc_free((unsigned __int64)v152);
          if ( v158 != v160 )
            _libc_free((unsigned __int64)v158);
          if ( v155 != (__int64 *)v157 )
            _libc_free((unsigned __int64)v155);
          v2 = 0;
LABEL_15:
          v7 += 11;
          if ( v7 != v142 )
          {
            while ( 1 )
            {
              v6 = *v7;
              if ( *v7 != -8192 && v6 != -4096 )
                break;
              v7 += 11;
              if ( v142 == v7 )
                return (unsigned int)v2;
            }
            if ( v142 != v7 )
              continue;
          }
          return (unsigned int)v2;
        }
      }
    }
  }
LABEL_2:
  LODWORD(v2) = 1;
  return (unsigned int)v2;
}
