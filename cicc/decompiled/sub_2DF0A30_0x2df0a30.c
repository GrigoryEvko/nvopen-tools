// Function: sub_2DF0A30
// Address: 0x2df0a30
//
__int64 __fastcall sub_2DF0A30(__int64 a1)
{
  __int64 (*v1)(void); // rax
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rbx
  _QWORD *v5; // rax
  _QWORD *v6; // rcx
  _QWORD *v7; // rsi
  _DWORD *v8; // rax
  _DWORD *v9; // rcx
  _DWORD *v10; // rdx
  __int64 *v11; // r14
  __int64 v12; // r12
  char v13; // bl
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // eax
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned int v19; // r12d
  unsigned int v20; // r13d
  __int64 v21; // rax
  unsigned int *v22; // r15
  __int64 v23; // rdx
  __int64 v24; // rcx
  _BYTE *v25; // rbx
  unsigned __int64 v26; // r15
  unsigned __int64 v27; // rdi
  __int64 *v28; // r14
  __int64 *v29; // r13
  __int64 *v30; // r15
  __int64 v31; // r12
  char v32; // bl
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 **v35; // rax
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 **v38; // rdx
  __int64 *v39; // r13
  __int64 v40; // r12
  unsigned int v41; // r14d
  unsigned int *v42; // r13
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rdx
  _BYTE *v46; // rbx
  unsigned __int64 v47; // r13
  unsigned __int64 v48; // rdi
  __int64 **v49; // rcx
  unsigned int v50; // eax
  __int64 *v51; // rsi
  __int64 i; // rbx
  _QWORD *v53; // rdx
  unsigned __int8 v54; // al
  _QWORD *v55; // r12
  __int64 v56; // rcx
  __int64 v57; // rax
  __int64 v58; // r13
  unsigned __int64 v59; // rdi
  unsigned __int64 v60; // r15
  unsigned __int64 v61; // r14
  unsigned __int64 v62; // rdi
  unsigned __int64 v63; // rbx
  unsigned __int64 v64; // rdi
  unsigned __int64 v65; // rbx
  unsigned __int64 v66; // rdi
  _QWORD *v67; // r12
  __int64 v68; // rcx
  __int64 v69; // r15
  unsigned __int64 v70; // rdi
  unsigned __int64 v71; // r13
  unsigned __int64 v72; // rbx
  unsigned __int64 v73; // rdi
  unsigned __int64 v74; // rbx
  unsigned __int64 v75; // rdi
  unsigned __int64 v76; // rbx
  unsigned __int64 v77; // rdi
  __int64 *v78; // r12
  __int64 v79; // rcx
  __int64 v80; // r15
  unsigned __int64 v81; // rdi
  unsigned __int64 v82; // r13
  unsigned __int64 v83; // rbx
  unsigned __int64 v84; // rdi
  unsigned __int64 v85; // rbx
  unsigned __int64 v86; // rdi
  unsigned __int64 v87; // rbx
  unsigned __int64 v88; // rdi
  __int64 *v89; // r12
  __int64 *v91; // r12
  __int64 v92; // rdx
  __int64 v93; // rax
  __int64 v94; // r15
  unsigned __int64 v95; // rdi
  unsigned __int64 v96; // r14
  unsigned __int64 v97; // r13
  unsigned __int64 v98; // rdi
  unsigned __int64 v99; // rbx
  unsigned __int64 v100; // rdi
  unsigned __int64 v101; // rbx
  unsigned __int64 v102; // rdi
  __int64 v103; // rdx
  __int64 v104; // rax
  __int64 v105; // r15
  unsigned __int64 v106; // rdi
  unsigned __int64 v107; // r14
  unsigned __int64 v108; // r13
  unsigned __int64 v109; // rdi
  unsigned __int64 v110; // rbx
  unsigned __int64 v111; // rdi
  unsigned __int64 v112; // rbx
  unsigned __int64 v113; // rdi
  _QWORD *v114; // rsi
  _QWORD *v115; // rdx
  __int64 v116; // rax
  __int64 v118; // [rsp+20h] [rbp-180h]
  __int64 v119; // [rsp+30h] [rbp-170h]
  __int64 v120; // [rsp+38h] [rbp-168h]
  unsigned __int64 v121; // [rsp+38h] [rbp-168h]
  __int64 **v122; // [rsp+40h] [rbp-160h]
  int v123; // [rsp+48h] [rbp-158h]
  __int64 *v124; // [rsp+50h] [rbp-150h]
  unsigned int v125; // [rsp+58h] [rbp-148h]
  unsigned __int8 v126; // [rsp+63h] [rbp-13Dh]
  unsigned int v127; // [rsp+64h] [rbp-13Ch]
  __int64 v128; // [rsp+68h] [rbp-138h]
  __int64 *v129; // [rsp+68h] [rbp-138h]
  char v130; // [rsp+70h] [rbp-130h]
  char v131; // [rsp+70h] [rbp-130h]
  __int64 *v132; // [rsp+70h] [rbp-130h]
  __int64 v133; // [rsp+70h] [rbp-130h]
  __int64 v134; // [rsp+70h] [rbp-130h]
  _QWORD *v135; // [rsp+78h] [rbp-128h]
  _QWORD *v136; // [rsp+78h] [rbp-128h]
  __int64 *v137; // [rsp+78h] [rbp-128h]
  __int64 v138[2]; // [rsp+80h] [rbp-120h] BYREF
  __int64 *v139; // [rsp+90h] [rbp-110h]
  __int64 *v140; // [rsp+A0h] [rbp-100h] BYREF
  __int64 *v141; // [rsp+A8h] [rbp-F8h]
  __int64 v142; // [rsp+B0h] [rbp-F0h]
  _QWORD **v143; // [rsp+C0h] [rbp-E0h] BYREF
  _QWORD ***v144; // [rsp+C8h] [rbp-D8h]
  __int64 v145; // [rsp+D0h] [rbp-D0h]
  unsigned __int64 v146; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v147; // [rsp+E8h] [rbp-B8h]
  _BYTE *v148; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v149; // [rsp+F8h] [rbp-A8h]
  _BYTE v150[96]; // [rsp+100h] [rbp-A0h] BYREF
  unsigned __int64 v151; // [rsp+160h] [rbp-40h] BYREF
  unsigned int v152; // [rsp+168h] [rbp-38h]

  sub_1049690(v138, *(_QWORD *)a1);
  v1 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 24) + 1504LL);
  if ( v1 == sub_2DE6890 )
  {
    v127 = 2;
    v118 = sub_B2BEC0(*(_QWORD *)a1);
  }
  else
  {
    v127 = v1();
    v118 = sub_B2BEC0(*(_QWORD *)a1);
    if ( v127 <= 1 )
    {
      v126 = 0;
      goto LABEL_148;
    }
  }
  v126 = 0;
  do
  {
    v142 = 0;
    v141 = (__int64 *)&v140;
    v140 = (__int64 *)&v140;
    v119 = *(_QWORD *)a1 + 72LL;
    v120 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
    if ( v120 == v119 )
    {
      v28 = (__int64 *)&v140;
    }
    else
    {
      do
      {
        if ( !v120 )
          BUG();
        v128 = *(_QWORD *)(v120 + 32);
        if ( v128 != v120 + 24 )
        {
          while ( 1 )
          {
            if ( !v128 )
              BUG();
            if ( *(_BYTE *)(v128 - 24) != 92 )
              goto LABEL_34;
            v2 = *(_QWORD *)(v128 - 16);
            if ( *(_BYTE *)(v2 + 8) == 18 )
              goto LABEL_34;
            v3 = sub_22077B0(0xA0u);
            *(_QWORD *)(v3 + 16) = off_49D4228;
            *(_QWORD *)(v3 + 64) = v3 + 48;
            *(_QWORD *)(v3 + 72) = v3 + 48;
            *(_QWORD *)(v3 + 112) = v3 + 96;
            *(_QWORD *)(v3 + 120) = v3 + 96;
            *(_DWORD *)(v3 + 48) = 0;
            *(_QWORD *)(v3 + 56) = 0;
            *(_QWORD *)(v3 + 80) = 0;
            *(_DWORD *)(v3 + 96) = 0;
            *(_QWORD *)(v3 + 104) = 0;
            *(_QWORD *)(v3 + 128) = 0;
            *(_QWORD *)(v3 + 152) = v2;
            v4 = *(unsigned int *)(v2 + 32);
            *(_QWORD *)(v3 + 24) = 0;
            *(_QWORD *)(v3 + 32) = 0;
            *(_QWORD *)(v3 + 136) = 0;
            v5 = (_QWORD *)sub_2207820(152 * v4 + 8);
            v6 = v5;
            if ( v5 )
            {
              *v5 = v4;
              v7 = v5 + 1;
              if ( v4 )
              {
                v8 = v5 + 1;
                v9 = &v6[19 * v4 + 1];
                do
                {
                  v10 = v8 + 8;
                  *v8 = -1;
                  v8 += 38;
                  *((_QWORD *)v8 - 18) = 0;
                  *((_QWORD *)v8 - 17) = v10;
                  *(v8 - 32) = 0;
                  *(v8 - 31) = 4;
                  *(v8 - 4) = 1;
                  *((_QWORD *)v8 - 3) = 0;
                  *((_QWORD *)v8 - 1) = 0;
                }
                while ( v9 != v8 );
              }
            }
            else
            {
              v7 = 0;
            }
            *(_QWORD *)(v3 + 144) = v7;
            sub_2208C80((_QWORD *)v3, (__int64)&v140);
            ++v142;
            if ( !(unsigned __int8)sub_2DEF330(v128 - 24, v141 + 2, v118) )
              break;
            v11 = v141;
            v12 = *(_QWORD *)(v141[19] + 24);
            v13 = sub_AE5020(v118, v12);
            v14 = sub_9208B0(v118, v12);
            v147 = v15;
            v146 = ((1LL << v13) + ((unsigned __int64)(v14 + 7) >> 3) - 1) >> v13 << v13;
            v16 = sub_CA1930(&v146);
            if ( *(_DWORD *)(v11[19] + 32) > 1u )
            {
              v19 = 1;
              v123 = v127 * v16;
              v20 = v127 * v16;
              while ( 1 )
              {
                v21 = v11[18];
                v22 = (unsigned int *)(v21 + 152LL * v19);
                LODWORD(v146) = *(_DWORD *)v21;
                v23 = *(_QWORD *)(v21 + 8);
                v148 = v150;
                v147 = v23;
                v149 = 0x400000000LL;
                v24 = *(unsigned int *)(v21 + 24);
                if ( (_DWORD)v24 )
                {
                  v134 = v21;
                  sub_2DEB050((__int64)&v148, (__int64 *)(v21 + 16), v23, v24, v17, v18);
                  v21 = v134;
                }
                v152 = *(_DWORD *)(v21 + 136);
                if ( v152 > 0x40 )
                  sub_C43780((__int64)&v151, (const void **)(v21 + 128));
                else
                  v151 = *(_QWORD *)(v21 + 128);
                sub_C46A40((__int64)&v151, v20);
                v130 = sub_2DEBCB0(v22, (int *)&v146);
                if ( v152 > 0x40 && v151 )
                  j_j___libc_free_0_0(v151);
                v25 = v148;
                v26 = (unsigned __int64)&v148[24 * (unsigned int)v149];
                if ( v148 != (_BYTE *)v26 )
                {
                  do
                  {
                    v26 -= 24LL;
                    if ( *(_DWORD *)(v26 + 16) > 0x40u )
                    {
                      v27 = *(_QWORD *)(v26 + 8);
                      if ( v27 )
                        j_j___libc_free_0_0(v27);
                    }
                  }
                  while ( v25 != (_BYTE *)v26 );
                  v26 = (unsigned __int64)v148;
                }
                if ( (_BYTE *)v26 != v150 )
                  _libc_free(v26);
                if ( !v130 )
                  break;
                ++v19;
                v20 += v123;
                if ( *(_DWORD *)(v11[19] + 32) <= v19 )
                  goto LABEL_34;
              }
              v91 = v141;
              --v142;
              sub_2208CA0(v141);
              v103 = v91[18];
              v91[2] = (__int64)off_49D4228;
              if ( v103 )
              {
                v104 = 152LL * *(_QWORD *)(v103 - 8);
                v105 = v103 + v104;
                if ( v103 != v103 + v104 )
                {
                  do
                  {
                    v105 -= 152;
                    if ( *(_DWORD *)(v105 + 136) > 0x40u )
                    {
                      v106 = *(_QWORD *)(v105 + 128);
                      if ( v106 )
                        j_j___libc_free_0_0(v106);
                    }
                    v107 = *(_QWORD *)(v105 + 16);
                    v108 = v107 + 24LL * *(unsigned int *)(v105 + 24);
                    if ( v107 != v108 )
                    {
                      do
                      {
                        v108 -= 24LL;
                        if ( *(_DWORD *)(v108 + 16) > 0x40u )
                        {
                          v109 = *(_QWORD *)(v108 + 8);
                          if ( v109 )
                            j_j___libc_free_0_0(v109);
                        }
                      }
                      while ( v107 != v108 );
                      v107 = *(_QWORD *)(v105 + 16);
                    }
                    if ( v107 != v105 + 32 )
                      _libc_free(v107);
                  }
                  while ( v91[18] != v105 );
                }
                j_j_j___libc_free_0_0(v105 - 8);
              }
              v110 = v91[13];
              while ( v110 )
              {
                sub_2DEAE80(*(_QWORD *)(v110 + 24));
                v111 = v110;
                v110 = *(_QWORD *)(v110 + 16);
                j_j___libc_free_0(v111);
              }
              v112 = v91[7];
              while ( v112 )
              {
                sub_2DEACB0(*(_QWORD *)(v112 + 24));
                v113 = v112;
                v112 = *(_QWORD *)(v112 + 16);
                j_j___libc_free_0(v113);
              }
              goto LABEL_196;
            }
LABEL_34:
            v128 = *(_QWORD *)(v128 + 8);
            if ( v120 + 24 == v128 )
              goto LABEL_35;
          }
          v91 = v141;
          --v142;
          sub_2208CA0(v141);
          v92 = v91[18];
          v91[2] = (__int64)off_49D4228;
          if ( v92 )
          {
            v93 = 152LL * *(_QWORD *)(v92 - 8);
            v94 = v92 + v93;
            if ( v92 != v92 + v93 )
            {
              do
              {
                v94 -= 152;
                if ( *(_DWORD *)(v94 + 136) > 0x40u )
                {
                  v95 = *(_QWORD *)(v94 + 128);
                  if ( v95 )
                    j_j___libc_free_0_0(v95);
                }
                v96 = *(_QWORD *)(v94 + 16);
                v97 = v96 + 24LL * *(unsigned int *)(v94 + 24);
                if ( v96 != v97 )
                {
                  do
                  {
                    v97 -= 24LL;
                    if ( *(_DWORD *)(v97 + 16) > 0x40u )
                    {
                      v98 = *(_QWORD *)(v97 + 8);
                      if ( v98 )
                        j_j___libc_free_0_0(v98);
                    }
                  }
                  while ( v96 != v97 );
                  v96 = *(_QWORD *)(v94 + 16);
                }
                if ( v96 != v94 + 32 )
                  _libc_free(v96);
              }
              while ( v91[18] != v94 );
            }
            j_j_j___libc_free_0_0(v94 - 8);
          }
          v99 = v91[13];
          while ( v99 )
          {
            sub_2DEAE80(*(_QWORD *)(v99 + 24));
            v100 = v99;
            v99 = *(_QWORD *)(v99 + 16);
            j_j___libc_free_0(v100);
          }
          v101 = v91[7];
          while ( v101 )
          {
            sub_2DEACB0(*(_QWORD *)(v101 + 24));
            v102 = v101;
            v101 = *(_QWORD *)(v101 + 16);
            j_j___libc_free_0(v102);
          }
LABEL_196:
          j_j___libc_free_0((unsigned __int64)v91);
          goto LABEL_34;
        }
LABEL_35:
        v120 = *(_QWORD *)(v120 + 8);
      }
      while ( v119 != v120 );
      v28 = v140;
    }
    v29 = (__int64 *)&v140;
    v145 = 0;
    v144 = &v143;
    v143 = &v143;
    v121 = v127;
    if ( v28 != (__int64 *)&v140 )
    {
      do
      {
        v30 = v29;
        while ( 1 )
        {
          v31 = *(_QWORD *)(v28[19] + 24);
          v32 = sub_AE5020(v118, v31);
          v33 = sub_9208B0(v118, v31);
          v147 = v34;
          v146 = ((1LL << v32) + ((unsigned __int64)(v33 + 7) >> 3) - 1) >> v32 << v32;
          v125 = sub_CA1930(&v146);
          v35 = (__int64 **)sub_22077B0(v121 * 8);
          v122 = v35;
          v38 = &v35[v121];
          do
          {
            if ( v35 )
              *v35 = v30;
            ++v35;
          }
          while ( v35 != v38 );
          v39 = v140;
          if ( v140 != v30 )
            break;
LABEL_152:
          v51 = *v122;
          if ( *v122 != v30 )
          {
            v29 = v30;
            goto LABEL_73;
          }
          j_j___libc_free_0((unsigned __int64)v122);
          v28 = (__int64 *)*v28;
          if ( v28 == v30 )
            goto LABEL_105;
        }
        while ( 1 )
        {
          if ( v39[19] == v28[19] && v39[3] == v28[3] && v39[4] == v28[4] )
          {
            v129 = v39;
            v40 = 1;
            v124 = v28;
            v41 = v125;
            do
            {
              v42 = (unsigned int *)v129[18];
              v43 = v124[18];
              LODWORD(v146) = *(_DWORD *)v43;
              v44 = *(_QWORD *)(v43 + 8);
              v148 = v150;
              v147 = v44;
              v149 = 0x400000000LL;
              v45 = *(unsigned int *)(v43 + 24);
              if ( (_DWORD)v45 )
              {
                v133 = v43;
                sub_2DEB050((__int64)&v148, (__int64 *)(v43 + 16), v45, 0x400000000LL, v36, v37);
                v43 = v133;
              }
              v152 = *(_DWORD *)(v43 + 136);
              if ( v152 > 0x40 )
                sub_C43780((__int64)&v151, (const void **)(v43 + 128));
              else
                v151 = *(_QWORD *)(v43 + 128);
              sub_C46A40((__int64)&v151, v41);
              v131 = sub_2DEBCB0(v42, (int *)&v146);
              if ( v152 > 0x40 && v151 )
                j_j___libc_free_0_0(v151);
              v46 = v148;
              v47 = (unsigned __int64)&v148[24 * (unsigned int)v149];
              if ( v148 != (_BYTE *)v47 )
              {
                do
                {
                  v47 -= 24LL;
                  if ( *(_DWORD *)(v47 + 16) > 0x40u )
                  {
                    v48 = *(_QWORD *)(v47 + 8);
                    if ( v48 )
                      j_j___libc_free_0_0(v48);
                  }
                }
                while ( v46 != (_BYTE *)v47 );
                v47 = (unsigned __int64)v148;
              }
              if ( (_BYTE *)v47 != v150 )
                _libc_free(v47);
              if ( v131 )
                v122[v40] = v129;
              ++v40;
              v41 += v125;
            }
            while ( v127 > (unsigned int)v40 );
            v39 = v129;
            v28 = v124;
            v49 = v122 + 1;
            v50 = 1;
            do
            {
              if ( *v49 == v30 )
                break;
              ++v50;
              ++v49;
            }
            while ( v127 > v50 );
            if ( v127 == v50 )
              break;
          }
          v39 = (__int64 *)*v39;
          if ( v39 == v30 )
            goto LABEL_152;
        }
        v29 = v30;
        v51 = v124;
        *v122 = v124;
LABEL_73:
        for ( i = 0; ; v51 = v122[i] )
        {
          v53 = (_QWORD *)*v51;
          if ( v51 != (__int64 *)&v143 && v53 != &v143 )
          {
            sub_2208C50((__int64)&v143, (__int64)v51, (__int64)v53);
            ++v145;
            --v142;
          }
          if ( v127 <= (unsigned int)++i )
            break;
        }
        j_j___libc_free_0((unsigned __int64)v122);
        v54 = sub_2DEDCB0(a1, &v143, v138);
        if ( v54 )
        {
          v126 = v54;
          v135 = v143;
        }
        else
        {
          v135 = v143;
          v114 = *v143;
          if ( *v143 != &v143 )
          {
            v115 = *v143;
            v116 = 0;
            do
            {
              v115 = (_QWORD *)*v115;
              ++v116;
            }
            while ( v115 != &v143 );
            v142 += v116;
            v145 -= v116;
            sub_2208C50((__int64)v140, (__int64)v114, (__int64)&v143);
            v135 = v143;
          }
        }
        if ( v135 != &v143 )
        {
          v132 = v29;
          do
          {
            v55 = v135;
            v56 = v135[18];
            v135 = (_QWORD *)*v135;
            v55[2] = off_49D4228;
            if ( v56 )
            {
              v57 = 152LL * *(_QWORD *)(v56 - 8);
              v58 = v56 + v57;
              if ( v56 != v56 + v57 )
              {
                do
                {
                  v58 -= 152;
                  if ( *(_DWORD *)(v58 + 136) > 0x40u )
                  {
                    v59 = *(_QWORD *)(v58 + 128);
                    if ( v59 )
                      j_j___libc_free_0_0(v59);
                  }
                  v60 = *(_QWORD *)(v58 + 16);
                  v61 = v60 + 24LL * *(unsigned int *)(v58 + 24);
                  if ( v60 != v61 )
                  {
                    do
                    {
                      v61 -= 24LL;
                      if ( *(_DWORD *)(v61 + 16) > 0x40u )
                      {
                        v62 = *(_QWORD *)(v61 + 8);
                        if ( v62 )
                          j_j___libc_free_0_0(v62);
                      }
                    }
                    while ( v60 != v61 );
                    v60 = *(_QWORD *)(v58 + 16);
                  }
                  if ( v60 != v58 + 32 )
                    _libc_free(v60);
                }
                while ( v55[18] != v58 );
              }
              j_j_j___libc_free_0_0(v58 - 8);
            }
            v63 = v55[13];
            while ( v63 )
            {
              sub_2DEAE80(*(_QWORD *)(v63 + 24));
              v64 = v63;
              v63 = *(_QWORD *)(v63 + 16);
              j_j___libc_free_0(v64);
            }
            v65 = v55[7];
            while ( v65 )
            {
              sub_2DEACB0(*(_QWORD *)(v65 + 24));
              v66 = v65;
              v65 = *(_QWORD *)(v65 + 16);
              j_j___libc_free_0(v66);
            }
            j_j___libc_free_0((unsigned __int64)v55);
          }
          while ( v135 != &v143 );
          v29 = v132;
        }
        v28 = v140;
        v145 = 0;
        v144 = &v143;
        v143 = &v143;
      }
      while ( v140 != v29 );
    }
LABEL_105:
    v136 = v143;
    while ( v136 != &v143 )
    {
      v67 = v136;
      v68 = v136[18];
      v136 = (_QWORD *)*v136;
      v67[2] = off_49D4228;
      if ( v68 )
      {
        v69 = v68 + 152LL * *(_QWORD *)(v68 - 8);
        if ( v68 != v69 )
        {
          do
          {
            v69 -= 152;
            if ( *(_DWORD *)(v69 + 136) > 0x40u )
            {
              v70 = *(_QWORD *)(v69 + 128);
              if ( v70 )
                j_j___libc_free_0_0(v70);
            }
            v71 = *(_QWORD *)(v69 + 16);
            v72 = v71 + 24LL * *(unsigned int *)(v69 + 24);
            if ( v71 != v72 )
            {
              do
              {
                v72 -= 24LL;
                if ( *(_DWORD *)(v72 + 16) > 0x40u )
                {
                  v73 = *(_QWORD *)(v72 + 8);
                  if ( v73 )
                    j_j___libc_free_0_0(v73);
                }
              }
              while ( v71 != v72 );
              v71 = *(_QWORD *)(v69 + 16);
            }
            if ( v71 != v69 + 32 )
              _libc_free(v71);
          }
          while ( v67[18] != v69 );
        }
        j_j_j___libc_free_0_0(v69 - 8);
      }
      v74 = v67[13];
      while ( v74 )
      {
        sub_2DEAE80(*(_QWORD *)(v74 + 24));
        v75 = v74;
        v74 = *(_QWORD *)(v74 + 16);
        j_j___libc_free_0(v75);
      }
      v76 = v67[7];
      while ( v76 )
      {
        sub_2DEACB0(*(_QWORD *)(v76 + 24));
        v77 = v76;
        v76 = *(_QWORD *)(v76 + 16);
        j_j___libc_free_0(v77);
      }
      j_j___libc_free_0((unsigned __int64)v67);
    }
    v137 = v140;
    while ( v137 != (__int64 *)&v140 )
    {
      v78 = v137;
      v79 = v137[18];
      v137 = (__int64 *)*v137;
      v78[2] = (__int64)off_49D4228;
      if ( v79 )
      {
        v80 = v79 + 152LL * *(_QWORD *)(v79 - 8);
        if ( v79 != v80 )
        {
          do
          {
            v80 -= 152;
            if ( *(_DWORD *)(v80 + 136) > 0x40u )
            {
              v81 = *(_QWORD *)(v80 + 128);
              if ( v81 )
                j_j___libc_free_0_0(v81);
            }
            v82 = *(_QWORD *)(v80 + 16);
            v83 = v82 + 24LL * *(unsigned int *)(v80 + 24);
            if ( v82 != v83 )
            {
              do
              {
                v83 -= 24LL;
                if ( *(_DWORD *)(v83 + 16) > 0x40u )
                {
                  v84 = *(_QWORD *)(v83 + 8);
                  if ( v84 )
                    j_j___libc_free_0_0(v84);
                }
              }
              while ( v82 != v83 );
              v82 = *(_QWORD *)(v80 + 16);
            }
            if ( v82 != v80 + 32 )
              _libc_free(v82);
          }
          while ( v78[18] != v80 );
        }
        j_j_j___libc_free_0_0(v80 - 8);
      }
      v85 = v78[13];
      while ( v85 )
      {
        sub_2DEAE80(*(_QWORD *)(v85 + 24));
        v86 = v85;
        v85 = *(_QWORD *)(v85 + 16);
        j_j___libc_free_0(v86);
      }
      v87 = v78[7];
      while ( v87 )
      {
        sub_2DEACB0(*(_QWORD *)(v87 + 24));
        v88 = v87;
        v87 = *(_QWORD *)(v87 + 16);
        j_j___libc_free_0(v88);
      }
      j_j___libc_free_0((unsigned __int64)v78);
    }
    --v127;
  }
  while ( v127 != 1 );
LABEL_148:
  v89 = v139;
  if ( v139 )
  {
    sub_FDC110(v139);
    j_j___libc_free_0((unsigned __int64)v89);
  }
  return v126;
}
