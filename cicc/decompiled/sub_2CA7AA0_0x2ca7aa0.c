// Function: sub_2CA7AA0
// Address: 0x2ca7aa0
//
__int64 __fastcall sub_2CA7AA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  __int64 v6; // r14
  __int64 v7; // r10
  __int64 v8; // rbx
  unsigned __int64 v9; // r12
  int v10; // eax
  __int64 v11; // r12
  int v12; // eax
  int v13; // r15d
  __int64 v14; // rax
  __int64 *v15; // r14
  __int64 v16; // r8
  __int64 *v18; // r12
  __int64 v19; // rbx
  char v20; // al
  bool v21; // zf
  __int64 *v22; // r13
  bool v23; // al
  unsigned __int8 *v24; // r12
  int v25; // r12d
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r13
  __int64 v29; // r15
  __int64 v30; // r14
  __int64 *v31; // r14
  bool v32; // al
  int v33; // r15d
  __int64 v34; // rax
  __int64 *v36; // rbx
  _BYTE *v37; // rsi
  unsigned int v38; // r15d
  int v39; // eax
  unsigned int v40; // edx
  unsigned __int64 v41; // rdi
  __int64 *v42; // rax
  _QWORD *v43; // rax
  __int64 v44; // r8
  __int64 v45; // r9
  unsigned __int8 *v46; // rax
  __int64 v47; // rdx
  __int64 v49; // rsi
  __int64 v50; // r12
  unsigned __int8 *v51; // r8
  unsigned __int8 v52; // al
  char *v53; // r14
  unsigned __int8 *v54; // r13
  __int64 *v55; // rax
  char v56; // al
  __int64 v57; // r15
  __int64 v58; // rbx
  __int64 v59; // rax
  __int64 v60; // r15
  __int64 v61; // r14
  __int64 v62; // rsi
  __int64 v63; // rcx
  __int64 v64; // rax
  unsigned int v65; // edx
  __int64 *v66; // r15
  __int64 v67; // r8
  __int64 v68; // rax
  __int64 v69; // r12
  char *v70; // rdx
  __int64 v71; // r13
  __int64 v72; // rax
  char *v73; // r14
  __int64 v74; // rcx
  __int64 v75; // rdx
  int v76; // r9d
  unsigned __int64 v77; // rax
  unsigned int i; // eax
  char **v79; // rdi
  unsigned int v80; // eax
  __int64 v81; // r9
  __int64 v82; // rdx
  __int64 v83; // rcx
  int v84; // r10d
  unsigned int j; // eax
  unsigned __int8 **v86; // rsi
  unsigned int v87; // eax
  char *v88; // r14
  _BYTE *v89; // rax
  __int64 v90; // rbx
  __int64 v91; // rax
  __int64 v92; // rdx
  __int64 v93; // r13
  int v94; // r13d
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rdx
  __int64 v98; // rax
  __int64 v99; // r13
  __int64 v100; // r15
  unsigned __int8 *v101; // rsi
  unsigned int v102; // eax
  unsigned __int8 **v103; // rdi
  unsigned __int8 *v104; // r10
  __int64 v105; // rdx
  __int64 v106; // rsi
  __int64 v107; // rcx
  __int64 v108; // r8
  __int64 v109; // r9
  unsigned __int64 v110; // r14
  __int64 v111; // rax
  _QWORD *v112; // r13
  unsigned __int8 *v113; // r13
  __int64 *v114; // rax
  int v115; // edx
  unsigned int v116; // eax
  int v117; // r9d
  char *v118; // rax
  int v119; // esi
  int v120; // edx
  unsigned int v121; // esi
  unsigned __int64 v122; // rdx
  int v123; // edi
  int v124; // ecx
  __int64 v125; // [rsp+20h] [rbp-190h]
  __int64 v126; // [rsp+28h] [rbp-188h]
  __int64 v129; // [rsp+40h] [rbp-170h]
  __int64 v131; // [rsp+50h] [rbp-160h]
  __int64 v132; // [rsp+50h] [rbp-160h]
  __int64 *v133; // [rsp+50h] [rbp-160h]
  __int64 v134; // [rsp+58h] [rbp-158h]
  __int64 v135; // [rsp+60h] [rbp-150h]
  __int64 *v136; // [rsp+60h] [rbp-150h]
  __int64 *v137; // [rsp+60h] [rbp-150h]
  __int64 v138; // [rsp+68h] [rbp-148h]
  char *v139; // [rsp+68h] [rbp-148h]
  __int64 v140; // [rsp+68h] [rbp-148h]
  __int64 v141; // [rsp+68h] [rbp-148h]
  unsigned __int8 *v142; // [rsp+70h] [rbp-140h] BYREF
  unsigned __int8 *v143; // [rsp+78h] [rbp-138h] BYREF
  unsigned __int64 v144; // [rsp+80h] [rbp-130h] BYREF
  __int64 v145; // [rsp+88h] [rbp-128h]
  __int64 v146; // [rsp+90h] [rbp-120h]
  unsigned __int64 v147; // [rsp+A0h] [rbp-110h] BYREF
  _BYTE *v148; // [rsp+A8h] [rbp-108h]
  __int64 v149; // [rsp+B0h] [rbp-100h]
  const char *v150; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v151; // [rsp+C8h] [rbp-E8h]
  __int64 v152; // [rsp+D0h] [rbp-E0h]
  __int64 v153; // [rsp+D8h] [rbp-D8h]
  char v154; // [rsp+E0h] [rbp-D0h]
  char v155; // [rsp+E1h] [rbp-CFh]
  const char *v156[2]; // [rsp+F0h] [rbp-C0h] BYREF
  _BYTE v157[112]; // [rsp+100h] [rbp-B0h] BYREF
  void *v158; // [rsp+170h] [rbp-40h]

  v7 = a2;
  v8 = a3;
  v9 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9 == a2 + 48 )
  {
    v11 = 0;
  }
  else
  {
    if ( !v9 )
      BUG();
    v10 = *(unsigned __int8 *)(v9 - 24);
    v11 = v9 - 24;
    if ( (unsigned int)(v10 - 30) >= 0xB )
      v11 = 0;
  }
  v12 = *(_DWORD *)(a3 + 8);
  if ( a6 )
  {
    if ( v12 )
    {
      v134 = v6;
      v13 = 0;
      v14 = 0;
      v15 = 0;
      v16 = v11;
      do
      {
        while ( 1 )
        {
          v18 = *(__int64 **)(*(_QWORD *)a3 + 8 * v14);
          v19 = v18[2];
          if ( v7 == *(_QWORD *)(v19 + 40) )
            break;
          v14 = (unsigned int)(v13 + 1);
          v13 = v14;
          if ( (_DWORD)v14 == *(_DWORD *)(a3 + 8) )
            goto LABEL_13;
        }
        v135 = v7;
        v138 = v16;
        v20 = sub_B19DB0(a4, v18[2], v16);
        v16 = v138;
        v7 = v135;
        v21 = v20 == 0;
        v14 = (unsigned int)(v13 + 1);
        if ( !v21 )
        {
          v15 = v18;
          v16 = v19;
        }
        ++v13;
      }
      while ( (_DWORD)v14 != *(_DWORD *)(a3 + 8) );
LABEL_13:
      v8 = a3;
      v22 = v15;
      v11 = v16;
      v6 = v134;
      if ( v22 )
      {
LABEL_14:
        v143 = 0;
        v23 = sub_D968A0(*v22);
        v139 = (char *)v22[3];
        if ( v23 )
        {
LABEL_62:
          v24 = v143;
          goto LABEL_17;
        }
        if ( !(_DWORD)qword_5011F68 )
        {
          v24 = (unsigned __int8 *)*v22;
          v143 = (unsigned __int8 *)*v22;
LABEL_17:
          if ( v24 && !sub_D968A0((__int64)v24) )
            v136 = sub_DCAF50(*(__int64 **)(a1 + 184), (__int64)v143, 0);
          else
            v136 = 0;
          v25 = 0;
          v26 = 0;
          if ( !*(_DWORD *)(v8 + 8) )
            return 1;
          while ( 1 )
          {
            v31 = *(__int64 **)(*(_QWORD *)v8 + 8 * v26);
            if ( sub_D968A0(*v31) )
              break;
            v32 = sub_D968A0(*v31);
            v27 = (__int64)v136;
            if ( !v32 )
            {
              if ( v136 )
                goto LABEL_24;
              v27 = *v31;
            }
LABEL_25:
            v28 = v31[3];
            v29 = (__int64)v139;
            v30 = v31[2];
            if ( v27 )
              goto LABEL_26;
LABEL_27:
            if ( *(_QWORD *)(v29 + 8) != *(_QWORD *)(v28 + 8) )
            {
              sub_23D0AB0((__int64)v156, v30, 0, 0, 0);
              v155 = 1;
              v150 = "scevcgptmp_";
              v154 = 3;
              v29 = sub_2C91010((__int64 *)v156, 49, v29, *(_QWORD *)(v28 + 8), (__int64)&v150, 0, v147, 0);
              nullsub_61();
              v158 = &unk_49DA100;
              nullsub_63();
              if ( v156[0] != v157 )
                _libc_free((unsigned __int64)v156[0]);
            }
            sub_BD2ED0(v30, v28, v29);
            v26 = (unsigned int)(v25 + 1);
            v25 = v26;
            if ( (_DWORD)v26 == *(_DWORD *)(v8 + 8) )
              return 1;
          }
          if ( !v136 )
          {
            v28 = v31[3];
            v29 = (__int64)v139;
            v30 = v31[2];
            goto LABEL_27;
          }
          if ( sub_D968A0(*v31) )
          {
            v28 = v31[3];
            v27 = (__int64)v136;
            v30 = v31[2];
LABEL_26:
            v131 = v27;
            v29 = (__int64)v139;
            if ( !sub_D968A0(v27) )
            {
              v42 = sub_DA3860(*(_QWORD **)(a1 + 184), (__int64)v139);
              v43 = sub_DC7ED0(*(__int64 **)(a1 + 184), (__int64)v42, v131, 0, 0);
              v44 = v129;
              LOWORD(v44) = 0;
              v129 = v44;
              v29 = (__int64)sub_F8DB90(a5, (__int64)v43, 0, v30 + 24, 0);
              if ( *(_BYTE *)v29 > 0x1Cu )
              {
                v45 = v30 + 24;
                if ( *(_QWORD *)(v29 + 40) != *(_QWORD *)(v30 + 40)
                  || (v56 = sub_B19DB0(a4, v29, v30), v45 = v30 + 24, !v56) )
                {
                  v132 = v45;
                  v46 = (unsigned __int8 *)sub_B47F80((_BYTE *)v29);
                  v157[17] = 1;
                  v29 = (__int64)v46;
                  v157[16] = 3;
                  v156[0] = "scevcgp_";
                  sub_BD6B50(v46, v156);
                  v47 = v126;
                  LOWORD(v47) = 0;
                  v126 = v47;
                  sub_B44220((_QWORD *)v29, v132, v47);
                }
              }
            }
            goto LABEL_27;
          }
LABEL_24:
          v27 = (__int64)sub_DC7ED0(*(__int64 **)(a1 + 184), *v31, (__int64)v136, 0, 0);
          goto LABEL_25;
        }
        v62 = v22[1];
        v63 = *(_QWORD *)(a1 + 8);
        v64 = *(unsigned int *)(a1 + 24);
        if ( (_DWORD)v64 )
        {
          v65 = (v64 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
          v66 = (__int64 *)(v63 + 32LL * v65);
          v67 = *v66;
          if ( v62 == *v66 )
          {
LABEL_76:
            if ( v66 != (__int64 *)(v63 + 32 * v64) )
            {
              v68 = *((unsigned int *)v66 + 4);
              if ( (_DWORD)v68 )
              {
                v69 = 0;
                v70 = (char *)v22[3];
                v137 = v22;
                v71 = 8 * v68;
                while ( 1 )
                {
                  v72 = v66[1];
                  v73 = *(char **)(v72 + v69);
                  if ( (unsigned __int8)*v73 > 0x1Cu && (unsigned __int8)*v70 > 0x1Cu )
                  {
                    if ( (unsigned __int8)sub_B19DB0(*(_QWORD *)(a1 + 200), *(_QWORD *)(v72 + v69), (__int64)v70) )
                    {
                      v139 = v73;
                      v24 = v143;
                      goto LABEL_17;
                    }
                    v70 = (char *)v137[3];
                  }
                  v69 += 8;
                  if ( v71 == v69 )
                  {
                    v139 = v70;
                    v22 = v137;
                    break;
                  }
                }
              }
            }
          }
          else
          {
            v117 = 1;
            while ( v67 != -4096 )
            {
              v65 = (v64 - 1) & (v117 + v65);
              v66 = (__int64 *)(v63 + 32LL * v65);
              v67 = *v66;
              if ( v62 == *v66 )
                goto LABEL_76;
              ++v117;
            }
          }
        }
        v49 = *v22;
        v74 = *(_QWORD *)(a1 + 120);
        v75 = *(unsigned int *)(a1 + 136);
        v156[1] = (const char *)*v22;
        v156[0] = v139;
        if ( (_DWORD)v75 )
        {
          v76 = 1;
          v77 = 0xBF58476D1CE4E5B9LL
              * (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4)
               | ((unsigned __int64)(((unsigned int)v139 >> 9) ^ ((unsigned int)v139 >> 4)) << 32));
          for ( i = (v75 - 1) & ((v77 >> 31) ^ v77); ; i = (v75 - 1) & v80 )
          {
            v79 = (char **)(v74 + 32LL * i);
            if ( *v79 == v139 && (char *)v49 == v79[1] )
              break;
            if ( *v79 == (char *)-4096LL && v79[1] == (char *)-4096LL )
              goto LABEL_58;
            v80 = v76 + i;
            ++v76;
          }
          if ( v79 != (char **)(v74 + 32 * v75) )
          {
            v88 = v79[2];
            if ( v88 )
            {
              v24 = (unsigned __int8 *)v79[3];
              if ( (unsigned __int8)*v88 <= 0x1Cu
                || (unsigned __int8)*v139 <= 0x1Cu
                || (v49 = (__int64)v79[2], (unsigned __int8)sub_B19DB0(*(_QWORD *)(a1 + 200), v49, (__int64)v139)) )
              {
                v143 = v24;
                v139 = v88;
                goto LABEL_17;
              }
              v139 = (char *)v22[3];
            }
          }
        }
LABEL_58:
        v50 = a1 + 112;
        v51 = sub_BD3990((unsigned __int8 *)v139, v49);
        v52 = *v51;
        if ( *v51 > 0x1Cu )
        {
          if ( v52 != 63 )
          {
LABEL_61:
            v53 = (char *)v22[3];
            v54 = (unsigned __int8 *)*v22;
            v139 = v53;
            v143 = v54;
            v55 = sub_2C9A8E0(v50, (__int64 *)v156);
            *v55 = (__int64)v53;
            v55[1] = (__int64)v54;
            goto LABEL_62;
          }
        }
        else if ( v52 != 5 || *((_WORD *)v51 + 1) != 34 )
        {
          goto LABEL_61;
        }
        v81 = *v22;
        v82 = *(unsigned int *)(a1 + 168);
        v83 = *(_QWORD *)(a1 + 152);
        if ( (_DWORD)v82 )
        {
          v84 = 1;
          for ( j = (v82 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4)
                      | ((unsigned __int64)(((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4)))); ; j = (v82 - 1) & v87 )
          {
            v86 = (unsigned __int8 **)(v83 + 32LL * j);
            if ( v51 == *v86 && (unsigned __int8 *)v81 == v86[1] )
              break;
            if ( *v86 == (unsigned __int8 *)-4096LL && v86[1] == (unsigned __int8 *)-4096LL )
              goto LABEL_129;
            v87 = v84 + j;
            ++v84;
          }
          if ( v86 != (unsigned __int8 **)(32 * v82 + v83) )
          {
            v24 = v86[3];
            v139 = (char *)v86[2];
            v143 = v24;
            goto LABEL_17;
          }
        }
LABEL_129:
        v106 = v22[3];
        v107 = *(_QWORD *)(v106 + 8);
        if ( *(_BYTE *)v106 <= 0x1Cu )
          v106 = 0;
        v139 = (char *)sub_2C9B570(a1, v106, a5, v107, (__int64)v51, v81, &v143);
        if ( !v139 )
          goto LABEL_61;
        if ( v143 && !sub_D968A0((__int64)v143) )
        {
LABEL_138:
          v113 = v143;
          v114 = sub_2C9A8E0(v50, (__int64 *)v156);
          v114[1] = (__int64)v113;
          *v114 = (__int64)v139;
          v24 = v143;
          goto LABEL_17;
        }
        v144 = v22[1];
        if ( (unsigned __int8)sub_2C95D00(a1, (__int64 *)&v144, &v147) )
        {
          v110 = v147;
          v111 = *(unsigned int *)(v147 + 16);
          v112 = (_QWORD *)(v147 + 8);
          if ( *(unsigned int *)(v147 + 20) < (unsigned __int64)(v111 + 1) )
          {
            sub_C8D5F0(v147 + 8, (const void *)(v147 + 24), v111 + 1, 8u, v108, v109);
            v111 = *(unsigned int *)(v110 + 16);
          }
LABEL_137:
          *(_QWORD *)(*v112 + 8 * v111) = v139;
          ++*((_DWORD *)v112 + 2);
          goto LABEL_138;
        }
        v118 = (char *)v147;
        v119 = *(_DWORD *)(a1 + 16);
        ++*(_QWORD *)a1;
        v150 = v118;
        v120 = v119 + 1;
        v121 = *(_DWORD *)(a1 + 24);
        if ( 4 * v120 >= 3 * v121 )
        {
          v121 *= 2;
        }
        else if ( v121 - *(_DWORD *)(a1 + 20) - v120 > v121 >> 3 )
        {
LABEL_159:
          *(_DWORD *)(a1 + 16) = v120;
          if ( *(_QWORD *)v118 != -4096 )
            --*(_DWORD *)(a1 + 20);
          v122 = v144;
          v112 = v118 + 8;
          *((_QWORD *)v118 + 2) = 0x100000000LL;
          *(_QWORD *)v118 = v122;
          *((_QWORD *)v118 + 1) = v118 + 24;
          v111 = 0;
          goto LABEL_137;
        }
        sub_2C9A2E0(a1, v121);
        sub_2C95D00(a1, (__int64 *)&v144, &v150);
        v120 = *(_DWORD *)(a1 + 16) + 1;
        v118 = (char *)v150;
        goto LABEL_159;
      }
    }
    goto LABEL_72;
  }
  if ( !v12 )
    goto LABEL_72;
  v140 = v11;
  v33 = 0;
  v22 = 0;
  v34 = 0;
  do
  {
    v36 = *(__int64 **)(*(_QWORD *)a3 + 8 * v34);
    v21 = !sub_D968A0(*v36);
    v34 = (unsigned int)(v33 + 1);
    if ( !v21 )
      v22 = v36;
    ++v33;
  }
  while ( (_DWORD)v34 != *(_DWORD *)(a3 + 8) );
  v8 = a3;
  v11 = v140;
  if ( !v22 )
LABEL_72:
    v22 = **(__int64 ***)v8;
  v144 = 0;
  v145 = 0;
  v146 = 0;
  v37 = (_BYTE *)v22[3];
  if ( *v37 <= 0x1Cu )
    goto LABEL_14;
  v142 = (unsigned __int8 *)v22[3];
  v150 = 0;
  v151 = 0;
  v152 = 0;
  v153 = 0;
  v147 = 0;
  v148 = 0;
  v149 = 0;
  v38 = sub_B19DB0(a4, (__int64)v37, v11);
  if ( !(_BYTE)v38 )
  {
    v39 = *v142;
    v40 = v39 - 84;
    LOBYTE(v40) = (_BYTE)v39 != 61 && (unsigned __int8)(v39 - 84) > 1u;
    v38 = v40;
    if ( (_BYTE)v40 )
    {
      sub_24454E0((__int64)&v147, v148, &v142);
      sub_2400480((__int64)v156, (__int64)&v150, (__int64 *)&v142);
      sub_2C95CD0((__int64)&v144, &v142);
      v89 = v148;
      if ( v148 == (_BYTE *)v147 )
      {
LABEL_154:
        sub_CF0530((__int64)&v150, &v144);
        if ( v147 )
          j_j___libc_free_0(v147);
        sub_C7D6A0(v151, 8LL * (unsigned int)v153, 8);
        v41 = v144;
        v57 = (__int64)(v145 - v144) >> 3;
        goto LABEL_66;
      }
      v133 = v22;
      v125 = v8;
      while ( 1 )
      {
        v90 = *((_QWORD *)v89 - 1);
        v148 = v89 - 8;
        if ( *(_BYTE *)v90 == 85 )
          break;
        LODWORD(v98) = *(_DWORD *)(v90 + 4) & 0x7FFFFFF;
LABEL_113:
        v99 = (unsigned int)v98;
        v100 = 0;
        if ( (_DWORD)v98 )
        {
          do
          {
            if ( *(_BYTE *)v90 == 85 )
            {
              v101 = *(unsigned __int8 **)(v90 + 32 * (v100 - (*(_DWORD *)(v90 + 4) & 0x7FFFFFF)));
              if ( !v101 )
                BUG();
            }
            else
            {
              if ( (*(_BYTE *)(v90 + 7) & 0x40) != 0 )
                v105 = *(_QWORD *)(v90 - 8);
              else
                v105 = v90 - 32LL * (*(_DWORD *)(v90 + 4) & 0x7FFFFFF);
              v101 = *(unsigned __int8 **)(v105 + 32 * v100);
            }
            if ( *v101 <= 0x1Cu )
              goto LABEL_121;
            v143 = v101;
            if ( (unsigned __int8)sub_B19DB0(a4, (__int64)v101, v11) )
              goto LABEL_121;
            if ( (_DWORD)v153 )
            {
              v102 = (v153 - 1) & (((unsigned int)v143 >> 9) ^ ((unsigned int)v143 >> 4));
              v103 = (unsigned __int8 **)(v151 + 8LL * v102);
              v104 = *v103;
              if ( v143 == *v103 )
              {
LABEL_120:
                if ( v103 != (unsigned __int8 **)(v151 + 8LL * (unsigned int)v153) )
                  goto LABEL_121;
              }
              else
              {
                v123 = 1;
                while ( v104 != (unsigned __int8 *)-4096LL )
                {
                  v124 = v123 + 1;
                  v102 = (v153 - 1) & (v102 + v123);
                  v103 = (unsigned __int8 **)(v151 + 8LL * v102);
                  v104 = *v103;
                  if ( v143 == *v103 )
                    goto LABEL_120;
                  v123 = v124;
                }
              }
            }
            v115 = *v143;
            v116 = v115 - 84;
            LOBYTE(v116) = (_BYTE)v115 != 61 && (unsigned __int8)(v115 - 84) > 1u;
            if ( !(_BYTE)v116 )
            {
              v22 = v133;
              v8 = v125;
              v38 = v116;
              goto LABEL_44;
            }
            sub_2400480((__int64)v156, (__int64)&v150, (__int64 *)&v143);
            sub_2C95CD0((__int64)&v144, &v143);
            sub_2C95CD0((__int64)&v147, &v143);
LABEL_121:
            ++v100;
          }
          while ( v99 != v100 );
        }
        v89 = v148;
        if ( v148 == (_BYTE *)v147 )
        {
          v22 = v133;
          v8 = v125;
          goto LABEL_154;
        }
      }
      if ( *(char *)(v90 + 7) < 0 )
      {
        v91 = sub_BD2BC0(v90);
        v93 = v91 + v92;
        if ( *(char *)(v90 + 7) >= 0 )
        {
          if ( (unsigned int)(v93 >> 4) )
LABEL_171:
            BUG();
        }
        else if ( (unsigned int)((v93 - sub_BD2BC0(v90)) >> 4) )
        {
          if ( *(char *)(v90 + 7) >= 0 )
            goto LABEL_171;
          v94 = *(_DWORD *)(sub_BD2BC0(v90) + 8);
          if ( *(char *)(v90 + 7) >= 0 )
            BUG();
          v95 = sub_BD2BC0(v90);
          v97 = 32LL * (unsigned int)(*(_DWORD *)(v95 + v96 - 4) - v94);
          goto LABEL_112;
        }
      }
      v97 = 0;
LABEL_112:
      v98 = (32LL * (*(_DWORD *)(v90 + 4) & 0x7FFFFFF) - 32 - v97) >> 5;
      goto LABEL_113;
    }
  }
LABEL_44:
  if ( v147 )
    j_j___libc_free_0(v147);
  sub_C7D6A0(v151, 8LL * (unsigned int)v153, 8);
  v41 = v144;
  if ( (_BYTE)v38 )
  {
    v57 = (__int64)(v145 - v144) >> 3;
LABEL_66:
    if ( (_DWORD)v57 )
    {
      v141 = v8;
      v58 = v6;
      v59 = 8LL * (unsigned int)v57;
      v60 = 0;
      v61 = v59;
      do
      {
        LOWORD(v58) = 0;
        sub_B444E0(*(_QWORD **)(v41 + v60), v11 + 24, v58);
        v41 = v144;
        v11 = *(_QWORD *)(v144 + v60);
        v60 += 8;
      }
      while ( v61 != v60 );
      v8 = v141;
    }
    if ( v41 )
      j_j___libc_free_0(v41);
    goto LABEL_14;
  }
  if ( v144 )
    j_j___libc_free_0(v144);
  return v38;
}
