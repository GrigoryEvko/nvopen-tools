// Function: sub_311BE60
// Address: 0x311be60
//
__int64 __fastcall sub_311BE60(__int64 a1, char a2)
{
  _QWORD *v3; // rax
  _QWORD *v4; // rdx
  unsigned __int64 *v5; // r15
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // r13
  unsigned __int64 *v9; // rax
  unsigned __int64 *v10; // r12
  unsigned __int64 *v11; // rbx
  unsigned __int64 *v12; // rax
  unsigned __int64 v13; // rdx
  unsigned __int64 *v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rcx
  unsigned __int64 *v19; // r14
  unsigned __int64 v20; // r15
  unsigned __int64 v21; // r12
  unsigned __int64 v22; // r12
  unsigned __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r11
  __int64 v26; // rsi
  __int64 v27; // rax
  int v28; // edx
  _DWORD *v29; // rcx
  _DWORD *v30; // rdx
  int v31; // r10d
  __int64 v32; // r12
  int v33; // ebx
  int v34; // esi
  unsigned int v35; // eax
  _DWORD *v36; // rdi
  int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rbx
  unsigned __int64 v40; // r13
  unsigned __int64 v41; // rdi
  __int64 v42; // rdx
  __int64 v43; // rcx
  double v44; // rbx
  _QWORD *v45; // rax
  __int64 v46; // rbx
  unsigned __int64 v47; // r12
  unsigned __int64 v48; // r13
  unsigned __int64 v49; // rdi
  unsigned __int64 *v50; // rax
  unsigned __int64 *v51; // r15
  __int64 *v52; // r13
  __int64 v53; // r9
  __int64 *v54; // rax
  unsigned __int64 v55; // r12
  unsigned __int64 v56; // r13
  unsigned __int64 v57; // r14
  unsigned __int64 *v58; // rbx
  unsigned __int64 *v59; // r15
  bool v60; // r9
  __int64 v61; // rax
  _QWORD *v62; // rax
  __int64 *v63; // rdx
  __int64 *v64; // r13
  _DWORD *v65; // rax
  __int64 v66; // rdx
  _DWORD *v67; // rbx
  __int64 v68; // rax
  __int64 v69; // rdi
  __int64 v70; // r14
  __int64 v71; // rax
  __int64 v72; // rdx
  int v73; // r11d
  unsigned int i; // ecx
  __int64 v75; // rax
  unsigned int v76; // ecx
  __int64 v77; // rax
  __int64 *v78; // rax
  __int64 *v79; // r10
  __int64 v80; // rsi
  __int64 v81; // r12
  __int64 v82; // rdx
  int v83; // r11d
  int v84; // edi
  int v85; // r11d
  int v86; // ebx
  unsigned int j; // eax
  unsigned __int64 v88; // r9
  unsigned int v89; // eax
  __int64 v90; // r13
  unsigned int v91; // [rsp+4h] [rbp-11Ch]
  unsigned int v92; // [rsp+8h] [rbp-118h]
  _QWORD *v94; // [rsp+18h] [rbp-108h]
  __int64 v95; // [rsp+28h] [rbp-F8h]
  unsigned __int64 *v97; // [rsp+38h] [rbp-E8h]
  __int64 v98; // [rsp+40h] [rbp-E0h]
  double v99; // [rsp+40h] [rbp-E0h]
  unsigned int v100; // [rsp+48h] [rbp-D8h]
  __int64 v101; // [rsp+48h] [rbp-D8h]
  unsigned __int64 v102; // [rsp+50h] [rbp-D0h]
  char v103; // [rsp+50h] [rbp-D0h]
  char v104; // [rsp+50h] [rbp-D0h]
  _QWORD *v105; // [rsp+58h] [rbp-C8h]
  unsigned __int64 v106; // [rsp+60h] [rbp-C0h]
  __int64 v107; // [rsp+60h] [rbp-C0h]
  unsigned __int64 *v108; // [rsp+60h] [rbp-C0h]
  __int64 v109; // [rsp+60h] [rbp-C0h]
  unsigned int v110; // [rsp+60h] [rbp-C0h]
  unsigned __int64 *v111; // [rsp+68h] [rbp-B8h]
  unsigned __int64 *v112; // [rsp+68h] [rbp-B8h]
  double v113; // [rsp+68h] [rbp-B8h]
  unsigned __int64 *v114; // [rsp+68h] [rbp-B8h]
  _DWORD *v115; // [rsp+68h] [rbp-B8h]
  __int64 *v116; // [rsp+68h] [rbp-B8h]
  __int64 *v117; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v118; // [rsp+78h] [rbp-A8h]
  _BYTE v119[64]; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v120; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v121; // [rsp+C8h] [rbp-58h] BYREF
  unsigned __int64 v122; // [rsp+D0h] [rbp-50h]
  __int64 *v123; // [rsp+D8h] [rbp-48h]
  __int64 *v124; // [rsp+E0h] [rbp-40h]
  __int64 v125; // [rsp+E8h] [rbp-38h]

  if ( !*(_DWORD *)(a1 + 16) )
    goto LABEL_2;
  v3 = *(_QWORD **)(a1 + 8);
  v4 = &v3[9 * *(unsigned int *)(a1 + 24)];
  v94 = v4;
  if ( v3 == v4 )
    goto LABEL_2;
  while ( *v3 > 0xFFFFFFFFFFFFFFFDLL )
  {
    v3 += 9;
    if ( v4 == v3 )
      goto LABEL_2;
  }
  v105 = v3;
  if ( v4 == v3 )
    goto LABEL_2;
  do
  {
    v5 = (unsigned __int64 *)v105[1];
    v6 = 8LL * *((unsigned int *)v105 + 4);
    v111 = &v5[(unsigned __int64)v6 / 8];
    v7 = v6 >> 3;
    if ( v6 )
    {
      while ( 1 )
      {
        v8 = v7;
        v9 = (unsigned __int64 *)sub_2207800(8 * v7);
        v10 = v9;
        if ( v9 )
          break;
        v7 >>= 1;
        if ( !v7 )
          goto LABEL_144;
      }
      v11 = &v9[v8];
      *v9 = *v5;
      v12 = v9 + 1;
      *v5 = 0;
      if ( v11 == v10 + 1 )
      {
        v14 = v10;
      }
      else
      {
        do
        {
          v13 = *(v12 - 1);
          *(v12++ - 1) = 0;
          *(v12 - 1) = v13;
        }
        while ( v11 != v12 );
        v14 = &v10[v8 - 1];
      }
      v15 = *v14;
      *v14 = 0;
      v16 = *v5;
      *v5 = v15;
      if ( v16 )
      {
        v17 = *(_QWORD *)(v16 + 24);
        if ( v17 )
        {
          v102 = v16;
          v106 = *(_QWORD *)(v16 + 24);
          sub_C7D6A0(*(_QWORD *)(v17 + 8), 16LL * *(unsigned int *)(v17 + 24), 8);
          j_j___libc_free_0(v106);
          v16 = v102;
        }
        j_j___libc_free_0(v16);
      }
      v18 = v7;
      v19 = v10;
      sub_311A8A0(v5, v111, v10, v18, a1);
      v112 = v10;
      do
      {
        v20 = *v19;
        if ( *v19 )
        {
          v21 = *(_QWORD *)(v20 + 24);
          if ( v21 )
          {
            sub_C7D6A0(*(_QWORD *)(v21 + 8), 16LL * *(unsigned int *)(v21 + 24), 8);
            j_j___libc_free_0(v21);
          }
          j_j___libc_free_0(v20);
        }
        ++v19;
      }
      while ( v11 != v19 );
      v22 = (unsigned __int64)v112;
    }
    else
    {
LABEL_144:
      v22 = 0;
      sub_31196C0(v5, v111, a1);
    }
    j_j___libc_free_0(v22);
    v98 = v105[1];
    v100 = *((_DWORD *)v105 + 4);
    if ( v100 <= 1 )
    {
LABEL_48:
      if ( a2 )
        goto LABEL_65;
      v117 = (__int64 *)v119;
      v118 = 0x600000000LL;
      v38 = *(_QWORD *)(*(_QWORD *)v98 + 24LL);
      if ( *(_DWORD *)(v38 + 16) )
      {
        v65 = *(_DWORD **)(v38 + 8);
        v66 = 4LL * *(unsigned int *)(v38 + 24);
        v115 = &v65[v66];
        if ( v65 != &v65[v66] )
        {
          while ( 1 )
          {
            while ( 1 )
            {
              v67 = v65;
              if ( *v65 != -1 )
                break;
              if ( v65[1] != -1 )
                goto LABEL_153;
              v65 += 4;
              if ( v115 == v65 )
                goto LABEL_50;
            }
            if ( *v65 != -2 || v65[1] != -2 )
              break;
            v65 += 4;
            if ( v115 == v65 )
              goto LABEL_50;
          }
LABEL_153:
          if ( v115 != v65 )
          {
            v110 = 0;
            do
            {
              if ( v100 <= 1 )
              {
LABEL_166:
                v77 = v110;
                if ( v110 >= (unsigned __int64)HIDWORD(v118) )
                {
                  v23 = v110 + 1LL;
                  v90 = *(_QWORD *)v67;
                  if ( HIDWORD(v118) < v23 )
                  {
                    sub_C8D5F0((__int64)&v117, v119, v110 + 1LL, 8u, v23, v24);
                    v77 = (unsigned int)v118;
                  }
                  v117[v77] = v90;
                  v110 = v118 + 1;
                  LODWORD(v118) = v118 + 1;
                }
                else
                {
                  v78 = &v117[v110];
                  if ( v78 )
                  {
                    *v78 = *(_QWORD *)v67;
                    v110 = v118;
                  }
                  LODWORD(v118) = ++v110;
                }
              }
              else
              {
                v68 = v105[1];
                v69 = v68 + 8;
                v70 = v68 + 8LL * (v100 - 2) + 16;
                while ( 1 )
                {
                  v71 = *(_QWORD *)(*(_QWORD *)v69 + 24LL);
                  v72 = *(unsigned int *)(v71 + 24);
                  v24 = *(_QWORD *)(v71 + 8);
                  if ( (_DWORD)v72 )
                  {
                    v23 = (unsigned int)v67[1];
                    v73 = 1;
                    for ( i = (v72 - 1)
                            & (((0xBF58476D1CE4E5B9LL
                               * ((unsigned int)(37 * v23) | ((unsigned __int64)(unsigned int)(37 * *v67) << 32))) >> 31)
                             ^ (756364221 * v23)); ; i = (v72 - 1) & v76 )
                    {
                      v75 = v24 + 16LL * i;
                      if ( *(_QWORD *)v67 == *(_QWORD *)v75 )
                        break;
                      if ( *(_DWORD *)v75 == -1 && *(_DWORD *)(v75 + 4) == -1 )
                        goto LABEL_163;
                      v76 = v73 + i;
                      ++v73;
                    }
                  }
                  else
                  {
LABEL_163:
                    v75 = v24 + 16 * v72;
                  }
                  if ( *(_QWORD *)(v75 + 8) != *((_QWORD *)v67 + 1) )
                    break;
                  v69 += 8;
                  if ( v70 == v69 )
                    goto LABEL_166;
                }
              }
              v67 += 4;
              if ( v67 == v115 )
                break;
              while ( 1 )
              {
                if ( *v67 == -1 )
                {
                  if ( v67[1] != -1 )
                    break;
                  goto LABEL_190;
                }
                if ( *v67 != -2 || v67[1] != -2 )
                  break;
LABEL_190:
                v67 += 4;
                if ( v115 == v67 )
                  goto LABEL_174;
              }
            }
            while ( v115 != v67 );
LABEL_174:
            v79 = v117;
            v116 = &v117[v110];
            if ( v117 != v116 )
            {
LABEL_175:
              v80 = v105[1];
              v81 = v80 + 8LL * *((unsigned int *)v105 + 4);
              if ( v80 == v81 )
                goto LABEL_184;
              while ( 1 )
              {
                while ( 1 )
                {
                  v82 = *(_QWORD *)(*(_QWORD *)v80 + 24LL);
                  v83 = *(_DWORD *)(v82 + 24);
                  v23 = *(_QWORD *)(v82 + 8);
                  if ( v83 )
                    break;
LABEL_183:
                  v80 += 8;
                  if ( v81 == v80 )
                    goto LABEL_184;
                }
                v84 = *((_DWORD *)v79 + 1);
                v85 = v83 - 1;
                v86 = 1;
                for ( j = v85
                        & (((0xBF58476D1CE4E5B9LL
                           * ((unsigned int)(37 * v84) | ((unsigned __int64)(unsigned int)(37 * *(_DWORD *)v79) << 32))) >> 31)
                         ^ (756364221 * v84)); ; j = v85 & v89 )
                {
                  v88 = v23 + 16LL * j;
                  if ( *(_DWORD *)v79 == *(_DWORD *)v88 && v84 == *(_DWORD *)(v88 + 4) )
                  {
                    *(_QWORD *)v88 = 0xFFFFFFFEFFFFFFFELL;
                    --*(_DWORD *)(v82 + 16);
                    ++*(_DWORD *)(v82 + 20);
                    goto LABEL_183;
                  }
                  if ( *(_DWORD *)v88 == -1 && *(_DWORD *)(v88 + 4) == -1 )
                    break;
                  v89 = v86 + j;
                  ++v86;
                }
                v80 += 8;
                if ( v81 == v80 )
                {
LABEL_184:
                  if ( ++v79 == v116 )
                  {
                    v116 = v117;
                    break;
                  }
                  goto LABEL_175;
                }
              }
            }
            if ( v116 != (__int64 *)v119 )
              _libc_free((unsigned __int64)v116);
          }
        }
      }
LABEL_50:
      v92 = *((_DWORD *)v105 + 4);
      v39 = v105[1];
      if ( v92 < dword_50328C8 )
        goto LABEL_111;
      v91 = *(_DWORD *)(*(_QWORD *)v39 + 16LL);
      if ( v91 < dword_50327E8 )
        goto LABEL_111;
      LODWORD(v121) = 0;
      v122 = 0;
      v117 = (__int64 *)v119;
      v118 = 0x800000000LL;
      v123 = &v121;
      v124 = &v121;
      v125 = 0;
      v95 = v39 + 8LL * v92;
      if ( v39 == v95 )
      {
        v40 = 0;
        v99 = 0.0;
        goto LABEL_62;
      }
      v101 = v39;
      v40 = 0;
      v99 = 0.0;
LABEL_54:
      LODWORD(v118) = 0;
      while ( v40 )
      {
        sub_3117FB0(*(_QWORD *)(v40 + 24));
        v41 = v40;
        v40 = *(_QWORD *)(v40 + 16);
        j_j___libc_free_0(v41);
      }
      v122 = 0;
      v123 = &v121;
      v124 = &v121;
      v125 = 0;
      v42 = *(_QWORD *)(*(_QWORD *)v101 + 24LL);
      if ( !*(_DWORD *)(v42 + 16)
        || (v50 = *(unsigned __int64 **)(v42 + 8), v114 = &v50[2 * *(unsigned int *)(v42 + 24)], v50 == v114) )
      {
LABEL_57:
        v40 = 0;
        goto LABEL_58;
      }
      while ( 1 )
      {
        v51 = v50;
        if ( *(_DWORD *)v50 == -1 )
        {
          if ( *((_DWORD *)v50 + 1) != -1 )
            goto LABEL_95;
        }
        else if ( *(_DWORD *)v50 != -2 || *((_DWORD *)v50 + 1) != -2 )
        {
LABEL_95:
          if ( v114 == v50 )
            goto LABEL_57;
          v43 = 0;
LABEL_97:
          v108 = v51 + 1;
          if ( v43 )
          {
            sub_A19EB0(&v120, v51 + 1);
            v43 = v125;
            goto LABEL_103;
          }
          v52 = &v117[(unsigned int)v118];
          if ( v117 == v52 )
          {
            if ( (unsigned int)v118 > 7uLL )
              goto LABEL_142;
            v53 = v51[1];
          }
          else
          {
            v53 = v51[1];
            v54 = v117;
            while ( *v54 != v53 )
            {
              if ( v52 == ++v54 )
                goto LABEL_127;
            }
            if ( v52 != v54 )
              goto LABEL_103;
LABEL_127:
            if ( (unsigned int)v118 > 7uLL )
            {
              v97 = v51;
              v58 = (unsigned __int64 *)&v117[(unsigned int)v118];
              v59 = (unsigned __int64 *)v117;
              do
              {
                v62 = sub_265B1C0(&v120, (__int64)&v121, v59);
                v64 = v63;
                if ( v63 )
                {
                  v60 = v62 || v63 == &v121 || *v59 < v63[4];
                  v104 = v60;
                  v61 = sub_22077B0(0x28u);
                  *(_QWORD *)(v61 + 32) = *v59;
                  sub_220F040(v104, v61, v64, &v121);
                  ++v125;
                }
                ++v59;
              }
              while ( v58 != v59 );
              v51 = v97;
LABEL_142:
              LODWORD(v118) = 0;
              sub_A19EB0(&v120, v108);
              v43 = v125;
LABEL_103:
              v51 += 2;
              if ( v51 != v114 )
              {
                do
                {
                  if ( *(_DWORD *)v51 == -1 )
                  {
                    if ( *((_DWORD *)v51 + 1) != -1 )
                      goto LABEL_106;
                  }
                  else if ( *(_DWORD *)v51 != -2 || *((_DWORD *)v51 + 1) != -2 )
                  {
LABEL_106:
                    if ( v114 != v51 )
                      goto LABEL_97;
                    goto LABEL_107;
                  }
                  v51 += 2;
                }
                while ( v114 != v51 );
                v40 = v122;
                if ( v43 )
                  goto LABEL_108;
LABEL_58:
                LODWORD(v43) = v118;
                if ( (unsigned int)qword_5032708 >= (unsigned int)v118 )
                  goto LABEL_59;
LABEL_109:
                sub_3117FB0(v40);
                if ( v117 != (__int64 *)v119 )
                {
                  _libc_free((unsigned __int64)v117);
                  v39 = v105[1];
                  goto LABEL_111;
                }
LABEL_208:
                v39 = v105[1];
LABEL_111:
                v55 = v39 + 8LL * *((unsigned int *)v105 + 4);
                if ( v39 != v55 )
                {
                  do
                  {
                    v56 = *(_QWORD *)(v55 - 8);
                    v55 -= 8LL;
                    if ( v56 )
                    {
                      v57 = *(_QWORD *)(v56 + 24);
                      if ( v57 )
                      {
                        sub_C7D6A0(*(_QWORD *)(v57 + 8), 16LL * *(unsigned int *)(v57 + 24), 8);
                        j_j___libc_free_0(v57);
                      }
                      j_j___libc_free_0(v56);
                    }
                  }
                  while ( v39 != v55 );
                  v55 = v105[1];
                }
                if ( (_QWORD *)v55 != v105 + 3 )
                {
                  v49 = v55;
                  goto LABEL_90;
                }
                goto LABEL_91;
              }
LABEL_107:
              v40 = v122;
              if ( !v43 )
                goto LABEL_58;
LABEL_108:
              if ( (unsigned int)qword_5032708 < (unsigned int)v43 )
                goto LABEL_109;
LABEL_59:
              if ( (_BYTE)qword_5032628 && !(_DWORD)v43 )
                goto LABEL_109;
              v101 += 8;
              v99 = (double)(int)v43 * *(double *)&qword_5032468 + *(double *)&qword_5032388 + v99;
              if ( v95 == v101 )
              {
LABEL_62:
                v113 = v99 + *(double *)&qword_50322A8;
                v44 = (double)(int)(v91 * (v92 - 1)) * *(double *)&qword_5032548;
                sub_3117FB0(v40);
                if ( v117 != (__int64 *)v119 )
                  _libc_free((unsigned __int64)v117);
                if ( v44 <= v113 )
                  goto LABEL_208;
                goto LABEL_65;
              }
              goto LABEL_54;
            }
          }
          if ( (unsigned __int64)(unsigned int)v118 + 1 > HIDWORD(v118) )
          {
            v109 = v53;
            sub_C8D5F0((__int64)&v117, v119, (unsigned int)v118 + 1LL, 8u, v23, v53);
            v53 = v109;
            v52 = &v117[(unsigned int)v118];
          }
          *v52 = v53;
          v43 = v125;
          LODWORD(v118) = v118 + 1;
          goto LABEL_103;
        }
        v50 += 2;
        if ( v114 == v50 )
          goto LABEL_57;
      }
    }
    v103 = 0;
    v25 = v98 + 8;
    v107 = v98 + 8LL * (v100 - 2) + 16;
    do
    {
LABEL_27:
      if ( *(_DWORD *)(*(_QWORD *)v25 + 16LL) != *(_DWORD *)(*(_QWORD *)v98 + 16LL) )
        goto LABEL_83;
      v26 = *(_QWORD *)(*(_QWORD *)v98 + 24LL);
      v27 = *(_QWORD *)(*(_QWORD *)v25 + 24LL);
      v28 = *(_DWORD *)(v26 + 16);
      if ( *(_DWORD *)(v27 + 16) != v28 )
        goto LABEL_83;
      if ( !v28 )
        goto LABEL_26;
      v29 = *(_DWORD **)(v26 + 8);
      v23 = (unsigned __int64)&v29[4 * *(unsigned int *)(v26 + 24)];
      if ( v29 == (_DWORD *)v23 )
        goto LABEL_26;
      while ( 1 )
      {
        v30 = v29;
        if ( *v29 != -1 )
          break;
        if ( v29[1] != -1 )
          goto LABEL_33;
LABEL_77:
        v29 += 4;
        if ( (_DWORD *)v23 == v29 )
          goto LABEL_26;
      }
      if ( *v29 == -2 && v29[1] == -2 )
        goto LABEL_77;
LABEL_33:
      if ( (_DWORD *)v23 == v29 )
        goto LABEL_26;
      v31 = *(_DWORD *)(v27 + 24);
      v32 = *(_QWORD *)(v27 + 8);
      v33 = v31 - 1;
      if ( !v31 )
      {
        v103 = 1;
        goto LABEL_46;
      }
      while ( 2 )
      {
        v34 = v30[1];
        v24 = 1;
        v35 = v33
            & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v34) | ((unsigned __int64)(unsigned int)(37 * *v30) << 32))) >> 31)
             ^ (756364221 * v34));
        while ( 2 )
        {
          v36 = (_DWORD *)(v32 + 16LL * v35);
          if ( *v30 != *v36 || v34 != v36[1] )
          {
            if ( *v36 != -1 || v36[1] != -1 )
            {
              v37 = v24 + v35;
              v24 = (unsigned int)(v24 + 1);
              v35 = v33 & v37;
              continue;
            }
            v103 = 1;
LABEL_46:
            v25 += 8;
            if ( v107 == v25 )
              goto LABEL_47;
            goto LABEL_27;
          }
          break;
        }
        v30 += 4;
        if ( v30 == (_DWORD *)v23 )
          break;
        while ( 2 )
        {
          if ( *v30 == -1 )
          {
            if ( v30[1] != -1 )
              goto LABEL_43;
LABEL_72:
            v30 += 4;
            if ( (_DWORD *)v23 == v30 )
              goto LABEL_26;
            continue;
          }
          break;
        }
        if ( *v30 == -2 && v30[1] == -2 )
          goto LABEL_72;
LABEL_43:
        if ( (_DWORD *)v23 != v30 )
          continue;
        break;
      }
LABEL_26:
      v25 += 8;
    }
    while ( v107 != v25 );
LABEL_47:
    if ( !v103 )
      goto LABEL_48;
LABEL_83:
    v46 = v98 + 8LL * v100;
    do
    {
      v47 = *(_QWORD *)(v46 - 8);
      v46 -= 8;
      if ( v47 )
      {
        v48 = *(_QWORD *)(v47 + 24);
        if ( v48 )
        {
          sub_C7D6A0(*(_QWORD *)(v48 + 8), 16LL * *(unsigned int *)(v48 + 24), 8);
          j_j___libc_free_0(v48);
        }
        j_j___libc_free_0(v47);
      }
    }
    while ( v98 != v46 );
    v49 = v105[1];
    if ( (_QWORD *)v49 != v105 + 3 )
LABEL_90:
      _libc_free(v49);
LABEL_91:
    *v105 = -2;
    --*(_DWORD *)(a1 + 16);
    ++*(_DWORD *)(a1 + 20);
LABEL_65:
    v105 += 9;
    v45 = v105;
    if ( v105 != v94 )
    {
      do
      {
        if ( *v45 <= 0xFFFFFFFFFFFFFFFDLL )
          break;
        v45 += 9;
      }
      while ( v94 != v45 );
      v105 = v45;
    }
  }
  while ( v105 != (_QWORD *)(*(_QWORD *)(a1 + 8) + 72LL * *(unsigned int *)(a1 + 24)) );
LABEL_2:
  *(_BYTE *)(a1 + 104) = 1;
  return a1;
}
