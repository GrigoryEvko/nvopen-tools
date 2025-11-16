// Function: sub_FEC920
// Address: 0xfec920
//
__int64 __fastcall sub_FEC920(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 *v5; // rax
  __int64 *v6; // r13
  __int64 *v7; // rbx
  int v8; // esi
  __int64 *v9; // rdi
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r9
  __int64 v13; // r14
  unsigned int v14; // esi
  unsigned int v15; // edx
  __int64 *v16; // r8
  unsigned int v17; // eax
  unsigned int v18; // edi
  char v19; // di
  char v20; // si
  unsigned int v21; // ecx
  __int64 *v22; // r15
  __int64 *v23; // rbx
  __int64 *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r8
  __int64 *v28; // rax
  __int64 v29; // rsi
  __int64 *v30; // r13
  __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // r10
  int v34; // edi
  int v35; // esi
  __int64 **v36; // r14
  int *v37; // r9
  __int64 *v38; // r12
  int v39; // esi
  __int64 v40; // rcx
  unsigned int v41; // edx
  __int64 v42; // rax
  int v43; // r12d
  unsigned __int64 v44; // rdx
  __int64 v45; // rsi
  unsigned __int64 v46; // r11
  char v47; // si
  __int64 *v48; // r14
  __int64 *v49; // r15
  __int64 *v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 *v53; // r13
  unsigned __int64 v54; // rdi
  _DWORD *v55; // rsi
  __int64 v56; // rax
  unsigned int *v57; // rbx
  unsigned int *v58; // r14
  __int64 v59; // rcx
  __int64 v60; // rax
  _DWORD *v61; // rdi
  __int64 v62; // r13
  __int64 v63; // r12
  bool v64; // al
  unsigned int *v65; // rbx
  _QWORD *v66; // rax
  __int64 v67; // rcx
  __int64 v68; // rdx
  __int64 v69; // r8
  __int64 v70; // r9
  int v71; // edx
  unsigned int v72; // r12d
  __int64 *v73; // rcx
  _DWORD *v74; // rcx
  __int64 *v75; // rdx
  _DWORD *v76; // rsi
  unsigned __int64 v77; // r11
  __int64 v78; // rax
  unsigned __int64 v79; // rdx
  __int64 v80; // rcx
  int v81; // r10d
  __int64 v82; // r8
  int v83; // esi
  __int64 *v84; // rdi
  unsigned int v85; // ecx
  __int64 v86; // rax
  int v87; // r9d
  __int64 *v88; // rdx
  int v89; // esi
  __int64 *v90; // rdi
  unsigned int v91; // ecx
  __int64 v92; // rax
  int v93; // r9d
  __int64 v94; // rcx
  __int64 *v95; // rbx
  __int64 v96; // rax
  __int64 *v97; // rdx
  __int64 v98; // rax
  int v100; // edx
  int v101; // [rsp+Ch] [rbp-204h]
  __int64 v102; // [rsp+10h] [rbp-200h]
  _QWORD **v106; // [rsp+50h] [rbp-1C0h]
  __int64 *v107; // [rsp+58h] [rbp-1B8h]
  int v108; // [rsp+60h] [rbp-1B0h]
  int v109; // [rsp+68h] [rbp-1A8h]
  __int64 v110; // [rsp+68h] [rbp-1A8h]
  __int64 *v111; // [rsp+70h] [rbp-1A0h]
  __int64 v112; // [rsp+78h] [rbp-198h]
  __int64 v113; // [rsp+80h] [rbp-190h] BYREF
  char *v114; // [rsp+88h] [rbp-188h] BYREF
  void *v115; // [rsp+90h] [rbp-180h] BYREF
  char *v116; // [rsp+98h] [rbp-178h] BYREF
  void *base; // [rsp+A0h] [rbp-170h] BYREF
  __int64 v118; // [rsp+A8h] [rbp-168h]
  _BYTE v119[16]; // [rsp+B0h] [rbp-160h] BYREF
  void *v120; // [rsp+C0h] [rbp-150h] BYREF
  __int64 v121; // [rsp+C8h] [rbp-148h]
  _BYTE v122[16]; // [rsp+D0h] [rbp-140h] BYREF
  _QWORD v123[2]; // [rsp+E0h] [rbp-130h] BYREF
  __int64 v124; // [rsp+F0h] [rbp-120h]
  __int64 v125; // [rsp+F8h] [rbp-118h]
  __int64 v126; // [rsp+100h] [rbp-110h]
  __int64 v127; // [rsp+108h] [rbp-108h]
  __int64 v128; // [rsp+110h] [rbp-100h]
  __int64 v129; // [rsp+118h] [rbp-F8h]
  char *v130; // [rsp+120h] [rbp-F0h]
  char *v131; // [rsp+128h] [rbp-E8h]
  __int64 v132; // [rsp+130h] [rbp-E0h]
  __int64 v133; // [rsp+138h] [rbp-D8h]
  __int64 v134; // [rsp+140h] [rbp-D0h]
  __int64 v135; // [rsp+148h] [rbp-C8h]
  char *v136; // [rsp+150h] [rbp-C0h] BYREF
  __int64 v137; // [rsp+158h] [rbp-B8h]
  __int64 *v138; // [rsp+160h] [rbp-B0h] BYREF
  unsigned int v139; // [rsp+168h] [rbp-A8h]
  _BYTE v140[48]; // [rsp+1E0h] [rbp-30h] BYREF

  v102 = a1 + 88;
  if ( a3 )
    v102 = *(_QWORD *)(a4 + 8);
  v4 = *(_QWORD *)(a2 + 16);
  v123[0] = 0;
  v123[1] = 0;
  v124 = 0;
  v125 = 0;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v130 = 0;
  v131 = 0;
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v135 = 0;
  sub_FEBEE0((int *)v123, v4);
  while ( 1 )
  {
    sub_FEC3C0((__int64)v123);
    if ( v131 == v130 )
      break;
    if ( (unsigned __int64)(v131 - v130) <= 8 )
      continue;
    v136 = 0;
    v113 = a3;
    base = v119;
    v118 = 0x400000000LL;
    v121 = 0x400000000LL;
    v5 = (__int64 *)&v138;
    v120 = v122;
    v137 = 1;
    do
    {
      *v5 = -4096;
      v5 += 2;
    }
    while ( v5 != (__int64 *)v140 );
    v6 = (__int64 *)v131;
    v7 = (__int64 *)v130;
    if ( v130 != v131 )
    {
      do
      {
        while ( 1 )
        {
          v13 = *v7;
          if ( (v137 & 1) == 0 )
            break;
          v8 = 7;
          v9 = (__int64 *)&v138;
LABEL_11:
          v10 = v8 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v11 = &v9[2 * v10];
          v12 = *v11;
          if ( v13 == *v11 )
            goto LABEL_12;
          v81 = 1;
          v16 = 0;
          while ( 1 )
          {
            if ( v12 == -4096 )
            {
              v15 = v137;
              v18 = 24;
              v14 = 8;
              if ( !v16 )
                v16 = v11;
              ++v136;
              v17 = ((unsigned int)v137 >> 1) + 1;
              if ( (v137 & 1) == 0 )
              {
                v14 = v139;
                goto LABEL_22;
              }
              goto LABEL_23;
            }
            if ( v12 != -8192 || v16 )
              v11 = v16;
            v10 = v8 & (v81 + v10);
            v12 = v9[2 * v10];
            if ( v13 == v12 )
              break;
            ++v81;
            v16 = v11;
            v11 = &v9[2 * v10];
          }
          v11 = &v9[2 * v10];
LABEL_12:
          ++v7;
          *((_BYTE *)v11 + 8) = 0;
          if ( v6 == v7 )
            goto LABEL_28;
        }
        v14 = v139;
        v9 = v138;
        if ( v139 )
        {
          v8 = v139 - 1;
          goto LABEL_11;
        }
        v15 = v137;
        ++v136;
        v16 = 0;
        v17 = ((unsigned int)v137 >> 1) + 1;
LABEL_22:
        v18 = 3 * v14;
LABEL_23:
        if ( v18 <= 4 * v17 )
        {
          sub_FEB670((__int64)&v136, 2 * v14);
          if ( (v137 & 1) != 0 )
          {
            v83 = 7;
            v84 = (__int64 *)&v138;
          }
          else
          {
            v84 = v138;
            if ( !v139 )
              goto LABEL_192;
            v83 = v139 - 1;
          }
          v15 = v137;
          v85 = v83 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v16 = &v84[2 * v85];
          v86 = *v16;
          if ( v13 == *v16 )
            goto LABEL_25;
          v87 = 1;
          v88 = 0;
          while ( v86 != -4096 )
          {
            if ( v86 == -8192 && !v88 )
              v88 = v16;
            v85 = v83 & (v87 + v85);
            v16 = &v84[2 * v85];
            v86 = *v16;
            if ( v13 == *v16 )
              goto LABEL_151;
            ++v87;
          }
        }
        else
        {
          if ( v14 - HIDWORD(v137) - v17 > v14 >> 3 )
            goto LABEL_25;
          sub_FEB670((__int64)&v136, v14);
          if ( (v137 & 1) != 0 )
          {
            v89 = 7;
            v90 = (__int64 *)&v138;
          }
          else
          {
            v90 = v138;
            if ( !v139 )
            {
LABEL_192:
              LODWORD(v137) = (2 * ((unsigned int)v137 >> 1) + 2) | v137 & 1;
              BUG();
            }
            v89 = v139 - 1;
          }
          v15 = v137;
          v91 = v89 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v16 = &v90[2 * v91];
          v92 = *v16;
          if ( v13 == *v16 )
            goto LABEL_25;
          v93 = 1;
          v88 = 0;
          while ( v92 != -4096 )
          {
            if ( v92 == -8192 && !v88 )
              v88 = v16;
            v91 = v89 & (v93 + v91);
            v16 = &v90[2 * v91];
            v92 = *v16;
            if ( v13 == *v16 )
              goto LABEL_151;
            ++v93;
          }
        }
        if ( v88 )
          v16 = v88;
LABEL_151:
        v15 = v137;
LABEL_25:
        LODWORD(v137) = (2 * (v15 >> 1) + 2) | v15 & 1;
        if ( *v16 != -4096 )
          --HIDWORD(v137);
        ++v7;
        *v16 = v13;
        *((_BYTE *)v16 + 8) = 0;
        *((_BYTE *)v16 + 8) = 0;
      }
      while ( v6 != v7 );
    }
LABEL_28:
    v19 = v137;
    v20 = v137 & 1;
    v21 = (unsigned int)v137 >> 1;
    if ( (unsigned int)v137 >> 1 )
    {
      if ( v20 )
      {
        v22 = (__int64 *)v140;
        v23 = (__int64 *)&v138;
      }
      else
      {
        v25 = v139;
        v24 = v138;
        v23 = v138;
        v82 = 2LL * v139;
        v22 = &v138[v82];
        if ( &v138[v82] == v138 )
          goto LABEL_36;
      }
      do
      {
        if ( *v23 != -8192 && *v23 != -4096 )
          break;
        v23 += 2;
      }
      while ( v23 != v22 );
    }
    else
    {
      if ( v20 )
      {
        v95 = (__int64 *)&v138;
        v96 = 16;
      }
      else
      {
        v95 = v138;
        v96 = 2LL * v139;
      }
      v23 = &v95[v96];
      v22 = v23;
    }
    if ( !v20 )
    {
      v24 = v138;
      v25 = v139;
LABEL_36:
      v26 = 2 * v25;
      goto LABEL_37;
    }
    v24 = (__int64 *)&v138;
    v26 = 16;
LABEL_37:
    v111 = &v24[v26];
    if ( &v24[v26] == v23 )
      goto LABEL_66;
    while ( 1 )
    {
      v27 = *v23;
      v28 = *(__int64 **)(*v23 + 24);
      v29 = *(unsigned int *)(*v23 + 4);
      v30 = *(__int64 **)(*v23 + 40);
      v31 = *(_QWORD *)(*v23 + 48);
      v32 = v29 + (((__int64)v28 - *(_QWORD *)(*v23 + 32)) >> 3);
      if ( v32 < 0 )
      {
        v45 = ~((unsigned __int64)~v32 >> 6);
        goto LABEL_61;
      }
      if ( v32 > 63 )
      {
        v45 = v32 >> 6;
LABEL_61:
        v33 = *(_QWORD *)(v31 + 8 * v45) + 8 * (v32 - (v45 << 6));
        goto LABEL_41;
      }
      v33 = (__int64)&v28[v29];
LABEL_41:
      v34 = v19 & 1;
      v35 = 8;
      v36 = (__int64 **)(v31 + 8);
      v37 = (int *)*v23;
      v38 = (__int64 *)&v138;
      if ( !v34 )
      {
        v38 = v138;
        v35 = v139;
      }
      v39 = v35 - 1;
LABEL_44:
      if ( (__int64 *)v33 != v28 )
      {
        while ( 1 )
        {
          v40 = *v28;
          if ( !(_BYTE)v34 && !v139 )
            break;
          v41 = v39 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
          v27 = v38[2 * v41];
          if ( v40 != v27 )
          {
            v109 = 1;
            while ( v27 != -4096 )
            {
              v41 = v39 & (v109 + v41);
              ++v109;
              v27 = v38[2 * v41];
              if ( v40 == v27 )
                goto LABEL_48;
            }
            break;
          }
LABEL_48:
          if ( v30 != ++v28 )
            goto LABEL_44;
          v28 = *v36++;
          v30 = v28 + 64;
          if ( (__int64 *)v33 == v28 )
            goto LABEL_50;
        }
        *((_BYTE *)v23 + 8) = 1;
        v42 = (unsigned int)v118;
        v43 = *v37;
        v44 = (unsigned int)v118 + 1LL;
        if ( v44 > HIDWORD(v118) )
        {
          sub_C8D5F0((__int64)&base, v119, v44, 4u, v27, (__int64)v37);
          v42 = (unsigned int)v118;
        }
        *((_DWORD *)base + v42) = v43;
        LODWORD(v118) = v118 + 1;
      }
      do
LABEL_50:
        v23 += 2;
      while ( v23 != v22 && (*v23 == -8192 || *v23 == -4096) );
      if ( v23 == v111 )
        break;
      v19 = v137;
    }
    v21 = (unsigned int)v137 >> 1;
LABEL_66:
    v46 = (unsigned int)v118;
    if ( (_DWORD)v118 == v21 )
    {
      if ( (unsigned int)v118 > 1uLL )
        qsort(base, (4LL * (unsigned int)v118) >> 2, 4u, (__compar_fn_t)sub_FE8120);
    }
    else
    {
      v47 = v137 & 1;
      if ( !v21 )
      {
        if ( v47 )
        {
          v97 = (__int64 *)&v138;
          v98 = 16;
        }
        else
        {
          v97 = v138;
          v98 = 2LL * v139;
        }
        v49 = &v97[v98];
        v48 = &v97[v98];
        goto LABEL_73;
      }
      if ( v47 )
      {
        v48 = (__int64 *)v140;
        v49 = (__int64 *)&v138;
        do
        {
LABEL_70:
          if ( *v49 != -4096 && *v49 != -8192 )
            break;
          v49 += 2;
        }
        while ( v49 != v48 );
LABEL_73:
        if ( !v47 )
        {
          v50 = v138;
          v51 = v139;
          goto LABEL_75;
        }
        v50 = (__int64 *)&v138;
        v52 = 16;
      }
      else
      {
        v51 = v139;
        v50 = v138;
        v49 = v138;
        v94 = 2LL * v139;
        v48 = &v138[v94];
        if ( &v138[v94] != v138 )
          goto LABEL_70;
LABEL_75:
        v52 = 2 * v51;
      }
      v53 = &v50[v52];
      if ( v49 != &v50[v52] )
      {
        while ( 2 )
        {
          v54 = v46;
          if ( *((_BYTE *)v49 + 8) )
            goto LABEL_78;
          v65 = (unsigned int *)*v49;
          v66 = *(_QWORD **)(*v49 + 24);
          v110 = *(_QWORD *)(*v49 + 40);
          v67 = *(unsigned int *)(*v49 + 4);
          v68 = v67 + (((__int64)v66 - *(_QWORD *)(*v49 + 32)) >> 3);
          if ( v68 < 0 )
          {
            v80 = ~((unsigned __int64)~v68 >> 6);
          }
          else
          {
            if ( v68 <= 63 )
            {
              v69 = (__int64)&v66[v67];
              goto LABEL_101;
            }
            v80 = v68 >> 6;
          }
          v69 = *(_QWORD *)(*(_QWORD *)(*v49 + 48) + 8 * v80) + 8 * (v68 - (v80 << 6));
LABEL_101:
          v70 = v139;
          v71 = 8;
          v106 = *(_QWORD ***)(*v49 + 48);
          v72 = *v65;
          if ( (v137 & 1) == 0 )
            v71 = v139;
          v73 = (__int64 *)&v138;
          if ( (v137 & 1) == 0 )
            v73 = v138;
          v107 = v73;
          v108 = v71 - 1;
          while ( (_QWORD *)v69 != v66 )
          {
            v74 = (_DWORD *)*v66;
            if ( *(_DWORD *)*v66 >= v72 )
            {
              if ( (v137 & 1) == 0 && !v139 )
                goto LABEL_112;
              v70 = v108 & (((unsigned int)v74 >> 9) ^ ((unsigned int)v74 >> 4));
              v75 = &v107[2 * v70];
              v76 = (_DWORD *)*v75;
              if ( v74 != (_DWORD *)*v75 )
              {
                v100 = 1;
                while ( v76 != (_DWORD *)-4096LL )
                {
                  v70 = v108 & (unsigned int)(v100 + v70);
                  v101 = v100 + 1;
                  v75 = &v107[2 * (unsigned int)v70];
                  v76 = (_DWORD *)*v75;
                  if ( v74 == (_DWORD *)*v75 )
                    goto LABEL_111;
                  v100 = v101;
                }
LABEL_112:
                v77 = v46 + 1;
                if ( v77 > HIDWORD(v118) )
                {
                  sub_C8D5F0((__int64)&base, v119, v77, 4u, v69, v70);
                  v54 = (unsigned int)v118;
                }
                *((_DWORD *)base + v54) = v72;
                v46 = (unsigned int)(v118 + 1);
                LODWORD(v118) = v118 + 1;
                v72 = *v65;
                break;
              }
LABEL_111:
              if ( !*((_BYTE *)v75 + 8) )
                goto LABEL_112;
            }
            if ( (_QWORD *)v110 == ++v66 )
            {
              v66 = *++v106;
              v110 = (__int64)(*v106 + 64);
            }
          }
          if ( *((_DWORD *)base + v46 - 1) != v72 )
          {
            v78 = (unsigned int)v121;
            v79 = (unsigned int)v121 + 1LL;
            if ( v79 > HIDWORD(v121) )
            {
              sub_C8D5F0((__int64)&v120, v122, v79, 4u, v69, v70);
              v78 = (unsigned int)v121;
            }
            *((_DWORD *)v120 + v78) = v72;
            v46 = (unsigned int)v118;
            LODWORD(v121) = v121 + 1;
          }
          do
LABEL_78:
            v49 += 2;
          while ( v49 != v48 && (*v49 == -4096 || *v49 == -8192) );
          if ( v53 == v49 )
            break;
          continue;
        }
      }
      if ( v46 > 1 )
        qsort(base, (__int64)(4 * v46) >> 2, 4u, (__compar_fn_t)sub_FE8120);
      if ( (unsigned int)v121 > 1uLL )
        qsort(v120, (4LL * (unsigned int)v121) >> 2, 4u, (__compar_fn_t)sub_FE8120);
    }
    if ( (v137 & 1) == 0 )
      sub_C7D6A0((__int64)v138, 16LL * v139, 8);
    v55 = (_DWORD *)a4;
    v115 = v120;
    v114 = (char *)v120 + 4 * (unsigned int)v121;
    v136 = (char *)base;
    v116 = (char *)base + 4 * (unsigned int)v118;
    v56 = sub_FE8F30(
            a1 + 88,
            a4,
            &v113,
            (const void **)&v136,
            (const void **)&v116,
            (const void **)&v115,
            (const void **)&v114);
    v57 = *(unsigned int **)(v56 + 112);
    v58 = &v57[*(unsigned int *)(v56 + 120)];
    if ( v57 != v58 )
    {
      v59 = v56 + 16;
      while ( 1 )
      {
        v62 = *(_QWORD *)(a1 + 64) + 24LL * *v57;
        v63 = *(_QWORD *)(v62 + 8);
        if ( !v63 )
          break;
        v60 = *(unsigned int *)(v63 + 12);
        v61 = *(_DWORD **)(v63 + 96);
        if ( (unsigned int)v60 > 1 )
        {
          v55 = &v61[v60];
          v112 = v59;
          v64 = sub_FDC990(v61, v55, (_DWORD *)v62);
          v59 = v112;
          if ( !v64 )
          {
            *(_QWORD *)(v62 + 8) = v112;
            goto LABEL_93;
          }
        }
        else if ( *(_DWORD *)v62 != *v61 )
        {
          break;
        }
        *(_QWORD *)v63 = v59;
LABEL_93:
        if ( v58 == ++v57 )
          goto LABEL_16;
      }
      *(_QWORD *)(v62 + 8) = v59;
      goto LABEL_93;
    }
LABEL_16:
    if ( v120 != v122 )
      _libc_free(v120, v55);
    if ( base != v119 )
      _libc_free(base, v55);
  }
  if ( v133 )
    j_j___libc_free_0(v133, v135 - v133);
  if ( v130 )
    j_j___libc_free_0(v130, v132 - (_QWORD)v130);
  if ( v127 )
    j_j___libc_free_0(v127, v129 - v127);
  sub_C7D6A0(v124, 16LL * (unsigned int)v126, 8);
  if ( a3 )
    return *(_QWORD *)v102;
  else
    return *(_QWORD *)(a1 + 88);
}
