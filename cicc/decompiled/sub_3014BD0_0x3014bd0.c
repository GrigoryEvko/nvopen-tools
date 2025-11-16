// Function: sub_3014BD0
// Address: 0x3014bd0
//
void __fastcall sub_3014BD0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  char v9; // dl
  __int64 v10; // r13
  __int64 v11; // rax
  int v12; // edx
  _BYTE *v13; // rax
  int v14; // eax
  unsigned __int64 v15; // rdx
  __int64 v16; // rdi
  int v17; // esi
  int v18; // eax
  __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  const __m128i *v21; // rbx
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // r8
  __m128i *v25; // rax
  __int64 v26; // r8
  __int64 i; // rbx
  unsigned __int8 *v28; // r12
  int v29; // ecx
  unsigned int v30; // ecx
  _QWORD *v31; // rax
  __int64 v32; // r12
  __int64 v33; // r14
  __int64 v34; // rax
  unsigned __int64 v35; // rdi
  __int64 v36; // rax
  int v37; // eax
  __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rbx
  char *v41; // rdx
  unsigned __int8 v42; // al
  __int64 v43; // rax
  __int64 v44; // rax
  _QWORD *v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rax
  int v49; // edx
  _BYTE *v50; // rax
  __int64 v51; // rax
  __int64 *v52; // rdx
  __int64 v53; // r13
  __int64 *v54; // rax
  __int64 v55; // r13
  __int64 v56; // r12
  __int64 v57; // rbx
  int v58; // edx
  _DWORD *v59; // r14
  __int64 *v60; // r13
  __int64 v61; // rdi
  __int64 v62; // r12
  _DWORD *v63; // r13
  __int64 v64; // rbx
  __int64 v65; // rax
  __int64 v66; // r9
  __int64 v67; // r14
  __int64 v68; // rcx
  _QWORD *v69; // rdx
  unsigned __int64 v70; // r8
  unsigned __int64 v71; // rsi
  unsigned __int64 v72; // rdx
  const __m128i *v73; // rbx
  unsigned __int64 v74; // rcx
  __int64 *v75; // r11
  __m128i *v76; // rdx
  __int64 j; // rax
  unsigned __int8 *v78; // r8
  int v79; // ecx
  unsigned int v80; // ecx
  unsigned int v81; // esi
  __int64 v82; // r9
  unsigned int v83; // r8d
  _QWORD *v84; // rax
  __int64 v85; // rdi
  unsigned int *v86; // rax
  __int64 v87; // rdx
  int v88; // esi
  _BYTE *v89; // rdx
  unsigned __int64 v90; // r10
  _QWORD *v91; // rdx
  __int64 v92; // rdi
  const void *v93; // rsi
  char *v94; // rbx
  __int64 *v95; // rdx
  int v96; // eax
  int v97; // edi
  int v98; // r9d
  int v99; // r9d
  __int64 v100; // r8
  unsigned int v101; // esi
  __int64 v102; // rax
  int v103; // ecx
  __int64 *v104; // r10
  int v105; // r9d
  int v106; // r9d
  __int64 v107; // r8
  int v108; // ecx
  unsigned int v109; // esi
  __int64 v110; // rax
  unsigned __int64 v111; // r9
  _QWORD *v112; // rax
  __int64 v113; // rdi
  const void *v114; // rsi
  char *v115; // rbx
  __int64 *v116; // rax
  __int64 *v117; // [rsp+0h] [rbp-190h]
  __int64 v118; // [rsp+8h] [rbp-188h]
  unsigned __int64 v119; // [rsp+10h] [rbp-180h]
  unsigned __int64 v120; // [rsp+18h] [rbp-178h]
  __int64 *v121; // [rsp+20h] [rbp-170h]
  unsigned __int8 *v122; // [rsp+20h] [rbp-170h]
  int v124; // [rsp+38h] [rbp-158h]
  __int64 *v125; // [rsp+38h] [rbp-158h]
  unsigned int v126; // [rsp+38h] [rbp-158h]
  unsigned __int64 v127; // [rsp+48h] [rbp-148h]
  __int64 v128; // [rsp+58h] [rbp-138h]
  int v129; // [rsp+64h] [rbp-12Ch]
  unsigned int v130; // [rsp+68h] [rbp-128h]
  _DWORD *v131; // [rsp+70h] [rbp-120h]
  unsigned __int64 v132; // [rsp+70h] [rbp-120h]
  __int64 v133; // [rsp+70h] [rbp-120h]
  __int64 v134; // [rsp+78h] [rbp-118h]
  unsigned int v135; // [rsp+78h] [rbp-118h]
  __int64 v136; // [rsp+78h] [rbp-118h]
  unsigned int v137; // [rsp+78h] [rbp-118h]
  __int64 v138; // [rsp+78h] [rbp-118h]
  __int64 *v139; // [rsp+78h] [rbp-118h]
  unsigned __int64 v140; // [rsp+80h] [rbp-110h] BYREF
  int v141; // [rsp+88h] [rbp-108h]
  int v142; // [rsp+8Ch] [rbp-104h]
  unsigned int v143; // [rsp+90h] [rbp-100h]
  int v144; // [rsp+94h] [rbp-FCh]
  _DWORD *v145; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v146; // [rsp+A8h] [rbp-E8h]
  _DWORD v147[8]; // [rsp+B0h] [rbp-E0h] BYREF
  _BYTE *v148; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v149; // [rsp+D8h] [rbp-B8h]
  _BYTE v150[176]; // [rsp+E0h] [rbp-B0h] BYREF

  if ( !*(_DWORD *)(a2 + 16) )
  {
    v2 = *(_QWORD *)(a1 + 80);
    v3 = a1 + 72;
    v148 = v150;
    v149 = 0x800000000LL;
    if ( a1 + 72 == v2 )
      goto LABEL_26;
    while ( 1 )
    {
      while ( 1 )
      {
        v5 = v2 - 24;
        if ( !v2 )
          v5 = 0;
        v6 = sub_AA4FF0(v5);
        if ( !v6 )
          BUG();
        v9 = *(_BYTE *)(v6 - 24);
        if ( v9 != 80 )
          break;
        if ( **(_BYTE **)(v6 - 56) == 21 )
          goto LABEL_12;
LABEL_5:
        v2 = *(_QWORD *)(v2 + 8);
        if ( v3 == v2 )
          goto LABEL_16;
      }
      if ( v9 != 39 || ***(_BYTE ***)(v6 - 32) != 21 )
        goto LABEL_5;
LABEL_12:
      v10 = v6 - 24;
      v11 = (unsigned int)v149;
      v12 = v149;
      if ( (unsigned int)v149 >= (unsigned __int64)HIDWORD(v149) )
      {
        v7 = v134 | 0xFFFFFFFFLL;
        v134 |= 0xFFFFFFFFuLL;
        if ( HIDWORD(v149) < (unsigned __int64)(unsigned int)v149 + 1 )
        {
          v133 = v7;
          sub_C8D5F0((__int64)&v148, v150, (unsigned int)v149 + 1LL, 0x10u, v7, v8);
          v11 = (unsigned int)v149;
          v7 = v133;
        }
        v116 = (__int64 *)&v148[16 * v11];
        *v116 = v10;
        v116[1] = v7;
        LODWORD(v149) = v149 + 1;
        goto LABEL_5;
      }
      v13 = &v148[16 * (unsigned int)v149];
      if ( v13 )
      {
        *(_QWORD *)v13 = v10;
        *((_DWORD *)v13 + 2) = -1;
        v12 = v149;
      }
      LODWORD(v149) = v12 + 1;
      v2 = *(_QWORD *)(v2 + 8);
      if ( v3 == v2 )
      {
LABEL_16:
        v14 = v149;
        if ( !(_DWORD)v149 )
        {
LABEL_26:
          v32 = *(_QWORD *)(a2 + 624);
          v33 = v32 + 24LL * *(unsigned int *)(a2 + 632);
          if ( v32 == v33 )
          {
LABEL_49:
            sub_30138B0(a1, a2);
            if ( v148 != v150 )
              _libc_free((unsigned __int64)v148);
            return;
          }
          while ( 1 )
          {
            v38 = *(_QWORD *)(v33 - 24);
            v33 -= 24;
            v39 = sub_AA4FF0(v38 & 0xFFFFFFFFFFFFFFF8LL);
            if ( !v39 )
              BUG();
            if ( *(_BYTE *)(v39 - 24) != 81 )
              break;
            if ( *(_DWORD *)(v33 + 16) == -1 )
            {
              v34 = *(_QWORD *)(v39 - 56);
              if ( (*(_BYTE *)(v34 + 2) & 1) != 0 )
              {
                v35 = *(_QWORD *)(*(_QWORD *)(v34 - 8) + 32LL);
                if ( v35 )
                  goto LABEL_31;
              }
LABEL_48:
              v37 = -1;
LABEL_34:
              *(_DWORD *)(v33 + 16) = v37;
            }
            if ( v32 == v33 )
              goto LABEL_49;
          }
          v40 = *(_QWORD *)(v39 - 8);
          if ( !v40 )
            goto LABEL_48;
          v136 = v39 - 24;
          while ( 1 )
          {
            v41 = *(char **)(v40 + 24);
            v42 = *v41;
            if ( (unsigned __int8)*v41 <= 0x1Cu )
              goto LABEL_43;
            if ( v42 == 37 )
            {
              if ( (v41[2] & 1) == 0 )
                goto LABEL_48;
              v35 = *(_QWORD *)&v41[32 * (1LL - (*((_DWORD *)v41 + 1) & 0x7FFFFFF))];
              if ( !v35 )
                goto LABEL_48;
LABEL_31:
              v36 = sub_AA4FF0(v35);
              if ( v36 )
                v36 -= 24;
              v145 = (_DWORD *)v36;
              v37 = *(_DWORD *)sub_3014430(a2, (__int64 *)&v145);
              goto LABEL_34;
            }
            if ( v42 == 34 )
              break;
            if ( v42 != 39 )
            {
              if ( v42 != 80 )
                goto LABEL_43;
              v145 = *(_DWORD **)(v40 + 24);
              v45 = sub_3014430(a2, (__int64 *)&v145);
              v46 = *(_QWORD *)(a2 + 624);
              v47 = *(int *)(v46 + 24LL * *(int *)v45 + 16);
              if ( (_DWORD)v47 == -1 )
                goto LABEL_43;
              v35 = *(_QWORD *)(v46 + 24 * v47) & 0xFFFFFFFFFFFFFFF8LL;
LABEL_53:
              if ( !v35 )
                goto LABEL_43;
              goto LABEL_54;
            }
            if ( (v41[2] & 1) == 0 )
              goto LABEL_43;
            v35 = *(_QWORD *)(*((_QWORD *)v41 - 1) + 32LL);
            if ( !v35 )
              goto LABEL_43;
LABEL_54:
            v43 = sub_AA4FF0(v35);
            if ( !v43 )
              BUG();
            if ( *(_BYTE *)(v43 - 24) == 39 )
            {
              v44 = **(_QWORD **)(v43 - 32);
              if ( !v44 )
                goto LABEL_31;
            }
            else
            {
              v44 = *(_QWORD *)(v43 - 56);
            }
            if ( v44 != v136 )
              goto LABEL_31;
LABEL_43:
            v40 = *(_QWORD *)(v40 + 8);
            if ( !v40 )
              goto LABEL_48;
          }
          v35 = *((_QWORD *)v41 - 8);
          goto LABEL_53;
        }
        while ( 2 )
        {
          v15 = (unsigned __int64)&v148[16 * v14 - 16];
          v16 = *(_QWORD *)v15;
          v17 = *(_DWORD *)(v15 + 8);
          LODWORD(v149) = v14 - 1;
          v128 = v16;
          v129 = v17;
          if ( *(_BYTE *)v16 == 80 )
          {
            v18 = *(_DWORD *)(v16 + 4);
            v19 = *(_QWORD *)(v16 + 40);
            HIDWORD(v146) = v17;
            v20 = *(unsigned int *)(a2 + 636);
            v21 = (const __m128i *)&v145;
            v147[0] = -1;
            LODWORD(v146) = 0;
            v145 = (_DWORD *)(v19 & 0xFFFFFFFFFFFFFFFBLL);
            v22 = *(_QWORD *)(a2 + 624);
            v147[1] = ((v18 & 0x7FFFFFF) != 1) + 1;
            v23 = *(unsigned int *)(a2 + 632);
            v24 = v23 + 1;
            if ( v23 + 1 > v20 )
            {
              v113 = a2 + 624;
              v114 = (const void *)(a2 + 640);
              if ( v22 > (unsigned __int64)&v145 || (unsigned __int64)&v145 >= v22 + 24 * v23 )
              {
                sub_C8D5F0(v113, v114, v24, 0x18u, v24, v8);
                v22 = *(_QWORD *)(a2 + 624);
                v23 = *(unsigned int *)(a2 + 632);
              }
              else
              {
                v115 = (char *)&v145 - v22;
                sub_C8D5F0(v113, v114, v24, 0x18u, v24, v8);
                v22 = *(_QWORD *)(a2 + 624);
                v23 = *(unsigned int *)(a2 + 632);
                v21 = (const __m128i *)&v115[v22];
              }
            }
            v25 = (__m128i *)(v22 + 24 * v23);
            *v25 = _mm_loadu_si128(v21);
            v25[1].m128i_i64[0] = v21[1].m128i_i64[0];
            v26 = *(unsigned int *)(a2 + 632);
            *(_DWORD *)(a2 + 632) = v26 + 1;
            for ( i = *(_QWORD *)(v128 + 16); i; i = *(_QWORD *)(i + 8) )
            {
              v28 = *(unsigned __int8 **)(i + 24);
              v29 = *v28;
              if ( (unsigned __int8)v29 > 0x1Cu )
              {
                v30 = v29 - 39;
                if ( v30 <= 0x38 && ((1LL << v30) & 0x100060000000001LL) != 0 )
                {
                  v48 = (unsigned int)v149;
                  v49 = v149;
                  if ( (unsigned int)v149 >= (unsigned __int64)HIDWORD(v149) )
                  {
                    v111 = (unsigned int)v26 | v120 & 0xFFFFFFFF00000000LL;
                    v120 = v111;
                    if ( HIDWORD(v149) < (unsigned __int64)(unsigned int)v149 + 1 )
                    {
                      v130 = v26;
                      v132 = v111;
                      sub_C8D5F0((__int64)&v148, v150, (unsigned int)v149 + 1LL, 0x10u, v26, v111);
                      v48 = (unsigned int)v149;
                      v26 = v130;
                      v111 = v132;
                    }
                    v112 = &v148[16 * v48];
                    *v112 = v28;
                    v112[1] = v111;
                    LODWORD(v149) = v149 + 1;
                  }
                  else
                  {
                    v50 = &v148[16 * (unsigned int)v149];
                    if ( v50 )
                    {
                      *(_QWORD *)v50 = v28;
                      *((_DWORD *)v50 + 2) = v26;
                      v49 = v149;
                    }
                    LODWORD(v149) = v49 + 1;
                  }
                }
              }
            }
            v135 = v26;
            v145 = (_DWORD *)v128;
            v31 = sub_3014430(a2, (__int64 *)&v145);
            v7 = v135;
            *(_DWORD *)v31 = v135;
            goto LABEL_25;
          }
          v51 = *(_QWORD *)(v16 - 8);
          v52 = (__int64 *)(v51 + 32);
          v53 = v51 + 32LL * (*(_DWORD *)(v16 + 4) & 0x7FFFFFF);
          v54 = (__int64 *)(v51 + 64);
          if ( (*(_BYTE *)(v16 + 2) & 1) == 0 )
            v54 = v52;
          v145 = v147;
          v146 = 0x400000000LL;
          v55 = v53 - (_QWORD)v54;
          v56 = v55 >> 5;
          v57 = v55 >> 5;
          if ( (unsigned __int64)v55 > 0x80 )
          {
            v139 = v54;
            sub_C8D5F0((__int64)&v145, v147, v55 >> 5, 8u, v7, v8);
            v58 = v146;
            v131 = v145;
            v59 = &v145[2 * (unsigned int)v146];
            v54 = v139;
          }
          else
          {
            v58 = 0;
            v131 = v147;
            v59 = v147;
          }
          if ( v55 > 0 )
          {
            v60 = v54;
            do
            {
              v61 = *v60;
              v59 += 2;
              v60 += 4;
              *((_QWORD *)v59 - 1) = sub_3011D80(v61);
              --v57;
            }
            while ( v57 );
            v58 = v146;
            v131 = v145;
          }
          v137 = -1;
          LODWORD(v146) = v58 + v56;
          v62 = (unsigned int)(v58 + v56);
          if ( &v131[2 * v62] == v131 )
          {
            v75 = (__int64 *)&v140;
LABEL_91:
            v140 = v128;
            *(_DWORD *)sub_3014430(a2, v75) = v137;
            if ( v145 != v147 )
              _libc_free((unsigned __int64)v145);
LABEL_25:
            v14 = v149;
            if ( !(_DWORD)v149 )
              goto LABEL_26;
            continue;
          }
          break;
        }
        v63 = &v131[2 * v62];
        while ( 2 )
        {
          v64 = *((_QWORD *)v63 - 1);
          v65 = sub_AA4FF0(v64);
          if ( !v65 )
            BUG();
          v67 = v65 - 24;
          v68 = *(_QWORD *)(v65 - 32LL * (*(_DWORD *)(v65 - 20) & 0x7FFFFFF) - 24);
          v69 = *(_QWORD **)(v68 + 24);
          if ( *(_DWORD *)(v68 + 32) > 0x40u )
            v69 = (_QWORD *)*v69;
          v70 = *(unsigned int *)(a2 + 632);
          v141 = (int)v69;
          v140 = v64 & 0xFFFFFFFFFFFFFFFBLL;
          v71 = *(unsigned int *)(a2 + 636);
          v142 = v129;
          v72 = v70 + 1;
          v73 = (const __m128i *)&v140;
          v74 = *(_QWORD *)(a2 + 624);
          v144 = 0;
          v143 = v137;
          v75 = (__int64 *)&v140;
          if ( v70 + 1 > v71 )
          {
            v92 = a2 + 624;
            v93 = (const void *)(a2 + 640);
            v138 = v65;
            if ( v74 > (unsigned __int64)&v140 || (v70 = v74 + 24 * v70, (unsigned __int64)&v140 >= v70) )
            {
              sub_C8D5F0(v92, v93, v72, 0x18u, v70, v66);
              v74 = *(_QWORD *)(a2 + 624);
              v70 = *(unsigned int *)(a2 + 632);
              v73 = (const __m128i *)&v140;
              v75 = (__int64 *)&v140;
              v65 = v138;
            }
            else
            {
              v94 = (char *)&v140 - v74;
              sub_C8D5F0(v92, v93, v72, 0x18u, v70, v66);
              v74 = *(_QWORD *)(a2 + 624);
              v70 = *(unsigned int *)(a2 + 632);
              v65 = v138;
              v75 = (__int64 *)&v140;
              v73 = (const __m128i *)&v94[v74];
            }
          }
          v76 = (__m128i *)(v74 + 24 * v70);
          *v76 = _mm_loadu_si128(v73);
          v76[1].m128i_i64[0] = v73[1].m128i_i64[0];
          v137 = *(_DWORD *)(a2 + 632);
          *(_DWORD *)(a2 + 632) = v137 + 1;
          for ( j = *(_QWORD *)(v65 - 8); j; j = *(_QWORD *)(j + 8) )
          {
            v78 = *(unsigned __int8 **)(j + 24);
            v79 = *v78;
            if ( (unsigned __int8)v79 > 0x1Cu )
            {
              v80 = v79 - 39;
              if ( v80 <= 0x38 && ((1LL << v80) & 0x100060000000001LL) != 0 )
              {
                v87 = (unsigned int)v149;
                v88 = v149;
                if ( (unsigned int)v149 >= (unsigned __int64)HIDWORD(v149) )
                {
                  v90 = v127 & 0xFFFFFFFF00000000LL | v137;
                  v127 = v90;
                  if ( HIDWORD(v149) < (unsigned __int64)(unsigned int)v149 + 1 )
                  {
                    v117 = v75;
                    v118 = j;
                    v119 = v90;
                    v122 = *(unsigned __int8 **)(j + 24);
                    sub_C8D5F0((__int64)&v148, v150, (unsigned int)v149 + 1LL, 0x10u, (__int64)v78, 1);
                    v87 = (unsigned int)v149;
                    v75 = v117;
                    j = v118;
                    v90 = v119;
                    v78 = v122;
                  }
                  v91 = &v148[16 * v87];
                  *v91 = v78;
                  v91[1] = v90;
                  LODWORD(v149) = v149 + 1;
                }
                else
                {
                  v89 = &v148[16 * (unsigned int)v149];
                  if ( v89 )
                  {
                    *(_QWORD *)v89 = v78;
                    *((_DWORD *)v89 + 2) = v137;
                    v88 = v149;
                  }
                  LODWORD(v149) = v88 + 1;
                }
              }
            }
          }
          v81 = *(_DWORD *)(a2 + 24);
          if ( v81 )
          {
            v82 = *(_QWORD *)(a2 + 8);
            v83 = (v81 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
            v84 = (_QWORD *)(v82 + 16LL * v83);
            v85 = *v84;
            if ( v67 == *v84 )
            {
LABEL_89:
              v86 = (unsigned int *)(v84 + 1);
              goto LABEL_90;
            }
            v124 = 1;
            v95 = 0;
            while ( v85 != -4096 )
            {
              if ( v85 == -8192 && !v95 )
                v95 = v84;
              v83 = (v81 - 1) & (v124 + v83);
              v84 = (_QWORD *)(v82 + 16LL * v83);
              v85 = *v84;
              if ( v67 == *v84 )
                goto LABEL_89;
              ++v124;
            }
            if ( !v95 )
              v95 = v84;
            v96 = *(_DWORD *)(a2 + 16);
            ++*(_QWORD *)a2;
            v97 = v96 + 1;
            if ( 4 * (v96 + 1) < 3 * v81 )
            {
              if ( v81 - *(_DWORD *)(a2 + 20) - v97 <= v81 >> 3 )
              {
                v121 = v75;
                v126 = ((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4);
                sub_30136D0(a2, v81);
                v105 = *(_DWORD *)(a2 + 24);
                if ( !v105 )
                {
LABEL_153:
                  ++*(_DWORD *)(a2 + 16);
                  BUG();
                }
                v106 = v105 - 1;
                v107 = *(_QWORD *)(a2 + 8);
                v104 = 0;
                v75 = v121;
                v97 = *(_DWORD *)(a2 + 16) + 1;
                v108 = 1;
                v109 = v106 & v126;
                v95 = (__int64 *)(v107 + 16LL * (v106 & v126));
                v110 = *v95;
                if ( v67 != *v95 )
                {
                  while ( v110 != -4096 )
                  {
                    if ( !v104 && v110 == -8192 )
                      v104 = v95;
                    v109 = v106 & (v108 + v109);
                    v95 = (__int64 *)(v107 + 16LL * v109);
                    v110 = *v95;
                    if ( v67 == *v95 )
                      goto LABEL_110;
                    ++v108;
                  }
                  goto LABEL_118;
                }
              }
              goto LABEL_110;
            }
          }
          else
          {
            ++*(_QWORD *)a2;
          }
          v125 = v75;
          sub_30136D0(a2, 2 * v81);
          v98 = *(_DWORD *)(a2 + 24);
          if ( !v98 )
            goto LABEL_153;
          v99 = v98 - 1;
          v100 = *(_QWORD *)(a2 + 8);
          v75 = v125;
          v101 = v99 & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
          v97 = *(_DWORD *)(a2 + 16) + 1;
          v95 = (__int64 *)(v100 + 16LL * v101);
          v102 = *v95;
          if ( v67 != *v95 )
          {
            v103 = 1;
            v104 = 0;
            while ( v102 != -4096 )
            {
              if ( !v104 && v102 == -8192 )
                v104 = v95;
              v101 = v99 & (v103 + v101);
              v95 = (__int64 *)(v100 + 16LL * v101);
              v102 = *v95;
              if ( v67 == *v95 )
                goto LABEL_110;
              ++v103;
            }
LABEL_118:
            if ( v104 )
              v95 = v104;
          }
LABEL_110:
          *(_DWORD *)(a2 + 16) = v97;
          if ( *v95 != -4096 )
            --*(_DWORD *)(a2 + 20);
          *v95 = v67;
          v86 = (unsigned int *)(v95 + 1);
          *((_DWORD *)v95 + 2) = 0;
LABEL_90:
          *v86 = v137;
          v63 -= 2;
          if ( v63 == v131 )
            goto LABEL_91;
          continue;
        }
      }
    }
  }
}
