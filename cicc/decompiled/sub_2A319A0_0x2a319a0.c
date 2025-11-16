// Function: sub_2A319A0
// Address: 0x2a319a0
//
_QWORD *__fastcall sub_2A319A0(
        const __m128i *a1,
        const __m128i *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 *a9)
{
  __int64 v9; // rbx
  unsigned __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // r13
  unsigned __int64 *v13; // r13
  unsigned __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // rax
  unsigned int v18; // r13d
  const char *v19; // r12
  __int64 v20; // r13
  _QWORD **v21; // rax
  int v22; // ecx
  int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // rsi
  const char *v26; // r12
  __int64 v27; // r13
  _QWORD *v28; // rdi
  __int64 v29; // r9
  __int64 v30; // rdx
  __int64 v31; // r13
  __int64 v32; // r14
  __int64 v33; // rdi
  __int64 v34; // rax
  unsigned int v35; // r9d
  int v36; // edx
  __int64 v37; // rax
  __int64 v38; // r15
  int v39; // edx
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 i; // r13
  __int64 v45; // rcx
  __int64 v46; // rax
  __int64 *v47; // r8
  __int64 v48; // rdi
  unsigned int v49; // esi
  __int64 v50; // rsi
  __int64 v51; // rdi
  _QWORD *v52; // rax
  __int64 *v54; // r12
  __int64 v55; // rdi
  unsigned __int64 v56; // rax
  const __m128i *v57; // r13
  __int64 v58; // rax
  __m128i *v59; // rdx
  const __m128i *v60; // rax
  __m128i v61; // xmm0
  __m128i *v62; // rdx
  const __m128i *v63; // rax
  __m128i v64; // xmm1
  __int64 v65; // rcx
  const void **v66; // r12
  int v67; // eax
  __int64 *v68; // rax
  __int64 v69; // rax
  unsigned int v70; // eax
  __int64 v71; // rax
  __int64 v72; // r12
  __int64 v73; // rax
  _QWORD *v74; // r12
  __int64 v75; // rax
  _QWORD *v76; // r13
  __int64 v77; // r14
  _QWORD **v78; // rax
  int v79; // ecx
  int v80; // edx
  __int64 *v81; // rax
  __int64 v82; // rsi
  __int64 v83; // rax
  unsigned __int64 *v84; // rbx
  __int64 v85; // rsi
  unsigned __int64 v86; // rcx
  const char *v87; // r13
  __int64 v88; // r14
  _QWORD *v89; // rdi
  __int64 v91; // r12
  __int64 v92; // rdx
  int v93; // ecx
  int v94; // edx
  __int64 *v95; // rax
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // r12
  __int64 v99; // r14
  int v100; // ebx
  const char *v101; // r12
  __int64 v102; // r13
  _QWORD **v103; // rax
  int v104; // ecx
  __int64 *v105; // rax
  __int64 v106; // rsi
  const char *v107; // r12
  __int64 v108; // r13
  _QWORD **v109; // rax
  int v110; // ecx
  __int64 *v111; // rax
  __int64 v112; // rsi
  const char *v113; // r12
  __int64 v114; // r13
  _QWORD **v115; // rax
  int v116; // ecx
  int v117; // edx
  __int64 *v118; // rax
  __int64 v119; // rsi
  unsigned __int64 v120; // [rsp+30h] [rbp-130h]
  unsigned __int64 v121; // [rsp+38h] [rbp-128h]
  __int64 v122; // [rsp+40h] [rbp-120h]
  __int64 v123; // [rsp+48h] [rbp-118h]
  __int64 v124; // [rsp+50h] [rbp-110h]
  unsigned __int64 v126; // [rsp+60h] [rbp-100h]
  __int64 v127; // [rsp+68h] [rbp-F8h]
  unsigned __int64 v128; // [rsp+68h] [rbp-F8h]
  __int64 v129; // [rsp+68h] [rbp-F8h]
  __int64 v130; // [rsp+68h] [rbp-F8h]
  __int64 v131; // [rsp+68h] [rbp-F8h]
  __int64 v132; // [rsp+68h] [rbp-F8h]
  _QWORD *v134; // [rsp+70h] [rbp-F0h]
  unsigned __int64 v135; // [rsp+70h] [rbp-F0h]
  __int64 v136; // [rsp+70h] [rbp-F0h]
  unsigned __int64 v138; // [rsp+78h] [rbp-E8h]
  __int64 v139; // [rsp+80h] [rbp-E0h]
  __int64 v140; // [rsp+80h] [rbp-E0h]
  unsigned int v141; // [rsp+80h] [rbp-E0h]
  __int64 *v142; // [rsp+80h] [rbp-E0h]
  _QWORD *v143; // [rsp+88h] [rbp-D8h]
  __int64 v144; // [rsp+88h] [rbp-D8h]
  __int64 v145; // [rsp+90h] [rbp-D0h]
  __int64 v146; // [rsp+98h] [rbp-C8h]
  const char *v147; // [rsp+A0h] [rbp-C0h] BYREF
  unsigned int v148; // [rsp+A8h] [rbp-B8h]
  const char *v149; // [rsp+B0h] [rbp-B0h] BYREF
  unsigned int v150; // [rsp+B8h] [rbp-A8h]
  const char *v151; // [rsp+C0h] [rbp-A0h] BYREF
  unsigned int v152; // [rsp+C8h] [rbp-98h]
  const char *v153; // [rsp+D0h] [rbp-90h] BYREF
  unsigned int v154; // [rsp+D8h] [rbp-88h]
  const char *v155; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v156; // [rsp+E8h] [rbp-78h]
  unsigned __int64 v157; // [rsp+F0h] [rbp-70h] BYREF
  unsigned int v158; // [rsp+F8h] [rbp-68h]
  const char *v159; // [rsp+100h] [rbp-60h] BYREF
  __int64 v160; // [rsp+108h] [rbp-58h]
  const char *v161; // [rsp+110h] [rbp-50h] BYREF
  unsigned int v162; // [rsp+118h] [rbp-48h]
  __int16 v163; // [rsp+120h] [rbp-40h]

  v9 = a7;
  v10 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)a1) >> 3);
  v139 = a5;
  if ( (_DWORD)v10 == 1 )
  {
    if ( a1->m128i_i64[0] == a3 && a1->m128i_i64[1] == a4 )
    {
      v54 = (__int64 *)(a3 + 24);
      LODWORD(v160) = *(_DWORD *)(a4 + 32);
      if ( (unsigned int)v160 > 0x40 )
        sub_C43780((__int64)&v159, (const void **)(a4 + 24));
      else
        v159 = *(const char **)(a4 + 24);
      sub_C46B40((__int64)&v159, v54);
      LODWORD(v156) = v160;
      v55 = a1[1].m128i_i64[0];
      v155 = v159;
      sub_2A31020(v55, a7, a6, (__int64)&v155);
      v143 = (_QWORD *)a1[1].m128i_i64[0];
      if ( (unsigned int)v156 > 0x40 && v155 )
        j_j___libc_free_0_0((unsigned __int64)v155);
      return v143;
    }
    v11 = *(_QWORD *)(a7 + 72);
    v163 = 259;
    v159 = "LeafBlock";
    v12 = sub_BD5C60(a5);
    v143 = (_QWORD *)sub_22077B0(0x50u);
    if ( v143 )
      sub_AA4D50((__int64)v143, v12, (__int64)&v159, 0, 0);
    v13 = *(unsigned __int64 **)(a7 + 32);
    sub_B2B790(v11 + 72, (__int64)v143);
    v14 = *v13;
    v15 = v143[3];
    v143[4] = v13;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    v143[3] = v14 | v15 & 7;
    *(_QWORD *)(v14 + 8) = v143 + 3;
    *v13 = *v13 & 7 | (unsigned __int64)(v143 + 3);
    sub_AA4C30((__int64)v143, *(_BYTE *)(v11 + 128));
    v16 = a1->m128i_i64[0];
    v17 = a1->m128i_i64[1];
    if ( a1->m128i_i64[0] == v17 )
    {
      sub_B43C20((__int64)&v155, (__int64)v143);
      v159 = "SwitchLeaf";
      v163 = 259;
      v134 = sub_BD2C40(72, unk_3F10FD0);
      if ( v134 )
      {
        v101 = v155;
        v102 = v156;
        v130 = a1->m128i_i64[0];
        v103 = *(_QWORD ***)(v139 + 8);
        v104 = *((unsigned __int8 *)v103 + 8);
        if ( (unsigned int)(v104 - 17) > 1 )
        {
          v106 = sub_BCB2A0(*v103);
        }
        else
        {
          BYTE4(v145) = (_BYTE)v104 == 18;
          LODWORD(v145) = *((_DWORD *)v103 + 8);
          v105 = (__int64 *)sub_BCB2A0(*v103);
          v106 = sub_BCE1B0(v105, v145);
        }
        sub_B523C0((__int64)v134, v106, 53, 32, v139, v130, (__int64)&v159, (__int64)v101, v102, 0);
      }
      goto LABEL_14;
    }
    if ( a3 == v16 )
    {
      sub_B43C20((__int64)&v155, (__int64)v143);
      v159 = "SwitchLeaf";
      v163 = 259;
      v134 = sub_BD2C40(72, unk_3F10FD0);
      if ( v134 )
      {
        v107 = v155;
        v108 = v156;
        v131 = a1->m128i_i64[1];
        v109 = *(_QWORD ***)(v139 + 8);
        v110 = *((unsigned __int8 *)v109 + 8);
        if ( (unsigned int)(v110 - 17) > 1 )
        {
          v112 = sub_BCB2A0(*v109);
        }
        else
        {
          BYTE4(v146) = (_BYTE)v110 == 18;
          LODWORD(v146) = *((_DWORD *)v109 + 8);
          v111 = (__int64 *)sub_BCB2A0(*v109);
          v112 = sub_BCE1B0(v111, v146);
        }
        sub_B523C0((__int64)v134, v112, 53, 41, v139, v131, (__int64)&v159, (__int64)v107, v108, 0);
      }
      goto LABEL_14;
    }
    if ( a4 == v17 )
    {
      sub_B43C20((__int64)&v155, (__int64)v143);
      v159 = "SwitchLeaf";
      v163 = 259;
      v134 = sub_BD2C40(72, unk_3F10FD0);
      if ( v134 )
      {
        v113 = v155;
        v114 = v156;
        v132 = a1->m128i_i64[0];
        v115 = *(_QWORD ***)(v139 + 8);
        v116 = *((unsigned __int8 *)v115 + 8);
        if ( (unsigned int)(v116 - 17) > 1 )
        {
          v119 = sub_BCB2A0(*v115);
        }
        else
        {
          v117 = *((_DWORD *)v115 + 8);
          BYTE4(v147) = (_BYTE)v116 == 18;
          LODWORD(v147) = v117;
          v118 = (__int64 *)sub_BCB2A0(*v115);
          v119 = sub_BCE1B0(v118, (__int64)v147);
        }
        sub_B523C0((__int64)v134, v119, 53, 39, v139, v132, (__int64)&v159, (__int64)v113, v114, 0);
      }
      goto LABEL_14;
    }
    v18 = *(_DWORD *)(v16 + 32);
    if ( v18 <= 0x40 )
    {
      if ( !*(_QWORD *)(v16 + 24) )
      {
LABEL_10:
        sub_B43C20((__int64)&v155, (__int64)v143);
        v159 = "SwitchLeaf";
        v163 = 259;
        v134 = sub_BD2C40(72, unk_3F10FD0);
        if ( !v134 )
        {
LABEL_14:
          v140 = a1[1].m128i_i64[0];
          sub_B43C20((__int64)&v159, (__int64)v143);
          v26 = v159;
          v27 = (unsigned __int16)v160;
          v28 = sub_BD2C40(72, 3u);
          if ( v28 )
            sub_B4C9A0((__int64)v28, v140, a8, (__int64)v134, 3u, v29, (__int64)v26, v27);
          v31 = sub_AA5930(a8);
          if ( v31 == v30 )
            goto LABEL_35;
          v32 = v30;
          while ( 1 )
          {
            v33 = *(_QWORD *)(v31 - 8);
            v34 = 0x1FFFFFFFE0LL;
            v35 = *(_DWORD *)(v31 + 72);
            v36 = *(_DWORD *)(v31 + 4) & 0x7FFFFFF;
            if ( !v36 )
              goto LABEL_23;
            v37 = 0;
            do
            {
              if ( a7 == *(_QWORD *)(v33 + 32LL * v35 + 8 * v37) )
              {
                v34 = 32 * v37;
LABEL_23:
                v38 = *(_QWORD *)(v33 + v34);
                if ( v36 == v35 )
                  goto LABEL_130;
                goto LABEL_24;
              }
              ++v37;
            }
            while ( v36 != (_DWORD)v37 );
            v38 = *(_QWORD *)(v33 + 0x1FFFFFFFE0LL);
            if ( v36 == v35 )
            {
LABEL_130:
              sub_B48D90(v31);
              v33 = *(_QWORD *)(v31 - 8);
              v36 = *(_DWORD *)(v31 + 4) & 0x7FFFFFF;
            }
LABEL_24:
            v39 = (v36 + 1) & 0x7FFFFFF;
            *(_DWORD *)(v31 + 4) = v39 | *(_DWORD *)(v31 + 4) & 0xF8000000;
            v40 = v33 + 32LL * (unsigned int)(v39 - 1);
            if ( *(_QWORD *)v40 )
            {
              v41 = *(_QWORD *)(v40 + 8);
              **(_QWORD **)(v40 + 16) = v41;
              if ( v41 )
                *(_QWORD *)(v41 + 16) = *(_QWORD *)(v40 + 16);
            }
            *(_QWORD *)v40 = v38;
            if ( v38 )
            {
              v42 = *(_QWORD *)(v38 + 16);
              *(_QWORD *)(v40 + 8) = v42;
              if ( v42 )
                *(_QWORD *)(v42 + 16) = v40 + 8;
              *(_QWORD *)(v40 + 16) = v38 + 16;
              *(_QWORD *)(v38 + 16) = v40;
            }
            *(_QWORD *)(*(_QWORD *)(v31 - 8)
                      + 32LL * *(unsigned int *)(v31 + 72)
                      + 8LL * ((*(_DWORD *)(v31 + 4) & 0x7FFFFFFu) - 1)) = v143;
            v43 = *(_QWORD *)(v31 + 32);
            if ( !v43 )
LABEL_195:
              BUG();
            v31 = 0;
            if ( *(_BYTE *)(v43 - 24) == 84 )
              v31 = v43 - 24;
            if ( v32 == v31 )
            {
LABEL_35:
              for ( i = *(_QWORD *)(v140 + 56); i; i = *(_QWORD *)(i + 8) )
              {
                if ( *(_BYTE *)(i - 24) != 84 )
                  return v143;
                v45 = a1->m128i_i64[0];
                v46 = a1->m128i_i64[1];
                v47 = (__int64 *)(a1->m128i_i64[0] + 24);
                LODWORD(v160) = *(_DWORD *)(v46 + 32);
                if ( (unsigned int)v160 > 0x40 )
                {
                  v142 = (__int64 *)(v45 + 24);
                  sub_C43780((__int64)&v159, (const void **)(v46 + 24));
                  v47 = v142;
                }
                else
                {
                  v159 = *(const char **)(v46 + 24);
                }
                sub_C46B40((__int64)&v159, v47);
                v141 = v160;
                LODWORD(v156) = v160;
                v135 = (unsigned __int64)v159;
                v155 = v159;
                if ( (unsigned int)v160 > 0x40 )
                  sub_C43690((__int64)&v159, 0, 0);
                else
                  v159 = 0;
                while ( (int)sub_C49970((__int64)&v159, (unsigned __int64 *)&v155) < 0 )
                {
                  if ( (*(_DWORD *)(i - 20) & 0x7FFFFFF) != 0 )
                  {
                    v48 = 0;
                    while ( 1 )
                    {
                      v49 = v48;
                      if ( a7 == *(_QWORD *)(*(_QWORD *)(i - 32) + 32LL * *(unsigned int *)(i + 48) + 8 * v48) )
                        break;
                      if ( (*(_DWORD *)(i - 20) & 0x7FFFFFF) == (_DWORD)++v48 )
                        goto LABEL_127;
                    }
                  }
                  else
                  {
LABEL_127:
                    v49 = -1;
                  }
                  sub_B48BF0(i - 24, v49, 1);
                  sub_C46250((__int64)&v159);
                }
                if ( (unsigned int)v160 > 0x40 && v159 )
                  j_j___libc_free_0_0((unsigned __int64)v159);
                v50 = 32LL * *(unsigned int *)(i + 48);
                if ( (*(_DWORD *)(i - 20) & 0x7FFFFFF) != 0 )
                {
                  v51 = *(_QWORD *)(i - 32);
                  v52 = (_QWORD *)(v51 + v50);
                  while ( a7 != *v52 )
                  {
                    if ( ++v52 == (_QWORD *)(v51 + v50 + 8 + 8LL * ((*(_DWORD *)(i - 20) & 0x7FFFFFFu) - 1)) )
                    {
                      v52 = (_QWORD *)(v51 + v50 + 0x7FFFFFFF8LL);
                      break;
                    }
                  }
                }
                else
                {
                  v52 = (_QWORD *)(*(_QWORD *)(i - 32) + v50 + 0x7FFFFFFF8LL);
                }
                *v52 = v143;
                if ( v141 > 0x40 )
                {
                  if ( v135 )
                    j_j___libc_free_0_0(v135);
                }
              }
              goto LABEL_195;
            }
          }
        }
        v19 = v155;
        v20 = v156;
        v127 = a1->m128i_i64[1];
        v21 = *(_QWORD ***)(v139 + 8);
        v22 = *((unsigned __int8 *)v21 + 8);
        if ( (unsigned int)(v22 - 17) <= 1 )
        {
          v23 = *((_DWORD *)v21 + 8);
          BYTE4(v149) = (_BYTE)v22 == 18;
          LODWORD(v149) = v23;
          v24 = (__int64 *)sub_BCB2A0(*v21);
          v25 = sub_BCE1B0(v24, (__int64)v149);
LABEL_13:
          sub_B523C0((__int64)v134, v25, 53, 37, v139, v127, (__int64)&v159, (__int64)v19, v20, 0);
          goto LABEL_14;
        }
        goto LABEL_135;
      }
    }
    else if ( v18 == (unsigned int)sub_C444A0(v16 + 24) )
    {
      goto LABEL_10;
    }
    v91 = sub_AD6890(v16, 0);
    sub_B43C20((__int64)&v155, (__int64)v143);
    v159 = sub_BD5D20(v139);
    v163 = 773;
    v160 = v92;
    v161 = ".off";
    v139 = sub_B504D0(13, v139, v91, (__int64)&v159, (__int64)v155, v156);
    v127 = sub_AD57C0(v91, (unsigned __int8 *)a1->m128i_i64[1], 0, 0);
    sub_B43C20((__int64)&v155, (__int64)v143);
    v159 = "SwitchLeaf";
    v163 = 259;
    v134 = sub_BD2C40(72, unk_3F10FD0);
    if ( !v134 )
      goto LABEL_14;
    v19 = v155;
    v20 = v156;
    v21 = *(_QWORD ***)(v139 + 8);
    v93 = *((unsigned __int8 *)v21 + 8);
    if ( (unsigned int)(v93 - 17) <= 1 )
    {
      v94 = *((_DWORD *)v21 + 8);
      BYTE4(v151) = (_BYTE)v93 == 18;
      LODWORD(v151) = v94;
      v95 = (__int64 *)sub_BCB2A0(*v21);
      v25 = sub_BCE1B0(v95, (__int64)v151);
      goto LABEL_13;
    }
LABEL_135:
    v25 = sub_BCB2A0(*v21);
    goto LABEL_13;
  }
  v56 = 24LL * ((unsigned int)v10 >> 1);
  v57 = (const __m128i *)((char *)a1 + v56);
  if ( v56 )
  {
    v58 = sub_22077B0(v56);
    v126 = v58;
    if ( a1 == v57 )
    {
      v128 = v58;
    }
    else
    {
      v59 = (__m128i *)v58;
      v60 = a1;
      do
      {
        if ( v59 )
        {
          v61 = _mm_loadu_si128(v60);
          v59[1].m128i_i64[0] = v60[1].m128i_i64[0];
          *v59 = v61;
        }
        v60 = (const __m128i *)((char *)v60 + 24);
        v59 = (__m128i *)((char *)v59 + 24);
      }
      while ( v57 != v60 );
      v128 = v126 + 8 * ((unsigned __int64)((char *)v57 - (char *)a1 - 24) >> 3) + 24;
    }
  }
  else
  {
    v126 = 0;
    v128 = 0;
  }
  v120 = (char *)a2 - (char *)v57;
  if ( (unsigned __int64)((char *)a2 - (char *)v57) > 0x7FFFFFFFFFFFFFF8LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v138 = 0;
  if ( v120 )
    v138 = sub_22077B0(v120);
  if ( a2 == v57 )
  {
    LODWORD(v121) = v138;
  }
  else
  {
    v62 = (__m128i *)v138;
    v63 = v57;
    do
    {
      if ( v62 )
      {
        v64 = _mm_loadu_si128(v63);
        v62[1].m128i_i64[0] = v63[1].m128i_i64[0];
        *v62 = v64;
      }
      v63 = (const __m128i *)((char *)v63 + 24);
      v62 = (__m128i *)((char *)v62 + 24);
    }
    while ( a2 != v63 );
    v121 = v138 + 8 * ((unsigned __int64)((char *)&a2[-2].m128i_u64[1] - (char *)v57) >> 3) + 24;
  }
  v65 = v57->m128i_i64[0];
  v123 = v57->m128i_i64[0];
  v66 = (const void **)(v57->m128i_i64[0] + 24);
  LODWORD(v156) = *(_DWORD *)(v57->m128i_i64[0] + 32);
  if ( (unsigned int)v156 > 0x40 )
    sub_C43780((__int64)&v155, (const void **)(v65 + 24));
  else
    v155 = *(const char **)(v65 + 24);
  sub_C46F20((__int64)&v155, 1u);
  v67 = v156;
  LODWORD(v156) = 0;
  LODWORD(v160) = v67;
  v159 = v155;
  v68 = (__int64 *)sub_BD5C60(v123);
  LODWORD(v122) = sub_ACCFD0(v68, (__int64)&v159);
  if ( (unsigned int)v160 > 0x40 && v159 )
    j_j___libc_free_0_0((unsigned __int64)v159);
  if ( (unsigned int)v156 > 0x40 && v155 )
    j_j___libc_free_0_0((unsigned __int64)v155);
  if ( *a9 != a9[1] )
  {
    v69 = *(_QWORD *)(v128 - 16);
    LODWORD(v160) = *(_DWORD *)(v69 + 32);
    if ( (unsigned int)v160 > 0x40 )
      sub_C43780((__int64)&v159, (const void **)(v69 + 24));
    else
      v159 = *(const char **)(v69 + 24);
    sub_C46A40((__int64)&v159, 1);
    v148 = v160;
    v147 = v159;
    LODWORD(v160) = *(_DWORD *)(v123 + 32);
    if ( (unsigned int)v160 > 0x40 )
      sub_C43780((__int64)&v159, v66);
    else
      v159 = *(const char **)(v123 + 24);
    sub_C46F20((__int64)&v159, 1u);
    v70 = v160;
    v149 = v159;
    v150 = v160;
    v152 = v148;
    if ( v148 > 0x40 )
    {
      sub_C43780((__int64)&v151, (const void **)&v147);
      v70 = v150;
    }
    else
    {
      v151 = v147;
    }
    v154 = v70;
    if ( v70 > 0x40 )
    {
      sub_C43780((__int64)&v153, (const void **)&v149);
      if ( (int)sub_C4C880((__int64)&v149, (__int64)&v147) < 0 )
      {
LABEL_170:
        if ( v154 > 0x40 && v153 )
          j_j___libc_free_0_0((unsigned __int64)v153);
        goto LABEL_98;
      }
    }
    else
    {
      v153 = v149;
      if ( (int)sub_C4C880((__int64)&v149, (__int64)&v147) < 0 )
      {
LABEL_98:
        if ( v152 > 0x40 && v151 )
          j_j___libc_free_0_0((unsigned __int64)v151);
        if ( v150 > 0x40 && v149 )
          j_j___libc_free_0_0((unsigned __int64)v149);
        if ( v148 > 0x40 && v147 )
          j_j___libc_free_0_0((unsigned __int64)v147);
        goto LABEL_107;
      }
    }
    v96 = a9[1];
    v144 = *a9;
    v97 = v96 - *a9;
    v98 = v97 >> 5;
    if ( v97 > 0 )
    {
      do
      {
        v99 = v144 + 32 * (v98 >> 1);
        LODWORD(v156) = v152;
        if ( v152 > 0x40 )
          sub_C43780((__int64)&v155, (const void **)&v151);
        else
          v155 = v151;
        v158 = v154;
        if ( v154 > 0x40 )
          sub_C43780((__int64)&v157, (const void **)&v153);
        else
          v157 = (unsigned __int64)v153;
        LODWORD(v160) = *(_DWORD *)(v99 + 8);
        if ( (unsigned int)v160 > 0x40 )
          sub_C43780((__int64)&v159, (const void **)v99);
        else
          v159 = *(const char **)v99;
        v162 = *(_DWORD *)(v99 + 24);
        if ( v162 > 0x40 )
        {
          sub_C43780((__int64)&v161, (const void **)(v99 + 16));
          v100 = sub_C4C880((__int64)&v161, (__int64)&v157);
          if ( v162 > 0x40 && v161 )
            j_j___libc_free_0_0((unsigned __int64)v161);
        }
        else
        {
          v161 = *(const char **)(v99 + 16);
          v100 = sub_C4C880((__int64)&v161, (__int64)&v157);
        }
        if ( (unsigned int)v160 > 0x40 && v159 )
          j_j___libc_free_0_0((unsigned __int64)v159);
        if ( v158 > 0x40 && v157 )
          j_j___libc_free_0_0(v157);
        if ( (unsigned int)v156 > 0x40 && v155 )
          j_j___libc_free_0_0((unsigned __int64)v155);
        if ( v100 < 0 )
        {
          v144 = v99 + 32;
          v98 = v98 - (v98 >> 1) - 1;
        }
        else
        {
          v98 >>= 1;
        }
      }
      while ( v98 > 0 );
      v9 = a7;
      v96 = a9[1];
    }
    if ( v144 != v96 && (int)sub_C4C880(v144, (__int64)&v151) <= 0 )
      v122 = *(_QWORD *)(v128 - 16);
    goto LABEL_170;
  }
LABEL_107:
  v71 = *(_QWORD *)(v9 + 72);
  v163 = 259;
  v124 = v71;
  v159 = "NodeBlock";
  v72 = sub_BD5C60(v139);
  v73 = sub_22077B0(0x50u);
  v143 = (_QWORD *)v73;
  if ( v73 )
    sub_AA4D50(v73, v72, (__int64)&v159, 0, 0);
  v159 = "Pivot";
  v163 = 259;
  v74 = sub_BD2C40(72, unk_3F10FD0);
  if ( v74 )
  {
    v75 = v57->m128i_i64[0];
    v76 = v74;
    v77 = v75;
    v78 = *(_QWORD ***)(v139 + 8);
    v79 = *((unsigned __int8 *)v78 + 8);
    if ( (unsigned int)(v79 - 17) > 1 )
    {
      v82 = sub_BCB2A0(*v78);
    }
    else
    {
      v80 = *((_DWORD *)v78 + 8);
      BYTE4(v155) = (_BYTE)v79 == 18;
      LODWORD(v155) = v80;
      v81 = (__int64 *)sub_BCB2A0(*v78);
      v82 = sub_BCE1B0(v81, (__int64)v155);
    }
    sub_B523C0((__int64)v74, v82, 53, 40, v139, v77, (__int64)&v159, 0, 0, 0);
  }
  else
  {
    v76 = 0;
  }
  v129 = sub_2A319A0(v126, v128, a3, v122, v139, (_DWORD)v143, v9, a8, (__int64)a9);
  v83 = sub_2A319A0(v138, v121, v123, a4, v139, (_DWORD)v143, v9, a8, (__int64)a9);
  v84 = *(unsigned __int64 **)(v9 + 32);
  v136 = v83;
  sub_B2B790(v124 + 72, (__int64)v143);
  v85 = v143[3];
  v86 = *v84;
  v143[4] = v84;
  v86 &= 0xFFFFFFFFFFFFFFF8LL;
  v143[3] = v86 | v85 & 7;
  *(_QWORD *)(v86 + 8) = v143 + 3;
  *v84 = *v84 & 7 | (unsigned __int64)(v143 + 3);
  sub_AA4C30((__int64)v143, *(_BYTE *)(v124 + 128));
  sub_B44240(v76, (__int64)v143, v143 + 6, 0);
  sub_B43C20((__int64)&v159, (__int64)v143);
  v87 = v159;
  v88 = (unsigned __int16)v160;
  v89 = sub_BD2C40(72, 3u);
  if ( v89 )
    sub_B4C9A0((__int64)v89, v129, v136, (__int64)v74, 3u, v136, (__int64)v87, v88);
  if ( v138 )
    j_j___libc_free_0(v138);
  if ( v126 )
    j_j___libc_free_0(v126);
  return v143;
}
