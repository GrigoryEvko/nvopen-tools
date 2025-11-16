// Function: sub_C59DA0
// Address: 0xc59da0
//
unsigned __int64 *__fastcall sub_C59DA0(unsigned __int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rsi
  int v5; // r12d
  __int64 v6; // r12
  __int64 v7; // rbx
  __m128i *v8; // rdx
  __int64 v9; // r12
  __int64 *v10; // rax
  _BYTE *v11; // r14
  void (__fastcall *v12)(__m128i *, __int64, __m128i *); // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  unsigned __int32 v21; // ebx
  __int64 v22; // r12
  __int64 v23; // rcx
  __int64 v24; // r9
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __m128i *v31; // rbx
  __m128i *v32; // r12
  size_t v34; // r12
  const void *v35; // r14
  _QWORD *v36; // rax
  __m128i v37; // xmm1
  __int64 v38; // r14
  __int64 v39; // rax
  __int64 *v40; // r15
  __int64 v41; // rsi
  void (__fastcall *v42)(__m128i *, __int64, __m128i *); // rax
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  unsigned __int32 v46; // ebx
  __int64 v47; // r15
  unsigned __int32 v48; // ebx
  __int64 v49; // r15
  __int64 v50; // rcx
  char v51; // bl
  __int64 v52; // r8
  __int64 v53; // r9
  _QWORD *v54; // rdi
  size_t v55; // rax
  unsigned __int64 v56; // r14
  __m128i *v57; // rax
  __m128i *v58; // rcx
  __int64 v59; // rdx
  _QWORD *v60; // r12
  size_t v61; // rax
  size_t v62; // r8
  __int64 v63; // rcx
  unsigned __int64 v64; // rsi
  int v65; // edx
  __m128i *v66; // rsi
  __m128i *v67; // rax
  __m128i *v68; // rcx
  __int64 v69; // rsi
  __int64 v70; // rdi
  __int64 v71; // rdx
  int v72; // eax
  __int64 v73; // r12
  char *v74; // rbx
  __int64 v75; // rsi
  __int64 v76; // rax
  signed __int64 v77; // r12
  unsigned __int64 v78; // rdx
  unsigned __int64 v79; // rcx
  __int64 v80; // r11
  size_t v81; // rdx
  char *v82; // r13
  char *v83; // r8
  __int64 v84; // rcx
  __int64 v85; // rdi
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // r12
  _QWORD *v90; // rdi
  __int64 v91; // rax
  void *v92; // rdi
  __int64 v93; // r14
  size_t v94; // rax
  char *v95; // r11
  unsigned __int64 v96; // rdx
  __m128i *v97; // rbx
  char *v98; // [rsp+8h] [rbp-428h]
  char *v99; // [rsp+8h] [rbp-428h]
  __int64 v100; // [rsp+10h] [rbp-420h]
  char *v101; // [rsp+10h] [rbp-420h]
  char *v102; // [rsp+10h] [rbp-420h]
  __int64 *v103; // [rsp+18h] [rbp-418h]
  __int64 v104; // [rsp+20h] [rbp-410h]
  __int64 v105; // [rsp+20h] [rbp-410h]
  char *v106; // [rsp+20h] [rbp-410h]
  size_t v107; // [rsp+20h] [rbp-410h]
  __int64 v109; // [rsp+58h] [rbp-3D8h]
  size_t v110; // [rsp+58h] [rbp-3D8h]
  char *s; // [rsp+70h] [rbp-3C0h]
  char *sa; // [rsp+70h] [rbp-3C0h]
  unsigned int v114; // [rsp+78h] [rbp-3B8h]
  _BYTE v115[32]; // [rsp+80h] [rbp-3B0h] BYREF
  _BYTE v116[32]; // [rsp+A0h] [rbp-390h] BYREF
  __m128i v117; // [rsp+C0h] [rbp-370h] BYREF
  __int64 v118; // [rsp+D0h] [rbp-360h] BYREF
  char v119; // [rsp+E0h] [rbp-350h]
  char v120; // [rsp+E1h] [rbp-34Fh]
  __m128i v121[2]; // [rsp+F0h] [rbp-340h] BYREF
  __int16 v122; // [rsp+110h] [rbp-320h]
  __m128i v123; // [rsp+120h] [rbp-310h] BYREF
  __int64 v124; // [rsp+130h] [rbp-300h] BYREF
  __int16 v125; // [rsp+140h] [rbp-2F0h]
  __m128i v126; // [rsp+150h] [rbp-2E0h] BYREF
  __int64 v127; // [rsp+160h] [rbp-2D0h] BYREF
  char v128; // [rsp+170h] [rbp-2C0h]
  char v129; // [rsp+171h] [rbp-2BFh]
  __m128i v130; // [rsp+180h] [rbp-2B0h] BYREF
  _BYTE v131[16]; // [rsp+190h] [rbp-2A0h] BYREF
  __int16 v132; // [rsp+1A0h] [rbp-290h]
  __m128i src[2]; // [rsp+1B0h] [rbp-280h] BYREF
  __m128i v134; // [rsp+1D0h] [rbp-260h] BYREF
  __int64 v135; // [rsp+1E0h] [rbp-250h]
  __int64 v136; // [rsp+1E8h] [rbp-248h]
  __int64 v137; // [rsp+1F0h] [rbp-240h]
  __int64 v138; // [rsp+1F8h] [rbp-238h]
  char v139; // [rsp+200h] [rbp-230h]
  char v140; // [rsp+208h] [rbp-228h]
  __m128i v141; // [rsp+210h] [rbp-220h] BYREF
  _QWORD v142[2]; // [rsp+220h] [rbp-210h] BYREF
  __m128i v143; // [rsp+230h] [rbp-200h]
  __int64 v144; // [rsp+240h] [rbp-1F0h]
  __int64 v145; // [rsp+248h] [rbp-1E8h]
  __int64 v146; // [rsp+250h] [rbp-1E0h]
  __int64 v147; // [rsp+258h] [rbp-1D8h]
  char v148; // [rsp+260h] [rbp-1D0h]
  __int64 v149; // [rsp+268h] [rbp-1C8h]
  __m128i v150; // [rsp+270h] [rbp-1C0h] BYREF
  _QWORD dest[2]; // [rsp+280h] [rbp-1B0h] BYREF
  __int64 v152; // [rsp+290h] [rbp-1A0h]
  char v153; // [rsp+2C8h] [rbp-168h]
  __m128i *v154; // [rsp+2D0h] [rbp-160h] BYREF
  __int64 v155; // [rsp+2D8h] [rbp-158h]
  _QWORD v156[2]; // [rsp+2E0h] [rbp-150h] BYREF
  __m128i v157; // [rsp+2F0h] [rbp-140h] BYREF
  __int64 v158; // [rsp+300h] [rbp-130h]
  __m128i *v159; // [rsp+360h] [rbp-D0h] BYREF
  __int64 v160; // [rsp+368h] [rbp-C8h]
  __m128i v161; // [rsp+370h] [rbp-C0h] BYREF
  __int64 v162; // [rsp+380h] [rbp-B0h]

  v154 = (__m128i *)v156;
  v155 = 0x300000000LL;
  v159 = &v161;
  sub_C4FB50((__int64 *)&v159, byte_3F871B3, (__int64)byte_3F871B3);
  v162 = *((unsigned int *)a3 + 2);
  v4 = 1;
  v5 = 0;
  if ( v156 )
  {
    v156[0] = &v157;
    if ( v159 == &v161 )
    {
      v157 = _mm_loadu_si128(&v161);
    }
    else
    {
      v156[0] = v159;
      v157.m128i_i64[0] = v161.m128i_i64[0];
    }
    v4 = v160;
    v159 = &v161;
    v160 = 0;
    v156[1] = v4;
    v5 = v155;
    v161.m128i_i8[0] = 0;
    v158 = v162;
  }
  v6 = (unsigned int)(v5 + 1);
  LODWORD(v155) = v6;
  if ( v159 != &v161 )
  {
    v4 = v161.m128i_i64[0] + 1;
    j_j___libc_free_0(v159, v161.m128i_i64[0] + 1);
    v6 = (unsigned int)v155;
  }
  if ( !*((_DWORD *)a3 + 2) )
  {
LABEL_29:
    v159 = 0;
    *a1 = 1;
    sub_9C66B0((__int64 *)&v159);
    goto LABEL_51;
  }
  v7 = 0;
  v114 = 0;
  while ( 1 )
  {
    v8 = v154;
    if ( v154->m128i_i64[5 * (unsigned int)v6 - 1] == v7 )
    {
      do
      {
        v9 = (unsigned int)(v6 - 1);
        LODWORD(v155) = v9;
        v10 = &v8->m128i_i64[5 * v9];
        if ( (__int64 *)*v10 != v10 + 2 )
        {
          v4 = v10[2] + 1;
          j_j___libc_free_0(*v10, v4);
          v8 = v154;
        }
        LODWORD(v6) = v155;
      }
      while ( v8->m128i_i64[5 * (unsigned int)v155 - 1] == v7 );
    }
    v109 = 8 * v7;
    v11 = *(_BYTE **)(*a3 + 8 * v7);
    if ( v11 )
    {
      if ( *v11 == 64 )
        break;
    }
    ++v114;
LABEL_28:
    v7 = v114;
    v6 = (unsigned int)v155;
    if ( *((_DWORD *)a3 + 2) == v114 )
      goto LABEL_29;
  }
  v160 = 0;
  v161.m128i_i64[0] = 128;
  v159 = (__m128i *)&v161.m128i_u64[1];
  LOWORD(v152) = 257;
  s = v11 + 1;
  if ( v11[1] )
  {
    v150.m128i_i64[0] = (__int64)(v11 + 1);
    LOBYTE(v152) = 3;
  }
  if ( !(unsigned __int8)sub_C81F30(&v150, 0) )
  {
LABEL_18:
    v4 = *(_QWORD *)(a2 + 16);
    v12 = *(void (__fastcall **)(__m128i *, __int64, __m128i *))(*(_QWORD *)v4 + 40LL);
    LOWORD(v152) = 257;
    if ( *s )
    {
      v150.m128i_i64[0] = (__int64)s;
      LOBYTE(v152) = 3;
    }
    v12(src, v4, &v150);
    if ( (v140 & 1) != 0 )
      goto LABEL_39;
    if ( !(unsigned __int8)sub_CA3E70(src) )
    {
      if ( (v140 & 1) == 0 )
      {
        if ( *(_BYTE *)(a2 + 58) )
          goto LABEL_60;
        ++v114;
        sub_2240A30(src);
        goto LABEL_25;
      }
LABEL_39:
      v21 = src[0].m128i_i32[0];
      v22 = src[0].m128i_i64[1];
      if ( *(_BYTE *)(a2 + 58) )
      {
        if ( !src[0].m128i_i32[0] )
        {
LABEL_60:
          v21 = 2;
          v22 = sub_2241E50(src, v4, v13, v14, v15);
        }
LABEL_43:
        (*(void (__fastcall **)(_BYTE *, __int64, _QWORD))(*(_QWORD *)v22 + 32LL))(v115, v22, v21);
        v126.m128i_i64[0] = (__int64)"': ";
        v143.m128i_i16[0] = 260;
        v141.m128i_i64[0] = (__int64)v115;
        v129 = 1;
        v128 = 3;
        v122 = 257;
        if ( *s )
        {
          v121[0].m128i_i64[0] = (__int64)s;
          LOBYTE(v122) = 3;
        }
        v117.m128i_i64[0] = (__int64)"cannot not open file '";
        v120 = 1;
        v119 = 3;
        sub_9C6370(&v123, &v117, v121, v23, (__int64)v115, v24);
        sub_9C6370(&v130, &v123, &v126, v25, v26, v27);
        sub_9C6370(&v150, &v130, &v141, v28, v29, v30);
        sub_CA0F50(v116, &v150);
        v4 = (__int64)v116;
        sub_C63F00(a1, v116, v21, v22);
        sub_2240A30(v116);
        sub_2240A30(v115);
        goto LABEL_46;
      }
      if ( src[0].m128i_i32[0] && (v22 != sub_2241E50(src, v4, v13, v14, v15) || v21 != 2) )
        goto LABEL_43;
      ++v114;
      goto LABEL_25;
    }
    v34 = src[0].m128i_u64[1];
    v35 = (const void *)src[0].m128i_i64[0];
    v36 = v142;
    v141.m128i_i64[0] = (__int64)v142;
    if ( src[0].m128i_i64[1] + src[0].m128i_i64[0] && !src[0].m128i_i64[0] )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v150.m128i_i64[0] = src[0].m128i_i64[1];
    if ( src[0].m128i_i64[1] > 0xFuLL )
    {
      v141.m128i_i64[0] = sub_22409D0(&v141, &v150, 0);
      v54 = (_QWORD *)v141.m128i_i64[0];
      v142[0] = v150.m128i_i64[0];
    }
    else
    {
      if ( src[0].m128i_i64[1] == 1 )
      {
        LOBYTE(v142[0]) = *(_BYTE *)src[0].m128i_i64[0];
        goto LABEL_70;
      }
      if ( !src[0].m128i_i64[1] )
      {
LABEL_70:
        v141.m128i_i64[1] = v34;
        *((_BYTE *)v36 + v34) = 0;
        v37 = _mm_load_si128(&v134);
        v144 = v135;
        v143 = v37;
        v145 = v136;
        v146 = v137;
        v147 = v138;
        v148 = v139;
        v149 = a2;
        v38 = (__int64)&v154[2].m128i_i64[1];
        if ( (unsigned __int64 *)((char *)v154 + 40 * (unsigned int)v155) != &v154[2].m128i_u64[1] )
        {
          v104 = v7;
          v39 = a2;
          v103 = a3;
          v40 = &v154->m128i_i64[5 * (unsigned int)v155];
          while ( 1 )
          {
            v41 = *(_QWORD *)(v39 + 16);
            v42 = *(void (__fastcall **)(__m128i *, __int64, __m128i *))(*(_QWORD *)v41 + 40LL);
            v132 = 260;
            v130.m128i_i64[0] = v38;
            v42(&v150, v41, &v130);
            if ( (v153 & 1) != 0 )
            {
              v132 = 260;
              v46 = v150.m128i_i32[0];
              v126.m128i_i64[0] = (__int64)"cannot open file: ";
              v47 = v150.m128i_i64[1];
              v130.m128i_i64[0] = v38;
              v129 = 1;
              v128 = 3;
              sub_9C6370(&v150, &v126, &v130, v43, v44, v45);
              sub_CA0F50(&v123, &v150);
              v4 = (__int64)&v123;
              sub_C63F00(a1, &v123, v46, v47);
              if ( (__int64 *)v123.m128i_i64[0] != &v124 )
              {
                v4 = v124 + 1;
                j_j___libc_free_0(v123.m128i_i64[0], v124 + 1);
              }
              goto LABEL_75;
            }
            v51 = sub_CA3AB0(&v141, &v150);
            if ( (v153 & 1) == 0 )
              sub_2240A30(&v150);
            if ( v51 )
              break;
            v38 += 40;
            if ( v40 == (__int64 *)v38 )
            {
              v7 = v104;
              a3 = v103;
              goto LABEL_92;
            }
            v39 = v149;
          }
          v123.m128i_i64[0] = v38;
          v130.m128i_i64[0] = (__int64)"'";
          v125 = 260;
          v121[0].m128i_i64[0] = (__int64)"recursive expansion of: '";
          v132 = 259;
          v122 = 259;
          sub_9C6370(&v126, v121, &v123, v50, v52, v53);
          sub_9C6370(&v150, &v126, &v130, v86, v87, v88);
          v89 = sub_2241E40();
          sub_CA0F50(&v117, &v150);
          v4 = (__int64)&v117;
          sub_C63F00(a1, &v117, 0, v89);
          if ( (__int64 *)v117.m128i_i64[0] != &v118 )
          {
            v4 = v118 + 1;
            j_j___libc_free_0(v117.m128i_i64[0], v118 + 1);
          }
LABEL_75:
          sub_2240A30(&v141);
LABEL_46:
          if ( (v140 & 1) == 0 )
            sub_2240A30(src);
          goto LABEL_48;
        }
LABEL_92:
        v130.m128i_i64[1] = 0;
        v130.m128i_i64[0] = (__int64)v131;
        v55 = strlen(s);
        v4 = a2;
        sub_C59290(&v150, a2, (__int64)s, v55, (__int64)&v130);
        v56 = v150.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
        if ( (v150.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          v150.m128i_i64[0] = 0;
          *a1 = v56 | 1;
          sub_9C66B0(v150.m128i_i64);
          if ( (_BYTE *)v130.m128i_i64[0] != v131 )
            _libc_free(v130.m128i_i64[0], a2);
          goto LABEL_75;
        }
        v150.m128i_i64[0] = 0;
        sub_9C66B0(v150.m128i_i64);
        v57 = v154;
        v58 = (__m128i *)((char *)v154 + 40 * (unsigned int)v155);
        if ( v58 != v154 )
        {
          v59 = v130.m128i_u32[2] - 1LL;
          do
          {
            v57[2].m128i_i64[0] += v59;
            v57 = (__m128i *)((char *)v57 + 40);
          }
          while ( v57 != v58 );
        }
        v60 = dest;
        v150.m128i_i64[0] = (__int64)dest;
        v61 = strlen(s);
        v126.m128i_i64[0] = v61;
        v62 = v61;
        if ( v61 > 0xF )
        {
          v107 = v61;
          v91 = sub_22409D0(&v150, &v126, 0);
          v62 = v107;
          v150.m128i_i64[0] = v91;
          v90 = (_QWORD *)v91;
          dest[0] = v126.m128i_i64[0];
        }
        else
        {
          if ( v61 == 1 )
          {
            LOBYTE(dest[0]) = *s;
LABEL_101:
            v150.m128i_i64[1] = v61;
            *((_BYTE *)v60 + v61) = 0;
            v63 = (unsigned int)v155;
            v64 = (unsigned int)v155 + 1LL;
            v152 = v130.m128i_u32[2] + v7;
            v65 = v155;
            if ( v64 > HIDWORD(v155) )
            {
              v97 = v154;
              if ( v154 > &v150 || &v150 >= (__m128i *)((char *)v154 + 40 * (unsigned int)v155) )
              {
                sub_C50060((__int64)&v154, v64);
                v63 = (unsigned int)v155;
                v66 = v154;
                v67 = &v150;
                v65 = v155;
              }
              else
              {
                sub_C50060((__int64)&v154, v64);
                v66 = v154;
                v63 = (unsigned int)v155;
                v67 = (__m128i *)((char *)v154 + (char *)&v150 - (char *)v97);
                v65 = v155;
              }
            }
            else
            {
              v66 = v154;
              v67 = &v150;
            }
            v68 = (__m128i *)((char *)v66 + 40 * v63);
            if ( v68 )
            {
              v68->m128i_i64[0] = (__int64)v68[1].m128i_i64;
              if ( (__m128i *)v67->m128i_i64[0] == &v67[1] )
              {
                v68[1] = _mm_loadu_si128(v67 + 1);
              }
              else
              {
                v68->m128i_i64[0] = v67->m128i_i64[0];
                v68[1].m128i_i64[0] = v67[1].m128i_i64[0];
              }
              v69 = v67->m128i_i64[1];
              v67->m128i_i64[0] = (__int64)v67[1].m128i_i64;
              v67->m128i_i64[1] = 0;
              v68->m128i_i64[1] = v69;
              v65 = v155;
              v67[1].m128i_i8[0] = 0;
              v68[2].m128i_i64[0] = v67[2].m128i_i64[0];
            }
            LODWORD(v155) = v65 + 1;
            sub_2240A30(&v150);
            v70 = *a3 + v109;
            v71 = *a3 + 8LL * *((unsigned int *)a3 + 2);
            v72 = *((_DWORD *)a3 + 2);
            if ( v71 != v70 + 8 )
            {
              memmove((void *)v70, (const void *)(v70 + 8), v71 - (v70 + 8));
              v72 = *((_DWORD *)a3 + 2);
            }
            v73 = v130.m128i_u32[2];
            v74 = (char *)v130.m128i_i64[0];
            v75 = (unsigned int)(v72 - 1);
            *((_DWORD *)a3 + 2) = v75;
            v76 = 8 * v75;
            v77 = 8 * v73;
            sa = &v74[v77];
            v105 = v77 >> 3;
            v78 = v75 + (v77 >> 3);
            v79 = *((unsigned int *)a3 + 3);
            if ( v109 == 8 * v75 )
            {
              if ( v78 > v79 )
              {
                sub_C8D5F0(a3, a3 + 2, v78, 8);
                v75 = *((unsigned int *)a3 + 2);
                v76 = 8 * v75;
              }
              if ( v74 != sa )
              {
                memcpy((void *)(*a3 + v76), v74, v77);
                v75 = *((unsigned int *)a3 + 2);
              }
              v4 = v105 + v75;
              *((_DWORD *)a3 + 2) = v4;
            }
            else
            {
              if ( v78 > v79 )
              {
                sub_C8D5F0(a3, a3 + 2, v78, 8);
                v75 = *((unsigned int *)a3 + 2);
                v76 = 8 * v75;
              }
              v80 = *a3;
              v81 = v76 - v109;
              v82 = (char *)(*a3 + v109);
              v83 = (char *)(*a3 + v76);
              v84 = (v76 - v109) >> 3;
              if ( v77 <= (unsigned __int64)(v76 - v109) )
              {
                v92 = (void *)(*a3 + v76);
                v93 = v76 - v77;
                v94 = v77;
                v95 = (char *)(v93 + v80);
                v96 = (v77 >> 3) + v75;
                if ( v96 > *((unsigned int *)a3 + 3) )
                {
                  v99 = v95;
                  v102 = v83;
                  sub_C8D5F0(a3, a3 + 2, v96, 8);
                  v75 = *((unsigned int *)a3 + 2);
                  v94 = v77;
                  v95 = v99;
                  v83 = v102;
                  v92 = (void *)(*a3 + 8 * v75);
                }
                if ( v83 != v95 )
                {
                  v98 = v83;
                  v101 = v95;
                  memmove(v92, v95, v94);
                  v75 = *((unsigned int *)a3 + 2);
                  v83 = v98;
                  v95 = v101;
                }
                v4 = (v77 >> 3) + v75;
                *((_DWORD *)a3 + 2) = v4;
                if ( v82 != v95 )
                {
                  v4 = (__int64)v82;
                  memmove(&v83[-(v93 - v109)], v82, v93 - v109);
                }
                if ( v74 != sa )
                {
                  v4 = (__int64)v74;
                  memmove(v82, v74, v77);
                }
              }
              else
              {
                v4 = v105 + v75;
                *((_DWORD *)a3 + 2) = v4;
                if ( v82 != v83 )
                {
                  v100 = (v76 - v109) >> 3;
                  v85 = v80 + 8LL * (unsigned int)v4;
                  v4 = (__int64)v82;
                  v106 = v83;
                  v110 = v76 - v109;
                  memcpy((void *)(v85 - v81), v82, v81);
                  v84 = v100;
                  v83 = v106;
                  v81 = v110;
                }
                if ( v84 )
                {
                  do
                  {
                    *(_QWORD *)&v82[8 * v56] = *(_QWORD *)&v74[8 * v56];
                    ++v56;
                  }
                  while ( v84 != v56 );
                  v74 += v81;
                }
                if ( sa != v74 )
                {
                  v4 = (__int64)v74;
                  memcpy(v83, v74, sa - v74);
                }
              }
            }
            if ( (_BYTE *)v130.m128i_i64[0] != v131 )
              _libc_free(v130.m128i_i64[0], v4);
            sub_2240A30(&v141);
            if ( (v140 & 1) == 0 )
              sub_2240A30(src);
LABEL_25:
            if ( v159 != (__m128i *)&v161.m128i_u64[1] )
              _libc_free(v159, v4);
            goto LABEL_28;
          }
          if ( !v61 )
            goto LABEL_101;
          v90 = dest;
        }
        memcpy(v90, s, v62);
        v61 = v126.m128i_i64[0];
        v60 = (_QWORD *)v150.m128i_i64[0];
        goto LABEL_101;
      }
      v54 = v142;
    }
    memcpy(v54, v35, v34);
    v34 = v150.m128i_i64[0];
    v36 = (_QWORD *)v141.m128i_i64[0];
    goto LABEL_70;
  }
  v16 = *(_QWORD *)(a2 + 32);
  if ( v16 )
  {
    v160 = 0;
    sub_C58CA0(&v159, *(_BYTE **)(a2 + 24), (_BYTE *)(*(_QWORD *)(a2 + 24) + v16));
LABEL_34:
    LOWORD(v152) = 257;
    v143.m128i_i16[0] = 257;
    v134.m128i_i16[0] = 257;
    v132 = 257;
    if ( v11[1] )
    {
      LOBYTE(v132) = 3;
      v130.m128i_i64[0] = (__int64)(v11 + 1);
    }
    sub_C81B70(&v159, &v130, src, &v141, &v150);
    v20 = v160;
    if ( (unsigned __int64)(v160 + 1) > v161.m128i_i64[0] )
    {
      sub_C8D290(&v159, &v161.m128i_u64[1], v160 + 1, 1);
      v20 = v160;
    }
    v159->m128i_i8[v20] = 0;
    s = (char *)v159;
    goto LABEL_18;
  }
  (*(void (__fastcall **)(__m128i *))(**(_QWORD **)(a2 + 16) + 80LL))(&v130);
  if ( (v132 & 1) == 0 )
  {
    v160 = 0;
    sub_C58CA0(&v159, v130.m128i_i64[0], (_BYTE *)(v130.m128i_i64[0] + v130.m128i_i64[1]));
    if ( (v132 & 1) == 0 )
      sub_2240A30(&v130);
    goto LABEL_34;
  }
  v143.m128i_i16[0] = 257;
  if ( v11[1] )
  {
    v143.m128i_i8[0] = 3;
    v141.m128i_i64[0] = (__int64)(v11 + 1);
  }
  src[0].m128i_i64[0] = (__int64)"cannot get absolute path for: ";
  v134.m128i_i16[0] = 259;
  sub_9C6370(&v150, src, &v141, v17, v18, v19);
  if ( (v132 & 1) != 0 )
  {
    v48 = v130.m128i_i32[0];
    v49 = v130.m128i_i64[1];
  }
  else
  {
    v48 = 0;
    v49 = sub_2241E40();
  }
  sub_CA0F50(&v126, &v150);
  v4 = (__int64)&v126;
  sub_C63F00(a1, &v126, v48, v49);
  if ( (__int64 *)v126.m128i_i64[0] != &v127 )
  {
    v4 = v127 + 1;
    j_j___libc_free_0(v126.m128i_i64[0], v127 + 1);
  }
  if ( (v132 & 1) == 0 )
    sub_2240A30(&v130);
LABEL_48:
  if ( v159 != (__m128i *)&v161.m128i_u64[1] )
    _libc_free(v159, v4);
  v6 = (unsigned int)v155;
LABEL_51:
  v31 = v154;
  v32 = (__m128i *)((char *)v154 + 40 * v6);
  if ( v154 != v32 )
  {
    do
    {
      v32 = (__m128i *)((char *)v32 - 40);
      if ( (__m128i *)v32->m128i_i64[0] != &v32[1] )
      {
        v4 = v32[1].m128i_i64[0] + 1;
        j_j___libc_free_0(v32->m128i_i64[0], v4);
      }
    }
    while ( v31 != v32 );
    v32 = v154;
  }
  if ( v32 != (__m128i *)v156 )
    _libc_free(v32, v4);
  return a1;
}
