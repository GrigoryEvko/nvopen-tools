// Function: sub_2287970
// Address: 0x2287970
//
void __fastcall sub_2287970(__int64 *a1, _QWORD *a2)
{
  __int64 v4; // r13
  __int64 v5; // r12
  size_t v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rax
  _QWORD *v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  size_t v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rdi
  __m128i *v18; // r8
  unsigned __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rax
  unsigned int v23; // edx
  unsigned __int64 v24; // r12
  __m128i si128; // xmm0
  __int64 v26; // rdx
  __m128i v27; // xmm0
  __int64 v28; // rax
  _WORD *v29; // rdx
  __int64 v30; // r8
  __int64 v31; // rax
  _QWORD *v32; // rdx
  __int64 v33; // rdi
  char *v34; // rax
  __int64 v35; // rdx
  size_t v36; // rdx
  unsigned __int8 *v37; // rsi
  __int64 v38; // rax
  __int64 v39; // rdx
  unsigned __int8 *v40; // rdi
  __int64 v41; // rdi
  _WORD *v42; // rdx
  unsigned __int64 v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rdx
  __int64 v46; // r14
  int v47; // r13d
  __int64 v48; // rax
  unsigned __int64 v49; // r15
  __int64 v50; // rdi
  __int64 v51; // rdx
  __int64 v52; // rdi
  _QWORD *v53; // rdx
  __int64 v54; // rdi
  _WORD *v55; // rdx
  __int64 v56; // rax
  __int64 i; // r13
  __int64 v58; // rax
  __int64 v59; // rdi
  _BYTE *v60; // rax
  __int64 v61; // rdi
  _BYTE *v62; // rax
  __int64 v63; // rdi
  _BYTE *v64; // rax
  __int64 v65; // rdi
  char *v66; // rax
  __int64 v67; // rdx
  unsigned int v68; // esi
  __int64 v69; // r8
  int v70; // r11d
  _QWORD *v71; // rdx
  unsigned int v72; // ecx
  _QWORD *v73; // rax
  __int64 v74; // rdi
  unsigned __int64 v75; // r13
  unsigned __int64 *v76; // rax
  char *v77; // rsi
  size_t v78; // rdx
  unsigned __int64 *v79; // rax
  unsigned __int64 *v80; // rax
  __int64 v81; // rax
  __int64 v82; // rdi
  _BYTE *v83; // rax
  __int64 v84; // rdi
  _BYTE *v85; // rax
  __int64 v86; // rdi
  _BYTE *v87; // rax
  int v88; // r8d
  int v89; // r8d
  __int64 v90; // r9
  __int64 v91; // rax
  int v92; // ecx
  __int64 v93; // r11
  size_t v94; // rdx
  __m128i *v95; // rdi
  int v96; // eax
  int v97; // edi
  int v98; // edi
  __int64 v99; // r9
  int v100; // esi
  __int64 v101; // r14
  _QWORD *v102; // rax
  __int64 v103; // r8
  int v104; // edi
  _QWORD *v105; // rsi
  __int64 v106; // [rsp+10h] [rbp-160h]
  __int64 v107; // [rsp+20h] [rbp-150h]
  unsigned __int64 v108; // [rsp+20h] [rbp-150h]
  __int64 v109; // [rsp+20h] [rbp-150h]
  __int64 v110; // [rsp+20h] [rbp-150h]
  __int64 v111; // [rsp+28h] [rbp-148h]
  __int64 v112; // [rsp+28h] [rbp-148h]
  __m128i *dest; // [rsp+30h] [rbp-140h]
  size_t v114; // [rsp+38h] [rbp-138h]
  __m128i v115; // [rsp+40h] [rbp-130h] BYREF
  __int64 v116[2]; // [rsp+50h] [rbp-120h] BYREF
  __int64 v117; // [rsp+60h] [rbp-110h] BYREF
  char *v118; // [rsp+70h] [rbp-100h] BYREF
  size_t v119; // [rsp+78h] [rbp-F8h]
  __int64 v120; // [rsp+80h] [rbp-F0h] BYREF
  __m128i *v121; // [rsp+90h] [rbp-E0h]
  size_t n; // [rsp+98h] [rbp-D8h]
  __m128i v123; // [rsp+A0h] [rbp-D0h] BYREF
  __m128i *v124; // [rsp+B0h] [rbp-C0h] BYREF
  size_t v125; // [rsp+B8h] [rbp-B8h]
  __m128i v126; // [rsp+C0h] [rbp-B0h] BYREF
  unsigned __int64 v127; // [rsp+D0h] [rbp-A0h] BYREF
  size_t v128; // [rsp+D8h] [rbp-98h]
  __m128i v129; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v130; // [rsp+F0h] [rbp-80h]
  unsigned __int8 *v131; // [rsp+100h] [rbp-70h] BYREF
  size_t v132; // [rsp+108h] [rbp-68h]
  _QWORD v133[12]; // [rsp+110h] [rbp-60h] BYREF

  v4 = a2[1];
  v5 = *(_QWORD *)a1[1];
  if ( v4 )
  {
    dest = &v115;
    v115.m128i_i8[0] = 0;
    if ( !(_BYTE)qword_4FDB308 )
    {
      v6 = 0;
      v121 = &v123;
      goto LABEL_4;
    }
    v68 = *(_DWORD *)(v5 + 40);
    if ( v68 )
    {
      v69 = *(_QWORD *)(v5 + 24);
      v70 = 1;
      v71 = 0;
      v72 = (v68 - 1) & (((unsigned int)v4 >> 4) ^ ((unsigned int)v4 >> 9));
      v73 = (_QWORD *)(v69 + 16LL * v72);
      v74 = *v73;
      if ( v4 == *v73 )
      {
LABEL_125:
        v75 = v73[1];
        sub_11FCE30(v116, v75, *(_QWORD *)(v5 + 48));
        if ( *(_QWORD *)(v5 + 48) >> 1 < v75 )
        {
          sub_11FCC80((__int64 *)&v118, 1.0);
          goto LABEL_127;
        }
LABEL_176:
        sub_11FCC80((__int64 *)&v118, 0.0);
LABEL_127:
        v131 = (unsigned __int8 *)v133;
        v132 = 0;
        LOBYTE(v133[0]) = 0;
        sub_2240E30((__int64)&v131, v119 + 7);
        if ( 0x3FFFFFFFFFFFFFFFLL - v132 <= 6 )
          goto LABEL_221;
        sub_2241490((unsigned __int64 *)&v131, "color=\"", 7u);
        sub_2241490((unsigned __int64 *)&v131, v118, v119);
        if ( 0x3FFFFFFFFFFFFFFFLL - v132 <= 0x1D )
          goto LABEL_221;
        v76 = sub_2241490((unsigned __int64 *)&v131, "ff\", style=filled, fillcolor=\"", 0x1Eu);
        v127 = (unsigned __int64)&v129;
        if ( (unsigned __int64 *)*v76 == v76 + 2 )
        {
          v129 = _mm_loadu_si128((const __m128i *)v76 + 1);
        }
        else
        {
          v127 = *v76;
          v129.m128i_i64[0] = v76[2];
        }
        v128 = v76[1];
        *v76 = (unsigned __int64)(v76 + 2);
        v77 = (char *)v116[0];
        v76[1] = 0;
        v78 = v116[1];
        *((_BYTE *)v76 + 16) = 0;
        v79 = sub_2241490(&v127, v77, v78);
        v124 = &v126;
        if ( (unsigned __int64 *)*v79 == v79 + 2 )
        {
          v126 = _mm_loadu_si128((const __m128i *)v79 + 1);
        }
        else
        {
          v124 = (__m128i *)*v79;
          v126.m128i_i64[0] = v79[2];
        }
        v125 = v79[1];
        *v79 = (unsigned __int64)(v79 + 2);
        v79[1] = 0;
        *((_BYTE *)v79 + 16) = 0;
        if ( 0x3FFFFFFFFFFFFFFFLL - v125 <= 2 )
LABEL_221:
          sub_4262D8((__int64)"basic_string::append");
        v80 = sub_2241490((unsigned __int64 *)&v124, "80\"", 3u);
        v121 = &v123;
        if ( (unsigned __int64 *)*v80 == v80 + 2 )
        {
          v123 = _mm_loadu_si128((const __m128i *)v80 + 1);
        }
        else
        {
          v121 = (__m128i *)*v80;
          v123.m128i_i64[0] = v80[2];
        }
        n = v80[1];
        *v80 = (unsigned __int64)(v80 + 2);
        v80[1] = 0;
        *((_BYTE *)v80 + 16) = 0;
        if ( v121 == &v123 )
        {
          v94 = n;
          if ( n )
          {
            if ( n == 1 )
              v115.m128i_i8[0] = v123.m128i_i8[0];
            else
              memcpy(&v115, &v123, n);
            v94 = n;
          }
          v114 = v94;
          v115.m128i_i8[v94] = 0;
          v95 = v121;
        }
        else
        {
          dest = v121;
          v114 = n;
          v115.m128i_i64[0] = v123.m128i_i64[0];
          v121 = &v123;
          v95 = &v123;
        }
        n = 0;
        v95->m128i_i8[0] = 0;
        if ( v121 != &v123 )
          j_j___libc_free_0((unsigned __int64)v121);
        if ( v124 != &v126 )
          j_j___libc_free_0((unsigned __int64)v124);
        if ( (__m128i *)v127 != &v129 )
          j_j___libc_free_0(v127);
        if ( v131 != (unsigned __int8 *)v133 )
          j_j___libc_free_0((unsigned __int64)v131);
        if ( v118 != (char *)&v120 )
          j_j___libc_free_0((unsigned __int64)v118);
        if ( (__int64 *)v116[0] != &v117 )
          j_j___libc_free_0(v116[0]);
        v6 = v114;
        v121 = &v123;
        if ( dest != &v115 )
        {
          v121 = dest;
          v123.m128i_i64[0] = v115.m128i_i64[0];
          goto LABEL_5;
        }
LABEL_4:
        v123 = _mm_load_si128(&v115);
LABEL_5:
        n = v6;
        goto LABEL_6;
      }
      while ( v74 != -4096 )
      {
        if ( v74 == -8192 && !v71 )
          v71 = v73;
        v72 = (v68 - 1) & (v70 + v72);
        v73 = (_QWORD *)(v69 + 16LL * v72);
        v74 = *v73;
        if ( v4 == *v73 )
          goto LABEL_125;
        ++v70;
      }
      if ( !v71 )
        v71 = v73;
      v96 = *(_DWORD *)(v5 + 32);
      ++*(_QWORD *)(v5 + 16);
      v92 = v96 + 1;
      if ( 4 * (v96 + 1) < 3 * v68 )
      {
        if ( v68 - *(_DWORD *)(v5 + 36) - v92 > v68 >> 3 )
        {
LABEL_173:
          *(_DWORD *)(v5 + 32) = v92;
          if ( *v71 != -4096 )
            --*(_DWORD *)(v5 + 36);
          *v71 = v4;
          v71[1] = 0;
          sub_11FCE30(v116, 0, *(_QWORD *)(v5 + 48));
          goto LABEL_176;
        }
        sub_A2B080(v5 + 16, v68);
        v97 = *(_DWORD *)(v5 + 40);
        if ( v97 )
        {
          v98 = v97 - 1;
          v99 = *(_QWORD *)(v5 + 24);
          v100 = 1;
          LODWORD(v101) = v98 & (((unsigned int)v4 >> 4) ^ ((unsigned int)v4 >> 9));
          v92 = *(_DWORD *)(v5 + 32) + 1;
          v102 = 0;
          v71 = (_QWORD *)(v99 + 16LL * (unsigned int)v101);
          v103 = *v71;
          if ( v4 != *v71 )
          {
            while ( v103 != -4096 )
            {
              if ( v103 == -8192 && !v102 )
                v102 = v71;
              v101 = v98 & (unsigned int)(v101 + v100);
              v71 = (_QWORD *)(v99 + 16 * v101);
              v103 = *v71;
              if ( v4 == *v71 )
                goto LABEL_173;
              ++v100;
            }
            if ( v102 )
              v71 = v102;
          }
          goto LABEL_173;
        }
LABEL_232:
        ++*(_DWORD *)(v5 + 32);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(v5 + 16);
    }
    sub_A2B080(v5 + 16, 2 * v68);
    v88 = *(_DWORD *)(v5 + 40);
    if ( v88 )
    {
      v89 = v88 - 1;
      v90 = *(_QWORD *)(v5 + 24);
      LODWORD(v91) = v89 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v92 = *(_DWORD *)(v5 + 32) + 1;
      v71 = (_QWORD *)(v90 + 16LL * (unsigned int)v91);
      v93 = *v71;
      if ( v4 != *v71 )
      {
        v104 = 1;
        v105 = 0;
        while ( v93 != -4096 )
        {
          if ( !v105 && v93 == -8192 )
            v105 = v71;
          v91 = v89 & (unsigned int)(v91 + v104);
          v71 = (_QWORD *)(v90 + 16 * v91);
          v93 = *v71;
          if ( v4 == *v71 )
            goto LABEL_173;
          ++v104;
        }
        if ( v105 )
          v71 = v105;
      }
      goto LABEL_173;
    }
    goto LABEL_232;
  }
  v123.m128i_i8[0] = 0;
  v121 = &v123;
  n = 0;
LABEL_6:
  v7 = *a1;
  v8 = *(_QWORD *)(*a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 24) - v8) <= 4 )
  {
    v7 = sub_CB6200(v7, "\tNode", 5u);
  }
  else
  {
    *(_DWORD *)v8 = 1685016073;
    *(_BYTE *)(v8 + 4) = 101;
    *(_QWORD *)(v7 + 32) += 5LL;
  }
  v9 = sub_CB5A80(v7, (unsigned __int64)a2);
  v10 = *(_QWORD **)(v9 + 32);
  if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 7u )
  {
    sub_CB6200(v9, " [shape=", 8u);
  }
  else
  {
    *v10 = 0x3D65706168735B20LL;
    *(_QWORD *)(v9 + 32) += 8LL;
  }
  v11 = *a1;
  v12 = *(_QWORD *)(*a1 + 32);
  v13 = *(_QWORD *)(*a1 + 24) - v12;
  if ( *((_BYTE *)a1 + 16) )
  {
    if ( v13 <= 4 )
    {
      sub_CB6200(v11, (unsigned __int8 *)"none,", 5u);
    }
    else
    {
      *(_DWORD *)v12 = 1701736302;
      *(_BYTE *)(v12 + 4) = 44;
      *(_QWORD *)(v11 + 32) += 5LL;
    }
LABEL_13:
    v14 = n;
    if ( !n )
      goto LABEL_14;
LABEL_105:
    v63 = sub_CB6200(*a1, (unsigned __int8 *)v121, v14);
    v64 = *(_BYTE **)(v63 + 32);
    if ( *(_BYTE **)(v63 + 24) == v64 )
    {
      sub_CB6200(v63, (unsigned __int8 *)",", 1u);
    }
    else
    {
      *v64 = 44;
      ++*(_QWORD *)(v63 + 32);
    }
    goto LABEL_14;
  }
  if ( v13 <= 6 )
  {
    sub_CB6200(v11, (unsigned __int8 *)"record,", 7u);
    goto LABEL_13;
  }
  *(_DWORD *)v12 = 1868785010;
  *(_WORD *)(v12 + 4) = 25714;
  *(_BYTE *)(v12 + 6) = 44;
  *(_QWORD *)(v11 + 32) += 7LL;
  v14 = n;
  if ( n )
    goto LABEL_105;
LABEL_14:
  v15 = *a1;
  v16 = *(_QWORD *)(*a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 24) - v16) <= 5 )
  {
    sub_CB6200(v15, "label=", 6u);
  }
  else
  {
    *(_DWORD *)v16 = 1700946284;
    *(_WORD *)(v16 + 4) = 15724;
    *(_QWORD *)(v15 + 32) += 6LL;
  }
  v17 = *a1;
  v18 = *(__m128i **)(*a1 + 32);
  v19 = *(_QWORD *)(*a1 + 24) - (_QWORD)v18;
  if ( *((_BYTE *)a1 + 16) )
  {
    v20 = a2[2];
    v21 = a2[3];
    if ( v20 == v21 )
    {
      v24 = 1;
    }
    else
    {
      v22 = v20 + 40;
      v23 = 0;
      do
      {
        ++v23;
        if ( v21 == v22 )
        {
          v24 = v23;
          goto LABEL_22;
        }
        v22 += 40;
      }
      while ( v23 != 64 );
      v24 = 65;
    }
LABEL_22:
    if ( v19 <= 0x30 )
    {
      v81 = sub_CB6200(v17, "<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"", 0x31u);
      v26 = *(_QWORD *)(v81 + 32);
      v17 = v81;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB60);
      v18[3].m128i_i8[0] = 34;
      *v18 = si128;
      v18[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CB70);
      v18[2] = _mm_load_si128((const __m128i *)&xmmword_3F8CB80);
      v26 = *(_QWORD *)(v17 + 32) + 49LL;
      *(_QWORD *)(v17 + 32) = v26;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v17 + 24) - v26) <= 0x2E )
    {
      v17 = sub_CB6200(v17, " cellpadding=\"0\"><tr><td align=\"text\" colspan=\"", 0x2Fu);
    }
    else
    {
      v27 = _mm_load_si128((const __m128i *)&xmmword_3F8CB90);
      qmemcpy((void *)(v26 + 32), "text\" colspan=\"", 15);
      *(__m128i *)v26 = v27;
      *(__m128i *)(v26 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F8CBA0);
      *(_QWORD *)(v17 + 32) += 47LL;
    }
    v28 = sub_CB59D0(v17, v24);
    v29 = *(_WORD **)(v28 + 32);
    if ( *(_QWORD *)(v28 + 24) - (_QWORD)v29 <= 1u )
    {
      sub_CB6200(v28, "\">", 2u);
    }
    else
    {
      *v29 = 15906;
      *(_QWORD *)(v28 + 32) += 2LL;
    }
  }
  else if ( v19 <= 1 )
  {
    sub_CB6200(v17, (unsigned __int8 *)"\"{", 2u);
  }
  else
  {
    v18->m128i_i16[0] = 31522;
    *(_QWORD *)(v17 + 32) += 2LL;
  }
  v30 = *a1;
  v31 = *(_QWORD *)(*(_QWORD *)a1[1] + 8LL);
  v32 = *(_QWORD **)(v31 + 56);
  if ( !*((_BYTE *)a1 + 16) )
  {
    if ( a2 == v32 )
    {
      v127 = (unsigned __int64)&v129;
      strcpy(v129.m128i_i8, "external caller");
      v128 = 15;
    }
    else if ( a2 == *(_QWORD **)(v31 + 64) )
    {
      v127 = (unsigned __int64)&v129;
      strcpy(v129.m128i_i8, "external callee");
      v128 = 15;
    }
    else
    {
      v65 = a2[1];
      if ( v65 )
      {
        v109 = *a1;
        v66 = (char *)sub_BD5D20(v65);
        v127 = (unsigned __int64)&v129;
        sub_2285B20((__int64 *)&v127, v66, (__int64)&v66[v67]);
        v30 = v109;
      }
      else
      {
        v127 = (unsigned __int64)&v129;
        strcpy(v129.m128i_i8, "external node");
        v128 = 13;
      }
    }
    v110 = v30;
    sub_C67200((__int64 *)&v131, (__int64)&v127);
    sub_CB6200(v110, v131, v132);
    if ( v131 != (unsigned __int8 *)v133 )
      j_j___libc_free_0((unsigned __int64)v131);
    v40 = (unsigned __int8 *)v127;
    if ( (__m128i *)v127 != &v129 )
      goto LABEL_36;
    goto LABEL_37;
  }
  if ( a2 == v32 )
  {
    v131 = (unsigned __int8 *)v133;
    qmemcpy(v133, "external caller", 15);
  }
  else
  {
    if ( a2 != *(_QWORD **)(v31 + 64) )
    {
      v33 = a2[1];
      if ( v33 )
      {
        v107 = *a1;
        v34 = (char *)sub_BD5D20(v33);
        v131 = (unsigned __int8 *)v133;
        sub_2285B20((__int64 *)&v131, v34, (__int64)&v34[v35]);
        v36 = v132;
        v37 = v131;
        v30 = v107;
      }
      else
      {
        v131 = (unsigned __int8 *)v133;
        v36 = 13;
        strcpy((char *)v133, "external node");
        v37 = (unsigned __int8 *)v133;
        v132 = 13;
      }
      goto LABEL_33;
    }
    v131 = (unsigned __int8 *)v133;
    qmemcpy(v133, "external callee", 15);
  }
  HIBYTE(v133[1]) = 0;
  v36 = 15;
  v132 = 15;
  v37 = (unsigned __int8 *)v133;
LABEL_33:
  v38 = sub_CB6200(v30, v37, v36);
  v39 = *(_QWORD *)(v38 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v38 + 24) - v39) <= 4 )
  {
    sub_CB6200(v38, "</td>", 5u);
  }
  else
  {
    *(_DWORD *)v39 = 1685335868;
    *(_BYTE *)(v39 + 4) = 62;
    *(_QWORD *)(v38 + 32) += 5LL;
  }
  v40 = v131;
  if ( v131 != (unsigned __int8 *)v133 )
LABEL_36:
    j_j___libc_free_0((unsigned __int64)v40);
LABEL_37:
  v126.m128i_i8[0] = 0;
  v124 = &v126;
  v133[3] = 0x100000000LL;
  v125 = 0;
  v132 = 0;
  v131 = (unsigned __int8 *)&unk_49DD210;
  memset(v133, 0, 24);
  v133[4] = &v124;
  sub_CB5980((__int64)&v131, 0, 0, 0);
  if ( (unsigned __int8)sub_2287290((__int64)a1, (__int64)&v131, (__int64)a2) )
  {
    if ( *((_BYTE *)a1 + 16) )
      goto LABEL_39;
    v82 = *a1;
    v83 = *(_BYTE **)(*a1 + 32);
    if ( *(_BYTE **)(*a1 + 24) == v83 )
    {
      sub_CB6200(v82, (unsigned __int8 *)"|", 1u);
    }
    else
    {
      *v83 = 124;
      ++*(_QWORD *)(v82 + 32);
    }
    v84 = *a1;
    if ( *((_BYTE *)a1 + 16) )
    {
LABEL_39:
      sub_CB6200(*a1, (unsigned __int8 *)v124, v125);
    }
    else
    {
      v85 = *(_BYTE **)(v84 + 32);
      if ( *(_BYTE **)(v84 + 24) == v85 )
      {
        v84 = sub_CB6200(v84, (unsigned __int8 *)"{", 1u);
      }
      else
      {
        *v85 = 123;
        ++*(_QWORD *)(v84 + 32);
      }
      v86 = sub_CB6200(v84, (unsigned __int8 *)v124, v125);
      v87 = *(_BYTE **)(v86 + 32);
      if ( *(_BYTE **)(v86 + 24) == v87 )
      {
        sub_CB6200(v86, (unsigned __int8 *)"}", 1u);
      }
      else
      {
        *v87 = 125;
        ++*(_QWORD *)(v86 + 32);
      }
    }
  }
  v41 = *a1;
  v42 = *(_WORD **)(*a1 + 32);
  v43 = *(_QWORD *)(*a1 + 24) - (_QWORD)v42;
  if ( *((_BYTE *)a1 + 16) )
  {
    if ( v43 <= 0xD )
    {
      sub_CB6200(v41, "</tr></table>>", 0xEu);
    }
    else
    {
      qmemcpy(v42, "</tr></table>>", 14);
      *(_QWORD *)(v41 + 32) += 14LL;
    }
  }
  else if ( v43 <= 1 )
  {
    sub_CB6200(v41, (unsigned __int8 *)"}\"", 2u);
  }
  else
  {
    *v42 = 8829;
    *(_QWORD *)(v41 + 32) += 2LL;
  }
  v44 = *a1;
  v45 = *(_QWORD *)(*a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 24) - v45) <= 2 )
  {
    sub_CB6200(v44, (unsigned __int8 *)"];\n", 3u);
  }
  else
  {
    *(_BYTE *)(v45 + 2) = 10;
    *(_WORD *)v45 = 15197;
    *(_QWORD *)(v44 + 32) += 3LL;
  }
  v46 = a2[2];
  v47 = 0;
  v111 = a2[3];
  if ( v111 != v46 )
  {
    v108 = (unsigned __int64)a2;
    do
    {
      v129.m128i_i8[8] = 0;
      if ( *(_BYTE *)(v46 + 24) )
      {
        v127 = 6;
        v128 = 0;
        v129.m128i_i64[0] = *(_QWORD *)(v46 + 16);
        if ( v129.m128i_i64[0] != 0 && v129.m128i_i64[0] != -4096 && v129.m128i_i64[0] != -8192 )
          sub_BD6050(&v127, *(_QWORD *)v46 & 0xFFFFFFFFFFFFFFF8LL);
        v129.m128i_i8[8] = 1;
        v130 = *(_QWORD *)(v46 + 32);
        v48 = sub_2285790((__int64)&v127);
        v129.m128i_i8[8] = 0;
        if ( v129.m128i_i64[0] != -4096 && v129.m128i_i64[0] != 0 && v129.m128i_i64[0] != -8192 )
        {
          v106 = v48;
          sub_BD60C0(&v127);
          v48 = v106;
        }
      }
      else
      {
        v130 = *(_QWORD *)(v46 + 32);
        v48 = sub_2285790((__int64)&v127);
      }
      if ( (_BYTE)qword_4FDB148 || *(_QWORD *)(v48 + 8) )
      {
        v129.m128i_i8[8] = 0;
        if ( *(_BYTE *)(v46 + 24) )
        {
          v127 = 6;
          v128 = 0;
          v129.m128i_i64[0] = *(_QWORD *)(v46 + 16);
          if ( v129.m128i_i64[0] != 0 && v129.m128i_i64[0] != -4096 && v129.m128i_i64[0] != -8192 )
            sub_BD6050(&v127, *(_QWORD *)v46 & 0xFFFFFFFFFFFFFFF8LL);
          v129.m128i_i8[8] = 1;
          v130 = *(_QWORD *)(v46 + 32);
          v56 = sub_2285790((__int64)&v127);
          v129.m128i_i8[8] = 0;
          v49 = v56;
          if ( v129.m128i_i64[0] != -4096 && v129.m128i_i64[0] != 0 && v129.m128i_i64[0] != -8192 )
            sub_BD60C0(&v127);
        }
        else
        {
          v130 = *(_QWORD *)(v46 + 32);
          v49 = sub_2285790((__int64)&v127);
        }
        if ( v49 )
        {
          sub_2285FB0(
            (__int64)&v127,
            v108,
            v46,
            (__int64 (__fastcall *)(unsigned __int64 *))sub_2285790,
            *(_QWORD *)a1[1]);
          v50 = *a1;
          v51 = *(_QWORD *)(*a1 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(*a1 + 24) - v51) <= 4 )
          {
            v50 = sub_CB6200(v50, "\tNode", 5u);
          }
          else
          {
            *(_DWORD *)v51 = 1685016073;
            *(_BYTE *)(v51 + 4) = 101;
            *(_QWORD *)(v50 + 32) += 5LL;
          }
          sub_CB5A80(v50, v108);
          v52 = *a1;
          v53 = *(_QWORD **)(*a1 + 32);
          if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v53 <= 7u )
          {
            v52 = sub_CB6200(v52, " -> Node", 8u);
          }
          else
          {
            *v53 = 0x65646F4E203E2D20LL;
            *(_QWORD *)(v52 + 32) += 8LL;
          }
          sub_CB5A80(v52, v49);
          if ( v128 )
          {
            v59 = *a1;
            v60 = *(_BYTE **)(*a1 + 32);
            if ( *(_BYTE **)(*a1 + 24) == v60 )
            {
              v59 = sub_CB6200(v59, (unsigned __int8 *)"[", 1u);
            }
            else
            {
              *v60 = 91;
              ++*(_QWORD *)(v59 + 32);
            }
            v61 = sub_CB6200(v59, (unsigned __int8 *)v127, v128);
            v62 = *(_BYTE **)(v61 + 32);
            if ( *(_BYTE **)(v61 + 24) == v62 )
            {
              sub_CB6200(v61, (unsigned __int8 *)"]", 1u);
            }
            else
            {
              *v62 = 93;
              ++*(_QWORD *)(v61 + 32);
            }
          }
          v54 = *a1;
          v55 = *(_WORD **)(*a1 + 32);
          if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v55 <= 1u )
          {
            sub_CB6200(v54, (unsigned __int8 *)";\n", 2u);
          }
          else
          {
            *v55 = 2619;
            *(_QWORD *)(v54 + 32) += 2LL;
          }
          if ( (__m128i *)v127 != &v129 )
            j_j___libc_free_0(v127);
        }
      }
      ++v47;
      v46 += 40;
    }
    while ( v47 != 64 && v111 != v46 );
    for ( i = v111; i != v46; v46 += 40 )
    {
      v129.m128i_i8[8] = 0;
      if ( *(_BYTE *)(v46 + 24) )
      {
        v127 = 6;
        v128 = 0;
        v129.m128i_i64[0] = *(_QWORD *)(v46 + 16);
        if ( v129.m128i_i64[0] != 0 && v129.m128i_i64[0] != -4096 && v129.m128i_i64[0] != -8192 )
          sub_BD6050(&v127, *(_QWORD *)v46 & 0xFFFFFFFFFFFFFFF8LL);
        v129.m128i_i8[8] = 1;
        v130 = *(_QWORD *)(v46 + 32);
        v58 = sub_2285790((__int64)&v127);
        v129.m128i_i8[8] = 0;
        if ( v129.m128i_i64[0] != -4096 && v129.m128i_i64[0] != 0 && v129.m128i_i64[0] != -8192 )
        {
          v112 = v58;
          sub_BD60C0(&v127);
          v58 = v112;
        }
      }
      else
      {
        v130 = *(_QWORD *)(v46 + 32);
        v58 = sub_2285790((__int64)&v127);
      }
      if ( (_BYTE)qword_4FDB148 || *(_QWORD *)(v58 + 8) )
        sub_22876D0((__int64 **)a1, v108, 64, v46, (__int64 (*)(void))sub_2285790);
    }
  }
  v131 = (unsigned __int8 *)&unk_49DD210;
  sub_CB5840((__int64)&v131);
  if ( v124 != &v126 )
    j_j___libc_free_0((unsigned __int64)v124);
  if ( v121 != &v123 )
    j_j___libc_free_0((unsigned __int64)v121);
}
