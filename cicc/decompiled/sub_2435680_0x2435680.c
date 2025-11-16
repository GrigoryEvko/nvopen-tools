// Function: sub_2435680
// Address: 0x2435680
//
void __fastcall sub_2435680(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  bool v4; // zf
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 *v7; // rdi
  __int64 v8; // rax
  __int64 *v9; // rdi
  __int64 v10; // rax
  const char *v11; // rbx
  size_t v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  _BYTE *v15; // rsi
  __int64 v16; // rdx
  __m128i *v17; // rax
  __m128i *v18; // rax
  char *v19; // rsi
  size_t v20; // rdx
  unsigned __int64 *v21; // rax
  __int64 *v22; // rbx
  __int64 v23; // rax
  __int64 *v24; // rdi
  __int64 v25; // rdx
  __int64 i; // r12
  __int8 *v27; // r8
  unsigned __int64 v28; // rcx
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  __m128i *v32; // rax
  __m128i *v33; // rdx
  __int64 v34; // rcx
  __m128i *v35; // rax
  char *v36; // rsi
  size_t v37; // rdx
  unsigned __int64 *v38; // rax
  __int64 *v39; // rbx
  __int64 v40; // rax
  __int64 *v41; // rdi
  __int64 v42; // rdx
  _BYTE *v43; // rsi
  _BYTE *v44; // rdx
  unsigned __int64 *v45; // rax
  __int64 v46; // rax
  __int64 *v47; // rdi
  __int64 v48; // rdx
  unsigned __int64 *v49; // rax
  __int64 v50; // rax
  __int64 *v51; // rdi
  __int64 v52; // rdx
  unsigned __int64 *v53; // rax
  __int64 v54; // rax
  __int64 *v55; // rdi
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 *v60; // rdi
  unsigned __int64 v61; // rax
  __int64 v62; // r12
  __int64 v63; // rdx
  __int64 v64; // rbx
  __int64 *v65; // rdi
  unsigned __int64 v66; // rax
  __int64 v67; // r12
  __int64 v68; // rdx
  __int64 v69; // rbx
  __int64 v70; // rax
  __int64 *v71; // rdi
  unsigned __int64 v72; // rax
  __int64 v73; // r12
  __int64 v74; // rdx
  __int64 v75; // rbx
  __int64 *v76; // rdi
  _QWORD *v77; // rax
  _BYTE *v78; // rax
  __int64 *v79; // rdi
  __int64 v80; // rax
  unsigned __int64 v81; // rax
  __int64 v82; // r12
  __int64 v83; // rdx
  __int64 v84; // rbx
  _QWORD *v85; // rdi
  __int64 *v86; // rdi
  unsigned __int64 v87; // rax
  __int64 *v88; // rdi
  __int64 *v89; // rdi
  __int64 v90; // rax
  __int64 *v91; // rdi
  __int64 v92; // rax
  unsigned __int64 v93; // [rsp+20h] [rbp-290h]
  unsigned __int64 v94; // [rsp+28h] [rbp-288h]
  __int64 v95; // [rsp+30h] [rbp-280h]
  __int64 v96; // [rsp+58h] [rbp-258h]
  unsigned __int64 v97; // [rsp+60h] [rbp-250h]
  size_t v99; // [rsp+C8h] [rbp-1E8h]
  _QWORD v100[2]; // [rsp+D0h] [rbp-1E0h] BYREF
  char *v101; // [rsp+E0h] [rbp-1D0h]
  size_t v102; // [rsp+E8h] [rbp-1C8h]
  _QWORD v103[2]; // [rsp+F0h] [rbp-1C0h] BYREF
  char *v104; // [rsp+100h] [rbp-1B0h]
  size_t v105; // [rsp+108h] [rbp-1A8h]
  _QWORD v106[2]; // [rsp+110h] [rbp-1A0h] BYREF
  __int64 v107[2]; // [rsp+120h] [rbp-190h] BYREF
  _QWORD v108[2]; // [rsp+130h] [rbp-180h] BYREF
  _BYTE *v109; // [rsp+140h] [rbp-170h] BYREF
  size_t v110; // [rsp+148h] [rbp-168h]
  _QWORD v111[2]; // [rsp+150h] [rbp-160h] BYREF
  char *v112; // [rsp+160h] [rbp-150h] BYREF
  size_t v113; // [rsp+168h] [rbp-148h]
  _QWORD v114[2]; // [rsp+170h] [rbp-140h] BYREF
  __m128i *v115; // [rsp+180h] [rbp-130h] BYREF
  __int64 v116; // [rsp+188h] [rbp-128h]
  __m128i v117; // [rsp+190h] [rbp-120h] BYREF
  __m128i *v118; // [rsp+1A0h] [rbp-110h] BYREF
  __int64 v119; // [rsp+1A8h] [rbp-108h]
  __m128i v120; // [rsp+1B0h] [rbp-100h] BYREF
  __m128i *v121; // [rsp+1C0h] [rbp-F0h] BYREF
  unsigned __int64 v122; // [rsp+1C8h] [rbp-E8h]
  __m128i v123; // [rsp+1D0h] [rbp-E0h] BYREF
  __int64 v124; // [rsp+1E0h] [rbp-D0h]
  _BYTE *v125; // [rsp+1F0h] [rbp-C0h]
  __int64 v126; // [rsp+1F8h] [rbp-B8h]
  _BYTE v127[32]; // [rsp+200h] [rbp-B0h] BYREF
  __int64 v128; // [rsp+220h] [rbp-90h]
  __int64 v129; // [rsp+228h] [rbp-88h]
  __int16 v130; // [rsp+230h] [rbp-80h]
  __int64 v131; // [rsp+238h] [rbp-78h]
  void **v132; // [rsp+240h] [rbp-70h]
  void **v133; // [rsp+248h] [rbp-68h]
  __int64 v134; // [rsp+250h] [rbp-60h]
  int v135; // [rsp+258h] [rbp-58h]
  __int16 v136; // [rsp+25Ch] [rbp-54h]
  char v137; // [rsp+25Eh] [rbp-52h]
  __int64 v138; // [rsp+260h] [rbp-50h]
  __int64 v139; // [rsp+268h] [rbp-48h]
  void *v140; // [rsp+270h] [rbp-40h] BYREF
  void *v141; // [rsp+278h] [rbp-38h] BYREF

  v3 = *a1;
  v125 = v127;
  v131 = v3;
  v132 = &v140;
  v133 = &v141;
  v126 = 0x200000000LL;
  v140 = &unk_49DA100;
  v134 = 0;
  v4 = *((_BYTE *)a1 + 171) == 0;
  v135 = 0;
  v136 = 512;
  v137 = 7;
  v138 = 0;
  v139 = 0;
  v128 = 0;
  v129 = 0;
  v130 = 0;
  v141 = &unk_49DA0B0;
  if ( v4 )
  {
    v5 = 0;
  }
  else
  {
    qmemcpy(v100, "_match_all", 10);
    v5 = 10;
  }
  v99 = v5;
  *((_BYTE *)v100 + v5) = 0;
  v4 = *((_BYTE *)a1 + 171) == 0;
  v121 = (__m128i *)a1[15];
  v122 = (unsigned __int64)v121;
  if ( v4 )
  {
    v6 = sub_BCF480((__int64 *)a1[14], &v121, 2, 0);
    v7 = (__int64 *)a1[14];
    v95 = v6;
    v121 = (__m128i *)a1[15];
    v97 = sub_BCF480(v7, &v121, 1, 0);
    v8 = a1[15];
    v121 = (__m128i *)a1[16];
    v122 = (unsigned __int64)v121;
    v123.m128i_i64[0] = v8;
    v94 = sub_BCF480(v121->m128i_i64, &v121, 3, 0);
    v9 = (__int64 *)a1[16];
    v122 = a1[18];
    v10 = a1[15];
    v121 = (__m128i *)v9;
    v123.m128i_i64[0] = v10;
    v93 = sub_BCF480(v9, &v121, 3, 0);
  }
  else
  {
    v86 = (__int64 *)a1[14];
    v123.m128i_i64[0] = a1[17];
    v87 = sub_BCF480(v86, &v121, 3, 0);
    v88 = (__int64 *)a1[14];
    v95 = v87;
    v121 = (__m128i *)a1[15];
    v122 = a1[17];
    v97 = sub_BCF480(v88, &v121, 2, 0);
    v89 = (__int64 *)a1[16];
    v123.m128i_i64[0] = a1[15];
    v90 = a1[17];
    v121 = (__m128i *)v89;
    v122 = (unsigned __int64)v89;
    v123.m128i_i64[1] = v90;
    v94 = sub_BCF480(v89, &v121, 4, 0);
    v91 = (__int64 *)a1[16];
    v122 = a1[18];
    v92 = a1[15];
    v121 = (__m128i *)v91;
    v123.m128i_i64[0] = v92;
    v123.m128i_i64[1] = a1[17];
    v93 = sub_BCF480(v91, &v121, 4, 0);
  }
  v11 = "load";
  v96 = 0;
  while ( 2 )
  {
    v101 = (char *)v103;
    v12 = strlen(v11);
    v13 = 0;
    if ( (v12 & 4) != 0 )
    {
      LODWORD(v103[0]) = *(_DWORD *)v11;
      v13 = 4;
    }
    if ( (v12 & 2) != 0 )
    {
      *(_WORD *)((char *)v103 + v13) = *(_WORD *)&v11[v13];
      v13 += 2;
    }
    if ( (v12 & 1) != 0 )
      *((_BYTE *)v103 + v13) = v11[v13];
    v102 = v12;
    *((_BYTE *)v103 + v12) = 0;
    v4 = *((_BYTE *)a1 + 161) == 0;
    v104 = (char *)v106;
    if ( v4 )
    {
      v14 = 0;
    }
    else
    {
      v106[0] = 0x74726F62616F6E5FLL;
      v14 = 1;
    }
    v15 = (_BYTE *)qword_4FE58A8;
    v105 = v14 * 8;
    v16 = qword_4FE58B0;
    LOBYTE(v106[v14]) = 0;
    v107[0] = (__int64)v108;
    sub_2434550(v107, v15, (__int64)&v15[v16]);
    sub_2241490((unsigned __int64 *)v107, v101, v102);
    if ( v107[1] == 0x3FFFFFFFFFFFFFFFLL )
      goto LABEL_113;
    v17 = (__m128i *)sub_2241490((unsigned __int64 *)v107, "N", 1u);
    v115 = &v117;
    if ( (__m128i *)v17->m128i_i64[0] == &v17[1] )
    {
      v117 = _mm_loadu_si128(v17 + 1);
    }
    else
    {
      v115 = (__m128i *)v17->m128i_i64[0];
      v117.m128i_i64[0] = v17[1].m128i_i64[0];
    }
    v116 = v17->m128i_i64[1];
    v17->m128i_i64[0] = (__int64)v17[1].m128i_i64;
    v17->m128i_i64[1] = 0;
    v17[1].m128i_i8[0] = 0;
    v18 = (__m128i *)sub_2241490((unsigned __int64 *)&v115, (char *)v100, v99);
    v118 = &v120;
    if ( (__m128i *)v18->m128i_i64[0] == &v18[1] )
    {
      v120 = _mm_loadu_si128(v18 + 1);
    }
    else
    {
      v118 = (__m128i *)v18->m128i_i64[0];
      v120.m128i_i64[0] = v18[1].m128i_i64[0];
    }
    v119 = v18->m128i_i64[1];
    v18->m128i_i64[0] = (__int64)v18[1].m128i_i64;
    v19 = v104;
    v18->m128i_i64[1] = 0;
    v20 = v105;
    v18[1].m128i_i8[0] = 0;
    v21 = sub_2241490((unsigned __int64 *)&v118, v19, v20);
    v121 = &v123;
    if ( (unsigned __int64 *)*v21 == v21 + 2 )
    {
      v123 = _mm_loadu_si128((const __m128i *)v21 + 1);
    }
    else
    {
      v121 = (__m128i *)*v21;
      v123.m128i_i64[0] = v21[2];
    }
    v122 = v21[1];
    *v21 = (unsigned __int64)(v21 + 2);
    v21[1] = 0;
    *((_BYTE *)v21 + 16) = 0;
    v22 = &a1[2 * v96 + 44];
    v23 = sub_BA8CA0(a2, (__int64)v121, v122, v95);
    v24 = (__int64 *)v121;
    v22[1] = v23;
    v22[2] = v25;
    if ( v24 != (__int64 *)&v123 )
      j_j___libc_free_0((unsigned __int64)v24);
    if ( v118 != &v120 )
      j_j___libc_free_0((unsigned __int64)v118);
    if ( v115 != &v117 )
      j_j___libc_free_0((unsigned __int64)v115);
    if ( (_QWORD *)v107[0] != v108 )
      j_j___libc_free_0(v107[0]);
    for ( i = 0; i != 5; ++i )
    {
      v27 = &v123.m128i_i8[5];
      v28 = 1LL << i;
      do
      {
        *--v27 = v28 % 0xA + 48;
        v29 = v28;
        v28 /= 0xAu;
      }
      while ( v29 > 9 );
      v112 = (char *)v114;
      sub_2434550((__int64 *)&v112, v27, (__int64)v123.m128i_i64 + 5);
      v109 = v111;
      sub_2434550((__int64 *)&v109, (_BYTE *)qword_4FE58A8, qword_4FE58A8 + qword_4FE58B0);
      sub_2241490((unsigned __int64 *)&v109, v101, v102);
      v30 = 15;
      if ( v109 != (_BYTE *)v111 )
        v30 = v111[0];
      if ( v110 + v113 <= v30 )
        goto LABEL_38;
      v31 = 15;
      if ( v112 != (char *)v114 )
        v31 = v114[0];
      if ( v110 + v113 <= v31 )
      {
        v32 = (__m128i *)sub_2241130((unsigned __int64 *)&v112, 0, 0, v109, v110);
        v33 = v32 + 1;
        v115 = &v117;
        v34 = v32->m128i_i64[0];
        if ( (__m128i *)v32->m128i_i64[0] != &v32[1] )
        {
LABEL_39:
          v115 = (__m128i *)v34;
          v117.m128i_i64[0] = v32[1].m128i_i64[0];
          goto LABEL_40;
        }
      }
      else
      {
LABEL_38:
        v32 = (__m128i *)sub_2241490((unsigned __int64 *)&v109, v112, v113);
        v33 = v32 + 1;
        v115 = &v117;
        v34 = v32->m128i_i64[0];
        if ( (__m128i *)v32->m128i_i64[0] != &v32[1] )
          goto LABEL_39;
      }
      v117 = _mm_loadu_si128(v32 + 1);
LABEL_40:
      v116 = v32->m128i_i64[1];
      v32->m128i_i64[0] = (__int64)v33;
      v32->m128i_i64[1] = 0;
      v32[1].m128i_i8[0] = 0;
      v35 = (__m128i *)sub_2241490((unsigned __int64 *)&v115, (char *)v100, v99);
      v118 = &v120;
      if ( (__m128i *)v35->m128i_i64[0] == &v35[1] )
      {
        v120 = _mm_loadu_si128(v35 + 1);
      }
      else
      {
        v118 = (__m128i *)v35->m128i_i64[0];
        v120.m128i_i64[0] = v35[1].m128i_i64[0];
      }
      v119 = v35->m128i_i64[1];
      v35->m128i_i64[0] = (__int64)v35[1].m128i_i64;
      v36 = v104;
      v35->m128i_i64[1] = 0;
      v37 = v105;
      v35[1].m128i_i8[0] = 0;
      v38 = sub_2241490((unsigned __int64 *)&v118, v36, v37);
      v121 = &v123;
      if ( (unsigned __int64 *)*v38 == v38 + 2 )
      {
        v123 = _mm_loadu_si128((const __m128i *)v38 + 1);
      }
      else
      {
        v121 = (__m128i *)*v38;
        v123.m128i_i64[0] = v38[2];
      }
      v122 = v38[1];
      *v38 = (unsigned __int64)(v38 + 2);
      v38[1] = 0;
      *((_BYTE *)v38 + 16) = 0;
      v39 = &a1[10 * v96 + 24 + 2 * i];
      v40 = sub_BA8CA0(a2, (__int64)v121, v122, v97);
      v41 = (__int64 *)v121;
      v39[1] = v40;
      v39[2] = v42;
      if ( v41 != (__int64 *)&v123 )
        j_j___libc_free_0((unsigned __int64)v41);
      if ( v118 != &v120 )
        j_j___libc_free_0((unsigned __int64)v118);
      if ( v115 != &v117 )
        j_j___libc_free_0((unsigned __int64)v115);
      if ( v109 != (_BYTE *)v111 )
        j_j___libc_free_0((unsigned __int64)v109);
      if ( v112 != (char *)v114 )
        j_j___libc_free_0((unsigned __int64)v112);
    }
    if ( v104 != (char *)v106 )
      j_j___libc_free_0((unsigned __int64)v104);
    if ( v101 != (char *)v103 )
      j_j___libc_free_0((unsigned __int64)v101);
    v11 = "store";
    if ( v96 != 1 )
    {
      v96 = 1;
      continue;
    }
    break;
  }
  if ( *((_BYTE *)a1 + 160) && !(_BYTE)qword_4FE57C8 )
  {
    v44 = v111;
    v109 = v111;
    v110 = 0;
    v43 = v111;
    LOBYTE(v111[0]) = 0;
  }
  else
  {
    v109 = v111;
    sub_2434550((__int64 *)&v109, (_BYTE *)qword_4FE58A8, qword_4FE58A8 + qword_4FE58B0);
    v43 = v109;
    v44 = &v109[v110];
  }
  v112 = (char *)v114;
  sub_2434550((__int64 *)&v112, v43, (__int64)v44);
  if ( 0x3FFFFFFFFFFFFFFFLL - v113 <= 6 )
    goto LABEL_113;
  sub_2241490((unsigned __int64 *)&v112, "memmove", 7u);
  v45 = sub_2241490((unsigned __int64 *)&v112, (char *)v100, v99);
  v121 = &v123;
  if ( (unsigned __int64 *)*v45 == v45 + 2 )
  {
    v123 = _mm_loadu_si128((const __m128i *)v45 + 1);
  }
  else
  {
    v121 = (__m128i *)*v45;
    v123.m128i_i64[0] = v45[2];
  }
  v122 = v45[1];
  *v45 = (unsigned __int64)(v45 + 2);
  v45[1] = 0;
  *((_BYTE *)v45 + 16) = 0;
  v46 = sub_BA8CA0(a2, (__int64)v121, v122, v94);
  v47 = (__int64 *)v121;
  a1[49] = v46;
  a1[50] = v48;
  if ( v47 != (__int64 *)&v123 )
    j_j___libc_free_0((unsigned __int64)v47);
  if ( v112 != (char *)v114 )
    j_j___libc_free_0((unsigned __int64)v112);
  v115 = &v117;
  sub_2434550((__int64 *)&v115, v109, (__int64)&v109[v110]);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v116) <= 5 )
    goto LABEL_113;
  sub_2241490((unsigned __int64 *)&v115, "memcpy", 6u);
  v49 = sub_2241490((unsigned __int64 *)&v115, (char *)v100, v99);
  v121 = &v123;
  if ( (unsigned __int64 *)*v49 == v49 + 2 )
  {
    v123 = _mm_loadu_si128((const __m128i *)v49 + 1);
  }
  else
  {
    v121 = (__m128i *)*v49;
    v123.m128i_i64[0] = v49[2];
  }
  v122 = v49[1];
  *v49 = (unsigned __int64)(v49 + 2);
  v49[1] = 0;
  *((_BYTE *)v49 + 16) = 0;
  v50 = sub_BA8CA0(a2, (__int64)v121, v122, v94);
  v51 = (__int64 *)v121;
  a1[51] = v50;
  a1[52] = v52;
  if ( v51 != (__int64 *)&v123 )
    j_j___libc_free_0((unsigned __int64)v51);
  if ( v115 != &v117 )
    j_j___libc_free_0((unsigned __int64)v115);
  v118 = &v120;
  sub_2434550((__int64 *)&v118, v109, (__int64)&v109[v110]);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v119) <= 5 )
LABEL_113:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)&v118, "memset", 6u);
  v53 = sub_2241490((unsigned __int64 *)&v118, (char *)v100, v99);
  v121 = &v123;
  if ( (unsigned __int64 *)*v53 == v53 + 2 )
  {
    v123 = _mm_loadu_si128((const __m128i *)v53 + 1);
  }
  else
  {
    v121 = (__m128i *)*v53;
    v123.m128i_i64[0] = v53[2];
  }
  v122 = v53[1];
  *v53 = (unsigned __int64)(v53 + 2);
  v53[1] = 0;
  *((_BYTE *)v53 + 16) = 0;
  v54 = sub_BA8CA0(a2, (__int64)v121, v122, v93);
  v55 = (__int64 *)v121;
  a1[53] = v54;
  a1[54] = v56;
  if ( v55 != (__int64 *)&v123 )
    j_j___libc_free_0((unsigned __int64)v55);
  if ( v118 != &v120 )
    j_j___libc_free_0((unsigned __int64)v118);
  v57 = a1[15];
  v58 = a1[17];
  v121 = &v123;
  v59 = a1[16];
  v60 = (__int64 *)a1[14];
  v123.m128i_i64[1] = v58;
  v123.m128i_i64[0] = v59;
  v124 = v57;
  v122 = 0x300000003LL;
  v61 = sub_BCF480(v60, &v123, 3, 0);
  v62 = sub_BA8C10(a2, (__int64)"__hwasan_tag_memory", 0x13u, v61, 0);
  v64 = v63;
  if ( v121 != &v123 )
    _libc_free((unsigned __int64)v121);
  a1[57] = v62;
  a1[58] = v64;
  v65 = (__int64 *)a1[17];
  v121 = &v123;
  v122 = 0;
  v66 = sub_BCF480(v65, &v123, 0, 0);
  v67 = sub_BA8C10(a2, (__int64)"__hwasan_generate_tag", 0x15u, v66, 0);
  v69 = v68;
  if ( v121 != &v123 )
    _libc_free((unsigned __int64)v121);
  v70 = a1[19];
  v71 = (__int64 *)a1[14];
  a1[59] = v67;
  a1[60] = v69;
  v123.m128i_i64[0] = v70;
  v121 = &v123;
  v122 = 0x100000001LL;
  v72 = sub_BCF480(v71, &v123, 1, 0);
  v73 = sub_BA8C10(a2, (__int64)"__hwasan_add_frame_record", 0x19u, v72, 0);
  v75 = v74;
  if ( v121 != &v123 )
    _libc_free((unsigned __int64)v121);
  a1[61] = v73;
  v76 = (__int64 *)a1[17];
  a1[62] = v75;
  v77 = sub_BCD420(v76, 0);
  v78 = sub_BA8D60(a2, (__int64)"__hwasan_shadow", 0xFu, (__int64)v77);
  v79 = (__int64 *)a1[14];
  a1[63] = (__int64)v78;
  v80 = a1[15];
  v121 = &v123;
  v123.m128i_i64[0] = v80;
  v122 = 0x100000001LL;
  v81 = sub_BCF480(v79, &v123, 1, 0);
  v82 = sub_BA8C10(a2, (__int64)"__hwasan_handle_vfork", 0x15u, v81, 0);
  v84 = v83;
  if ( v121 != &v123 )
    _libc_free((unsigned __int64)v121);
  a1[55] = v82;
  v85 = v109;
  a1[56] = v84;
  if ( v85 != v111 )
    j_j___libc_free_0((unsigned __int64)v85);
  nullsub_61();
  v140 = &unk_49DA100;
  nullsub_63();
  if ( v125 != v127 )
    _libc_free((unsigned __int64)v125);
}
