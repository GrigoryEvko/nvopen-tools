// Function: sub_11F8520
// Address: 0x11f8520
//
void *__fastcall sub_11F8520(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r9
  _QWORD *v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  size_t v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // r12
  unsigned __int64 v18; // rax
  unsigned int v19; // edx
  int v20; // eax
  unsigned __int64 v21; // r13
  __int64 v22; // rdi
  __m128i *v23; // rdx
  __m128i si128; // xmm0
  __int64 v25; // rdx
  __m128i v26; // xmm0
  __int64 v27; // rax
  _WORD *v28; // rdx
  __int64 v29; // r13
  _QWORD *v30; // rdx
  char v31; // al
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rdi
  _WORD *v35; // rdx
  unsigned __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rdx
  unsigned __int64 v39; // rax
  __int64 v40; // r12
  unsigned int v41; // r15d
  __int64 v42; // rax
  unsigned __int64 v43; // r13
  size_t v44; // rax
  int v45; // esi
  __int64 v46; // rdi
  __int64 v47; // rdx
  __int64 v48; // rdi
  _QWORD *v49; // rdx
  __int64 v50; // rdi
  _WORD *v51; // rdx
  void *result; // rax
  unsigned __int64 v53; // r13
  __int64 v54; // rcx
  __int64 v55; // rcx
  __int64 v56; // rcx
  __int64 v57; // rax
  size_t v58; // rcx
  __int64 v59; // rax
  size_t v60; // rcx
  __int64 v61; // rsi
  __int64 v62; // rdx
  __int64 v63; // rax
  size_t v64; // rcx
  __m128i *v65; // rax
  __int64 v66; // rcx
  __m128i *v67; // rax
  __int64 v68; // rcx
  __m128i *v69; // rdi
  __int64 v70; // rdi
  _BYTE *v71; // rax
  __int64 v72; // rdi
  _WORD *v73; // rdx
  __int64 v74; // rdi
  _WORD *v75; // rdx
  __int64 v76; // rdi
  _BYTE *v77; // rax
  __int64 v78; // rdi
  _BYTE *v79; // rax
  unsigned int v80; // r15d
  __int64 v81; // rax
  size_t v82; // r13
  int v83; // r13d
  __int64 v84; // rdi
  __int64 v85; // rdx
  __int64 v86; // rdi
  _QWORD *v87; // rdx
  __int64 v88; // rdi
  _WORD *v89; // rdx
  __int64 v90; // rdi
  _WORD *v91; // rdx
  __int64 v92; // rdi
  _BYTE *v93; // rax
  __int64 v94; // rdi
  _BYTE *v95; // rax
  __int64 v96; // rdi
  _BYTE *v97; // rax
  __int64 v98; // rdi
  _BYTE *v99; // rax
  __int64 v100; // rdi
  _BYTE *v101; // rax
  __int64 v102; // rax
  __int64 v103; // rax
  size_t v104; // [rsp+20h] [rbp-170h]
  unsigned __int64 v105; // [rsp+28h] [rbp-168h]
  unsigned __int64 v106; // [rsp+30h] [rbp-160h]
  int v107; // [rsp+38h] [rbp-158h]
  _QWORD v108[2]; // [rsp+40h] [rbp-150h] BYREF
  __int64 v109; // [rsp+50h] [rbp-140h] BYREF
  __int64 *v110; // [rsp+60h] [rbp-130h] BYREF
  __int64 v111; // [rsp+68h] [rbp-128h]
  __int64 v112; // [rsp+70h] [rbp-120h] BYREF
  __m128i *v113; // [rsp+80h] [rbp-110h]
  size_t v114; // [rsp+88h] [rbp-108h]
  __m128i v115; // [rsp+90h] [rbp-100h] BYREF
  __m128i *v116; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v117; // [rsp+A8h] [rbp-E8h]
  __m128i v118; // [rsp+B0h] [rbp-E0h] BYREF
  __m128i *v119; // [rsp+C0h] [rbp-D0h] BYREF
  size_t v120; // [rsp+C8h] [rbp-C8h]
  __m128i v121; // [rsp+D0h] [rbp-C0h] BYREF
  __m128i *v122; // [rsp+E0h] [rbp-B0h] BYREF
  size_t v123; // [rsp+E8h] [rbp-A8h]
  __m128i v124; // [rsp+F0h] [rbp-A0h] BYREF
  __m128i *v125; // [rsp+100h] [rbp-90h] BYREF
  size_t v126; // [rsp+108h] [rbp-88h]
  __m128i v127; // [rsp+110h] [rbp-80h] BYREF
  unsigned __int8 *v128; // [rsp+120h] [rbp-70h] BYREF
  size_t v129; // [rsp+128h] [rbp-68h]
  _QWORD v130[12]; // [rsp+130h] [rbp-60h] BYREF

  v4 = a1[1];
  v5 = *(_QWORD *)v4;
  if ( *(_BYTE *)(*(_QWORD *)v4 + 40LL) )
  {
    v53 = sub_FDD860(*(__int64 **)(v5 + 8), a2);
    sub_11FCE30(v108, v53, *(_QWORD *)(v5 + 32));
    if ( v53 <= *(_QWORD *)(v5 + 32) >> 1 )
      sub_11FCC80(&v110, 0.0);
    else
      sub_11FCC80(&v110, 1.0);
    v128 = (unsigned __int8 *)v130;
    v129 = 0;
    LOBYTE(v130[0]) = 0;
    sub_2240E30(&v128, v111 + 7);
    if ( 0x3FFFFFFFFFFFFFFFLL - v129 <= 6 )
      goto LABEL_195;
    sub_2241490(&v128, "color=\"", 7, v54);
    sub_2241490(&v128, v110, v111, v55);
    if ( 0x3FFFFFFFFFFFFFFFLL - v129 <= 0x11 )
      goto LABEL_195;
    v57 = sub_2241490(&v128, "ff\", style=filled,", 18, v56);
    v125 = &v127;
    if ( *(_QWORD *)v57 == v57 + 16 )
    {
      v127 = _mm_loadu_si128((const __m128i *)(v57 + 16));
    }
    else
    {
      v125 = *(__m128i **)v57;
      v127.m128i_i64[0] = *(_QWORD *)(v57 + 16);
    }
    v58 = *(_QWORD *)(v57 + 8);
    v126 = v58;
    *(_QWORD *)v57 = v57 + 16;
    *(_QWORD *)(v57 + 8) = 0;
    *(_BYTE *)(v57 + 16) = 0;
    if ( 0x3FFFFFFFFFFFFFFFLL - v126 <= 0xB )
      goto LABEL_195;
    v59 = sub_2241490(&v125, " fillcolor=\"", 12, v58);
    v122 = &v124;
    if ( *(_QWORD *)v59 == v59 + 16 )
    {
      v124 = _mm_loadu_si128((const __m128i *)(v59 + 16));
    }
    else
    {
      v122 = *(__m128i **)v59;
      v124.m128i_i64[0] = *(_QWORD *)(v59 + 16);
    }
    v123 = *(_QWORD *)(v59 + 8);
    v60 = v123;
    *(_QWORD *)v59 = v59 + 16;
    v61 = v108[0];
    *(_QWORD *)(v59 + 8) = 0;
    v62 = v108[1];
    *(_BYTE *)(v59 + 16) = 0;
    v63 = sub_2241490(&v122, v61, v62, v60);
    v119 = &v121;
    if ( *(_QWORD *)v63 == v63 + 16 )
    {
      v121 = _mm_loadu_si128((const __m128i *)(v63 + 16));
    }
    else
    {
      v119 = *(__m128i **)v63;
      v121.m128i_i64[0] = *(_QWORD *)(v63 + 16);
    }
    v64 = *(_QWORD *)(v63 + 8);
    v120 = v64;
    *(_QWORD *)v63 = v63 + 16;
    *(_QWORD *)(v63 + 8) = 0;
    *(_BYTE *)(v63 + 16) = 0;
    if ( 0x3FFFFFFFFFFFFFFFLL - v120 <= 2 )
      goto LABEL_195;
    v65 = (__m128i *)sub_2241490(&v119, "70\"", 3, v64);
    v116 = &v118;
    if ( (__m128i *)v65->m128i_i64[0] == &v65[1] )
    {
      v118 = _mm_loadu_si128(v65 + 1);
    }
    else
    {
      v116 = (__m128i *)v65->m128i_i64[0];
      v118.m128i_i64[0] = v65[1].m128i_i64[0];
    }
    v66 = v65->m128i_i64[1];
    v65[1].m128i_i8[0] = 0;
    v117 = v66;
    v65->m128i_i64[0] = (__int64)v65[1].m128i_i64;
    v65->m128i_i64[1] = 0;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v117) <= 0x12 )
LABEL_195:
      sub_4262D8((__int64)"basic_string::append");
    v67 = (__m128i *)sub_2241490(&v116, " fontname=\"Courier\"", 19, v66);
    v113 = &v115;
    if ( (__m128i *)v67->m128i_i64[0] == &v67[1] )
    {
      v115 = _mm_loadu_si128(v67 + 1);
    }
    else
    {
      v113 = (__m128i *)v67->m128i_i64[0];
      v115.m128i_i64[0] = v67[1].m128i_i64[0];
    }
    v68 = v67->m128i_i64[1];
    v67[1].m128i_i8[0] = 0;
    v114 = v68;
    v67->m128i_i64[0] = (__int64)v67[1].m128i_i64;
    v69 = v116;
    v67->m128i_i64[1] = 0;
    if ( v69 != &v118 )
      j_j___libc_free_0(v69, v118.m128i_i64[0] + 1);
    if ( v119 != &v121 )
      j_j___libc_free_0(v119, v121.m128i_i64[0] + 1);
    if ( v122 != &v124 )
      j_j___libc_free_0(v122, v124.m128i_i64[0] + 1);
    if ( v125 != &v127 )
      j_j___libc_free_0(v125, v127.m128i_i64[0] + 1);
    if ( v128 != (unsigned __int8 *)v130 )
      j_j___libc_free_0(v128, v130[0] + 1LL);
    v119 = &v121;
    if ( v113 == &v115 )
    {
      v121 = _mm_load_si128(&v115);
    }
    else
    {
      v119 = v113;
      v121.m128i_i64[0] = v115.m128i_i64[0];
    }
    v120 = v114;
    if ( v110 != &v112 )
      j_j___libc_free_0(v110, v112 + 1);
    if ( (__int64 *)v108[0] != &v109 )
      j_j___libc_free_0(v108[0], v109 + 1);
  }
  else
  {
    v121.m128i_i8[0] = 0;
    v119 = &v121;
    v120 = 0;
  }
  v6 = *a1;
  v7 = *(_QWORD *)(*a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 24) - v7) <= 4 )
  {
    v6 = sub_CB6200(v6, "\tNode", 5u);
  }
  else
  {
    *(_DWORD *)v7 = 1685016073;
    *(_BYTE *)(v7 + 4) = 101;
    *(_QWORD *)(v6 + 32) += 5LL;
  }
  v8 = sub_CB5A80(v6, a2);
  v10 = *(_QWORD **)(v8 + 32);
  if ( *(_QWORD *)(v8 + 24) - (_QWORD)v10 <= 7u )
  {
    sub_CB6200(v8, " [shape=", 8u);
  }
  else
  {
    *v10 = 0x3D65706168735B20LL;
    *(_QWORD *)(v8 + 32) += 8LL;
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
LABEL_10:
    v14 = v120;
    if ( !v120 )
      goto LABEL_11;
LABEL_106:
    v70 = sub_CB6200(*a1, (unsigned __int8 *)v119, v14);
    v71 = *(_BYTE **)(v70 + 32);
    if ( *(_BYTE **)(v70 + 24) == v71 )
    {
      sub_CB6200(v70, (unsigned __int8 *)",", 1u);
    }
    else
    {
      *v71 = 44;
      ++*(_QWORD *)(v70 + 32);
    }
    goto LABEL_11;
  }
  if ( v13 <= 6 )
  {
    sub_CB6200(v11, (unsigned __int8 *)"record,", 7u);
    goto LABEL_10;
  }
  *(_DWORD *)v12 = 1868785010;
  *(_WORD *)(v12 + 4) = 25714;
  *(_BYTE *)(v12 + 6) = 44;
  *(_QWORD *)(v11 + 32) += 7LL;
  v14 = v120;
  if ( v120 )
    goto LABEL_106;
LABEL_11:
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
  v17 = a2 + 48;
  if ( *((_BYTE *)a1 + 16) )
  {
    v18 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v18 == v17 )
      goto LABEL_186;
    if ( !v18 )
      goto LABEL_134;
    if ( (unsigned int)*(unsigned __int8 *)(v18 - 24) - 30 > 0xA || (v19 = sub_B46E30(v18 - 24)) == 0 )
    {
LABEL_186:
      v21 = 1;
    }
    else
    {
      v20 = 0;
      do
      {
        if ( v19 == ++v20 )
        {
          v21 = v19;
          goto LABEL_22;
        }
      }
      while ( v20 != 64 );
      v21 = 65;
    }
LABEL_22:
    v22 = *a1;
    v23 = *(__m128i **)(*a1 + 32);
    if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v23 <= 0x30u )
    {
      v102 = sub_CB6200(v22, "<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"", 0x31u);
      v25 = *(_QWORD *)(v102 + 32);
      v22 = v102;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB60);
      v23[3].m128i_i8[0] = 34;
      *v23 = si128;
      v23[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CB70);
      v23[2] = _mm_load_si128((const __m128i *)&xmmword_3F8CB80);
      v25 = *(_QWORD *)(v22 + 32) + 49LL;
      *(_QWORD *)(v22 + 32) = v25;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v22 + 24) - v25) <= 0x2E )
    {
      v22 = sub_CB6200(v22, " cellpadding=\"0\"><tr><td align=\"text\" colspan=\"", 0x2Fu);
    }
    else
    {
      v26 = _mm_load_si128((const __m128i *)&xmmword_3F8CB90);
      qmemcpy((void *)(v25 + 32), "text\" colspan=\"", 15);
      *(__m128i *)v25 = v26;
      *(__m128i *)(v25 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F8CBA0);
      *(_QWORD *)(v22 + 32) += 47LL;
    }
    v27 = sub_CB59D0(v22, v21);
    v28 = *(_WORD **)(v27 + 32);
    if ( *(_QWORD *)(v27 + 24) - (_QWORD)v28 <= 1u )
    {
      sub_CB6200(v27, "\">", 2u);
    }
    else
    {
      v9 = 15906;
      *v28 = 15906;
      *(_QWORD *)(v27 + 32) += 2LL;
    }
  }
  else
  {
    v72 = *a1;
    v73 = *(_WORD **)(*a1 + 32);
    if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v73 <= 1u )
    {
      sub_CB6200(v72, (unsigned __int8 *)"\"{", 2u);
    }
    else
    {
      *v73 = 31522;
      *(_QWORD *)(v72 + 32) += 2LL;
    }
  }
  v29 = *a1;
  v30 = *(_QWORD **)a1[1];
  v31 = *((_BYTE *)a1 + 24);
  if ( *((_BYTE *)a1 + 16) )
  {
    if ( v31 )
      sub_11F3900((__int64)&v128, (unsigned __int8 *)a2);
    else
      sub_11F8430(
        (__int64 *)&v128,
        a2,
        v30,
        0,
        0,
        v9,
        (void (__fastcall *)(__int64, __int64 *, unsigned int *, __int64))sub_11F32A0,
        (__int64)sub_11F32F0);
    v32 = sub_CB6200(v29, v128, v129);
    v33 = *(_QWORD *)(v32 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v32 + 24) - v33) <= 4 )
    {
      sub_CB6200(v32, "</td>", 5u);
    }
    else
    {
      *(_DWORD *)v33 = 1685335868;
      *(_BYTE *)(v33 + 4) = 62;
      *(_QWORD *)(v32 + 32) += 5LL;
    }
    if ( v128 != (unsigned __int8 *)v130 )
      j_j___libc_free_0(v128, v130[0] + 1LL);
  }
  else
  {
    if ( v31 )
      sub_11F3900((__int64)&v125, (unsigned __int8 *)a2);
    else
      sub_11F8430(
        (__int64 *)&v125,
        a2,
        v30,
        0,
        0,
        v9,
        (void (__fastcall *)(__int64, __int64 *, unsigned int *, __int64))sub_11F32A0,
        (__int64)sub_11F32F0);
    sub_C67200((__int64 *)&v128, (__int64)&v125);
    sub_CB6200(v29, v128, v129);
    if ( v128 != (unsigned __int8 *)v130 )
      j_j___libc_free_0(v128, v130[0] + 1LL);
    if ( v125 != &v127 )
      j_j___libc_free_0(v125, v127.m128i_i64[0] + 1);
  }
  v122 = &v124;
  v130[3] = 0x100000000LL;
  v123 = 0;
  v124.m128i_i8[0] = 0;
  v128 = (unsigned __int8 *)&unk_49DD210;
  v129 = 0;
  memset(v130, 0, 24);
  v130[4] = &v122;
  sub_CB5980((__int64)&v128, 0, 0, 0);
  if ( (unsigned __int8)sub_11F5CE0((__int64)a1, (__int64)&v128, a2) )
  {
    if ( *((_BYTE *)a1 + 16) )
      goto LABEL_37;
    v96 = *a1;
    v97 = *(_BYTE **)(*a1 + 32);
    if ( *(_BYTE **)(*a1 + 24) == v97 )
    {
      sub_CB6200(v96, (unsigned __int8 *)"|", 1u);
    }
    else
    {
      *v97 = 124;
      ++*(_QWORD *)(v96 + 32);
    }
    v98 = *a1;
    if ( *((_BYTE *)a1 + 16) )
    {
LABEL_37:
      sub_CB6200(*a1, (unsigned __int8 *)v122, v123);
    }
    else
    {
      v99 = *(_BYTE **)(v98 + 32);
      if ( *(_BYTE **)(v98 + 24) == v99 )
      {
        v98 = sub_CB6200(v98, (unsigned __int8 *)"{", 1u);
      }
      else
      {
        *v99 = 123;
        ++*(_QWORD *)(v98 + 32);
      }
      v100 = sub_CB6200(v98, (unsigned __int8 *)v122, v123);
      v101 = *(_BYTE **)(v100 + 32);
      if ( *(_BYTE **)(v100 + 24) == v101 )
      {
        sub_CB6200(v100, (unsigned __int8 *)"}", 1u);
      }
      else
      {
        *v101 = 125;
        ++*(_QWORD *)(v100 + 32);
      }
    }
  }
  v34 = *a1;
  v35 = *(_WORD **)(*a1 + 32);
  v36 = *(_QWORD *)(*a1 + 24) - (_QWORD)v35;
  if ( *((_BYTE *)a1 + 16) )
  {
    if ( v36 <= 0xD )
    {
      sub_CB6200(v34, "</tr></table>>", 0xEu);
    }
    else
    {
      qmemcpy(v35, "</tr></table>>", 14);
      *(_QWORD *)(v34 + 32) += 14LL;
    }
  }
  else if ( v36 <= 1 )
  {
    sub_CB6200(v34, (unsigned __int8 *)"}\"", 2u);
  }
  else
  {
    *v35 = 8829;
    *(_QWORD *)(v34 + 32) += 2LL;
  }
  v37 = *a1;
  v38 = *(_QWORD *)(*a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(*a1 + 24) - v38) <= 2 )
  {
    sub_CB6200(v37, (unsigned __int8 *)"];\n", 3u);
  }
  else
  {
    *(_BYTE *)(v38 + 2) = 10;
    *(_WORD *)v38 = 15197;
    *(_QWORD *)(v37 + 32) += 3LL;
  }
  v39 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v39 == v17 )
    goto LABEL_66;
  if ( !v39 )
LABEL_134:
    BUG();
  v40 = v39 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v39 - 24) - 30 <= 0xA )
  {
    v107 = sub_B46E30(v40);
    if ( v107 )
    {
      v106 = a2;
      v41 = 0;
      do
      {
        v42 = sub_B46EC0(v40, v41);
        if ( (unsigned __int8)sub_11F7FC0((__int64)(a1 + 3), v42, *(_QWORD *)a1[1]) )
          goto LABEL_48;
        v43 = sub_B46EC0(v40, v41);
        if ( !v43 )
          goto LABEL_48;
        sub_11F3610((__int64)&v125, v106, v41);
        v44 = v126;
        if ( v125 != &v127 )
        {
          v104 = v126;
          j_j___libc_free_0(v125, v127.m128i_i64[0] + 1);
          v44 = v104;
        }
        v45 = -1;
        if ( v44 )
          v45 = v41;
        sub_11F46F0((__int64)&v125, v106, v41, *(_QWORD *)a1[1]);
        v46 = *a1;
        v47 = *(_QWORD *)(*a1 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(*a1 + 24) - v47) <= 4 )
        {
          v46 = sub_CB6200(v46, "\tNode", 5u);
        }
        else
        {
          *(_DWORD *)v47 = 1685016073;
          *(_BYTE *)(v47 + 4) = 101;
          *(_QWORD *)(v46 + 32) += 5LL;
        }
        sub_CB5A80(v46, v106);
        if ( v45 != -1 )
        {
          v74 = *a1;
          v75 = *(_WORD **)(*a1 + 32);
          if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v75 <= 1u )
          {
            v103 = sub_CB6200(v74, ":s", 2u);
            sub_CB59F0(v103, v45);
          }
          else
          {
            *v75 = 29498;
            *(_QWORD *)(v74 + 32) += 2LL;
            sub_CB59F0(v74, v45);
          }
        }
        v48 = *a1;
        v49 = *(_QWORD **)(*a1 + 32);
        if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v49 <= 7u )
        {
          v48 = sub_CB6200(v48, " -> Node", 8u);
        }
        else
        {
          *v49 = 0x65646F4E203E2D20LL;
          *(_QWORD *)(v48 + 32) += 8LL;
        }
        sub_CB5A80(v48, v43);
        if ( v126 )
        {
          v76 = *a1;
          v77 = *(_BYTE **)(*a1 + 32);
          if ( *(_BYTE **)(*a1 + 24) == v77 )
          {
            v76 = sub_CB6200(v76, (unsigned __int8 *)"[", 1u);
          }
          else
          {
            *v77 = 91;
            ++*(_QWORD *)(v76 + 32);
          }
          v78 = sub_CB6200(v76, (unsigned __int8 *)v125, v126);
          v79 = *(_BYTE **)(v78 + 32);
          if ( *(_BYTE **)(v78 + 24) == v79 )
          {
            sub_CB6200(v78, (unsigned __int8 *)"]", 1u);
          }
          else
          {
            *v79 = 93;
            ++*(_QWORD *)(v78 + 32);
          }
        }
        v50 = *a1;
        v51 = *(_WORD **)(*a1 + 32);
        if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v51 <= 1u )
        {
          sub_CB6200(v50, (unsigned __int8 *)";\n", 2u);
        }
        else
        {
          *v51 = 2619;
          *(_QWORD *)(v50 + 32) += 2LL;
        }
        if ( v125 == &v127 )
        {
LABEL_48:
          if ( ++v41 == v107 )
            goto LABEL_66;
        }
        else
        {
          ++v41;
          j_j___libc_free_0(v125, v127.m128i_i64[0] + 1);
          if ( v41 == v107 )
            goto LABEL_66;
        }
      }
      while ( v41 != 64 );
      v80 = 64;
      do
      {
        v81 = sub_B46EC0(v40, v80);
        if ( !(unsigned __int8)sub_11F7FC0((__int64)(a1 + 3), v81, *(_QWORD *)a1[1]) )
        {
          v105 = sub_B46EC0(v40, v80);
          if ( v105 )
          {
            sub_11F3610((__int64)&v125, v106, v80);
            v82 = v126;
            if ( v125 != &v127 )
              j_j___libc_free_0(v125, v127.m128i_i64[0] + 1);
            v83 = v82 == 0 ? -1 : 0x40;
            sub_11F46F0((__int64)&v125, v106, v80, *(_QWORD *)a1[1]);
            v84 = *a1;
            v85 = *(_QWORD *)(*a1 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(*a1 + 24) - v85) <= 4 )
            {
              v84 = sub_CB6200(v84, "\tNode", 5u);
            }
            else
            {
              *(_DWORD *)v85 = 1685016073;
              *(_BYTE *)(v85 + 4) = 101;
              *(_QWORD *)(v84 + 32) += 5LL;
            }
            sub_CB5A80(v84, v106);
            if ( v83 != -1 )
            {
              v90 = *a1;
              v91 = *(_WORD **)(*a1 + 32);
              if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v91 <= 1u )
              {
                v90 = sub_CB6200(v90, ":s", 2u);
              }
              else
              {
                *v91 = 29498;
                *(_QWORD *)(v90 + 32) += 2LL;
              }
              sub_CB59F0(v90, 64);
            }
            v86 = *a1;
            v87 = *(_QWORD **)(*a1 + 32);
            if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v87 <= 7u )
            {
              v86 = sub_CB6200(v86, " -> Node", 8u);
            }
            else
            {
              *v87 = 0x65646F4E203E2D20LL;
              *(_QWORD *)(v86 + 32) += 8LL;
            }
            sub_CB5A80(v86, v105);
            if ( v126 )
            {
              v92 = *a1;
              v93 = *(_BYTE **)(*a1 + 32);
              if ( *(_BYTE **)(*a1 + 24) == v93 )
              {
                v92 = sub_CB6200(v92, (unsigned __int8 *)"[", 1u);
              }
              else
              {
                *v93 = 91;
                ++*(_QWORD *)(v92 + 32);
              }
              v94 = sub_CB6200(v92, (unsigned __int8 *)v125, v126);
              v95 = *(_BYTE **)(v94 + 32);
              if ( *(_BYTE **)(v94 + 24) == v95 )
              {
                sub_CB6200(v94, (unsigned __int8 *)"]", 1u);
              }
              else
              {
                *v95 = 93;
                ++*(_QWORD *)(v94 + 32);
              }
            }
            v88 = *a1;
            v89 = *(_WORD **)(*a1 + 32);
            if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v89 <= 1u )
            {
              sub_CB6200(v88, (unsigned __int8 *)";\n", 2u);
            }
            else
            {
              *v89 = 2619;
              *(_QWORD *)(v88 + 32) += 2LL;
            }
            if ( v125 != &v127 )
              j_j___libc_free_0(v125, v127.m128i_i64[0] + 1);
          }
        }
        ++v80;
      }
      while ( v107 != v80 );
    }
  }
LABEL_66:
  v128 = (unsigned __int8 *)&unk_49DD210;
  result = sub_CB5840((__int64)&v128);
  if ( v122 != &v124 )
    result = (void *)j_j___libc_free_0(v122, v124.m128i_i64[0] + 1);
  if ( v119 != &v121 )
    return (void *)j_j___libc_free_0(v119, v121.m128i_i64[0] + 1);
  return result;
}
