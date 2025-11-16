// Function: sub_DF0A60
// Address: 0xdf0a60
//
__int64 __fastcall sub_DF0A60(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r13
  __m128i *v4; // rdx
  __m128i v5; // xmm0
  __m128i v6; // xmm0
  _BYTE *v7; // rax
  __int64 result; // rax
  __int64 *v9; // r12
  __int64 *j; // rbx
  __int64 v11; // rdx
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // r14
  __int64 i; // r13
  __int64 v18; // rax
  _BYTE *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rax
  int v22; // ecx
  __int64 v23; // rdi
  int v24; // ecx
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // r10
  __int64 *v28; // r15
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r15
  __int64 *v32; // r15
  __m128i *v33; // rdx
  __m128i si128; // xmm0
  __int64 v35; // rbx
  _WORD *v36; // rdx
  __int64 v37; // r15
  int v38; // eax
  _WORD *v39; // rdx
  __m128i *v40; // rax
  __m128i *v41; // rax
  __m128i *v42; // rax
  __int64 v43; // rax
  __m128i *v44; // rax
  _BYTE *v45; // rsi
  __m128i *v46; // rax
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __m128i *v51; // rax
  __m128i *v52; // rax
  __int8 *v53; // rax
  const __m128i *v54; // rsi
  __int64 v55; // rdx
  __int64 v56; // r8
  __int64 v57; // r9
  const __m128i *v58; // rax
  const __m128i *v59; // rdi
  unsigned __int64 v60; // rcx
  __int64 v61; // rax
  __m128i *v62; // rdx
  __int64 v63; // rcx
  __m128i *v64; // rsi
  const __m128i *v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  const __m128i *v68; // rcx
  __m128i *v69; // r10
  unsigned __int64 v70; // rbx
  __int64 v71; // rax
  __m128i *v72; // rdx
  const __m128i *v73; // rax
  __m128i *v74; // rax
  __m128i *v75; // r13
  __int64 v76; // rcx
  __int64 v77; // r14
  char *v78; // rbx
  __int64 *v79; // rax
  __int64 *v80; // rdx
  __int64 v81; // r14
  __int64 *v82; // rax
  char v83; // dl
  __m128i *v84; // rdx
  __int64 v85; // rsi
  _WORD *v86; // rdx
  __int64 v87; // r13
  int v88; // eax
  __int64 v89; // r15
  __int64 v90; // r15
  int v91; // eax
  int v92; // r8d
  __int64 v93; // [rsp+18h] [rbp-358h]
  unsigned __int64 v94; // [rsp+20h] [rbp-350h]
  __int64 v95; // [rsp+20h] [rbp-350h]
  __int64 v96; // [rsp+28h] [rbp-348h]
  char *v97; // [rsp+30h] [rbp-340h]
  __int64 *v98; // [rsp+38h] [rbp-338h]
  __int64 v99; // [rsp+40h] [rbp-330h]
  _QWORD v101[16]; // [rsp+50h] [rbp-320h] BYREF
  __m128i v102; // [rsp+D0h] [rbp-2A0h] BYREF
  __int64 v103; // [rsp+E0h] [rbp-290h]
  int v104; // [rsp+E8h] [rbp-288h]
  char v105; // [rsp+ECh] [rbp-284h]
  _QWORD v106[8]; // [rsp+F0h] [rbp-280h] BYREF
  __m128i *v107; // [rsp+130h] [rbp-240h] BYREF
  __m128i *v108; // [rsp+138h] [rbp-238h]
  __int64 v109; // [rsp+140h] [rbp-230h]
  __int64 v110; // [rsp+150h] [rbp-220h] BYREF
  __int64 *v111; // [rsp+158h] [rbp-218h]
  unsigned int v112; // [rsp+160h] [rbp-210h]
  unsigned int v113; // [rsp+164h] [rbp-20Ch]
  char v114; // [rsp+16Ch] [rbp-204h]
  _BYTE v115[64]; // [rsp+170h] [rbp-200h] BYREF
  __m128i *v116; // [rsp+1B0h] [rbp-1C0h] BYREF
  __m128i *v117; // [rsp+1B8h] [rbp-1B8h]
  __int64 v118; // [rsp+1C0h] [rbp-1B0h]
  char v119[8]; // [rsp+1D0h] [rbp-1A0h] BYREF
  __int64 v120; // [rsp+1D8h] [rbp-198h]
  char v121; // [rsp+1ECh] [rbp-184h]
  _BYTE v122[64]; // [rsp+1F0h] [rbp-180h] BYREF
  __m128i *v123; // [rsp+230h] [rbp-140h]
  __m128i *v124; // [rsp+238h] [rbp-138h]
  __int8 *v125; // [rsp+240h] [rbp-130h]
  __m128i v126; // [rsp+250h] [rbp-120h] BYREF
  __int64 v127; // [rsp+260h] [rbp-110h] BYREF
  unsigned int v128; // [rsp+268h] [rbp-108h]
  char v129; // [rsp+26Ch] [rbp-104h]
  char v130[64]; // [rsp+270h] [rbp-100h] BYREF
  const __m128i *v131; // [rsp+2B0h] [rbp-C0h]
  const __m128i *v132; // [rsp+2B8h] [rbp-B8h]
  __int64 v133; // [rsp+2C0h] [rbp-B0h]
  char v134[8]; // [rsp+2C8h] [rbp-A8h] BYREF
  __int64 v135; // [rsp+2D0h] [rbp-A0h]
  char v136; // [rsp+2E4h] [rbp-8Ch]
  _BYTE v137[64]; // [rsp+2E8h] [rbp-88h] BYREF
  const __m128i *v138; // [rsp+328h] [rbp-48h]
  const __m128i *v139; // [rsp+330h] [rbp-40h]
  __int8 *v140; // [rsp+338h] [rbp-38h]

  v2 = a2;
  v3 = a1;
  if ( !(_BYTE)qword_4F88CE8 )
    goto LABEL_2;
  sub_904010(a2, "Classifying expressions for: ");
  sub_A5BF40(*(unsigned __int8 **)a1, a2, 0, 0);
  sub_904010(a2, "\n");
  v12 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  v13 = *(_QWORD *)a1 + 72LL;
  v99 = v13;
  if ( v13 == v12 )
  {
    v14 = 0;
  }
  else
  {
    do
    {
      if ( !v12 )
LABEL_178:
        BUG();
      v14 = *(_QWORD *)(v12 + 32);
      if ( v14 != v12 + 24 )
        break;
      v12 = *(_QWORD *)(v12 + 8);
    }
    while ( v13 != v12 );
  }
  v15 = v12;
  i = v14;
  while ( v15 != v99 )
  {
    if ( !i )
      BUG();
    if ( !sub_D97040(a1, *(_QWORD *)(i - 16)) || (unsigned __int8)(*(_BYTE *)(i - 24) - 82) <= 1u )
      goto LABEL_18;
    sub_A69870(i - 24, (_BYTE *)a2, 0);
    v19 = *(_BYTE **)(a2 + 32);
    if ( (unsigned __int64)v19 >= *(_QWORD *)(a2 + 24) )
    {
      sub_CB5D20(a2, 10);
    }
    else
    {
      *(_QWORD *)(a2 + 32) = v19 + 1;
      *v19 = 10;
    }
    sub_904010(a2, "  -->  ");
    v98 = sub_DD8400(a1, i - 24);
    sub_D955C0((__int64)v98, a2);
    if ( !sub_D96A50((__int64)v98) )
    {
      sub_904010(a2, " U: ");
      v89 = sub_DBB9F0(a1, (__int64)v98, 0, 0);
      v126.m128i_i32[2] = *(_DWORD *)(v89 + 8);
      if ( v126.m128i_i32[2] > 0x40u )
        sub_C43780((__int64)&v126, (const void **)v89);
      else
        v126.m128i_i64[0] = *(_QWORD *)v89;
      v128 = *(_DWORD *)(v89 + 24);
      if ( v128 > 0x40 )
        sub_C43780((__int64)&v127, (const void **)(v89 + 16));
      else
        v127 = *(_QWORD *)(v89 + 16);
      sub_ABE8C0((__int64)&v126, a2);
      if ( v128 > 0x40 && v127 )
        j_j___libc_free_0_0(v127);
      if ( v126.m128i_i32[2] > 0x40u && v126.m128i_i64[0] )
        j_j___libc_free_0_0(v126.m128i_i64[0]);
      sub_904010(a2, " S: ");
      v90 = sub_DBB9F0(a1, (__int64)v98, 1u, 0);
      v126.m128i_i32[2] = *(_DWORD *)(v90 + 8);
      if ( v126.m128i_i32[2] > 0x40u )
        sub_C43780((__int64)&v126, (const void **)v90);
      else
        v126.m128i_i64[0] = *(_QWORD *)v90;
      v128 = *(_DWORD *)(v90 + 24);
      if ( v128 > 0x40 )
        sub_C43780((__int64)&v127, (const void **)(v90 + 16));
      else
        v127 = *(_QWORD *)(v90 + 16);
      sub_ABE8C0((__int64)&v126, a2);
      if ( v128 > 0x40 && v127 )
        j_j___libc_free_0_0(v127);
      if ( v126.m128i_i32[2] > 0x40u && v126.m128i_i64[0] )
        j_j___libc_free_0_0(v126.m128i_i64[0]);
    }
    v20 = *(_QWORD *)(i + 16);
    v21 = *(_QWORD *)(a1 + 48);
    v22 = *(_DWORD *)(v21 + 24);
    v23 = *(_QWORD *)(v21 + 8);
    if ( v22 )
    {
      v24 = v22 - 1;
      v25 = v24 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v26 = (__int64 *)(v23 + 16LL * v25);
      v27 = *v26;
      if ( v20 == *v26 )
      {
LABEL_31:
        v97 = (char *)v26[1];
        v28 = sub_DDF4E0(a1, (__int64 **)v98, v97);
        if ( v98 == v28 )
          goto LABEL_42;
        sub_904010(a2, "  -->  ");
        sub_D955C0((__int64)v28, a2);
        if ( sub_D96A50((__int64)v28) )
          goto LABEL_42;
        goto LABEL_33;
      }
      v91 = 1;
      while ( v27 != -4096 )
      {
        v92 = v91 + 1;
        v25 = v24 & (v91 + v25);
        v26 = (__int64 *)(v23 + 16LL * v25);
        v27 = *v26;
        if ( v20 == *v26 )
          goto LABEL_31;
        v91 = v92;
      }
    }
    v28 = sub_DDF4E0(a1, (__int64 **)v98, 0);
    if ( v98 == v28 )
      goto LABEL_43;
    sub_904010(a2, "  -->  ");
    sub_D955C0((__int64)v28, a2);
    if ( sub_D96A50((__int64)v28) )
      goto LABEL_43;
    v97 = 0;
LABEL_33:
    sub_904010(a2, " U: ");
    v29 = sub_DBB9F0(a1, (__int64)v28, 0, 0);
    v126.m128i_i32[2] = *(_DWORD *)(v29 + 8);
    if ( v126.m128i_i32[2] > 0x40u )
    {
      v96 = v29;
      sub_C43780((__int64)&v126, (const void **)v29);
      v29 = v96;
    }
    else
    {
      v126.m128i_i64[0] = *(_QWORD *)v29;
    }
    v128 = *(_DWORD *)(v29 + 24);
    if ( v128 > 0x40 )
      sub_C43780((__int64)&v127, (const void **)(v29 + 16));
    else
      v127 = *(_QWORD *)(v29 + 16);
    sub_ABE8C0((__int64)&v126, a2);
    sub_969240(&v127);
    sub_969240(v126.m128i_i64);
    sub_904010(a2, " S: ");
    v30 = sub_DBB9F0(a1, (__int64)v28, 1u, 0);
    v31 = v30;
    v126.m128i_i32[2] = *(_DWORD *)(v30 + 8);
    if ( v126.m128i_i32[2] > 0x40u )
      sub_C43780((__int64)&v126, (const void **)v30);
    else
      v126.m128i_i64[0] = *(_QWORD *)v30;
    v128 = *(_DWORD *)(v31 + 24);
    if ( v128 > 0x40 )
      sub_C43780((__int64)&v127, (const void **)(v31 + 16));
    else
      v127 = *(_QWORD *)(v31 + 16);
    sub_ABE8C0((__int64)&v126, a2);
    sub_969240(&v127);
    sub_969240(v126.m128i_i64);
LABEL_42:
    if ( v97 )
    {
      sub_904010(a2, "\t\tExits: ");
      v32 = sub_DDF4E0(a1, (__int64 **)v98, *(char **)v97);
      if ( sub_DADE90(a1, (__int64)v32, (__int64)v97) )
        sub_D955C0((__int64)v32, a2);
      else
        sub_904010(a2, "<<Unknown>>");
      v33 = *(__m128i **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v33 <= 0x15u )
      {
        sub_CB6200(a2, "\t\tLoopDispositions: { ", 0x16u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F74EF0);
        v33[1].m128i_i32[0] = 540701550;
        v33[1].m128i_i16[2] = 8315;
        *v33 = si128;
        *(_QWORD *)(a2 + 32) += 22LL;
      }
      v35 = (__int64)v97;
      while ( 1 )
      {
        sub_A5BF40(**(unsigned __int8 ***)(v35 + 32), a2, 0, 0);
        v36 = *(_WORD **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v36 <= 1u )
        {
          v37 = sub_CB6200(a2, (unsigned __int8 *)": ", 2u);
        }
        else
        {
          v37 = a2;
          *v36 = 8250;
          *(_QWORD *)(a2 + 32) += 2LL;
        }
        v38 = sub_DAD860(a1, (__int64)v98, v35);
        sub_D9A040(v37, v38);
        v35 = *(_QWORD *)v35;
        if ( !v35 )
          break;
        v39 = *(_WORD **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v39 <= 1u )
        {
          sub_CB6200(a2, (unsigned __int8 *)", ", 2u);
        }
        else
        {
          *v39 = 8236;
          *(_QWORD *)(a2 + 32) += 2LL;
        }
      }
      memset(v101, 0, 0x78u);
      v103 = 0x100000008LL;
      v102.m128i_i64[1] = (__int64)v106;
      v101[1] = &v101[4];
      v106[0] = v97;
      v126.m128i_i64[0] = (__int64)v97;
      LODWORD(v101[2]) = 8;
      BYTE4(v101[3]) = 1;
      v107 = 0;
      v108 = 0;
      v109 = 0;
      v104 = 0;
      v105 = 1;
      v102.m128i_i64[0] = 1;
      LOBYTE(v127) = 0;
      sub_DAD500((__int64)&v107, &v126);
      sub_C8CF70((__int64)v119, v122, 8, (__int64)&v101[4], (__int64)v101);
      v40 = (__m128i *)v101[12];
      memset(&v101[12], 0, 24);
      v123 = v40;
      v124 = (__m128i *)v101[13];
      v125 = (__int8 *)v101[14];
      sub_C8CF70((__int64)&v110, v115, 8, (__int64)v106, (__int64)&v102);
      v41 = v107;
      v107 = 0;
      v116 = v41;
      v42 = v108;
      v108 = 0;
      v117 = v42;
      v43 = v109;
      v109 = 0;
      v118 = v43;
      sub_C8CF70((__int64)&v126, v130, 8, (__int64)v115, (__int64)&v110);
      v44 = v116;
      v45 = v137;
      v116 = 0;
      v131 = v44;
      v46 = v117;
      v117 = 0;
      v132 = v46;
      v47 = v118;
      v118 = 0;
      v133 = v47;
      sub_C8CF70((__int64)v134, v137, 8, (__int64)v122, (__int64)v119);
      v51 = v123;
      v123 = 0;
      v138 = v51;
      v52 = v124;
      v124 = 0;
      v139 = v52;
      v53 = v125;
      v125 = 0;
      v140 = v53;
      if ( v116 )
      {
        v45 = (_BYTE *)(v118 - (_QWORD)v116);
        j_j___libc_free_0(v116, v118 - (_QWORD)v116);
      }
      if ( !v114 )
        _libc_free(v111, v45);
      if ( v123 )
      {
        v45 = (_BYTE *)(v125 - (__int8 *)v123);
        j_j___libc_free_0(v123, v125 - (__int8 *)v123);
      }
      if ( !v121 )
        _libc_free(v120, v45);
      if ( v107 )
      {
        v45 = (_BYTE *)(v109 - (_QWORD)v107);
        j_j___libc_free_0(v107, v109 - (_QWORD)v107);
      }
      if ( !v105 )
        _libc_free(v102.m128i_i64[1], v45);
      if ( v101[12] )
      {
        v45 = (_BYTE *)(v101[14] - v101[12]);
        j_j___libc_free_0(v101[12], v101[14] - v101[12]);
      }
      if ( !BYTE4(v101[3]) )
        _libc_free(v101[1], v45);
      v54 = (const __m128i *)v115;
      sub_C8CD80((__int64)&v110, (__int64)v115, (__int64)&v126, v48, v49, v50);
      v58 = v132;
      v59 = v131;
      v116 = 0;
      v117 = 0;
      v118 = 0;
      v60 = (char *)v132 - (char *)v131;
      if ( v132 == v131 )
      {
        v62 = 0;
      }
      else
      {
        if ( v60 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_177;
        v94 = (char *)v132 - (char *)v131;
        v61 = sub_22077B0((char *)v132 - (char *)v131);
        v59 = v131;
        v60 = v94;
        v62 = (__m128i *)v61;
        v58 = v132;
      }
      v63 = (__int64)v62->m128i_i64 + v60;
      v116 = v62;
      v117 = v62;
      v118 = v63;
      if ( v59 != v58 )
      {
        v64 = v62;
        v65 = v59;
        do
        {
          if ( v64 )
          {
            *v64 = _mm_loadu_si128(v65);
            v56 = v65[1].m128i_i64[0];
            v64[1].m128i_i64[0] = v56;
          }
          v65 = (const __m128i *)((char *)v65 + 24);
          v64 = (__m128i *)((char *)v64 + 24);
        }
        while ( v58 != v65 );
        v63 = 0x1FFFFFFFFFFFFFFFLL;
        v62 = (__m128i *)((char *)v62
                        + 24
                        * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)&v58[-2].m128i_u64[1] - (char *)v59) >> 3))
                         & 0x1FFFFFFFFFFFFFFFLL)
                        + 24);
      }
      v117 = v62;
      v59 = (const __m128i *)v119;
      sub_C8CD80((__int64)v119, (__int64)v122, (__int64)v134, v63, v56, v57);
      v68 = v139;
      v54 = v138;
      v123 = 0;
      v124 = 0;
      v69 = 0;
      v125 = 0;
      v70 = (char *)v139 - (char *)v138;
      if ( v139 != v138 )
      {
        if ( v70 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_177:
          sub_4261EA(v59, v54, v55);
        v71 = sub_22077B0((char *)v139 - (char *)v138);
        v68 = v139;
        v54 = v138;
        v69 = (__m128i *)v71;
      }
      v123 = v69;
      v125 = &v69->m128i_i8[v70];
      v72 = v69;
      v124 = v69;
      if ( v68 != v54 )
      {
        v73 = v54;
        do
        {
          if ( v72 )
          {
            *v72 = _mm_loadu_si128(v73);
            v72[1].m128i_i64[0] = v73[1].m128i_i64[0];
          }
          v73 = (const __m128i *)((char *)v73 + 24);
          v72 = (__m128i *)((char *)v72 + 24);
        }
        while ( v73 != v68 );
        v72 = (__m128i *)((char *)v69
                        + 24
                        * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)&v73[-2].m128i_u64[1] - (char *)v54) >> 3))
                         & 0x1FFFFFFFFFFFFFFFLL)
                        + 24);
      }
      v93 = i;
      v74 = v116;
      v124 = v72;
      v75 = v117;
      v95 = v15;
      v76 = (char *)v117 - (char *)v116;
      if ( (char *)v117 - (char *)v116 == (char *)v72 - (char *)v69 )
        goto LABEL_108;
LABEL_94:
      v77 = v75[-2].m128i_i64[1];
      v78 = v97;
      if ( (char *)v77 != v97 )
      {
        sub_904010(a2, ", ");
        sub_A5BF40(**(unsigned __int8 ***)(v77 + 32), a2, 0, 0);
        v86 = *(_WORD **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v86 <= 1u )
        {
          v87 = sub_CB6200(a2, (unsigned __int8 *)": ", 2u);
        }
        else
        {
          v87 = a2;
          *v86 = 8250;
          *(_QWORD *)(a2 + 32) += 2LL;
        }
        v88 = sub_DAD860(a1, (__int64)v98, v77);
        sub_D9A040(v87, v88);
        v75 = v117;
LABEL_104:
        v78 = (char *)v75[-2].m128i_i64[1];
      }
      if ( !v75[-1].m128i_i8[8] )
      {
        v79 = (__int64 *)*((_QWORD *)v78 + 1);
        v75[-1].m128i_i8[8] = 1;
        v75[-1].m128i_i64[0] = (__int64)v79;
        if ( *((__int64 **)v78 + 2) == v79 )
          goto LABEL_103;
        goto LABEL_97;
      }
      while ( 1 )
      {
        while ( 1 )
        {
          v79 = (__int64 *)v75[-1].m128i_i64[0];
          if ( *((__int64 **)v78 + 2) == v79 )
          {
LABEL_103:
            v117 = (__m128i *)((char *)v117 - 24);
            v74 = v116;
            v75 = v117;
            if ( v117 != v116 )
              goto LABEL_104;
LABEL_107:
            v69 = v123;
            v76 = (char *)v75 - (char *)v74;
            if ( (char *)v75 - (char *)v74 == (char *)v124 - (char *)v123 )
            {
LABEL_108:
              if ( v75 == v74 )
              {
LABEL_115:
                v15 = v95;
                i = v93;
                v85 = v125 - (__int8 *)v69;
                if ( v69 )
                  j_j___libc_free_0(v69, v85);
                if ( !v121 )
                  _libc_free(v120, v85);
                if ( v116 )
                {
                  v85 = v118 - (_QWORD)v116;
                  j_j___libc_free_0(v116, v118 - (_QWORD)v116);
                }
                if ( !v114 )
                  _libc_free(v111, v85);
                if ( v138 )
                {
                  v85 = v140 - (__int8 *)v138;
                  j_j___libc_free_0(v138, v140 - (__int8 *)v138);
                }
                if ( !v136 )
                  _libc_free(v135, v85);
                if ( v131 )
                {
                  v85 = v133 - (_QWORD)v131;
                  j_j___libc_free_0(v131, v133 - (_QWORD)v131);
                }
                if ( !v129 )
                  _libc_free(v126.m128i_i64[1], v85);
                sub_904010(a2, " }");
                sub_904010(a2, "\n");
                goto LABEL_18;
              }
              v84 = v69;
              while ( 1 )
              {
                v76 = v84->m128i_i64[0];
                if ( v74->m128i_i64[0] != v84->m128i_i64[0] )
                  break;
                v76 = v74[1].m128i_u8[0];
                if ( (_BYTE)v76 != v84[1].m128i_i8[0] )
                  break;
                if ( (_BYTE)v76 )
                {
                  v76 = v84->m128i_i64[1];
                  if ( v74->m128i_i64[1] != v76 )
                    break;
                }
                v74 = (__m128i *)((char *)v74 + 24);
                v84 = (__m128i *)((char *)v84 + 24);
                if ( v75 == v74 )
                  goto LABEL_115;
              }
            }
            goto LABEL_94;
          }
LABEL_97:
          v80 = v79 + 1;
          v75[-1].m128i_i64[0] = (__int64)(v79 + 1);
          v81 = *v79;
          if ( v114 )
            break;
LABEL_105:
          sub_C8CC70((__int64)&v110, v81, (__int64)v80, v76, v66, v67);
          if ( v83 )
            goto LABEL_106;
        }
        v82 = v111;
        v80 = &v111[v113];
        if ( v111 == v80 )
        {
LABEL_132:
          if ( v113 < v112 )
          {
            ++v113;
            *v80 = v81;
            ++v110;
LABEL_106:
            v102.m128i_i64[0] = v81;
            LOBYTE(v103) = 0;
            sub_DAD500((__int64)&v116, &v102);
            v74 = v116;
            v75 = v117;
            goto LABEL_107;
          }
          goto LABEL_105;
        }
        while ( v81 != *v82 )
        {
          if ( v80 == ++v82 )
            goto LABEL_132;
        }
      }
    }
LABEL_43:
    sub_904010(a2, "\n");
LABEL_18:
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v15 + 32) )
    {
      v18 = v15 - 24;
      if ( !v15 )
        v18 = 0;
      if ( i != v18 + 48 )
        break;
      v15 = *(_QWORD *)(v15 + 8);
      if ( v99 == v15 )
        break;
      if ( !v15 )
        goto LABEL_178;
    }
  }
  v3 = a1;
  v2 = a2;
LABEL_2:
  v4 = *(__m128i **)(v2 + 32);
  if ( *(_QWORD *)(v2 + 24) - (_QWORD)v4 <= 0x26u )
  {
    sub_CB6200(v2, "Determining loop execution counts for: ", 0x27u);
  }
  else
  {
    v5 = _mm_load_si128((const __m128i *)&xmmword_3F74F00);
    v4[2].m128i_i32[0] = 1868963955;
    v4[2].m128i_i16[2] = 14962;
    *v4 = v5;
    v6 = _mm_load_si128((const __m128i *)&xmmword_3F74F10);
    v4[2].m128i_i8[6] = 32;
    v4[1] = v6;
    *(_QWORD *)(v2 + 32) += 39LL;
  }
  sub_A5BF40(*(unsigned __int8 **)v3, v2, 0, 0);
  v7 = *(_BYTE **)(v2 + 32);
  if ( *(_BYTE **)(v2 + 24) == v7 )
  {
    sub_CB6200(v2, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v7 = 10;
    ++*(_QWORD *)(v2 + 32);
  }
  result = *(_QWORD *)(v3 + 48);
  v9 = *(__int64 **)(result + 40);
  for ( j = *(__int64 **)(result + 32); v9 != j; result = sub_DEFA70(v2, (__int64 *)v3, v11) )
    v11 = *j++;
  return result;
}
