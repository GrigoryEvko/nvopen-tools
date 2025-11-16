// Function: sub_1096640
// Address: 0x1096640
//
__int64 __fastcall sub_1096640(__int64 a1, __int64 a2)
{
  char v4; // di
  char *v5; // r10
  int v6; // edx
  unsigned __int8 v7; // dl
  unsigned __int8 *v8; // rcx
  unsigned __int8 *v9; // r8
  char *v10; // r12
  char v11; // al
  _BYTE *v12; // rax
  unsigned int v13; // r12d
  char *v14; // rax
  char v15; // dl
  __int64 v16; // rdx
  __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // rdi
  _BYTE *v21; // rbx
  char v22; // al
  _BYTE *v23; // rax
  _BYTE *v24; // rdx
  __int64 v25; // rax
  unsigned __int8 *v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rdx
  unsigned __int64 v30; // rcx
  __int64 v31; // r8
  __m128i *v32; // rax
  __int64 v33; // rcx
  __m128i *v34; // rax
  unsigned __int8 *v36; // rax
  unsigned __int8 *v37; // rcx
  __int64 v38; // rax
  __m128i *v39; // rax
  __m128i si128; // xmm0
  __int64 v41; // rdx
  __int64 v42; // rdi
  char v43; // dl
  _BYTE *v44; // rcx
  unsigned __int8 v45; // dl
  _BYTE *v46; // rax
  _BYTE *v47; // rbx
  __int64 v48; // r12
  __int64 v49; // rax
  __int64 v50; // rax
  __m128i v51; // xmm0
  __m128i *v52; // rax
  __int64 v53; // rcx
  __m128i *v54; // rax
  char v55; // al
  unsigned __int8 *v56; // r12
  __int64 v57; // r14
  __int64 v58; // r12
  __int64 v59; // rax
  __int64 v60; // rdx
  unsigned __int64 v61; // rcx
  __int64 v62; // r8
  __m128i *v63; // rax
  __int64 v64; // rcx
  __m128i *v65; // rax
  __int64 v66; // rax
  __m128i v67; // xmm0
  unsigned __int8 *v68; // r8
  __int64 v69; // rsi
  unsigned __int8 *v70; // rax
  __int64 v71; // rsi
  unsigned __int8 *v72; // rdi
  __int64 v73; // rax
  _BYTE *v74; // rax
  __int64 v75; // rax
  unsigned __int8 *v76; // r12
  unsigned __int8 v77; // dl
  unsigned int v78; // r12d
  __int64 v79; // rax
  __int64 v80; // rdx
  unsigned __int64 v81; // rcx
  __int64 v82; // r8
  __m128i *v83; // rax
  __int64 v84; // rcx
  __m128i *v85; // rax
  __int64 v86; // rdx
  __m128i *v87; // rax
  __m128i v88; // xmm0
  __int64 v89; // rdx
  __m128i *v90; // rax
  __m128i v91; // xmm0
  __int64 v92; // rax
  __m128i v93; // xmm0
  __int64 v94; // [rsp+8h] [rbp-1C8h]
  unsigned int v95; // [rsp+8h] [rbp-1C8h]
  __int64 v96; // [rsp+10h] [rbp-1C0h] BYREF
  unsigned int v97; // [rsp+18h] [rbp-1B8h]
  __m128i v98; // [rsp+20h] [rbp-1B0h] BYREF
  __int64 v99; // [rsp+30h] [rbp-1A0h] BYREF
  unsigned int v100; // [rsp+38h] [rbp-198h]
  __m128i v101; // [rsp+40h] [rbp-190h] BYREF
  __int64 v102; // [rsp+50h] [rbp-180h] BYREF
  unsigned int v103; // [rsp+58h] [rbp-178h]
  __int64 v104; // [rsp+60h] [rbp-170h] BYREF
  unsigned int v105; // [rsp+68h] [rbp-168h]
  __m128i v106; // [rsp+70h] [rbp-160h] BYREF
  _QWORD v107[2]; // [rsp+80h] [rbp-150h] BYREF
  __int64 v108; // [rsp+90h] [rbp-140h] BYREF
  __m128i *v109; // [rsp+A0h] [rbp-130h] BYREF
  __int64 v110; // [rsp+A8h] [rbp-128h]
  __m128i v111; // [rsp+B0h] [rbp-120h] BYREF
  _QWORD v112[2]; // [rsp+C0h] [rbp-110h] BYREF
  __int64 v113; // [rsp+D0h] [rbp-100h] BYREF
  __m128i *v114; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v115; // [rsp+E8h] [rbp-E8h]
  __m128i v116; // [rsp+F0h] [rbp-E0h] BYREF
  _QWORD v117[2]; // [rsp+100h] [rbp-D0h] BYREF
  __int64 v118; // [rsp+110h] [rbp-C0h] BYREF
  __m128i *v119; // [rsp+120h] [rbp-B0h] BYREF
  __int64 v120; // [rsp+128h] [rbp-A8h]
  __m128i v121; // [rsp+130h] [rbp-A0h] BYREF
  _QWORD v122[2]; // [rsp+140h] [rbp-90h] BYREF
  __int64 v123; // [rsp+150h] [rbp-80h] BYREF
  __int64 v124; // [rsp+160h] [rbp-70h] BYREF
  __int64 v125; // [rsp+168h] [rbp-68h]
  __m128i v126; // [rsp+170h] [rbp-60h] BYREF
  __m128i v127; // [rsp+180h] [rbp-50h] BYREF
  _OWORD v128[4]; // [rsp+190h] [rbp-40h] BYREF

  v4 = *(_BYTE *)(a2 + 117);
  if ( !v4 )
    goto LABEL_23;
  v5 = *(char **)(a2 + 152);
  v6 = *(v5 - 1);
  if ( (unsigned int)(v6 - 48) <= 9 )
  {
    v7 = v6 - 48;
    if ( v7 <= 1u )
    {
      v9 = 0;
      v8 = 0;
    }
    else
    {
      v8 = (unsigned __int8 *)(v5 - 1);
      v9 = 0;
      if ( v7 >= 0xAu )
        v9 = (unsigned __int8 *)(v5 - 1);
    }
    v10 = *(char **)(a2 + 152);
    v11 = *v5;
    if ( word_3F64060[(unsigned __int8)*v5] == 0xFFFF )
    {
LABEL_99:
      if ( v11 == 46 )
      {
        *(_QWORD *)(a2 + 152) = v10 + 1;
        sub_1095CD0(a1, a2);
        return a1;
      }
      v55 = v11 & 0xDF;
      if ( *(_BYTE *)(a2 + 116) && v55 == 82 )
      {
        v75 = *(_QWORD *)(a2 + 104);
        v76 = (unsigned __int8 *)(v10 + 1);
        *(_QWORD *)(a2 + 152) = v76;
        *(_DWORD *)a1 = 6;
        *(_QWORD *)(a1 + 8) = v75;
        *(_QWORD *)(a1 + 16) = &v76[-v75];
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      }
      if ( v55 == 72 )
      {
        v95 = 16;
        v56 = (unsigned __int8 *)(v10 + 1);
        *(_QWORD *)(a2 + 152) = v56;
LABEL_104:
        v57 = *(_QWORD *)(a2 + 104);
        v97 = 128;
        sub_C43690((__int64)&v96, 0, 1);
        v58 = (__int64)&v56[-v57];
        v127.m128i_i64[0] = v57;
        v59 = v58 - 1;
        if ( !v58 )
          v59 = 0;
        v127.m128i_i64[1] = v59;
        if ( !sub_C94210(&v127, v95, (unsigned __int64 *)&v96) )
        {
          sub_1095380((_QWORD *)(a2 + 152));
          sub_1095480(a1, v57, v58, (__int64)&v96);
          goto LABEL_118;
        }
        sub_1095810((__int64)v107, v95, v60, v61, v62);
        v63 = (__m128i *)sub_2241130(v107, 0, 0, "invalid ", 8);
        v109 = &v111;
        if ( (__m128i *)v63->m128i_i64[0] == &v63[1] )
        {
          v111 = _mm_loadu_si128(v63 + 1);
        }
        else
        {
          v109 = (__m128i *)v63->m128i_i64[0];
          v111.m128i_i64[0] = v63[1].m128i_i64[0];
        }
        v64 = v63->m128i_i64[1];
        v63[1].m128i_i8[0] = 0;
        v110 = v64;
        v63->m128i_i64[0] = (__int64)v63[1].m128i_i64;
        v63->m128i_i64[1] = 0;
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v110) > 6 )
        {
          v65 = (__m128i *)sub_2241490(&v109, " number", 7, v64);
          v127.m128i_i64[0] = (__int64)v128;
          if ( (__m128i *)v65->m128i_i64[0] == &v65[1] )
          {
            v128[0] = _mm_loadu_si128(v65 + 1);
          }
          else
          {
            v127.m128i_i64[0] = v65->m128i_i64[0];
            *(_QWORD *)&v128[0] = v65[1].m128i_i64[0];
          }
          v127.m128i_i64[1] = v65->m128i_i64[1];
          v65->m128i_i64[0] = (__int64)v65[1].m128i_i64;
          v65->m128i_i64[1] = 0;
          v65[1].m128i_i8[0] = 0;
          sub_1095C00(a1, a2, *(_QWORD *)(a2 + 104), (__int64)&v127);
          if ( (_OWORD *)v127.m128i_i64[0] != v128 )
            j_j___libc_free_0(v127.m128i_i64[0], *(_QWORD *)&v128[0] + 1LL);
          if ( v109 != &v111 )
            j_j___libc_free_0(v109, v111.m128i_i64[0] + 1);
          if ( (__int64 *)v107[0] != &v108 )
            j_j___libc_free_0(v107[0], v108 + 1);
LABEL_118:
          if ( v97 <= 0x40 )
            return a1;
          v20 = v96;
          if ( !v96 )
            return a1;
LABEL_63:
          j_j___libc_free_0_0(v20);
          return a1;
        }
LABEL_188:
        sub_4262D8((__int64)"basic_string::append");
      }
      if ( v55 == 84 )
      {
        v95 = 10;
        v56 = (unsigned __int8 *)(v10 + 1);
        *(_QWORD *)(a2 + 152) = v56;
        goto LABEL_104;
      }
      if ( ((v55 - 79) & 0xFD) == 0 )
      {
        v95 = 8;
        v56 = (unsigned __int8 *)(v10 + 1);
        *(_QWORD *)(a2 + 152) = v56;
        goto LABEL_104;
      }
      if ( v55 == 89 )
      {
        v95 = 2;
        v56 = (unsigned __int8 *)(v10 + 1);
        *(_QWORD *)(a2 + 152) = v56;
        goto LABEL_104;
      }
      if ( v9 && v10 == (char *)(v9 + 1) && *(_DWORD *)(a2 + 124) <= 0xDu && (*v9 & 0xDF) == 0x44 )
      {
        v95 = 10;
        v56 = *(unsigned __int8 **)(a2 + 152);
        goto LABEL_104;
      }
      if ( v8 && v10 == (char *)(v8 + 1) && *(_DWORD *)(a2 + 124) <= 0xBu && (*v8 & 0xDF) == 0x42 )
      {
        v95 = 2;
        v56 = *(unsigned __int8 **)(a2 + 152);
        goto LABEL_104;
      }
      *(_QWORD *)(a2 + 152) = v5;
      goto LABEL_22;
    }
    while ( v11 <= 49 )
    {
      if ( v11 <= 47 )
        goto LABEL_9;
LABEL_13:
      *(_QWORD *)(a2 + 152) = ++v10;
      v11 = *v10;
      if ( word_3F64060[(unsigned __int8)*v10] == 0xFFFF )
        goto LABEL_99;
    }
    if ( (unsigned __int8)(v11 - 50) > 7u )
    {
LABEL_9:
      if ( !v9 )
        v9 = (unsigned __int8 *)v10;
    }
    if ( !v8 )
      v8 = (unsigned __int8 *)v10;
    goto LABEL_13;
  }
LABEL_22:
  if ( *(_BYTE *)(a2 + 120) )
  {
    v26 = *(unsigned __int8 **)(a2 + 152);
    if ( (unsigned int)(__int16)word_3F64060[*v26] <= 0xF )
    {
      do
        v27 = *++v26;
      while ( (unsigned int)(__int16)word_3F64060[v27] <= 0xF );
    }
    v28 = *(_QWORD *)(a2 + 104);
    *(_QWORD *)(a2 + 152) = v26;
    v100 = 128;
    v98.m128i_i64[0] = v28;
    v98.m128i_i64[1] = (__int64)&v26[-v28];
    sub_C43690((__int64)&v99, 0, 1);
    if ( sub_C94210(&v98, *(_DWORD *)(a2 + 124), (unsigned __int64 *)&v99) )
    {
      sub_1095810((__int64)v112, *(_DWORD *)(a2 + 124), v29, v30, v31);
      v32 = (__m128i *)sub_2241130(v112, 0, 0, "invalid ", 8);
      v114 = &v116;
      if ( (__m128i *)v32->m128i_i64[0] == &v32[1] )
      {
        v116 = _mm_loadu_si128(v32 + 1);
      }
      else
      {
        v114 = (__m128i *)v32->m128i_i64[0];
        v116.m128i_i64[0] = v32[1].m128i_i64[0];
      }
      v33 = v32->m128i_i64[1];
      v32[1].m128i_i8[0] = 0;
      v115 = v33;
      v32->m128i_i64[0] = (__int64)v32[1].m128i_i64;
      v32->m128i_i64[1] = 0;
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v115) <= 6 )
        goto LABEL_188;
      v34 = (__m128i *)sub_2241490(&v114, " number", 7, v33);
      v127.m128i_i64[0] = (__int64)v128;
      if ( (__m128i *)v34->m128i_i64[0] == &v34[1] )
      {
        v128[0] = _mm_loadu_si128(v34 + 1);
      }
      else
      {
        v127.m128i_i64[0] = v34->m128i_i64[0];
        *(_QWORD *)&v128[0] = v34[1].m128i_i64[0];
      }
      v127.m128i_i64[1] = v34->m128i_i64[1];
      v34->m128i_i64[0] = (__int64)v34[1].m128i_i64;
      v34->m128i_i64[1] = 0;
      v34[1].m128i_i8[0] = 0;
      sub_1095C00(a1, a2, *(_QWORD *)(a2 + 104), (__int64)&v127);
      if ( (_OWORD *)v127.m128i_i64[0] != v128 )
        j_j___libc_free_0(v127.m128i_i64[0], *(_QWORD *)&v128[0] + 1LL);
      if ( v114 != &v116 )
        j_j___libc_free_0(v114, v116.m128i_i64[0] + 1);
      if ( (__int64 *)v112[0] != &v113 )
        j_j___libc_free_0(v112[0], v113 + 1);
    }
    else
    {
      sub_1095480(a1, v98.m128i_i64[0], v98.m128i_i64[1], (__int64)&v99);
    }
    if ( v100 > 0x40 )
    {
      v20 = v99;
      if ( v99 )
        goto LABEL_63;
    }
    return a1;
  }
LABEL_23:
  if ( *(_BYTE *)(a2 + 119) )
  {
    v21 = *(_BYTE **)(a2 + 152);
    v22 = *(v21 - 1);
    if ( v22 == 36 )
    {
      v36 = v21 + 1;
      if ( word_3F64060[(unsigned __int8)*v21] != 0xFFFF )
      {
        do
        {
          v37 = v36;
          *(_QWORD *)(a2 + 152) = v36++;
        }
        while ( word_3F64060[*v37] != 0xFFFF );
      }
      LODWORD(v125) = 128;
      sub_C43690((__int64)&v124, 0, 0);
      v38 = *(_QWORD *)(a2 + 152);
      v127.m128i_i64[0] = (__int64)v21;
      v127.m128i_i64[1] = v38 - (_QWORD)v21;
      if ( !sub_C94210(&v127, 0x10u, (unsigned __int64 *)&v124) )
        goto LABEL_43;
      v127.m128i_i64[0] = (__int64)v128;
      v122[0] = 26;
      v39 = (__m128i *)sub_22409D0(&v127, v122, 0);
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F900A0);
      v127.m128i_i64[0] = (__int64)v39;
      *(_QWORD *)&v128[0] = v122[0];
      qmemcpy(&v39[1], "mal number", 10);
      *v39 = si128;
    }
    else
    {
      if ( v22 != 37 )
        goto LABEL_24;
      v23 = v21 + 1;
      if ( (unsigned __int8)(*v21 - 48) <= 1u )
      {
        do
        {
          v24 = v23;
          *(_QWORD *)(a2 + 152) = v23++;
        }
        while ( (unsigned __int8)(*v24 - 48) <= 1u );
      }
      LODWORD(v125) = 128;
      sub_C43690((__int64)&v124, 0, 0);
      v25 = *(_QWORD *)(a2 + 152);
      v127.m128i_i64[0] = (__int64)v21;
      v127.m128i_i64[1] = v25 - (_QWORD)v21;
      if ( !sub_C94210(&v127, 2u, (unsigned __int64 *)&v124) )
      {
LABEL_43:
        sub_1095480(a1, *(_QWORD *)(a2 + 104), *(_QWORD *)(a2 + 152) - *(_QWORD *)(a2 + 104), (__int64)&v124);
        goto LABEL_44;
      }
      v127.m128i_i64[0] = (__int64)v128;
      v122[0] = 21;
      v66 = sub_22409D0(&v127, v122, 0);
      v67 = _mm_load_si128((const __m128i *)&xmmword_3F90120);
      v127.m128i_i64[0] = v66;
      *(_QWORD *)&v128[0] = v122[0];
      *(_DWORD *)(v66 + 16) = 1700949365;
      *(_BYTE *)(v66 + 20) = 114;
      *(__m128i *)v66 = v67;
    }
    v127.m128i_i64[1] = v122[0];
    *(_BYTE *)(v127.m128i_i64[0] + v122[0]) = 0;
    v41 = *(_QWORD *)(a2 + 104);
    goto LABEL_71;
  }
LABEL_24:
  if ( *(_BYTE *)(a2 + 128) || (v12 = *(_BYTE **)(a2 + 152), *(v12 - 1) != 48) || *v12 == 46 )
  {
    v13 = sub_10953C0((__int64 *)(a2 + 152), 0xAu, v4);
    v14 = *(char **)(a2 + 152);
    if ( *(_BYTE *)(a2 + 128) == 1 || v13 == 16 )
      goto LABEL_31;
    v15 = *v14;
    if ( *v14 == 46 )
    {
      *(_QWORD *)(a2 + 152) = v14 + 1;
    }
    else if ( v15 != 101 && v15 != 69 )
    {
LABEL_31:
      v16 = *(_QWORD *)(a2 + 104);
      v103 = 128;
      v101.m128i_i64[0] = v16;
      v101.m128i_i64[1] = (__int64)&v14[-v16];
      sub_C43690((__int64)&v102, 0, 1);
      if ( sub_C94210(&v101, v13, (unsigned __int64 *)&v102) )
      {
        sub_1095810((__int64)v117, v13, v17, v18, v19);
        v52 = (__m128i *)sub_2241130(v117, 0, 0, "invalid ", 8);
        v119 = &v121;
        if ( (__m128i *)v52->m128i_i64[0] == &v52[1] )
        {
          v121 = _mm_loadu_si128(v52 + 1);
        }
        else
        {
          v119 = (__m128i *)v52->m128i_i64[0];
          v121.m128i_i64[0] = v52[1].m128i_i64[0];
        }
        v53 = v52->m128i_i64[1];
        v52[1].m128i_i8[0] = 0;
        v120 = v53;
        v52->m128i_i64[0] = (__int64)v52[1].m128i_i64;
        v52->m128i_i64[1] = 0;
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v120) <= 6 )
          goto LABEL_188;
        v54 = (__m128i *)sub_2241490(&v119, " number", 7, v53);
        v127.m128i_i64[0] = (__int64)v128;
        if ( (__m128i *)v54->m128i_i64[0] == &v54[1] )
        {
          v128[0] = _mm_loadu_si128(v54 + 1);
        }
        else
        {
          v127.m128i_i64[0] = v54->m128i_i64[0];
          *(_QWORD *)&v128[0] = v54[1].m128i_i64[0];
        }
        v127.m128i_i64[1] = v54->m128i_i64[1];
        v54->m128i_i64[0] = (__int64)v54[1].m128i_i64;
        v54->m128i_i64[1] = 0;
        v54[1].m128i_i8[0] = 0;
        sub_1095C00(a1, a2, *(_QWORD *)(a2 + 104), (__int64)&v127);
        if ( (_OWORD *)v127.m128i_i64[0] != v128 )
          j_j___libc_free_0(v127.m128i_i64[0], *(_QWORD *)&v128[0] + 1LL);
        if ( v119 != &v121 )
          j_j___libc_free_0(v119, v121.m128i_i64[0] + 1);
        if ( (__int64 *)v117[0] != &v118 )
          j_j___libc_free_0(v117[0], v118 + 1);
      }
      else
      {
        if ( !*(_BYTE *)(a2 + 128) )
          sub_1095380((_QWORD *)(a2 + 152));
        sub_1095480(a1, v101.m128i_i64[0], v101.m128i_i64[1], (__int64)&v102);
      }
      if ( v103 > 0x40 )
      {
        v20 = v102;
        if ( v102 )
          goto LABEL_63;
      }
      return a1;
    }
    sub_1095CD0(a1, a2);
    return a1;
  }
  v43 = *v12 & 0xDF;
  if ( !v4 && v43 == 66 )
  {
    v44 = v12 + 1;
    *(_QWORD *)(a2 + 152) = v12 + 1;
    v45 = v12[1] - 48;
    if ( v45 > 9u )
    {
      v86 = *(_QWORD *)(a2 + 104);
      *(_QWORD *)(a2 + 152) = v12;
      *(_DWORD *)a1 = 4;
      *(_QWORD *)(a1 + 8) = v86;
      *(_QWORD *)(a1 + 16) = &v12[-v86];
      *(_DWORD *)(a1 + 32) = 64;
      *(_QWORD *)(a1 + 24) = 0;
      return a1;
    }
    v46 = v12 + 2;
    if ( v45 <= 1u )
    {
      do
      {
        v47 = v46;
        *(_QWORD *)(a2 + 152) = v46++;
      }
      while ( (unsigned __int8)(*v47 - 48) <= 1u );
      if ( v44 != v47 )
      {
        v48 = *(_QWORD *)(a2 + 104);
        LODWORD(v125) = 128;
        v94 = (__int64)&v47[-v48];
        sub_C43690((__int64)&v124, 0, 1);
        v49 = 0;
        if ( (unsigned __int64)&v47[-v48] > 1 )
        {
          v47 = (_BYTE *)(v48 + 2);
          v49 = v94 - 2;
        }
        v127.m128i_i64[0] = (__int64)v47;
        v127.m128i_i64[1] = v49;
        if ( !sub_C94210(&v127, 2u, (unsigned __int64 *)&v124) )
        {
          sub_1095380((_QWORD *)(a2 + 152));
          sub_1095480(a1, v48, v94, (__int64)&v124);
          goto LABEL_44;
        }
        v127.m128i_i64[0] = (__int64)v128;
        v122[0] = 21;
        v50 = sub_22409D0(&v127, v122, 0);
        v51 = _mm_load_si128((const __m128i *)&xmmword_3F90120);
        v127.m128i_i64[0] = v50;
        *(_QWORD *)&v128[0] = v122[0];
        *(_DWORD *)(v50 + 16) = 1700949365;
        *(_BYTE *)(v50 + 20) = 114;
        *(__m128i *)v50 = v51;
        v127.m128i_i64[1] = v122[0];
        *(_BYTE *)(v127.m128i_i64[0] + v122[0]) = 0;
        sub_1095C00(a1, a2, *(_QWORD *)(a2 + 104), (__int64)&v127);
        v42 = v127.m128i_i64[0];
        if ( (_OWORD *)v127.m128i_i64[0] == v128 )
          goto LABEL_44;
        goto LABEL_72;
      }
    }
    v124 = 21;
    v127.m128i_i64[0] = (__int64)v128;
    v92 = sub_22409D0(&v127, &v124, 0);
    v93 = _mm_load_si128((const __m128i *)&xmmword_3F90120);
    v127.m128i_i64[0] = v92;
    *(_QWORD *)&v128[0] = v124;
    *(_DWORD *)(v92 + 16) = 1700949365;
    *(_BYTE *)(v92 + 20) = 114;
    *(__m128i *)v92 = v93;
    v127.m128i_i64[1] = v124;
    *(_BYTE *)(v127.m128i_i64[0] + v124) = 0;
    v89 = *(_QWORD *)(a2 + 104);
LABEL_172:
    sub_1095C00(a1, a2, v89, (__int64)&v127);
    if ( (_OWORD *)v127.m128i_i64[0] != v128 )
      j_j___libc_free_0(v127.m128i_i64[0], *(_QWORD *)&v128[0] + 1LL);
    return a1;
  }
  if ( v43 == 88 )
  {
    v68 = v12 + 1;
    *(_QWORD *)(a2 + 152) = v12 + 1;
    v69 = (unsigned __int8)v12[1];
    if ( word_3F64060[v69] == 0xFFFF )
    {
      if ( (v69 & 0xDF) != 0x50 && (_BYTE)v69 != 46 )
        goto LABEL_171;
      v77 = 1;
    }
    else
    {
      v70 = v12 + 2;
      do
      {
        *(_QWORD *)(a2 + 152) = v70;
        v71 = *v70;
        v72 = v70++;
      }
      while ( word_3F64060[v71] != 0xFFFF );
      if ( (v71 & 0xDF) != 0x50 && (_BYTE)v71 != 46 )
      {
        if ( v68 != v72 )
        {
          LODWORD(v125) = 128;
          sub_C43690((__int64)&v124, 0, 0);
          v73 = *(_QWORD *)(a2 + 152);
          v127.m128i_i64[0] = *(_QWORD *)(a2 + 104);
          v127.m128i_i64[1] = v73 - v127.m128i_i64[0];
          if ( !sub_C94210(&v127, 0, (unsigned __int64 *)&v124) )
          {
            if ( *(_BYTE *)(a2 + 117) )
            {
              v74 = *(_BYTE **)(a2 + 152);
              if ( (*v74 & 0xDF) == 0x48 )
                *(_QWORD *)(a2 + 152) = v74 + 1;
            }
            sub_1095380((_QWORD *)(a2 + 152));
            sub_1095480(a1, *(_QWORD *)(a2 + 104), *(_QWORD *)(a2 + 152) - *(_QWORD *)(a2 + 104), (__int64)&v124);
            goto LABEL_44;
          }
          v127.m128i_i64[0] = (__int64)v128;
          v122[0] = 26;
          v90 = (__m128i *)sub_22409D0(&v127, v122, 0);
          v91 = _mm_load_si128((const __m128i *)&xmmword_3F900A0);
          v127.m128i_i64[0] = (__int64)v90;
          *(_QWORD *)&v128[0] = v122[0];
          qmemcpy(&v90[1], "mal number", 10);
          *v90 = v91;
          v127.m128i_i64[1] = v122[0];
          *(_BYTE *)(v127.m128i_i64[0] + v122[0]) = 0;
          v41 = *(_QWORD *)(a2 + 104);
LABEL_71:
          sub_1095C00(a1, a2, v41, (__int64)&v127);
          v42 = v127.m128i_i64[0];
          if ( (_OWORD *)v127.m128i_i64[0] != v128 )
LABEL_72:
            j_j___libc_free_0(v42, *(_QWORD *)&v128[0] + 1LL);
LABEL_44:
          if ( (unsigned int)v125 > 0x40 )
          {
            v20 = v124;
            if ( v124 )
              goto LABEL_63;
          }
          return a1;
        }
LABEL_171:
        v124 = 26;
        v127.m128i_i64[0] = (__int64)v128;
        v87 = (__m128i *)sub_22409D0(&v127, &v124, 0);
        v88 = _mm_load_si128((const __m128i *)&xmmword_3F900A0);
        v127.m128i_i64[0] = (__int64)v87;
        *(_QWORD *)&v128[0] = v124;
        qmemcpy(&v87[1], "mal number", 10);
        *v87 = v88;
        v127.m128i_i64[1] = v124;
        *(_BYTE *)(v127.m128i_i64[0] + v124) = 0;
        v89 = *(_QWORD *)(a2 + 152) - 2LL;
        goto LABEL_172;
      }
      v77 = v68 == v72;
    }
    sub_1095E80(a1, a2, v77);
    return a1;
  }
  v105 = 128;
  sub_C43690((__int64)&v104, 0, 1);
  v78 = sub_10953C0((__int64 *)(a2 + 152), 8u, *(_BYTE *)(a2 + 117));
  v79 = *(_QWORD *)(a2 + 152);
  v106.m128i_i64[0] = *(_QWORD *)(a2 + 104);
  v106.m128i_i64[1] = v79 - v106.m128i_i64[0];
  if ( sub_C94210(&v106, v78, (unsigned __int64 *)&v104) )
  {
    sub_1095810((__int64)v122, v78, v80, v81, v82);
    v83 = (__m128i *)sub_2241130(v122, 0, 0, "invalid ", 8);
    v124 = (__int64)&v126;
    if ( (__m128i *)v83->m128i_i64[0] == &v83[1] )
    {
      v126 = _mm_loadu_si128(v83 + 1);
    }
    else
    {
      v124 = v83->m128i_i64[0];
      v126.m128i_i64[0] = v83[1].m128i_i64[0];
    }
    v84 = v83->m128i_i64[1];
    v125 = v84;
    v83->m128i_i64[0] = (__int64)v83[1].m128i_i64;
    v83->m128i_i64[1] = 0;
    v83[1].m128i_i8[0] = 0;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v125) <= 6 )
      goto LABEL_188;
    v85 = (__m128i *)sub_2241490(&v124, " number", 7, v84);
    v127.m128i_i64[0] = (__int64)v128;
    if ( (__m128i *)v85->m128i_i64[0] == &v85[1] )
    {
      v128[0] = _mm_loadu_si128(v85 + 1);
    }
    else
    {
      v127.m128i_i64[0] = v85->m128i_i64[0];
      *(_QWORD *)&v128[0] = v85[1].m128i_i64[0];
    }
    v127.m128i_i64[1] = v85->m128i_i64[1];
    v85->m128i_i64[0] = (__int64)v85[1].m128i_i64;
    v85->m128i_i64[1] = 0;
    v85[1].m128i_i8[0] = 0;
    sub_1095C00(a1, a2, *(_QWORD *)(a2 + 104), (__int64)&v127);
    if ( (_OWORD *)v127.m128i_i64[0] != v128 )
      j_j___libc_free_0(v127.m128i_i64[0], *(_QWORD *)&v128[0] + 1LL);
    if ( (__m128i *)v124 != &v126 )
      j_j___libc_free_0(v124, v126.m128i_i64[0] + 1);
    if ( (__int64 *)v122[0] != &v123 )
      j_j___libc_free_0(v122[0], v123 + 1);
  }
  else
  {
    if ( v78 == 16 )
      ++*(_QWORD *)(a2 + 152);
    sub_1095380((_QWORD *)(a2 + 152));
    sub_1095480(a1, v106.m128i_i64[0], v106.m128i_i64[1], (__int64)&v104);
  }
  if ( v105 > 0x40 )
  {
    v20 = v104;
    if ( v104 )
      goto LABEL_63;
  }
  return a1;
}
