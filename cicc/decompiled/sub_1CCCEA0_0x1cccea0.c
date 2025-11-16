// Function: sub_1CCCEA0
// Address: 0x1cccea0
//
__int64 __fastcall sub_1CCCEA0(
        __int64 *a1,
        __int64 **a2,
        __int64 a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  unsigned int v12; // r8d
  __int64 *v14; // rbx
  __int64 *v15; // r14
  unsigned __int8 *v16; // r13
  size_t v17; // rdx
  size_t v18; // r15
  unsigned int v19; // r8d
  _QWORD *v20; // r9
  __int64 v21; // rax
  unsigned int v22; // r8d
  __int64 *v23; // r9
  __int64 v24; // r10
  void *v25; // rdi
  __int64 *v26; // rbx
  __int64 *v27; // r14
  unsigned __int8 *v28; // r13
  size_t v29; // rdx
  size_t v30; // r15
  unsigned int v31; // r8d
  _QWORD *v32; // r9
  __int64 v33; // rax
  unsigned int v34; // r8d
  __int64 *v35; // r9
  __int64 v36; // r10
  void *v37; // rdi
  __int64 *v38; // r13
  __int64 *v39; // r14
  const char *v40; // rax
  size_t v41; // rdx
  __int64 **v42; // rbx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rsi
  size_t v46; // rax
  __int64 (*v47)(void); // rax
  __int8 v48; // al
  __int64 v49; // rax
  __int64 v50; // rax
  void *v51; // rax
  __int64 v52; // rax
  void *v53; // rax
  _QWORD *v54; // r13
  _BYTE *v55; // r9
  size_t v56; // r8
  char **v57; // rax
  _BYTE *v58; // rdi
  char **v59; // rax
  size_t v60; // rdi
  char *v61; // rsi
  char *v62; // r8
  __int64 v63; // rsi
  __int64 v64; // rdx
  double v65; // xmm4_8
  double v66; // xmm5_8
  __int64 v67; // rax
  unsigned __int8 v68; // al
  unsigned __int8 v69; // r8
  __m128i v70; // xmm0
  __m128i v71; // xmm2
  __m128i v72; // xmm1
  unsigned __int64 v73; // r8
  __int64 v74; // r12
  __int64 v75; // rbx
  unsigned __int64 v76; // rdi
  unsigned __int64 v77; // r9
  __int64 v78; // r12
  __int64 v79; // rbx
  unsigned __int64 v80; // rdi
  size_t v81; // rdx
  __int64 v82; // rdx
  __int64 v83; // rax
  char **v84; // rdi
  __int64 v85; // r12
  __int64 v86; // r14
  __int64 *v88; // [rsp+10h] [rbp-110h]
  __int64 *v89; // [rsp+10h] [rbp-110h]
  __int64 v90; // [rsp+10h] [rbp-110h]
  __int64 v91; // [rsp+10h] [rbp-110h]
  unsigned int v92; // [rsp+18h] [rbp-108h]
  unsigned int v93; // [rsp+18h] [rbp-108h]
  __int64 v94; // [rsp+18h] [rbp-108h]
  __int64 v95; // [rsp+18h] [rbp-108h]
  __int64 *v96; // [rsp+18h] [rbp-108h]
  __int64 *v97; // [rsp+18h] [rbp-108h]
  __int64 *v98; // [rsp+20h] [rbp-100h]
  __int64 *v99; // [rsp+20h] [rbp-100h]
  unsigned int v100; // [rsp+20h] [rbp-100h]
  unsigned int v101; // [rsp+20h] [rbp-100h]
  unsigned int src; // [rsp+28h] [rbp-F8h]
  unsigned int srca; // [rsp+28h] [rbp-F8h]
  unsigned __int8 srcb; // [rsp+28h] [rbp-F8h]
  _BYTE *srcc; // [rsp+28h] [rbp-F8h]
  size_t n; // [rsp+30h] [rbp-F0h]
  unsigned __int8 na; // [rsp+30h] [rbp-F0h]
  size_t nb; // [rsp+30h] [rbp-F0h]
  size_t nc; // [rsp+30h] [rbp-F0h]
  __int64 **v110; // [rsp+38h] [rbp-E8h]
  unsigned __int8 v111; // [rsp+38h] [rbp-E8h]
  unsigned __int8 v112; // [rsp+38h] [rbp-E8h]
  unsigned __int8 v113; // [rsp+38h] [rbp-E8h]
  unsigned __int8 v114; // [rsp+38h] [rbp-E8h]
  unsigned __int8 v115; // [rsp+38h] [rbp-E8h]
  _QWORD *v116; // [rsp+48h] [rbp-D8h] BYREF
  unsigned __int64 v117; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v118; // [rsp+58h] [rbp-C8h]
  __int64 v119; // [rsp+60h] [rbp-C0h]
  __m128i v120; // [rsp+70h] [rbp-B0h] BYREF
  void (__fastcall *v121)(__m128i *, __m128i *, __int64); // [rsp+80h] [rbp-A0h]
  __int64 v122; // [rsp+88h] [rbp-98h]
  __m128i v123; // [rsp+90h] [rbp-90h] BYREF
  void (__fastcall *v124)(__m128i *, __m128i *, __int64); // [rsp+A0h] [rbp-80h]
  __int64 v125; // [rsp+A8h] [rbp-78h]
  size_t v126[2]; // [rsp+B0h] [rbp-70h] BYREF
  char *v127; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v128; // [rsp+C8h] [rbp-58h]
  unsigned __int64 v129; // [rsp+D0h] [rbp-50h]
  __int64 v130; // [rsp+D8h] [rbp-48h]
  __int64 v131; // [rsp+E0h] [rbp-40h]

  v12 = sub_1CCA460((__int64)a1);
  if ( !(_BYTE)v12 )
    return v12;
  v14 = (__int64 *)a1[2];
  v15 = a1 + 1;
  v117 = 0;
  v119 = 0x1000000000LL;
  v118 = 0;
  if ( a1 + 1 != v14 )
  {
    while ( 1 )
    {
      if ( !v14 )
        goto LABEL_133;
      if ( (*((_BYTE *)v14 - 33) & 0x20) == 0 )
        goto LABEL_5;
      v16 = (unsigned __int8 *)sub_1649960((__int64)(v14 - 7));
      v18 = v17;
      v19 = sub_16D19C0((__int64)&v117, v16, v17);
      v20 = (_QWORD *)(v117 + 8LL * v19);
      if ( !*v20 )
        goto LABEL_11;
      if ( *v20 == -8 )
      {
        LODWORD(v119) = v119 - 1;
LABEL_11:
        v88 = (__int64 *)(v117 + 8LL * v19);
        v92 = v19;
        v21 = malloc(v18 + 17);
        v22 = v92;
        v23 = v88;
        v24 = v21;
        if ( !v21 )
        {
          if ( v18 == -17 )
          {
            v50 = malloc(1u);
            v22 = v92;
            v23 = v88;
            v24 = 0;
            if ( v50 )
            {
              v25 = (void *)(v50 + 16);
              v24 = v50;
LABEL_41:
              v94 = v24;
              v98 = v23;
              src = v22;
              v51 = memcpy(v25, v16, v18);
              v24 = v94;
              v23 = v98;
              v22 = src;
              v25 = v51;
              goto LABEL_13;
            }
          }
          v91 = v24;
          v97 = v23;
          v101 = v22;
          sub_16BD1C0("Allocation failed", 1u);
          v22 = v101;
          v23 = v97;
          v24 = v91;
        }
        v25 = (void *)(v24 + 16);
        if ( v18 + 1 > 1 )
          goto LABEL_41;
LABEL_13:
        *((_BYTE *)v25 + v18) = 0;
        *(_QWORD *)v24 = v18;
        *(_BYTE *)(v24 + 8) = 0;
        *v23 = v24;
        ++HIDWORD(v118);
        sub_16D1CD0((__int64)&v117, v22);
        v14 = (__int64 *)v14[1];
        if ( v15 == v14 )
          break;
      }
      else
      {
LABEL_5:
        v14 = (__int64 *)v14[1];
        if ( v15 == v14 )
          break;
      }
    }
  }
  v26 = (__int64 *)a1[6];
  v27 = a1 + 5;
  if ( a1 + 5 != v26 )
  {
    while ( 1 )
    {
      if ( !v26 )
        goto LABEL_133;
      if ( (*((_BYTE *)v26 - 25) & 0x20) == 0 )
        goto LABEL_16;
      v28 = (unsigned __int8 *)sub_1649960((__int64)(v26 - 6));
      v30 = v29;
      v31 = sub_16D19C0((__int64)&v117, v28, v29);
      v32 = (_QWORD *)(v117 + 8LL * v31);
      if ( !*v32 )
        goto LABEL_22;
      if ( *v32 == -8 )
      {
        LODWORD(v119) = v119 - 1;
LABEL_22:
        v89 = (__int64 *)(v117 + 8LL * v31);
        v93 = v31;
        v33 = malloc(v30 + 17);
        v34 = v93;
        v35 = v89;
        v36 = v33;
        if ( !v33 )
        {
          if ( v30 == -17 )
          {
            v52 = malloc(1u);
            v34 = v93;
            v35 = v89;
            v36 = 0;
            if ( v52 )
            {
              v37 = (void *)(v52 + 16);
              v36 = v52;
LABEL_44:
              v95 = v36;
              v99 = v35;
              srca = v34;
              v53 = memcpy(v37, v28, v30);
              v36 = v95;
              v35 = v99;
              v34 = srca;
              v37 = v53;
              goto LABEL_24;
            }
          }
          v90 = v36;
          v96 = v35;
          v100 = v34;
          sub_16BD1C0("Allocation failed", 1u);
          v34 = v100;
          v35 = v96;
          v36 = v90;
        }
        v37 = (void *)(v36 + 16);
        if ( v30 + 1 > 1 )
          goto LABEL_44;
LABEL_24:
        *((_BYTE *)v37 + v30) = 0;
        *(_QWORD *)v36 = v30;
        *(_BYTE *)(v36 + 8) = 0;
        *v35 = v36;
        ++HIDWORD(v118);
        sub_16D1CD0((__int64)&v117, v34);
        v26 = (__int64 *)v26[1];
        if ( v27 == v26 )
          break;
      }
      else
      {
LABEL_16:
        v26 = (__int64 *)v26[1];
        if ( v27 == v26 )
          break;
      }
    }
  }
  v38 = (__int64 *)a1[4];
  v39 = a1 + 3;
  if ( v38 != a1 + 3 )
  {
    while ( v38 )
    {
      if ( (*((_BYTE *)v38 - 33) & 0x20) == 0 || sub_15E4F60((__int64)(v38 - 7)) )
      {
        v38 = (__int64 *)v38[1];
        if ( v39 == v38 )
          goto LABEL_32;
      }
      else
      {
        v40 = sub_1649960((__int64)(v38 - 7));
        sub_167C570((__int64)&v117, v40, v41);
        v38 = (__int64 *)v38[1];
        if ( v39 == v38 )
          goto LABEL_32;
      }
    }
LABEL_133:
    BUG();
  }
LABEL_32:
  v110 = &a2[a3];
  if ( v110 == a2 )
  {
LABEL_70:
    v120.m128i_i64[0] = (__int64)&v117;
    v70 = _mm_loadu_si128(&v120);
    v71 = _mm_loadu_si128((const __m128i *)v126);
    v121 = 0;
    v122 = v125;
    v72 = _mm_loadu_si128(&v123);
    v127 = (char *)sub_1CCA570;
    v124 = 0;
    v125 = v128;
    v128 = (__int64)sub_1CCA510;
    v129 = 0;
    v130 = 0;
    v131 = 0x1000000000LL;
    v120 = v72;
    v123 = v71;
    *(__m128i *)v126 = v70;
    sub_18708C0((__int64)v126, (__int64)a1, 0);
    if ( HIDWORD(v130) )
    {
      v73 = v129;
      if ( (_DWORD)v130 )
      {
        v74 = 8LL * (unsigned int)v130;
        v75 = 0;
        do
        {
          v76 = *(_QWORD *)(v73 + v75);
          if ( v76 != -8 && v76 )
          {
            _libc_free(v76);
            v73 = v129;
          }
          v75 += 8;
        }
        while ( v74 != v75 );
      }
    }
    else
    {
      v73 = v129;
    }
    _libc_free(v73);
    if ( v127 )
      ((void (__fastcall *)(size_t *, size_t *, __int64))v127)(v126, v126, 3);
    if ( v124 )
      v124(&v123, &v123, 3);
    if ( v121 )
      v121(&v120, &v120, 3);
    v69 = 0;
    goto LABEL_84;
  }
  v42 = a2;
  while ( 1 )
  {
    sub_16C2450(&v116, **v42, (*v42)[1], (__int64)byte_3F871B3, 0);
    v45 = *a1;
    v46 = v116[2] - v116[1];
    v126[0] = v116[1];
    v126[1] = v46;
    v47 = *(__int64 (**)(void))(*v116 + 16LL);
    if ( (char *)v47 == (char *)sub_12BCB10 )
    {
      v128 = 14;
      v127 = "Unknown buffer";
    }
    else
    {
      v127 = (char *)v47();
      v128 = v82;
    }
    sub_15099C0((__int64)&v123, v45, 0, 0, v43, v44, a4, (__m128i *)v126[0], v126[1]);
    v48 = v123.m128i_i8[8];
    v123.m128i_i8[8] &= ~2u;
    if ( (v48 & 1) != 0 )
    {
      v49 = v123.m128i_i64[0];
      v123.m128i_i64[0] = 0;
      v126[0] = v49 | 1;
      if ( (v49 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_16BCAE0(v126, v45, v49 | 1);
      v54 = 0;
    }
    else
    {
      v54 = (_QWORD *)v123.m128i_i64[0];
    }
    v55 = (_BYTE *)a1[30];
    if ( !v55 )
    {
      LOBYTE(v127) = 0;
      v81 = 0;
      v126[0] = (size_t)&v127;
      v58 = (_BYTE *)v54[30];
LABEL_93:
      v54[31] = v81;
      v58[v81] = 0;
      v59 = (char **)v126[0];
      goto LABEL_54;
    }
    v56 = a1[31];
    v126[0] = (size_t)&v127;
    v120.m128i_i64[0] = v56;
    if ( v56 > 0xF )
    {
      srcc = v55;
      nc = v56;
      v83 = sub_22409D0(v126, &v120, 0);
      v56 = nc;
      v55 = srcc;
      v126[0] = v83;
      v84 = (char **)v83;
      v127 = (char *)v120.m128i_i64[0];
    }
    else
    {
      if ( v56 == 1 )
      {
        LOBYTE(v127) = *v55;
        v57 = &v127;
        goto LABEL_50;
      }
      if ( !v56 )
      {
        v57 = &v127;
        goto LABEL_50;
      }
      v84 = &v127;
    }
    memcpy(v84, v55, v56);
    v56 = v120.m128i_i64[0];
    v57 = (char **)v126[0];
LABEL_50:
    v126[1] = v56;
    *((_BYTE *)v57 + v56) = 0;
    v58 = (_BYTE *)v54[30];
    v59 = (char **)v58;
    if ( (char **)v126[0] == &v127 )
    {
      v81 = v126[1];
      if ( v126[1] )
      {
        if ( v126[1] == 1 )
          *v58 = (_BYTE)v127;
        else
          memcpy(v58, &v127, v126[1]);
        v81 = v126[1];
        v58 = (_BYTE *)v54[30];
      }
      goto LABEL_93;
    }
    v60 = v126[1];
    v61 = v127;
    if ( v59 == v54 + 32 )
    {
      v54[30] = v126[0];
      v54[31] = v60;
      v54[32] = v61;
    }
    else
    {
      v62 = (char *)v54[32];
      v54[30] = v126[0];
      v54[31] = v60;
      v54[32] = v61;
      if ( v59 )
      {
        v126[0] = (size_t)v59;
        v127 = v62;
        goto LABEL_54;
      }
    }
    v126[0] = (size_t)&v127;
    v59 = &v127;
LABEL_54:
    v126[1] = 0;
    *(_BYTE *)v59 = 0;
    if ( (char **)v126[0] != &v127 )
      j_j___libc_free_0(v126[0], v127 + 1);
    v63 = sub_1632FA0((__int64)a1);
    sub_1632B40((__int64)v54, v63);
    v127 = 0;
    if ( (v123.m128i_i8[8] & 2) != 0 )
      goto LABEL_102;
    v67 = v123.m128i_i64[0];
    v63 = (__int64)&v120;
    v123.m128i_i64[0] = 0;
    v120.m128i_i64[0] = v67;
    v68 = sub_167F6E0(a1, &v120, 3, (__m128i *)v126, *(double *)a4.m128i_i64, a5, a6, a7, v65, v66, a10, a11);
    v69 = v68;
    if ( v120.m128i_i64[0] )
    {
      srcb = v68;
      n = v120.m128i_i64[0];
      sub_1633490(v120.m128i_i64[0]);
      v63 = 736;
      j_j___libc_free_0(n, 736);
      v69 = srcb;
    }
    if ( v127 )
    {
      na = v69;
      v63 = (__int64)v126;
      ((void (__fastcall *)(size_t *, size_t *, __int64))v127)(v126, v126, 3);
      v69 = na;
    }
    if ( v69 )
      break;
    if ( !(unsigned __int8)sub_1CCA460((__int64)a1) )
    {
      if ( (v123.m128i_i8[8] & 2) == 0 )
      {
        v86 = v123.m128i_i64[0];
        if ( (v123.m128i_i8[8] & 1) != 0 )
        {
          if ( v123.m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v123.m128i_i64[0] + 8LL))(v123.m128i_i64[0]);
        }
        else if ( v123.m128i_i64[0] )
        {
          sub_1633490(v123.m128i_i64[0]);
          j_j___libc_free_0(v86, 736);
        }
        if ( v116 )
          (*(void (__fastcall **)(_QWORD *))(*v116 + 8LL))(v116);
        goto LABEL_70;
      }
LABEL_102:
      sub_1264230(&v123, v63, v64);
    }
    if ( (v123.m128i_i8[8] & 2) != 0 )
      goto LABEL_102;
    if ( (v123.m128i_i8[8] & 1) != 0 )
    {
      if ( v123.m128i_i64[0] )
        (*(void (**)(void))(*(_QWORD *)v123.m128i_i64[0] + 8LL))();
    }
    else if ( v123.m128i_i64[0] )
    {
      nb = v123.m128i_i64[0];
      sub_1633490(v123.m128i_i64[0]);
      j_j___libc_free_0(nb, 736);
    }
    if ( v116 )
      (*(void (__fastcall **)(_QWORD *))(*v116 + 8LL))(v116);
    if ( v110 == ++v42 )
      goto LABEL_70;
  }
  if ( (v123.m128i_i8[8] & 2) != 0 )
    goto LABEL_102;
  v85 = v123.m128i_i64[0];
  if ( (v123.m128i_i8[8] & 1) != 0 )
  {
    if ( v123.m128i_i64[0] )
    {
      v115 = v69;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v123.m128i_i64[0] + 8LL))(v123.m128i_i64[0]);
      v69 = v115;
    }
  }
  else if ( v123.m128i_i64[0] )
  {
    v113 = v69;
    sub_1633490(v123.m128i_i64[0]);
    j_j___libc_free_0(v85, 736);
    v69 = v113;
  }
  if ( v116 )
  {
    v114 = v69;
    (*(void (__fastcall **)(_QWORD *))(*v116 + 8LL))(v116);
    v69 = v114;
  }
LABEL_84:
  v77 = v117;
  if ( HIDWORD(v118) && (_DWORD)v118 )
  {
    v78 = 8LL * (unsigned int)v118;
    v79 = 0;
    do
    {
      v80 = *(_QWORD *)(v77 + v79);
      if ( v80 != -8 && v80 )
      {
        v111 = v69;
        _libc_free(v80);
        v77 = v117;
        v69 = v111;
      }
      v79 += 8;
    }
    while ( v79 != v78 );
  }
  v112 = v69;
  _libc_free(v77);
  return v112;
}
