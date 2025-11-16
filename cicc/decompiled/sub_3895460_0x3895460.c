// Function: sub_3895460
// Address: 0x3895460
//
unsigned __int64 __fastcall sub_3895460(__int64 a1, __int64 a2, int *a3, int a4, unsigned int a5, _QWORD *a6)
{
  __int64 v8; // rdi
  _BYTE *v10; // rsi
  __int64 v11; // rdx
  size_t v12; // r13
  int *v13; // r14
  int *v14; // r13
  _QWORD *v15; // rax
  _QWORD *v16; // r13
  _QWORD *v17; // r14
  _QWORD *v18; // rsi
  __int64 v19; // rax
  unsigned __int64 v20; // r14
  _QWORD *v21; // r10
  __int64 *v22; // r14
  unsigned __int64 v23; // r13
  __int64 v24; // r11
  _QWORD *v25; // rax
  _QWORD *v26; // rcx
  _QWORD *v27; // r9
  _QWORD *v28; // rdx
  __int64 v29; // rdi
  __int64 v30; // rsi
  _QWORD *v31; // rsi
  _QWORD *v32; // rdx
  __int64 v33; // r9
  __int64 v34; // rdi
  __int64 v35; // rsi
  __int64 v36; // rdx
  __int64 v37; // rax
  _QWORD *v38; // rax
  _QWORD *v39; // rdx
  char v40; // di
  size_t v41; // r13
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r14
  __int64 v45; // rdx
  _QWORD *v46; // rax
  _QWORD *v47; // r14
  _QWORD *v48; // r13
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // rcx
  __int64 v52; // rdx
  unsigned __int64 **v53; // rsi
  unsigned __int64 **i; // rax
  unsigned __int64 *v55; // rdx
  int *v56; // rax
  unsigned __int64 v57; // rdi
  unsigned __int64 v58; // r13
  __int64 v59; // rax
  __int64 v60; // rdi
  __int64 v61; // rcx
  __int64 v62; // rdx
  __int64 *v63; // rax
  __int64 *v64; // r8
  unsigned __int64 j; // rsi
  __int64 v66; // rdx
  int *v67; // rax
  unsigned __int64 v68; // rdi
  unsigned __int64 v69; // r13
  char *v70; // rsi
  __int64 v71; // rdx
  unsigned __int64 result; // rax
  char *v73; // rsi
  __int64 v74; // rax
  _QWORD *v75; // rax
  _QWORD *v76; // r13
  _QWORD *v77; // r14
  unsigned __int64 v78; // rcx
  char *v79; // rax
  _QWORD *v80; // r9
  __int64 v81; // rdi
  __int64 v82; // rax
  _QWORD *v83; // rax
  _QWORD *v84; // rdx
  __int64 v85; // r9
  __int64 v86; // r11
  _QWORD *v87; // rcx
  char v88; // di
  __int64 v89; // rsi
  _QWORD *v90; // r9
  __int64 v91; // rax
  _QWORD *v92; // rax
  _QWORD *v93; // rdx
  char v94; // di
  unsigned __int64 v95; // rdi
  unsigned __int64 v96; // rdi
  __int64 v97; // [rsp+8h] [rbp-148h]
  _QWORD *v98; // [rsp+10h] [rbp-140h]
  __int64 v99; // [rsp+10h] [rbp-140h]
  void *v100; // [rsp+10h] [rbp-140h]
  unsigned __int64 v101; // [rsp+18h] [rbp-138h]
  _QWORD *v102; // [rsp+18h] [rbp-138h]
  unsigned __int64 v103; // [rsp+18h] [rbp-138h]
  _QWORD *v104; // [rsp+18h] [rbp-138h]
  __int64 v105; // [rsp+18h] [rbp-138h]
  _QWORD *v106; // [rsp+18h] [rbp-138h]
  _QWORD *v107; // [rsp+18h] [rbp-138h]
  _QWORD *v108; // [rsp+20h] [rbp-130h]
  __int64 v109; // [rsp+20h] [rbp-130h]
  unsigned __int64 v110; // [rsp+20h] [rbp-130h]
  _QWORD *v111; // [rsp+20h] [rbp-130h]
  __int64 v112; // [rsp+20h] [rbp-130h]
  _QWORD *v113; // [rsp+20h] [rbp-130h]
  _QWORD *v114; // [rsp+20h] [rbp-130h]
  _QWORD *v115; // [rsp+20h] [rbp-130h]
  __int64 v116; // [rsp+28h] [rbp-128h]
  _QWORD *v117; // [rsp+28h] [rbp-128h]
  unsigned __int64 v118; // [rsp+28h] [rbp-128h]
  __int64 v119; // [rsp+28h] [rbp-128h]
  _QWORD *v120; // [rsp+28h] [rbp-128h]
  _QWORD *v121; // [rsp+28h] [rbp-128h]
  __int64 v122; // [rsp+28h] [rbp-128h]
  __int64 v123; // [rsp+30h] [rbp-120h]
  __int64 v124; // [rsp+30h] [rbp-120h]
  int *v125; // [rsp+30h] [rbp-120h]
  __int64 v126; // [rsp+30h] [rbp-120h]
  __int64 v127; // [rsp+30h] [rbp-120h]
  _QWORD *v128; // [rsp+30h] [rbp-120h]
  _QWORD *v129; // [rsp+30h] [rbp-120h]
  unsigned __int64 v130; // [rsp+30h] [rbp-120h]
  _QWORD *v131; // [rsp+30h] [rbp-120h]
  _QWORD *v133; // [rsp+38h] [rbp-118h]
  _QWORD *v134; // [rsp+38h] [rbp-118h]
  _QWORD *v135; // [rsp+38h] [rbp-118h]
  unsigned __int64 v136; // [rsp+48h] [rbp-108h] BYREF
  unsigned __int64 v137[2]; // [rsp+50h] [rbp-100h] BYREF
  int *v138; // [rsp+60h] [rbp-F0h] BYREF
  size_t v139; // [rsp+68h] [rbp-E8h]
  _QWORD v140[2]; // [rsp+70h] [rbp-E0h] BYREF
  __m128i v141; // [rsp+80h] [rbp-D0h] BYREF
  _QWORD *v142; // [rsp+90h] [rbp-C0h]
  _QWORD *v143; // [rsp+98h] [rbp-B8h]
  __int64 v144; // [rsp+A0h] [rbp-B0h]

  v136 = 0;
  if ( a3 )
  {
    v74 = *(_QWORD *)(a1 + 184);
    v138 = a3;
    v127 = v74;
    if ( *(_BYTE *)(v74 + 178) )
    {
      v141.m128i_i64[0] = 0;
    }
    else
    {
      v141.m128i_i64[1] = 0;
      v141.m128i_i64[0] = (__int64)byte_3F871B3;
    }
    v142 = 0;
    v143 = 0;
    v144 = 0;
    v75 = sub_142DA40((_QWORD *)v74, (unsigned __int64 *)&v138, &v141);
    v76 = v143;
    v77 = v142;
    v118 = (unsigned __int64)(v75 + 4);
    if ( v143 != v142 )
    {
      do
      {
        if ( *v77 )
          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v77 + 8LL))(*v77);
        ++v77;
      }
      while ( v76 != v77 );
      v77 = v142;
    }
    if ( v77 )
      j_j___libc_free_0((unsigned __int64)v77);
    v136 = v118 & 0xFFFFFFFFFFFFFFFBLL | (4LL * *(unsigned __int8 *)(v127 + 178));
    goto LABEL_15;
  }
  v8 = *(_QWORD *)(a1 + 176);
  v10 = *(_BYTE **)a2;
  v11 = *(_QWORD *)(a2 + 8);
  if ( v8 )
  {
    v116 = sub_1632000(v8, (__int64)v10, v11);
    v123 = *(_QWORD *)(a1 + 184);
    sub_15E4EB0((__int64 *)&v138, v116);
    v12 = v139;
    v13 = v138;
    sub_16C1840(&v141);
    sub_16C1A90(v141.m128i_i32, v13, v12);
    sub_16C1AA0(&v141, v137);
    v14 = (int *)v137[0];
    if ( v138 != (int *)v140 )
      j_j___libc_free_0((unsigned __int64)v138);
    v138 = v14;
    if ( *(_BYTE *)(v123 + 178) )
    {
      v141.m128i_i64[0] = 0;
    }
    else
    {
      v141.m128i_i64[1] = 0;
      v141.m128i_i64[0] = (__int64)byte_3F871B3;
    }
    v142 = 0;
    v143 = 0;
    v144 = 0;
    v15 = sub_142DA40((_QWORD *)v123, (unsigned __int64 *)&v138, &v141);
    v16 = v143;
    v17 = v142;
    v108 = v15;
    v101 = (unsigned __int64)(v15 + 4);
    if ( v143 != v142 )
    {
      do
      {
        if ( *v17 )
          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v17 + 8LL))(*v17);
        ++v17;
      }
      while ( v16 != v17 );
      v17 = v142;
    }
    if ( v17 )
      j_j___libc_free_0((unsigned __int64)v17);
    v108[5] = v116;
    v136 = v101 & 0xFFFFFFFFFFFFFFFBLL | (4LL * *(unsigned __int8 *)(v123 + 178));
    goto LABEL_15;
  }
  v99 = a2;
  sub_15E4CF0((__int64 *)&v138, v10, v11, a4, *(_BYTE **)(a1 + 1464), *(_QWORD *)(a1 + 1472));
  v41 = v139;
  v125 = v138;
  sub_16C1840(&v141);
  sub_16C1A90(v141.m128i_i32, v125, v41);
  sub_16C1AA0(&v141, v137);
  v110 = v137[0];
  v42 = v99;
  if ( v138 != (int *)v140 )
  {
    j_j___libc_free_0((unsigned __int64)v138);
    v42 = v99;
  }
  v43 = *(_QWORD *)(v42 + 8);
  v44 = *(_QWORD *)(a1 + 184);
  v138 = (int *)v140;
  v126 = v44;
  sub_3887850((__int64 *)&v138, *(_BYTE **)v42, *(_QWORD *)v42 + v43);
  v100 = sub_16D3940((void ***)(v44 + 384), v138, v139);
  v97 = v45;
  v137[0] = v110;
  if ( *(_BYTE *)(v44 + 178) )
  {
    v141.m128i_i64[0] = 0;
  }
  else
  {
    v141.m128i_i64[1] = 0;
    v141.m128i_i64[0] = (__int64)byte_3F871B3;
  }
  v142 = 0;
  v143 = 0;
  v144 = 0;
  v46 = sub_142DA40((_QWORD *)v44, v137, &v141);
  v47 = v143;
  v48 = v142;
  v111 = v46;
  v103 = (unsigned __int64)(v46 + 4);
  if ( v143 != v142 )
  {
    do
    {
      if ( *v48 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v48 + 8LL))(*v48);
      ++v48;
    }
    while ( v47 != v48 );
    v48 = v142;
  }
  if ( v48 )
    j_j___libc_free_0((unsigned __int64)v48);
  v111[5] = v100;
  v111[6] = v97;
  v136 = v103 & 0xFFFFFFFFFFFFFFFBLL | (4LL * *(unsigned __int8 *)(v126 + 178));
  if ( v138 == (int *)v140 )
  {
LABEL_15:
    v18 = a6;
    v19 = *a6;
    if ( !*a6 )
      goto LABEL_56;
    goto LABEL_16;
  }
  j_j___libc_free_0((unsigned __int64)v138);
  v18 = a6;
  v19 = *a6;
  if ( !*a6 )
    goto LABEL_56;
LABEL_16:
  v20 = v136;
  v141.m128i_i64[0] = v19;
  v21 = *(_QWORD **)(a1 + 184);
  *v18 = 0;
  v22 = (__int64 *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
  v23 = *(_QWORD *)(v19 + 16);
  v24 = *v22;
  if ( v23 != *v22 && v23 )
  {
    v25 = (_QWORD *)v21[18];
    v26 = v21 + 17;
    v27 = v21 + 17;
    if ( !v25 )
    {
LABEL_38:
      v133 = v21 + 16;
      goto LABEL_39;
    }
    v28 = (_QWORD *)v21[18];
    do
    {
      while ( 1 )
      {
        v29 = v28[2];
        v30 = v28[3];
        if ( v23 <= v28[4] )
          break;
        v28 = (_QWORD *)v28[3];
        if ( !v30 )
          goto LABEL_23;
      }
      v27 = v28;
      v28 = (_QWORD *)v28[2];
    }
    while ( v29 );
LABEL_23:
    if ( v26 == v27 || v23 < v27[4] )
    {
LABEL_32:
      v27 = v26;
      do
      {
        while ( 1 )
        {
          v35 = v25[2];
          v36 = v25[3];
          if ( v23 <= v25[4] )
            break;
          v25 = (_QWORD *)v25[3];
          if ( !v36 )
            goto LABEL_36;
        }
        v27 = v25;
        v25 = (_QWORD *)v25[2];
      }
      while ( v35 );
LABEL_36:
      if ( v26 != v27 && v23 >= v27[4] )
        goto LABEL_85;
      goto LABEL_38;
    }
    v31 = v21 + 17;
    v32 = (_QWORD *)v21[18];
    do
    {
      while ( 1 )
      {
        v33 = v32[2];
        v34 = v32[3];
        if ( v23 <= v32[4] )
          break;
        v32 = (_QWORD *)v32[3];
        if ( !v34 )
          goto LABEL_29;
      }
      v31 = v32;
      v32 = (_QWORD *)v32[2];
    }
    while ( v33 );
LABEL_29:
    if ( v26 == v31 || v23 < v31[4] )
    {
      v104 = v21 + 17;
      v112 = *v22;
      v134 = v21;
      v82 = sub_22077B0(0x30u);
      *(_QWORD *)(v82 + 32) = v23;
      *(_QWORD *)(v82 + 40) = 0;
      v119 = v82;
      v128 = v134;
      v133 = v134 + 16;
      v83 = sub_142EC30(v133, v31, (unsigned __int64 *)(v82 + 32));
      v85 = v119;
      v86 = v112;
      v87 = v104;
      if ( v84 )
      {
        v88 = v104 == v84 || v83 || v23 < v84[4];
        v89 = v119;
        v105 = v112;
        v113 = v128;
        v120 = v87;
        v129 = (_QWORD *)v85;
        sub_220F040(v88, v89, v84, v87);
        v21 = v113;
        v90 = v129;
        v26 = v120;
        v24 = v105;
        ++v113[21];
      }
      else
      {
        v95 = v119;
        v107 = v83;
        v115 = v87;
        v122 = v86;
        j_j___libc_free_0(v95);
        v26 = v115;
        v24 = v122;
        v21 = v128;
        v90 = v107;
      }
      v25 = (_QWORD *)v21[18];
      if ( v24 == v90[5] )
      {
        if ( !v25 )
        {
          v27 = v26;
LABEL_39:
          v109 = v24;
          v117 = v27;
          v98 = v26;
          v102 = v21;
          v37 = sub_22077B0(0x30u);
          *(_QWORD *)(v37 + 32) = v23;
          *(_QWORD *)(v37 + 40) = 0;
          v124 = v37;
          v38 = sub_142EC30(v133, v117, (unsigned __int64 *)(v37 + 32));
          if ( v39 )
          {
            v40 = v98 == v39 || v38 || v23 < v39[4];
            sub_220F040(v40, v124, v39, v98);
            v27 = (_QWORD *)v124;
            v24 = v109;
            ++v102[21];
          }
          else
          {
            v96 = v124;
            v131 = v38;
            j_j___libc_free_0(v96);
            v24 = v109;
            v27 = v131;
          }
LABEL_85:
          v27[5] = v24;
          goto LABEL_86;
        }
        goto LABEL_32;
      }
      if ( !v25 )
      {
        v80 = v26;
        goto LABEL_133;
      }
    }
    else if ( v24 == v31[5] )
    {
      goto LABEL_32;
    }
    v80 = v26;
    do
    {
      if ( v23 > v25[4] )
      {
        v25 = (_QWORD *)v25[3];
      }
      else
      {
        v80 = v25;
        v25 = (_QWORD *)v25[2];
      }
    }
    while ( v25 );
    if ( v26 != v80 && v23 >= v80[4] )
      goto LABEL_138;
    v133 = v21 + 16;
LABEL_133:
    v121 = v80;
    v106 = v26;
    v114 = v21;
    v91 = sub_22077B0(0x30u);
    *(_QWORD *)(v91 + 32) = v23;
    *(_QWORD *)(v91 + 40) = 0;
    v130 = v91;
    v92 = sub_142EC30(v133, v121, (unsigned __int64 *)(v91 + 32));
    if ( v93 )
    {
      v94 = v106 == v93 || v92 || v23 < v93[4];
      sub_220F040(v94, v130, v93, v106);
      v80 = (_QWORD *)v130;
      ++v114[21];
    }
    else
    {
      v135 = v92;
      j_j___libc_free_0(v130);
      v80 = v135;
    }
LABEL_138:
    v80[5] = 0;
  }
LABEL_86:
  v73 = (char *)v22[4];
  if ( v73 == (char *)v22[5] )
  {
    sub_142DF10(v22 + 3, v73, &v141);
    v81 = v141.m128i_i64[0];
  }
  else
  {
    if ( v73 )
    {
      *(_QWORD *)v73 = v141.m128i_i64[0];
      v22[4] += 8;
      goto LABEL_56;
    }
    v81 = v141.m128i_i64[0];
    v22[4] = 8;
  }
  if ( v81 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v81 + 8LL))(v81);
LABEL_56:
  v49 = *(_QWORD *)(a1 + 1240);
  if ( v49 )
  {
    v50 = a1 + 1232;
    do
    {
      while ( 1 )
      {
        v51 = *(_QWORD *)(v49 + 16);
        v52 = *(_QWORD *)(v49 + 24);
        if ( *(_DWORD *)(v49 + 32) >= a5 )
          break;
        v49 = *(_QWORD *)(v49 + 24);
        if ( !v52 )
          goto LABEL_61;
      }
      v50 = v49;
      v49 = *(_QWORD *)(v49 + 16);
    }
    while ( v51 );
LABEL_61:
    if ( v50 != a1 + 1232 && *(_DWORD *)(v50 + 32) <= a5 )
    {
      v53 = *(unsigned __int64 ***)(v50 + 48);
      for ( i = *(unsigned __int64 ***)(v50 + 40); v53 != i; *v55 = v136 )
      {
        v55 = *i;
        i += 2;
      }
      v56 = sub_220F330((int *)v50, (_QWORD *)(a1 + 1232));
      v57 = *((_QWORD *)v56 + 5);
      v58 = (unsigned __int64)v56;
      if ( v57 )
        j_j___libc_free_0(v57);
      j_j___libc_free_0(v58);
      --*(_QWORD *)(a1 + 1264);
    }
  }
  v59 = *(_QWORD *)(a1 + 1288);
  if ( v59 )
  {
    v60 = a1 + 1280;
    do
    {
      while ( 1 )
      {
        v61 = *(_QWORD *)(v59 + 16);
        v62 = *(_QWORD *)(v59 + 24);
        if ( *(_DWORD *)(v59 + 32) >= a5 )
          break;
        v59 = *(_QWORD *)(v59 + 24);
        if ( !v62 )
          goto LABEL_73;
      }
      v60 = v59;
      v59 = *(_QWORD *)(v59 + 16);
    }
    while ( v61 );
LABEL_73:
    if ( v60 != a1 + 1280 && *(_DWORD *)(v60 + 32) <= a5 )
    {
      v63 = *(__int64 **)(v60 + 40);
      v64 = *(__int64 **)(v60 + 48);
      for ( j = v136 & 0xFFFFFFFFFFFFFFF8LL; v64 != v63; *(_QWORD *)(v66 + 64) = **(_QWORD **)(j + 24) )
      {
        v66 = *v63;
        v63 += 2;
      }
      v67 = sub_220F330((int *)v60, (_QWORD *)(a1 + 1280));
      v68 = *((_QWORD *)v67 + 5);
      v69 = (unsigned __int64)v67;
      if ( v68 )
        j_j___libc_free_0(v68);
      j_j___libc_free_0(v69);
      --*(_QWORD *)(a1 + 1312);
    }
  }
  v70 = *(char **)(a1 + 1328);
  v71 = *(_QWORD *)(a1 + 1320);
  result = (__int64)&v70[-v71] >> 3;
  if ( a5 == result )
  {
    if ( v70 == *(char **)(a1 + 1336) )
    {
      return sub_142E0C0((char **)(a1 + 1320), v70, &v136);
    }
    else
    {
      if ( v70 )
      {
        result = v136;
        *(_QWORD *)v70 = v136;
        v70 = *(char **)(a1 + 1328);
      }
      *(_QWORD *)(a1 + 1328) = v70 + 8;
    }
  }
  else
  {
    if ( a5 > result )
    {
      v78 = a5 + 1;
      if ( v78 > result )
      {
        sub_3891340((_QWORD *)(a1 + 1320), v78 - result);
        v71 = *(_QWORD *)(a1 + 1320);
      }
      else if ( v78 < result )
      {
        v79 = (char *)(v71 + 8 * v78);
        if ( v70 != v79 )
          *(_QWORD *)(a1 + 1328) = v79;
      }
    }
    result = v136;
    *(_QWORD *)(v71 + 8LL * a5) = v136;
  }
  return result;
}
