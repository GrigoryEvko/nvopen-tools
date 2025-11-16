// Function: sub_298D780
// Address: 0x298d780
//
__int64 __fastcall sub_298D780(__int64 a1, _QWORD *a2, __int64 *a3)
{
  char *v3; // r12
  __int64 *v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  const __m128i *v8; // rsi
  __int64 *v9; // rdi
  const __m128i *v10; // rdx
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 v13; // r8
  unsigned __int64 v14; // rbx
  __int64 v15; // rax
  __m128i *v16; // rdi
  __m128i *v17; // rdx
  const __m128i *v18; // rax
  const __m128i *v19; // rcx
  const __m128i *v20; // r8
  unsigned __int64 v21; // rbx
  __int64 v22; // rax
  __m128i *v23; // rsi
  __m128i *v24; // rdx
  const __m128i *v25; // rax
  unsigned __int64 v26; // rax
  __int64 *v27; // rdx
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rdx
  __m128i *v30; // rcx
  char v31; // al
  __int64 v34; // rax
  __int64 v35; // r12
  __int64 *v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 *v40; // rax
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  unsigned __int64 v47; // rax
  __int64 v48; // r8
  __int64 v49; // r9
  const __m128i *v50; // rcx
  __m128i *v51; // rax
  __int64 v52; // rcx
  const __m128i *v53; // rax
  const __m128i *v54; // rcx
  unsigned __int64 v55; // r14
  __int64 v56; // rax
  unsigned __int64 v57; // rdi
  __m128i *v58; // rdx
  __m128i *v59; // rax
  __int64 v60; // rax
  unsigned __int64 v61; // rdi
  unsigned __int64 v62; // rcx
  unsigned __int64 v63; // rax
  char v64; // si
  __int64 *v65; // rax
  __int64 v66; // rbx
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // r9
  __int64 v71; // rcx
  __int64 v72; // r8
  unsigned __int64 v73; // r14
  __int64 v74; // rax
  __m128i *v75; // rdi
  __m128i *v76; // rdx
  const __m128i *v77; // rax
  const __m128i *v78; // r8
  __int64 v79; // rax
  __m128i *v80; // rcx
  __m128i *v81; // rdx
  const __m128i *v82; // rax
  unsigned __int64 v83; // rax
  __int64 v84; // rax
  unsigned __int64 v85; // rax
  unsigned __int64 v86; // rdx
  __m128i *v87; // rsi
  char v88; // al
  char v91; // [rsp+1Fh] [rbp-641h]
  const __m128i *v92; // [rsp+28h] [rbp-638h]
  unsigned int v94; // [rsp+38h] [rbp-628h]
  unsigned int v95; // [rsp+3Ch] [rbp-624h]
  const __m128i *v96; // [rsp+58h] [rbp-608h]
  _BYTE v97[32]; // [rsp+60h] [rbp-600h] BYREF
  _BYTE v98[64]; // [rsp+80h] [rbp-5E0h] BYREF
  __m128i *v99; // [rsp+C0h] [rbp-5A0h]
  __m128i *v100; // [rsp+C8h] [rbp-598h]
  __int8 *v101; // [rsp+D0h] [rbp-590h]
  _BYTE v102[32]; // [rsp+E0h] [rbp-580h] BYREF
  char v103[64]; // [rsp+100h] [rbp-560h] BYREF
  __m128i *v104; // [rsp+140h] [rbp-520h]
  unsigned __int64 j; // [rsp+148h] [rbp-518h]
  __int8 *v106; // [rsp+150h] [rbp-510h]
  _BYTE v107[32]; // [rsp+160h] [rbp-500h] BYREF
  _BYTE v108[64]; // [rsp+180h] [rbp-4E0h] BYREF
  __m128i *v109; // [rsp+1C0h] [rbp-4A0h]
  __m128i *v110; // [rsp+1C8h] [rbp-498h]
  __int8 *v111; // [rsp+1D0h] [rbp-490h]
  _BYTE v112[32]; // [rsp+1E0h] [rbp-480h] BYREF
  _BYTE v113[64]; // [rsp+200h] [rbp-460h] BYREF
  __m128i *v114; // [rsp+240h] [rbp-420h]
  unsigned __int64 v115; // [rsp+248h] [rbp-418h]
  __int8 *v116; // [rsp+250h] [rbp-410h]
  char v117[8]; // [rsp+260h] [rbp-400h] BYREF
  unsigned __int64 v118; // [rsp+268h] [rbp-3F8h]
  char v119; // [rsp+27Ch] [rbp-3E4h]
  char v120[64]; // [rsp+280h] [rbp-3E0h] BYREF
  __m128i *v121; // [rsp+2C0h] [rbp-3A0h]
  __int64 v122; // [rsp+2C8h] [rbp-398h]
  __int8 *v123; // [rsp+2D0h] [rbp-390h]
  __int64 v124; // [rsp+2E0h] [rbp-380h] BYREF
  __int64 *v125; // [rsp+2E8h] [rbp-378h]
  __int64 v126; // [rsp+2F0h] [rbp-370h]
  int v127; // [rsp+2F8h] [rbp-368h]
  char v128; // [rsp+2FCh] [rbp-364h]
  _QWORD v129[8]; // [rsp+300h] [rbp-360h] BYREF
  unsigned __int64 v130; // [rsp+340h] [rbp-320h] BYREF
  unsigned __int64 i; // [rsp+348h] [rbp-318h]
  unsigned __int64 v132; // [rsp+350h] [rbp-310h]
  unsigned __int64 v133[15]; // [rsp+360h] [rbp-300h] BYREF
  _BYTE v134[96]; // [rsp+3D8h] [rbp-288h] BYREF
  const __m128i *v135; // [rsp+438h] [rbp-228h]
  const __m128i *v136; // [rsp+440h] [rbp-220h]
  __m128i v137; // [rsp+450h] [rbp-210h] BYREF
  char v138; // [rsp+468h] [rbp-1F8h]
  char v139; // [rsp+46Ch] [rbp-1F4h]
  char v140[64]; // [rsp+470h] [rbp-1F0h] BYREF
  const __m128i *v141; // [rsp+4B0h] [rbp-1B0h]
  const __m128i *v142; // [rsp+4B8h] [rbp-1A8h]
  unsigned __int64 v143; // [rsp+4C0h] [rbp-1A0h]
  char v144[8]; // [rsp+4C8h] [rbp-198h] BYREF
  unsigned __int64 v145; // [rsp+4D0h] [rbp-190h]
  char v146; // [rsp+4E4h] [rbp-17Ch]
  char v147[64]; // [rsp+4E8h] [rbp-178h] BYREF
  const __m128i *v148; // [rsp+528h] [rbp-138h]
  unsigned __int64 v149; // [rsp+530h] [rbp-130h]
  unsigned __int64 v150; // [rsp+538h] [rbp-128h]
  _QWORD v151[12]; // [rsp+540h] [rbp-120h] BYREF
  __int64 v152; // [rsp+5A0h] [rbp-C0h]
  __int64 v153; // [rsp+5A8h] [rbp-B8h]
  _BYTE v154[96]; // [rsp+5B8h] [rbp-A8h] BYREF
  const __m128i *v155; // [rsp+618h] [rbp-48h]
  const __m128i *v156; // [rsp+620h] [rbp-40h]

  *(_QWORD *)(a1 + 48) = a3;
  v4 = (__int64 *)sub_AA48A0(*a2 & 0xFFFFFFFFFFFFFFF8LL);
  v94 = sub_B6ED60(v4, "structurizecfg.uniform", 0x16u);
  sub_22DE850((__int64)&v137, a2);
  sub_22DE7C0((__int64)v133, a2);
  sub_23FD870(v151, v133, &v137);
  if ( v133[12] )
    j_j___libc_free_0(v133[12]);
  if ( !BYTE4(v133[3]) )
    _libc_free(v133[1]);
  if ( v141 )
    j_j___libc_free_0((unsigned __int64)v141);
  if ( !v139 )
    _libc_free(v137.m128i_u64[1]);
  v8 = (const __m128i *)v108;
  v9 = (__int64 *)v107;
  sub_C8CD80((__int64)v107, (__int64)v108, (__int64)v151, v5, v6, v7);
  v12 = v153;
  v13 = v152;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v14 = v153 - v152;
  if ( v153 == v152 )
  {
    v16 = 0;
  }
  else
  {
    if ( v14 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_167;
    v15 = sub_22077B0(v153 - v152);
    v12 = v153;
    v13 = v152;
    v16 = (__m128i *)v15;
  }
  v109 = v16;
  v110 = v16;
  v111 = &v16->m128i_i8[v14];
  if ( v13 != v12 )
  {
    v17 = v16;
    v18 = (const __m128i *)v13;
    do
    {
      if ( v17 )
      {
        *v17 = _mm_loadu_si128(v18);
        v17[1] = _mm_loadu_si128(v18 + 1);
        v17[2].m128i_i64[0] = v18[2].m128i_i64[0];
      }
      v18 = (const __m128i *)((char *)v18 + 40);
      v17 = (__m128i *)((char *)v17 + 40);
    }
    while ( v18 != (const __m128i *)v12 );
    v16 = (__m128i *)((char *)v16 + 8 * (((unsigned __int64)&v18[-3].m128i_u64[1] - v13) >> 3) + 40);
  }
  v110 = v16;
  v8 = (const __m128i *)v113;
  v9 = (__int64 *)v112;
  sub_C8CD80((__int64)v112, (__int64)v113, (__int64)v154, v12, v13, v11);
  v19 = v156;
  v20 = v155;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v21 = (char *)v156 - (char *)v155;
  if ( v156 == v155 )
  {
    v23 = 0;
  }
  else
  {
    if ( v21 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_167;
    v22 = sub_22077B0((char *)v156 - (char *)v155);
    v19 = v156;
    v20 = v155;
    v23 = (__m128i *)v22;
  }
  v114 = v23;
  v115 = (unsigned __int64)v23;
  v116 = &v23->m128i_i8[v21];
  if ( v20 == v19 )
  {
    v26 = (unsigned __int64)v23;
  }
  else
  {
    v24 = v23;
    v25 = v20;
    do
    {
      if ( v24 )
      {
        *v24 = _mm_loadu_si128(v25);
        v24[1] = _mm_loadu_si128(v25 + 1);
        v24[2].m128i_i64[0] = v25[2].m128i_i64[0];
      }
      v25 = (const __m128i *)((char *)v25 + 40);
      v24 = (__m128i *)((char *)v24 + 40);
    }
    while ( v25 != v19 );
    v26 = (unsigned __int64)&v23[2].m128i_u64[((unsigned __int64)((char *)&v25[-3].m128i_u64[1] - (char *)v20) >> 3) + 1];
  }
  v115 = v26;
  v95 = 0;
  v91 = 1;
  while ( 1 )
  {
    v30 = v109;
    if ( (char *)v110 - (char *)v109 != v26 - (_QWORD)v23 )
      goto LABEL_28;
    if ( v109 == v110 )
      break;
    while ( v30->m128i_i64[0] == v23->m128i_i64[0] )
    {
      v31 = v30[2].m128i_i8[0];
      if ( v31 != v23[2].m128i_i8[0] )
        break;
      if ( v31 )
      {
        if ( !(((v30->m128i_i64[1] >> 1) & 3) != 0
             ? ((v23->m128i_i64[1] >> 1) & 3) == ((v30->m128i_i64[1] >> 1) & 3)
             : v30[1].m128i_i32[2] == v23[1].m128i_i32[2]) )
          break;
      }
      v30 = (__m128i *)((char *)v30 + 40);
      v23 = (__m128i *)((char *)v23 + 40);
      if ( v110 == v30 )
        goto LABEL_44;
    }
LABEL_28:
    v27 = (__int64 *)v110[-3].m128i_i64[1];
    if ( (*v27 & 4) != 0 )
    {
      memset(v133, 0, sizeof(v133));
      LODWORD(v133[2]) = 8;
      BYTE4(v133[3]) = 1;
      v133[1] = (unsigned __int64)&v133[4];
      v34 = *v27;
      v35 = v27[4];
      v125 = v129;
      v130 = 0;
      i = 0;
      v132 = 0;
      v126 = 0x100000008LL;
      v127 = 0;
      v128 = 1;
      v129[0] = v34 & 0xFFFFFFFFFFFFFFF8LL;
      v124 = 1;
      v137.m128i_i64[0] = v34 & 0xFFFFFFFFFFFFFFF8LL;
      v138 = 0;
      sub_298D040((__int64)&v130, &v137);
      if ( !v128 )
        goto LABEL_114;
      v40 = v125;
      v37 = HIDWORD(v126);
      v36 = &v125[HIDWORD(v126)];
      if ( v125 != v36 )
      {
        while ( v35 != *v40 )
        {
          if ( v36 == ++v40 )
            goto LABEL_117;
        }
        goto LABEL_53;
      }
LABEL_117:
      if ( HIDWORD(v126) < (unsigned int)v126 )
      {
        ++HIDWORD(v126);
        *v36 = v35;
        ++v124;
      }
      else
      {
LABEL_114:
        sub_C8CC70((__int64)&v124, v35, (__int64)v36, v37, v38, v39);
      }
LABEL_53:
      sub_C8CF70((__int64)&v137, v140, 8, (__int64)v129, (__int64)&v124);
      v41 = v130;
      v130 = 0;
      v141 = (const __m128i *)v41;
      v42 = i;
      i = 0;
      v142 = (const __m128i *)v42;
      v43 = v132;
      v132 = 0;
      v143 = v43;
      sub_C8CF70((__int64)v144, v147, 8, (__int64)&v133[4], (__int64)v133);
      v47 = v133[12];
      memset(&v133[12], 0, 24);
      v148 = (const __m128i *)v47;
      v149 = v133[13];
      v150 = v133[14];
      if ( v130 )
        j_j___libc_free_0(v130);
      if ( !v128 )
        _libc_free((unsigned __int64)v125);
      if ( v133[12] )
        j_j___libc_free_0(v133[12]);
      if ( !BYTE4(v133[3]) )
        _libc_free(v133[1]);
      v3 = v117;
      v9 = (__int64 *)v117;
      sub_C8CD80((__int64)v117, (__int64)v120, (__int64)&v137, v44, v45, v46);
      v50 = v142;
      v10 = v141;
      v121 = 0;
      v122 = 0;
      v123 = 0;
      v8 = (const __m128i *)((char *)v142 - (char *)v141);
      if ( v142 == v141 )
      {
        v51 = 0;
      }
      else
      {
        if ( (unsigned __int64)v8 > 0x7FFFFFFFFFFFFFE0LL )
          goto LABEL_167;
        v92 = (const __m128i *)((char *)v142 - (char *)v141);
        v51 = (__m128i *)sub_22077B0((char *)v142 - (char *)v141);
        v50 = v142;
        v10 = v141;
        v8 = v92;
      }
      v121 = v51;
      v122 = (__int64)v51;
      v123 = &v8->m128i_i8[(_QWORD)v51];
      if ( v50 == v10 )
      {
        v52 = (__int64)v51;
      }
      else
      {
        v52 = (__int64)v51->m128i_i64 + (char *)v50 - (char *)v10;
        do
        {
          if ( v51 )
          {
            *v51 = _mm_loadu_si128(v10);
            v51[1] = _mm_loadu_si128(v10 + 1);
          }
          v51 += 2;
          v10 += 2;
        }
        while ( (__m128i *)v52 != v51 );
      }
      v9 = &v124;
      v8 = (const __m128i *)v129;
      v122 = v52;
      sub_C8CD80((__int64)&v124, (__int64)v129, (__int64)v144, v52, v48, v49);
      v53 = (const __m128i *)v149;
      v54 = v148;
      v130 = 0;
      i = 0;
      v132 = 0;
      v55 = v149 - (_QWORD)v148;
      if ( (const __m128i *)v149 == v148 )
      {
        v57 = 0;
        goto LABEL_72;
      }
      if ( v55 <= 0x7FFFFFFFFFFFFFE0LL )
      {
        v56 = sub_22077B0(v149 - (_QWORD)v148);
        v54 = v148;
        v57 = v56;
        v53 = (const __m128i *)v149;
LABEL_72:
        v130 = v57;
        i = v57;
        v132 = v57 + v55;
        if ( v53 == v54 )
        {
          v59 = (__m128i *)v57;
        }
        else
        {
          v58 = (__m128i *)v57;
          v59 = (__m128i *)(v57 + (char *)v53 - (char *)v54);
          do
          {
            if ( v58 )
            {
              *v58 = _mm_loadu_si128(v54);
              v58[1] = _mm_loadu_si128(v54 + 1);
            }
            v58 += 2;
            v54 += 2;
          }
          while ( v59 != v58 );
        }
        for ( i = (unsigned __int64)v59; ; v59 = (__m128i *)i )
        {
          v62 = (unsigned __int64)v121;
          if ( (__m128i *)(v122 - (_QWORD)v121) == (__m128i *)((char *)v59 - v57) )
          {
            if ( v121 == (__m128i *)v122 )
              goto LABEL_94;
            v63 = v57;
            while ( *(_QWORD *)v62 == *(_QWORD *)v63 )
            {
              v64 = *(_BYTE *)(v62 + 24);
              if ( v64 != *(_BYTE *)(v63 + 24) || v64 && *(_DWORD *)(v62 + 16) != *(_DWORD *)(v63 + 16) )
                break;
              v62 += 32LL;
              v63 += 32LL;
              if ( v122 == v62 )
                goto LABEL_94;
            }
          }
          v60 = *(_QWORD *)(v122 - 32);
          v61 = *(_QWORD *)(v60 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v61 == v60 + 48 || !v61 || (unsigned int)*(unsigned __int8 *)(v61 - 24) - 30 > 0xA )
            goto LABEL_168;
          if ( *(_BYTE *)(v61 - 24) == 31 && (*(_DWORD *)(v61 - 20) & 0x7FFFFFF) == 3 )
          {
            if ( v94 )
            {
              if ( (*(_BYTE *)(v61 - 17) & 0x20) == 0 || !sub_B91C10(v61 - 24, v94) )
              {
LABEL_112:
                if ( (_BYTE)qword_5007828 )
                {
                  v91 = 0;
                  v57 = v130;
LABEL_94:
                  if ( v57 )
                    j_j___libc_free_0(v57);
                  if ( !v128 )
                    _libc_free((unsigned __int64)v125);
                  if ( v121 )
                    j_j___libc_free_0((unsigned __int64)v121);
                  if ( !v119 )
                    _libc_free(v118);
                  if ( v148 )
                    j_j___libc_free_0((unsigned __int64)v148);
                  if ( !v146 )
                    _libc_free(v145);
                  if ( v141 )
                    j_j___libc_free_0((unsigned __int64)v141);
                  if ( !v139 )
                    _libc_free(v137.m128i_u64[1]);
                  goto LABEL_34;
                }
                sub_23FD540((__int64)&v124);
                sub_23FD540((__int64)v117);
                sub_23FD540((__int64)v144);
                sub_23FD540((__int64)&v137);
LABEL_161:
                LODWORD(v3) = 0;
                sub_23FD500((__int64)v112);
                sub_23FD500((__int64)v107);
                sub_23FD500((__int64)v154);
                sub_23FD500((__int64)v151);
                return (unsigned int)v3;
              }
            }
            else if ( !*(_QWORD *)(v61 + 24) )
            {
              goto LABEL_112;
            }
          }
          sub_23EC7E0((__int64)v117);
          v57 = v130;
        }
      }
LABEL_167:
      sub_4261EA(v9, v8, v10);
    }
    v28 = *v27 & 0xFFFFFFFFFFFFFFF8LL;
    v29 = *(_QWORD *)(v28 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v29 == v28 + 48 || !v29 || (unsigned int)*(unsigned __int8 *)(v29 - 24) - 30 > 0xA )
LABEL_168:
      BUG();
    if ( *(_BYTE *)(v29 - 24) == 31 && (*(_DWORD *)(v29 - 20) & 0x7FFFFFF) == 3 )
    {
      if ( !(unsigned __int8)sub_10563D0(a3, (unsigned __int8 *)(v29 - 24)) )
      {
        ++v95;
        goto LABEL_34;
      }
      goto LABEL_161;
    }
LABEL_34:
    sub_22DE410((__int64)v107);
    v23 = v114;
    v26 = v115;
  }
LABEL_44:
  sub_23FD500((__int64)v112);
  sub_23FD500((__int64)v107);
  sub_23FD500((__int64)v154);
  sub_23FD500((__int64)v151);
  LOBYTE(v3) = v91 | (v95 <= 1);
  if ( (_BYTE)v3 )
  {
    v65 = (__int64 *)sub_B2BE50(*(_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 72));
    v66 = sub_B9C770(v65, 0, 0, 0, 1);
    sub_22DE850((__int64)v151, a2);
    sub_22DE7C0((__int64)&v137, a2);
    sub_23FD870(v133, &v137, v151);
    sub_23FD500((__int64)&v137);
    sub_23FD500((__int64)v151);
    v8 = (const __m128i *)v98;
    v9 = (__int64 *)v97;
    sub_C8CD80((__int64)v97, (__int64)v98, (__int64)v133, v67, v68, v69);
    v71 = v133[13];
    v72 = v133[12];
    v99 = 0;
    v100 = 0;
    v101 = 0;
    v73 = v133[13] - v133[12];
    if ( v133[13] == v133[12] )
    {
      v75 = 0;
    }
    else
    {
      if ( v73 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_167;
      v74 = sub_22077B0(v133[13] - v133[12]);
      v71 = v133[13];
      v72 = v133[12];
      v75 = (__m128i *)v74;
    }
    v99 = v75;
    v100 = v75;
    v101 = &v75->m128i_i8[v73];
    if ( v71 != v72 )
    {
      v76 = v75;
      v77 = (const __m128i *)v72;
      do
      {
        if ( v76 )
        {
          *v76 = _mm_loadu_si128(v77);
          v76[1] = _mm_loadu_si128(v77 + 1);
          v76[2].m128i_i64[0] = v77[2].m128i_i64[0];
        }
        v77 = (const __m128i *)((char *)v77 + 40);
        v76 = (__m128i *)((char *)v76 + 40);
      }
      while ( (const __m128i *)v71 != v77 );
      v75 = (__m128i *)((char *)v75 + 8 * ((unsigned __int64)(v71 - 40 - v72) >> 3) + 40);
    }
    v100 = v75;
    v9 = (__int64 *)v102;
    sub_C8CD80((__int64)v102, (__int64)v103, (__int64)v134, v71, v72, v70);
    v8 = v136;
    v78 = v135;
    v104 = 0;
    j = 0;
    v106 = 0;
    v10 = (const __m128i *)((char *)v136 - (char *)v135);
    if ( v136 == v135 )
    {
      v80 = 0;
    }
    else
    {
      if ( (unsigned __int64)v10 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_167;
      v96 = (const __m128i *)((char *)v136 - (char *)v135);
      v79 = sub_22077B0((char *)v136 - (char *)v135);
      v8 = v136;
      v78 = v135;
      v10 = v96;
      v80 = (__m128i *)v79;
    }
    v104 = v80;
    j = (unsigned __int64)v80;
    v106 = &v10->m128i_i8[(_QWORD)v80];
    if ( v78 == v8 )
    {
      v83 = (unsigned __int64)v80;
    }
    else
    {
      v81 = v80;
      v82 = v78;
      do
      {
        if ( v81 )
        {
          *v81 = _mm_loadu_si128(v82);
          v81[1] = _mm_loadu_si128(v82 + 1);
          v81[2].m128i_i64[0] = v82[2].m128i_i64[0];
        }
        v82 = (const __m128i *)((char *)v82 + 40);
        v81 = (__m128i *)((char *)v81 + 40);
      }
      while ( v8 != v82 );
      v83 = (unsigned __int64)&v80[2].m128i_u64[((unsigned __int64)((char *)&v8[-3].m128i_u64[1] - (char *)v78) >> 3)
                                              + 1];
    }
    for ( j = v83; ; v83 = j )
    {
      v87 = v99;
      if ( (char *)v100 - (char *)v99 == v83 - (_QWORD)v80 )
      {
        if ( v99 == v100 )
        {
LABEL_158:
          LODWORD(v3) = (unsigned __int8)v3;
          sub_23FD500((__int64)v102);
          sub_23FD500((__int64)v97);
          sub_23FD500((__int64)v134);
          sub_23FD500((__int64)v133);
          return (unsigned int)v3;
        }
        while ( v87->m128i_i64[0] == v80->m128i_i64[0] )
        {
          v88 = v87[2].m128i_i8[0];
          if ( v88 != v80[2].m128i_i8[0] )
            break;
          if ( v88 )
          {
            if ( !(((v87->m128i_i64[1] >> 1) & 3) != 0
                 ? ((v80->m128i_i64[1] >> 1) & 3) == ((v87->m128i_i64[1] >> 1) & 3)
                 : v87[1].m128i_i32[2] == v80[1].m128i_i32[2]) )
              break;
          }
          v87 = (__m128i *)((char *)v87 + 40);
          v80 = (__m128i *)((char *)v80 + 40);
          if ( v100 == v87 )
            goto LABEL_158;
        }
      }
      v84 = *(_QWORD *)v100[-3].m128i_i64[1];
      if ( (v84 & 4) == 0 )
      {
        v85 = v84 & 0xFFFFFFFFFFFFFFF8LL;
        v86 = *(_QWORD *)(v85 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v86 != v85 + 48 )
        {
          if ( !v86 )
            BUG();
          if ( (unsigned int)*(unsigned __int8 *)(v86 - 24) - 30 <= 0xA )
            sub_B99FD0(v86 - 24, v94, v66);
        }
      }
      sub_22DE410((__int64)v97);
      v80 = v104;
    }
  }
  return (unsigned int)v3;
}
