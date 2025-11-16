// Function: sub_2759640
// Address: 0x2759640
//
__int64 __fastcall sub_2759640(__int64 a1, __int64 a2, _QWORD **a3, unsigned __int8 *a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v8; // rax
  __int64 v9; // rcx
  unsigned __int64 v10; // r8
  __int64 v11; // r9
  __m128i v12; // kr00_16
  __int64 v13; // rax
  __int64 v14; // rcx
  __m128i *v15; // r12
  unsigned __int64 v16; // rsi
  __int64 v17; // rdx
  _QWORD *v18; // rax
  __int64 v19; // rdx
  unsigned int v20; // eax
  char v21; // r15
  __m128i *v22; // rcx
  __int64 v23; // rsi
  char *v24; // rdx
  __int64 v25; // rax
  __int8 *v26; // rax
  unsigned __int64 v27; // rdi
  __int64 v28; // rbx
  __int64 v29; // r14
  __int64 v30; // r12
  __int64 v31; // rbx
  unsigned __int8 *v32; // r14
  char v33; // al
  unsigned int v34; // r13d
  __int64 v35; // rbx
  unsigned __int64 v36; // r12
  unsigned __int64 v37; // rdi
  __int64 v39; // r12
  unsigned __int8 *v40; // rax
  int v41; // esi
  __int64 v42; // rdx
  __int64 v43; // r14
  unsigned int v44; // esi
  __int64 v45; // rax
  __int64 v46; // rdi
  __int64 *v47; // rdx
  int v48; // eax
  __int64 v49; // r8
  unsigned __int8 *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rsi
  __int64 v53; // rdx
  _QWORD *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rdx
  int v57; // esi
  _QWORD *v58; // rax
  __int64 v59; // rdx
  _QWORD *v60; // rsi
  __int64 v61; // rcx
  __int64 v62; // rdx
  _QWORD *v63; // rdx
  __int64 v64; // rax
  __int8 *v65; // r12
  unsigned int v66; // esi
  __int64 *v67; // rdi
  __int8 *v68; // r14
  unsigned int v69; // esi
  __int64 v71; // [rsp+30h] [rbp-790h]
  __int64 v72; // [rsp+38h] [rbp-788h]
  __int64 v73; // [rsp+40h] [rbp-780h]
  unsigned __int8 *v74; // [rsp+40h] [rbp-780h]
  __int64 v75; // [rsp+50h] [rbp-770h]
  unsigned __int8 *v76; // [rsp+68h] [rbp-758h]
  __int64 v77; // [rsp+70h] [rbp-750h]
  char *v78; // [rsp+78h] [rbp-748h]
  unsigned __int64 v79; // [rsp+88h] [rbp-738h]
  __int64 v81; // [rsp+98h] [rbp-728h]
  int v82; // [rsp+98h] [rbp-728h]
  unsigned __int8 *v83; // [rsp+98h] [rbp-728h]
  __int64 v84; // [rsp+A0h] [rbp-720h] BYREF
  __int64 v85; // [rsp+A8h] [rbp-718h]
  __int64 v86; // [rsp+B0h] [rbp-710h]
  unsigned int v87; // [rsp+B8h] [rbp-708h]
  __m128i v88; // [rsp+C0h] [rbp-700h] BYREF
  unsigned __int8 *v89; // [rsp+D0h] [rbp-6F0h]
  __int64 v90; // [rsp+D8h] [rbp-6E8h]
  char *v91; // [rsp+E0h] [rbp-6E0h]
  unsigned __int64 v92; // [rsp+E8h] [rbp-6D8h]
  __m128i *v93; // [rsp+F0h] [rbp-6D0h] BYREF
  unsigned __int8 *v94; // [rsp+F8h] [rbp-6C8h]
  __int64 v95; // [rsp+100h] [rbp-6C0h]
  char *v96; // [rsp+108h] [rbp-6B8h]
  _BYTE *v97; // [rsp+110h] [rbp-6B0h] BYREF
  __int64 v98; // [rsp+118h] [rbp-6A8h]
  _BYTE v99[32]; // [rsp+120h] [rbp-6A0h] BYREF
  __int64 v100; // [rsp+140h] [rbp-680h]
  unsigned __int8 *v101; // [rsp+148h] [rbp-678h]
  unsigned __int8 *v102; // [rsp+150h] [rbp-670h]
  __int64 v103; // [rsp+158h] [rbp-668h]
  char *v104; // [rsp+160h] [rbp-660h] BYREF
  unsigned __int64 v105; // [rsp+168h] [rbp-658h] BYREF
  __int64 v106; // [rsp+170h] [rbp-650h] BYREF
  _BYTE v107[40]; // [rsp+178h] [rbp-648h] BYREF
  __m128i v108; // [rsp+1A0h] [rbp-620h] BYREF
  unsigned __int8 *v109; // [rsp+1B0h] [rbp-610h]
  __int64 v110; // [rsp+1B8h] [rbp-608h]
  char *v111; // [rsp+1C0h] [rbp-600h]
  _BYTE *v112; // [rsp+1C8h] [rbp-5F8h] BYREF
  __int64 v113; // [rsp+1D0h] [rbp-5F0h]
  _BYTE v114[40]; // [rsp+1D8h] [rbp-5E8h] BYREF
  __int64 v115; // [rsp+200h] [rbp-5C0h] BYREF
  __int64 v116; // [rsp+208h] [rbp-5B8h]
  _BYTE v117[1456]; // [rsp+210h] [rbp-5B0h] BYREF

  v115 = (__int64)v117;
  v116 = 0x1000000000LL;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  if ( !a1 )
    BUG();
  v5 = *(_QWORD *)(a2 + 40);
  v72 = *(_QWORD *)(a1 + 32);
  v8 = 0;
  if ( a2 )
    v8 = a2 + 24;
  v71 = v8;
  v75 = *(_QWORD *)(a1 + 40);
  if ( *(_BYTE *)a2 == 85
    && (v64 = *(_QWORD *)(a2 - 32)) != 0
    && !*(_BYTE *)v64
    && *(_QWORD *)(v64 + 24) == *(_QWORD *)(a2 + 80)
    && (*(_BYTE *)(v64 + 33) & 0x20) != 0
    && ((*(_DWORD *)(v64 + 36) - 243) & 0xFFFFFFFD) == 0 )
  {
    sub_D67210(&v88, a2);
    v76 = v89;
    v77 = v90;
    v78 = v91;
    v79 = v92;
    v12 = v88;
  }
  else
  {
    sub_D66840(&v108, (_BYTE *)a2);
    v12 = v108;
    v76 = v109;
    v77 = v110;
    v78 = v111;
    v79 = (unsigned __int64)v112;
  }
  v100 = v12.m128i_i64[0];
  v101 = a4;
  v102 = 0;
  v103 = 0;
  v104 = (char *)&v106;
  v105 = 0x400000000LL;
  if ( *(_BYTE *)v12.m128i_i64[0] <= 0x1Cu )
  {
    v108.m128i_i64[1] = v12.m128i_i64[0];
    v108.m128i_i64[0] = v5;
    v109 = a4;
    v110 = 0;
    v111 = 0;
    v112 = v114;
    v113 = 0x400000000LL;
  }
  else
  {
    v106 = v12.m128i_i64[0];
    v108.m128i_i64[1] = v12.m128i_i64[0];
    LODWORD(v105) = 1;
    v108.m128i_i64[0] = v5;
    v109 = a4;
    v110 = 0;
    v111 = 0;
    v112 = v114;
    v113 = 0x400000000LL;
    sub_2753AD0((__int64)&v112, &v104, 0x400000000LL, v9, v10, v11);
  }
  v13 = (unsigned int)v116;
  v14 = v115;
  v15 = &v108;
  v16 = (unsigned int)v116 + 1LL;
  v17 = (unsigned int)v116;
  if ( v16 > HIDWORD(v116) )
  {
    if ( v115 > (unsigned __int64)&v108
      || (v17 = 5LL * (unsigned int)v116, (unsigned __int64)&v108 >= v115 + 88 * (unsigned __int64)(unsigned int)v116) )
    {
      sub_102D890((__int64)&v115, v16, v17, v115, v10, v11);
      v13 = (unsigned int)v116;
      v14 = v115;
      v15 = &v108;
      LODWORD(v17) = v116;
    }
    else
    {
      v65 = &v108.m128i_i8[-v115];
      sub_102D890((__int64)&v115, v16, v17, v115, v10, v11);
      v14 = v115;
      v13 = (unsigned int)v116;
      v15 = (__m128i *)&v65[v115];
      LODWORD(v17) = v116;
    }
  }
  v18 = (_QWORD *)(v14 + 88 * v13);
  if ( v18 )
  {
    *v18 = v15->m128i_i64[0];
    v18[1] = v15->m128i_i64[1];
    v18[2] = v15[1].m128i_i64[0];
    v18[3] = v15[1].m128i_i64[1];
    v19 = v15[2].m128i_i64[0];
    v18[6] = 0x400000000LL;
    v18[4] = v19;
    v18[5] = v18 + 7;
    v10 = v15[3].m128i_u32[0];
    if ( (_DWORD)v10 )
      sub_2753AD0((__int64)(v18 + 5), (char **)&v15[2].m128i_i64[1], (__int64)(v18 + 7), v14, v10, v11);
    LODWORD(v17) = v116;
  }
  LODWORD(v116) = v17 + 1;
  if ( v112 != v114 )
    _libc_free((unsigned __int64)v112);
  if ( v104 != (char *)&v106 )
    _libc_free((unsigned __int64)v104);
  v20 = v116;
  v21 = 1;
  if ( !(_DWORD)v116 )
  {
LABEL_51:
    v34 = 1;
    goto LABEL_36;
  }
  while ( 1 )
  {
    v22 = (__m128i *)v115;
    v23 = v115 + 88LL * v20 - 88;
    v100 = *(_QWORD *)v23;
    v101 = *(unsigned __int8 **)(v23 + 8);
    v102 = *(unsigned __int8 **)(v23 + 16);
    v103 = *(_QWORD *)(v23 + 24);
    v24 = *(char **)(v23 + 32);
    v105 = (unsigned __int64)v107;
    v104 = v24;
    v106 = 0x400000000LL;
    if ( *(_DWORD *)(v23 + 48) )
    {
      sub_2753AD0((__int64)&v105, (char **)(v23 + 40), (__int64)v24, v115, v10, v11);
      v22 = (__m128i *)v115;
      v20 = v116;
    }
    v25 = v20 - 1;
    LODWORD(v116) = v25;
    v26 = &v22->m128i_i8[88 * v25];
    v27 = *((_QWORD *)v26 + 5);
    if ( (__int8 *)v27 != v26 + 56 )
      _libc_free(v27);
    v28 = v100;
    v29 = v72;
    v81 = (__int64)v101;
    if ( v100 != v75 )
      v29 = *(_QWORD *)(v100 + 56);
    v30 = v100 + 48;
    if ( v21 )
      v30 = v71;
    if ( v29 != v30 )
    {
      v73 = v100;
      v31 = v29;
      while ( 1 )
      {
        if ( v31 )
        {
          v32 = (unsigned __int8 *)(v31 - 24);
          v33 = sub_B46490(v31 - 24);
          if ( a2 != v31 - 24 && v33 )
            goto LABEL_32;
        }
        else if ( (unsigned __int8)sub_B46490(0) )
        {
          v32 = 0;
LABEL_32:
          LOBYTE(v113) = 1;
          v108.m128i_i64[0] = v81;
          v108.m128i_i64[1] = v12.m128i_i64[1];
          v109 = v76;
          v110 = v77;
          v111 = v78;
          v112 = (_BYTE *)v79;
          if ( (sub_CF63E0(*a3, v32, &v108, (__int64)(a3 + 1)) & 2) != 0 )
            goto LABEL_33;
        }
        v31 = *(_QWORD *)(v31 + 8);
        if ( v31 == v30 )
        {
          v28 = v73;
          break;
        }
      }
    }
    if ( v28 != v75 )
    {
      v39 = *(_QWORD *)(v28 + 16);
      if ( v39 )
        break;
    }
LABEL_48:
    if ( (_BYTE *)v105 != v107 )
      _libc_free(v105);
    v20 = v116;
    v21 = 0;
    if ( !(_DWORD)v116 )
      goto LABEL_51;
  }
  while ( 1 )
  {
    v40 = *(unsigned __int8 **)(v39 + 24);
    v41 = *v40;
    v42 = (unsigned int)(v41 - 30);
    if ( (unsigned __int8)(v41 - 30) <= 0xAu )
      break;
    v39 = *(_QWORD *)(v39 + 8);
    if ( !v39 )
      goto LABEL_48;
  }
LABEL_54:
  v43 = *((_QWORD *)v40 + 5);
  v97 = v99;
  v93 = (__m128i *)v101;
  v94 = v102;
  v95 = v103;
  v96 = v104;
  v98 = 0x400000000LL;
  if ( !(_DWORD)v106 )
  {
LABEL_55:
    v44 = v87;
    v22 = v93;
    if ( v87 )
    {
      v11 = v85;
      v10 = (v87 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
      v45 = v85 + 16 * v10;
      v46 = *(_QWORD *)v45;
      if ( v43 == *(_QWORD *)v45 )
      {
LABEL_57:
        if ( *(__m128i **)(v45 + 8) != v93 )
          goto LABEL_58;
        goto LABEL_77;
      }
      v82 = 1;
      v47 = 0;
      while ( v46 != -4096 )
      {
        if ( v46 == -8192 && !v47 )
          v47 = (__int64 *)v45;
        v10 = (v87 - 1) & (v82 + (_DWORD)v10);
        v45 = v85 + 16LL * (unsigned int)v10;
        v46 = *(_QWORD *)v45;
        if ( v43 == *(_QWORD *)v45 )
          goto LABEL_57;
        ++v82;
      }
      v44 = v87;
      if ( !v47 )
        v47 = (__int64 *)v45;
      ++v84;
      v48 = v86 + 1;
      if ( 4 * ((int)v86 + 1) < 3 * v87 )
      {
        v49 = v87 >> 3;
        if ( v87 - HIDWORD(v86) - v48 > (unsigned int)v49 )
          goto LABEL_66;
        v74 = (unsigned __int8 *)v93;
        sub_116E750((__int64)&v84, v87);
        if ( !v87 )
        {
LABEL_151:
          LODWORD(v86) = v86 + 1;
          BUG();
        }
        v67 = 0;
        v22 = (__m128i *)v74;
        v49 = 1;
        v69 = (v87 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
        v48 = v86 + 1;
        v47 = (__int64 *)(v85 + 16LL * v69);
        v11 = *v47;
        if ( v43 == *v47 )
          goto LABEL_66;
        while ( v11 != -4096 )
        {
          if ( !v67 && v11 == -8192 )
            v67 = v47;
          v69 = (v87 - 1) & (v49 + v69);
          v47 = (__int64 *)(v85 + 16LL * v69);
          v11 = *v47;
          if ( v43 == *v47 )
            goto LABEL_66;
          v49 = (unsigned int)(v49 + 1);
        }
        goto LABEL_125;
      }
    }
    else
    {
      ++v84;
    }
    v83 = (unsigned __int8 *)v93;
    sub_116E750((__int64)&v84, 2 * v44);
    if ( !v87 )
      goto LABEL_151;
    v22 = (__m128i *)v83;
    v66 = (v87 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
    v48 = v86 + 1;
    v47 = (__int64 *)(v85 + 16LL * v66);
    v11 = *v47;
    if ( v43 == *v47 )
      goto LABEL_66;
    v49 = 1;
    v67 = 0;
    while ( v11 != -4096 )
    {
      if ( v11 == -8192 && !v67 )
        v67 = v47;
      v66 = (v87 - 1) & (v49 + v66);
      v47 = (__int64 *)(v85 + 16LL * v66);
      v11 = *v47;
      if ( v43 == *v47 )
        goto LABEL_66;
      v49 = (unsigned int)(v49 + 1);
    }
LABEL_125:
    if ( v67 )
      v47 = v67;
LABEL_66:
    LODWORD(v86) = v48;
    if ( *v47 != -4096 )
      --HIDWORD(v86);
    *v47 = v43;
    v50 = (unsigned __int8 *)v93;
    v47[1] = (__int64)v22;
    v108.m128i_i64[1] = (__int64)v50;
    v108.m128i_i64[0] = v43;
    v109 = v94;
    v110 = v95;
    v111 = v96;
    v112 = v114;
    v113 = 0x400000000LL;
    if ( (_DWORD)v98 )
      sub_2753C30((__int64)&v112, (__int64)&v97, (__int64)v47, (unsigned int)v98, v49, v11);
    v51 = (unsigned int)v116;
    v52 = v115;
    v22 = &v108;
    v10 = (unsigned int)v116 + 1LL;
    v53 = (unsigned int)v116;
    if ( v10 > HIDWORD(v116) )
    {
      if ( v115 > (unsigned __int64)&v108
        || (v53 = 5LL * (unsigned int)v116, (unsigned __int64)&v108 >= v115 + 88 * (unsigned __int64)(unsigned int)v116) )
      {
        sub_102D890((__int64)&v115, (unsigned int)v116 + 1LL, v53, (__int64)&v108, v10, v11);
        v51 = (unsigned int)v116;
        v52 = v115;
        v22 = &v108;
        LODWORD(v53) = v116;
      }
      else
      {
        v68 = &v108.m128i_i8[-v115];
        sub_102D890((__int64)&v115, (unsigned int)v116 + 1LL, v53, (__int64)v108.m128i_i64 - v115, v10, v11);
        v52 = v115;
        v51 = (unsigned int)v116;
        v22 = (__m128i *)&v68[v115];
        LODWORD(v53) = v116;
      }
    }
    v54 = (_QWORD *)(v52 + 88 * v51);
    if ( v54 )
    {
      *v54 = v22->m128i_i64[0];
      v54[1] = v22->m128i_i64[1];
      v54[2] = v22[1].m128i_i64[0];
      v54[3] = v22[1].m128i_i64[1];
      v55 = v22[2].m128i_i64[0];
      v54[6] = 0x400000000LL;
      v54[4] = v55;
      v54[5] = v54 + 7;
      v56 = v22[3].m128i_u32[0];
      if ( (_DWORD)v56 )
        sub_2753AD0((__int64)(v54 + 5), (char **)&v22[2].m128i_i64[1], v56, (__int64)v22, v10, v11);
      LODWORD(v53) = v116;
    }
    LODWORD(v116) = v53 + 1;
    if ( v112 != v114 )
      _libc_free((unsigned __int64)v112);
LABEL_77:
    if ( v97 != v99 )
      _libc_free((unsigned __int64)v97);
    while ( 1 )
    {
      v39 = *(_QWORD *)(v39 + 8);
      if ( !v39 )
        goto LABEL_48;
      v40 = *(unsigned __int8 **)(v39 + 24);
      v57 = *v40;
      v42 = (unsigned int)(v57 - 30);
      if ( (unsigned __int8)(v57 - 30) <= 0xAu )
        goto LABEL_54;
    }
  }
  sub_2753C30((__int64)&v97, (__int64)&v105, v42, (__int64)v22, v10, v11);
  v58 = v97;
  v59 = 8LL * (unsigned int)v98;
  v60 = &v97[v59];
  v61 = v59 >> 3;
  v62 = v59 >> 5;
  if ( !v62 )
    goto LABEL_100;
  v63 = &v97[32 * v62];
  do
  {
    if ( v28 == *(_QWORD *)(*v58 + 40LL) )
      goto LABEL_91;
    if ( v28 == *(_QWORD *)(v58[1] + 40LL) )
    {
      ++v58;
      goto LABEL_91;
    }
    if ( v28 == *(_QWORD *)(v58[2] + 40LL) )
    {
      v58 += 2;
      goto LABEL_91;
    }
    if ( v28 == *(_QWORD *)(v58[3] + 40LL) )
    {
      v58 += 3;
      goto LABEL_91;
    }
    v58 += 4;
  }
  while ( v63 != v58 );
  v61 = v60 - v58;
LABEL_100:
  if ( v61 == 2 )
  {
LABEL_104:
    if ( v28 == *(_QWORD *)(*v58 + 40LL) )
      goto LABEL_91;
    ++v58;
LABEL_106:
    if ( v28 == *(_QWORD *)(*v58 + 40LL) )
      goto LABEL_91;
    goto LABEL_55;
  }
  if ( v61 != 3 )
  {
    if ( v61 != 1 )
      goto LABEL_55;
    goto LABEL_106;
  }
  if ( v28 != *(_QWORD *)(*v58 + 40LL) )
  {
    ++v58;
    goto LABEL_104;
  }
LABEL_91:
  if ( v60 == v58 || sub_104A900((__int64 *)&v93) && sub_104B4A0((unsigned __int8 **)&v93, v28, v43, a5, 0) )
    goto LABEL_55;
LABEL_58:
  if ( v97 != v99 )
    _libc_free((unsigned __int64)v97);
LABEL_33:
  if ( (_BYTE *)v105 != v107 )
    _libc_free(v105);
  v34 = 0;
LABEL_36:
  sub_C7D6A0(v85, 16LL * v87, 8);
  v35 = v115;
  v36 = v115 + 88LL * (unsigned int)v116;
  if ( v115 != v36 )
  {
    do
    {
      v36 -= 88LL;
      v37 = *(_QWORD *)(v36 + 40);
      if ( v37 != v36 + 56 )
        _libc_free(v37);
    }
    while ( v35 != v36 );
    v36 = v115;
  }
  if ( (_BYTE *)v36 != v117 )
    _libc_free(v36);
  return v34;
}
