// Function: sub_26E7C00
// Address: 0x26e7c00
//
__int64 __fastcall sub_26E7C00(__int64 a1)
{
  const __m128i *v1; // r15
  size_t v2; // rbx
  const char *v3; // rax
  char *v4; // r8
  size_t v5; // rdx
  int *v6; // r12
  size_t v7; // r13
  int v8; // r14d
  size_t v9; // r10
  int v10; // eax
  int v11; // r14d
  int v12; // eax
  int v13; // r11d
  unsigned int j; // r9d
  size_t v15; // rcx
  const void *v16; // rsi
  bool v17; // al
  unsigned int v18; // r9d
  int v19; // eax
  size_t v20; // rbx
  unsigned __int64 *v21; // r14
  unsigned __int64 v22; // rdi
  _QWORD *v23; // r8
  _QWORD *v24; // rax
  _QWORD *v25; // rsi
  __int64 *v26; // r14
  __int64 *v27; // rax
  __int64 *v28; // rdi
  int *v29; // r14
  size_t v30; // r12
  unsigned __int64 *v31; // r13
  __int64 v32; // rsi
  __int64 v33; // rdi
  __int64 v35; // rax
  unsigned __int64 v36; // r8
  __m128i v37; // xmm0
  unsigned __int64 v38; // rsi
  _QWORD *v39; // r10
  _QWORD *v40; // rax
  _QWORD *v41; // rdi
  unsigned __int64 v42; // rsi
  __int64 v43; // r10
  unsigned __int64 v44; // rtt
  __int64 v45; // r11
  unsigned __int64 v46; // r9
  __int64 **v47; // r12
  __int64 *v48; // rax
  __int64 *v49; // rcx
  __int64 v50; // r8
  unsigned __int64 v51; // rdx
  _QWORD *v52; // rax
  unsigned __int64 v53; // rdi
  unsigned __int64 v54; // r9
  unsigned __int64 v55; // r8
  _QWORD *v56; // r11
  unsigned __int64 v57; // r10
  _QWORD *v58; // rax
  _QWORD *v59; // rsi
  unsigned __int64 v60; // rdx
  char v61; // al
  unsigned __int64 v62; // rdx
  void *v63; // r8
  unsigned __int64 v64; // r10
  __int64 v65; // r9
  _QWORD *v66; // rax
  void *v67; // rax
  _QWORD *v68; // rax
  _QWORD *v69; // rcx
  _QWORD *v70; // rdi
  unsigned __int64 v71; // r11
  _QWORD *v72; // rsi
  unsigned __int64 v73; // rdx
  _QWORD **v74; // rax
  int v75; // r14d
  _QWORD *v76; // rdi
  int v77; // eax
  size_t v78; // r9
  int v79; // r11d
  unsigned int v80; // ecx
  const void *v81; // r14
  bool v82; // al
  _QWORD *v83; // rdi
  int v84; // r14d
  int v85; // eax
  int v86; // r11d
  unsigned int i; // ecx
  const void *v88; // rsi
  bool v89; // al
  int v90; // eax
  unsigned int v91; // ecx
  int v92; // eax
  int v93; // [rsp+8h] [rbp-148h]
  int v94; // [rsp+8h] [rbp-148h]
  char *v95; // [rsp+10h] [rbp-140h]
  size_t v96; // [rsp+10h] [rbp-140h]
  size_t v97; // [rsp+10h] [rbp-140h]
  size_t v98; // [rsp+18h] [rbp-138h]
  unsigned int v99; // [rsp+18h] [rbp-138h]
  unsigned int v100; // [rsp+18h] [rbp-138h]
  size_t v101; // [rsp+20h] [rbp-130h]
  size_t v102; // [rsp+20h] [rbp-130h]
  size_t v103; // [rsp+20h] [rbp-130h]
  size_t v104; // [rsp+20h] [rbp-130h]
  int v105; // [rsp+28h] [rbp-128h]
  unsigned __int64 v106; // [rsp+28h] [rbp-128h]
  unsigned __int64 v107; // [rsp+28h] [rbp-128h]
  char *v108; // [rsp+28h] [rbp-128h]
  char *s1c; // [rsp+30h] [rbp-120h]
  unsigned int s1; // [rsp+30h] [rbp-120h]
  void *s1a; // [rsp+30h] [rbp-120h]
  void *s1d; // [rsp+30h] [rbp-120h]
  char *s1e; // [rsp+30h] [rbp-120h]
  int s1b; // [rsp+30h] [rbp-120h]
  char *s1f; // [rsp+30h] [rbp-120h]
  char *s1g; // [rsp+30h] [rbp-120h]
  size_t n; // [rsp+38h] [rbp-118h]
  size_t na; // [rsp+38h] [rbp-118h]
  size_t nb; // [rsp+38h] [rbp-118h]
  __int64 nf; // [rsp+38h] [rbp-118h]
  size_t ng; // [rsp+38h] [rbp-118h]
  size_t nc; // [rsp+38h] [rbp-118h]
  size_t nd; // [rsp+38h] [rbp-118h]
  size_t ne; // [rsp+38h] [rbp-118h]
  __int64 v125; // [rsp+40h] [rbp-110h]
  _QWORD v127[2]; // [rsp+50h] [rbp-100h] BYREF
  __int64 v128; // [rsp+60h] [rbp-F0h] BYREF
  size_t v129; // [rsp+68h] [rbp-E8h]
  __int64 v130; // [rsp+70h] [rbp-E0h]
  __int64 v131; // [rsp+78h] [rbp-D8h]
  __int64 v132[26]; // [rsp+80h] [rbp-D0h] BYREF

  v1 = *(const __m128i **)(a1 + 216);
  v128 = 0;
  v129 = 0;
  v130 = 0;
  v131 = 0;
  if ( !v1 )
    goto LABEL_37;
  do
  {
    v2 = 0;
    v3 = sub_BD5D20(v1->m128i_i64[1]);
    v4 = (char *)v1[1].m128i_i64[0];
    v125 = v5;
    v6 = (int *)v3;
    v7 = v5;
    if ( v4 )
      v2 = v1[1].m128i_u64[1];
    v8 = v131;
    if ( !(_DWORD)v131 )
    {
      ++v128;
LABEL_6:
      n = (size_t)v4;
      sub_BA8070((__int64)&v128, 2 * v8);
      v9 = 0;
      v4 = (char *)n;
      if ( !(_DWORD)v131 )
        goto LABEL_7;
      v83 = (_QWORD *)n;
      s1f = (char *)n;
      v84 = v131 - 1;
      ne = v129;
      v85 = sub_C94890(v83, v2);
      v4 = s1f;
      v86 = 1;
      v78 = 0;
      for ( i = v84 & v85; ; i = v84 & v91 )
      {
        v9 = ne + 16LL * i;
        v88 = *(const void **)v9;
        if ( *(_QWORD *)v9 == -1 )
          goto LABEL_122;
        v89 = v4 + 2 == 0;
        if ( v88 != (const void *)-2LL )
        {
          if ( v2 != *(_QWORD *)(v9 + 8) )
            goto LABEL_113;
          v93 = v86;
          v96 = v78;
          v99 = i;
          if ( !v2 )
            goto LABEL_7;
          v103 = ne + 16LL * i;
          s1g = v4;
          v90 = memcmp(v4, v88, v2);
          v4 = s1g;
          v9 = v103;
          i = v99;
          v78 = v96;
          v86 = v93;
          v89 = v90 == 0;
        }
        if ( v89 )
          goto LABEL_7;
        if ( !v78 && v88 == (const void *)-2LL )
          v78 = v9;
LABEL_113:
        v91 = v86 + i;
        ++v86;
      }
    }
    s1c = (char *)v1[1].m128i_i64[0];
    v11 = v131 - 1;
    na = v129;
    v12 = sub_C94890(v4, v2);
    v4 = s1c;
    v13 = 1;
    v9 = 0;
    for ( j = v11 & v12; ; j = v11 & v18 )
    {
      v15 = na + 16LL * j;
      v16 = *(const void **)v15;
      v17 = v4 + 1 == 0;
      if ( *(_QWORD *)v15 != -1 )
      {
        v17 = v4 + 2 == 0;
        if ( v16 != (const void *)-2LL )
        {
          if ( v2 != *(_QWORD *)(v15 + 8) )
            goto LABEL_15;
          v98 = na + 16LL * j;
          v101 = v9;
          v105 = v13;
          s1 = j;
          if ( !v2 )
            goto LABEL_22;
          v95 = v4;
          v19 = memcmp(v4, v16, v2);
          v4 = v95;
          j = s1;
          v13 = v105;
          v9 = v101;
          v15 = v98;
          v17 = v19 == 0;
        }
      }
      if ( v17 )
        goto LABEL_22;
      if ( v16 == (const void *)-1LL )
        break;
LABEL_15:
      if ( v16 == (const void *)-2LL && !v9 )
        v9 = v15;
      v18 = v13 + j;
      ++v13;
    }
    v8 = v131;
    if ( !v9 )
      v9 = v15;
    ++v128;
    v10 = v130 + 1;
    if ( 4 * ((int)v130 + 1) >= (unsigned int)(3 * v131) )
      goto LABEL_6;
    if ( (int)v131 - (v10 + HIDWORD(v130)) > (unsigned int)v131 >> 3 )
      goto LABEL_8;
    nc = (size_t)v4;
    sub_BA8070((__int64)&v128, v131);
    v75 = v131;
    v9 = 0;
    v4 = (char *)nc;
    if ( !(_DWORD)v131 )
      goto LABEL_7;
    v76 = (_QWORD *)nc;
    s1e = (char *)nc;
    nd = v129;
    v77 = sub_C94890(v76, v2);
    v4 = s1e;
    s1b = v75 - 1;
    v78 = 0;
    v79 = 1;
    v80 = (v75 - 1) & v77;
    while ( 2 )
    {
      v9 = nd + 16LL * v80;
      v81 = *(const void **)v9;
      if ( *(_QWORD *)v9 != -1 )
      {
        v82 = v4 + 2 == 0;
        if ( v81 != (const void *)-2LL )
        {
          if ( v2 != *(_QWORD *)(v9 + 8) )
          {
LABEL_98:
            if ( v81 != (const void *)-2LL || v78 )
              v9 = v78;
            v78 = v9;
            v80 = s1b & (v79 + v80);
            ++v79;
            continue;
          }
          v94 = v79;
          v97 = v78;
          v100 = v80;
          if ( !v2 )
            goto LABEL_7;
          v104 = nd + 16LL * v80;
          v108 = v4;
          v92 = memcmp(v4, v81, v2);
          v4 = v108;
          v9 = v104;
          v80 = v100;
          v78 = v97;
          v79 = v94;
          v82 = v92 == 0;
        }
        if ( v82 )
          goto LABEL_7;
        if ( v81 == (const void *)-1LL )
          goto LABEL_118;
        goto LABEL_98;
      }
      break;
    }
LABEL_122:
    if ( v4 == (char *)-1LL )
      goto LABEL_7;
LABEL_118:
    if ( v78 )
      v9 = v78;
LABEL_7:
    v10 = v130 + 1;
LABEL_8:
    LODWORD(v130) = v10;
    if ( *(_QWORD *)v9 != -1 )
      --HIDWORD(v130);
    *(_QWORD *)v9 = v4;
    *(_QWORD *)(v9 + 8) = v2;
LABEL_22:
    v20 = v7;
    v21 = *(unsigned __int64 **)(a1 + 256);
    if ( v6 )
    {
      sub_C7D030(v132);
      sub_C7D280((int *)v132, v6, v7);
      sub_C7D290(v132, v127);
      v20 = v127[0];
    }
    v22 = v21[1];
    v23 = *(_QWORD **)(*v21 + 8 * (v20 % v22));
    if ( !v23 )
    {
LABEL_38:
      v35 = sub_22077B0(0x20u);
      v36 = v35;
      if ( v35 )
        *(_QWORD *)v35 = 0;
      v37 = _mm_loadu_si128(v1 + 1);
      *(_QWORD *)(v35 + 8) = v20;
      *(__m128i *)(v35 + 16) = v37;
      v38 = v21[1];
      v39 = *(_QWORD **)(*v21 + 8 * (v20 % v38));
      if ( v39 )
      {
        v40 = (_QWORD *)*v39;
        if ( v20 == *(_QWORD *)(*v39 + 8LL) )
        {
LABEL_45:
          if ( *v39 )
          {
            j_j___libc_free_0(v36);
            goto LABEL_30;
          }
        }
        else
        {
          while ( 1 )
          {
            v41 = (_QWORD *)*v40;
            if ( !*v40 )
              break;
            v39 = v40;
            if ( v20 % v38 != v41[1] % v38 )
              break;
            v40 = (_QWORD *)*v40;
            if ( v20 == v41[1] )
              goto LABEL_45;
          }
        }
      }
      nb = v36;
      v61 = sub_222DA10((__int64)(v21 + 4), v38, v21[3], 1);
      v63 = (void *)nb;
      v64 = v62;
      if ( !v61 )
      {
        v65 = 8 * (v20 % v38);
        v66 = *(_QWORD **)(*v21 + v65);
        if ( v66 )
          goto LABEL_70;
LABEL_85:
        *(_QWORD *)v63 = v21[2];
        v21[2] = (unsigned __int64)v63;
        if ( *(_QWORD *)v63 )
          *(_QWORD *)(*v21 + 8 * (*(_QWORD *)(*(_QWORD *)v63 + 8LL) % v21[1])) = v63;
        *(_QWORD *)(*v21 + v65) = v21 + 2;
        goto LABEL_71;
      }
      if ( v62 == 1 )
      {
        v69 = v21 + 6;
        v21[6] = 0;
        s1a = v21 + 6;
      }
      else
      {
        if ( v62 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(v21 + 4, v38, v62);
        v102 = nb;
        v106 = v62;
        nf = 8 * v62;
        v67 = (void *)sub_22077B0(8 * v62);
        v68 = memset(v67, 0, nf);
        v63 = (void *)v102;
        v64 = v106;
        v69 = v68;
        s1a = v21 + 6;
      }
      v70 = (_QWORD *)v21[2];
      v21[2] = 0;
      if ( !v70 )
      {
LABEL_82:
        if ( s1a != (void *)*v21 )
        {
          v107 = v64;
          s1d = v63;
          ng = (size_t)v69;
          j_j___libc_free_0(*v21);
          v64 = v107;
          v63 = s1d;
          v69 = (_QWORD *)ng;
        }
        v21[1] = v64;
        *v21 = (unsigned __int64)v69;
        v65 = 8 * (v20 % v64);
        v66 = (_QWORD *)v69[(unsigned __int64)v65 / 8];
        if ( !v66 )
          goto LABEL_85;
LABEL_70:
        *(_QWORD *)v63 = *v66;
        **(_QWORD **)(*v21 + v65) = v63;
LABEL_71:
        ++v21[3];
        goto LABEL_30;
      }
      v71 = 0;
      while ( 1 )
      {
        v72 = v70;
        v70 = (_QWORD *)*v70;
        v73 = v72[1] % v64;
        v74 = (_QWORD **)&v69[v73];
        if ( *v74 )
          break;
        *v72 = v21[2];
        v21[2] = (unsigned __int64)v72;
        *v74 = v21 + 2;
        if ( !*v72 )
        {
          v71 = v73;
LABEL_78:
          if ( !v70 )
            goto LABEL_82;
          continue;
        }
        v69[v71] = v72;
        v71 = v73;
        if ( !v70 )
          goto LABEL_82;
      }
      *v72 = **v74;
      **v74 = v72;
      goto LABEL_78;
    }
    v24 = (_QWORD *)*v23;
    if ( *(_QWORD *)(*v23 + 8LL) != v20 )
    {
      do
      {
        v25 = (_QWORD *)*v24;
        if ( !*v24 )
          goto LABEL_38;
        v23 = v24;
        if ( v20 % v22 != v25[1] % v22 )
          goto LABEL_38;
        v24 = (_QWORD *)*v24;
      }
      while ( v25[1] != v20 );
    }
    if ( !*v23 )
      goto LABEL_38;
LABEL_30:
    v26 = *(__int64 **)(a1 + 264);
    if ( v6 )
    {
      sub_C7D030(v132);
      sub_C7D280((int *)v132, v6, v7);
      sub_C7D290(v132, v127);
      v125 = v127[0];
    }
    v132[0] = v125;
    v27 = sub_26C56D0(v26, v132);
    v28 = v27;
    if ( v27 )
    {
      v42 = v26[1];
      v43 = *v26;
      v44 = v27[1];
      v45 = 8 * (v44 % v42);
      v46 = v44 % v42;
      v47 = (__int64 **)(*v26 + v45);
      v48 = *v47;
      do
      {
        v49 = v48;
        v48 = (__int64 *)*v48;
      }
      while ( v28 != v48 );
      v50 = *v28;
      if ( *v47 != v49 )
      {
        if ( v50 )
        {
          v51 = *(_QWORD *)(v50 + 8) % v42;
          if ( v46 != v51 )
          {
            *(_QWORD *)(v43 + 8 * v51) = v49;
            v50 = *v28;
          }
        }
        goto LABEL_53;
      }
      if ( v50 )
      {
        v60 = *(_QWORD *)(v50 + 8) % v42;
        if ( v46 != v60 )
        {
          *(_QWORD *)(v43 + 8 * v60) = v49;
          v47 = (__int64 **)(*v26 + v45);
          if ( *v47 != v26 + 2 )
            goto LABEL_66;
LABEL_102:
          v26[2] = v50;
LABEL_66:
          *v47 = 0;
          v50 = *v28;
        }
LABEL_53:
        *v49 = v50;
        j_j___libc_free_0((unsigned __int64)v28);
        --v26[3];
        goto LABEL_33;
      }
      if ( v49 == v26 + 2 )
        goto LABEL_102;
      goto LABEL_66;
    }
LABEL_33:
    v29 = (int *)v1[1].m128i_i64[0];
    v30 = v1[1].m128i_u64[1];
    v31 = *(unsigned __int64 **)(a1 + 264);
    if ( v29 )
    {
      sub_C7D030(v132);
      sub_C7D280((int *)v132, v29, v30);
      sub_C7D290(v132, v127);
      v30 = v127[0];
    }
    v132[0] = v30;
    if ( !sub_26C56D0(v31, v132) )
    {
      v52 = (_QWORD *)sub_22077B0(0x18u);
      v53 = (unsigned __int64)v52;
      if ( v52 )
        *v52 = 0;
      v54 = v132[0];
      v52[2] = v1->m128i_i64[1];
      v52[1] = v54;
      v55 = v31[1];
      v56 = *(_QWORD **)(*v31 + 8 * (v54 % v55));
      v57 = v54 % v55;
      if ( v56 )
      {
        v58 = (_QWORD *)*v56;
        if ( v54 == *(_QWORD *)(*v56 + 8LL) )
        {
LABEL_61:
          if ( *v56 )
          {
            j_j___libc_free_0(v53);
            goto LABEL_36;
          }
        }
        else
        {
          while ( 1 )
          {
            v59 = (_QWORD *)*v58;
            if ( !*v58 )
              break;
            v56 = v58;
            if ( v57 != v59[1] % v55 )
              break;
            v58 = (_QWORD *)*v58;
            if ( v54 == v59[1] )
              goto LABEL_61;
          }
        }
      }
      sub_26DFD20(v31, v57, v54, v53, 1);
    }
LABEL_36:
    v1 = (const __m128i *)v1->m128i_i64[0];
  }
  while ( v1 );
LABEL_37:
  sub_26E68F0(*(_QWORD **)(a1 + 8), (__int64)&v128);
  v32 = (unsigned int)v131;
  v33 = v129;
  *(_QWORD *)(*(_QWORD *)(a1 + 8) + 96LL) = *(_QWORD *)(a1 + 256);
  return sub_C7D6A0(v33, 16 * v32, 8);
}
