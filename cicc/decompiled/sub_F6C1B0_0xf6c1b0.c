// Function: sub_F6C1B0
// Address: 0xf6c1b0
//
_QWORD *__fastcall sub_F6C1B0(_QWORD *a1, __int64 p_dest, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v8; // r14
  unsigned __int8 *v9; // rdi
  bool v10; // zf
  _QWORD *v11; // rax
  unsigned int v12; // r12d
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 *v16; // rax
  __int64 *v17; // rdx
  __int64 *v18; // rax
  char v19; // dl
  __int64 v20; // r12
  __int64 i; // r15
  unsigned __int8 *v22; // rdi
  unsigned __int64 v23; // rax
  __int64 v24; // r15
  int v25; // r14d
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 *v28; // r12
  unsigned int j; // ebx
  __int64 v30; // rdx
  __int64 k; // rax
  __int64 v32; // r14
  __int64 v33; // rcx
  __int64 v34; // r12
  char *v35; // rax
  char v36; // dl
  __int64 *v37; // rax
  __int64 *v38; // rdx
  _QWORD *v39; // r13
  __int64 v40; // rax
  const __m128i *v41; // r15
  __int64 v42; // rax
  signed __int64 v43; // rax
  __int64 v44; // r14
  const __m128i *v45; // r12
  __m128i v46; // xmm5
  __m128i v47; // xmm6
  __m128i v48; // xmm4
  __m128i v49; // xmm2
  __int64 v50; // rbx
  __int64 v51; // r12
  char v52; // al
  unsigned __int64 v53; // rdx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // rax
  __int64 *v57; // rbx
  __int64 v58; // r12
  __int64 *v59; // rax
  unsigned __int64 v60; // rax
  __int64 v61; // r13
  unsigned int v62; // r15d
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // r12
  _QWORD *v66; // rax
  __int64 v67; // r13
  int v68; // ebx
  __int64 v69; // r14
  char v70; // al
  __m128i v72; // xmm2
  __int64 v73; // rax
  __m128i v74; // xmm0
  __m128i v75; // xmm5
  __int64 v76; // [rsp+10h] [rbp-1B0h]
  __int64 v77; // [rsp+18h] [rbp-1A8h]
  const void *v78; // [rsp+18h] [rbp-1A8h]
  __int64 v79; // [rsp+18h] [rbp-1A8h]
  int v81; // [rsp+30h] [rbp-190h]
  const __m128i *v82; // [rsp+30h] [rbp-190h]
  int v83; // [rsp+30h] [rbp-190h]
  __int64 v84; // [rsp+38h] [rbp-188h]
  _QWORD *v85; // [rsp+40h] [rbp-180h] BYREF
  __int64 v86; // [rsp+48h] [rbp-178h]
  _QWORD v87[4]; // [rsp+50h] [rbp-170h] BYREF
  __m128i v88; // [rsp+70h] [rbp-150h] BYREF
  __m128i v89; // [rsp+80h] [rbp-140h]
  __m128i v90; // [rsp+90h] [rbp-130h]
  char v91; // [rsp+A0h] [rbp-120h]
  __int64 v92; // [rsp+B0h] [rbp-110h] BYREF
  __int64 *v93; // [rsp+B8h] [rbp-108h]
  __int64 v94; // [rsp+C0h] [rbp-100h]
  int v95; // [rsp+C8h] [rbp-F8h]
  char v96; // [rsp+CCh] [rbp-F4h]
  __int64 v97; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v98; // [rsp+F0h] [rbp-D0h] BYREF
  char *v99; // [rsp+F8h] [rbp-C8h]
  __int64 v100; // [rsp+100h] [rbp-C0h]
  int v101; // [rsp+108h] [rbp-B8h]
  unsigned __int8 v102; // [rsp+10Ch] [rbp-B4h]
  char v103; // [rsp+110h] [rbp-B0h] BYREF
  void *dest; // [rsp+130h] [rbp-90h] BYREF
  __int64 v105; // [rsp+138h] [rbp-88h]
  _BYTE v106[48]; // [rsp+140h] [rbp-80h] BYREF
  __int64 v107; // [rsp+170h] [rbp-50h]
  char v108; // [rsp+178h] [rbp-48h]
  __int64 v109; // [rsp+180h] [rbp-40h]

  v6 = a4 + 48;
  v8 = *(_QWORD *)(a4 + 56);
  dest = v106;
  v105 = 0x600000000LL;
  v85 = v87;
  v86 = 0x400000002LL;
  v93 = &v97;
  v84 = p_dest;
  v107 = 0;
  v108 = 1;
  v109 = 0;
  v87[0] = a3;
  v87[1] = a4;
  v94 = 0x100000004LL;
  v95 = 0;
  v96 = 1;
  v97 = a4;
  v92 = 1;
  if ( a4 + 48 != v8 )
  {
    do
    {
      v9 = (unsigned __int8 *)(v8 - 24);
      if ( !v8 )
        v9 = 0;
      if ( (unsigned __int8)sub_B46970(v9) )
        break;
      v8 = *(_QWORD *)(v8 + 8);
    }
    while ( v6 != v8 );
  }
  v77 = a5;
  v10 = v6 == v8;
  v11 = v87;
  v12 = 2;
  v108 = v10;
  while ( 1 )
  {
    v13 = *(_QWORD *)v84;
    v14 = v12--;
    v15 = v11[v14 - 1];
    LODWORD(v86) = v12;
    if ( !*(_BYTE *)(v13 + 84) )
    {
      p_dest = v15;
      if ( sub_C8CA60(v13 + 56, v15) )
      {
        if ( v96 )
          goto LABEL_14;
        goto LABEL_23;
      }
LABEL_18:
      v12 = v86;
      goto LABEL_19;
    }
    v16 = *(__int64 **)(v13 + 64);
    v17 = &v16[*(unsigned int *)(v13 + 76)];
    if ( v16 != v17 )
      break;
LABEL_19:
    if ( !v12 )
      goto LABEL_41;
LABEL_20:
    v11 = v85;
  }
  while ( v15 != *v16 )
  {
    if ( v17 == ++v16 )
      goto LABEL_19;
  }
  if ( !v96 )
  {
LABEL_23:
    p_dest = v15;
    sub_C8CC70((__int64)&v92, v15, (__int64)v17, a4, a5, a6);
    if ( v19 )
      goto LABEL_24;
    goto LABEL_18;
  }
LABEL_14:
  v18 = v93;
  a4 = HIDWORD(v94);
  v17 = &v93[HIDWORD(v94)];
  if ( v93 != v17 )
  {
    while ( v15 != *v18 )
    {
      if ( v17 == ++v18 )
        goto LABEL_51;
    }
    goto LABEL_18;
  }
LABEL_51:
  if ( HIDWORD(v94) >= (unsigned int)v94 )
    goto LABEL_23;
  a4 = (unsigned int)++HIDWORD(v94);
  *v17 = v15;
  ++v92;
LABEL_24:
  v20 = *(_QWORD *)(v15 + 56);
  for ( i = v15 + 48; i != v20; v20 = *(_QWORD *)(v20 + 8) )
  {
    v22 = (unsigned __int8 *)(v20 - 24);
    if ( !v20 )
      v22 = 0;
    if ( (unsigned __int8)sub_B46970(v22) )
      break;
  }
  v108 &= i == v20;
  v23 = *(_QWORD *)(v15 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( i != v23 )
  {
    if ( !v23 )
LABEL_125:
      BUG();
    v24 = v23 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v23 - 24) - 30 <= 0xA )
    {
      v25 = sub_B46E30(v24);
      v81 = v25;
      v26 = (unsigned int)v86;
      v27 = v25 + (unsigned __int64)(unsigned int)v86;
      a4 = HIDWORD(v86);
      if ( v27 <= HIDWORD(v86) )
        goto LABEL_34;
      goto LABEL_82;
    }
    v27 = (unsigned int)v86;
    if ( HIDWORD(v86) >= (unsigned int)v86 )
    {
      v81 = 0;
    }
    else
    {
LABEL_81:
      v81 = 0;
      v25 = 0;
      v24 = 0;
LABEL_82:
      p_dest = (__int64)v87;
      sub_C8D5F0((__int64)&v85, v87, v27, 8u, a5, a6);
      v26 = (unsigned int)v86;
LABEL_34:
      v28 = &v85[v26];
      if ( v25 )
      {
        for ( j = 0; j != v25; ++j )
        {
          if ( v28 )
          {
            p_dest = j;
            *v28 = sub_B46EC0(v24, j);
          }
          ++v28;
        }
        v12 = v86 + v81;
        goto LABEL_40;
      }
      LODWORD(v27) = v26;
    }
    v12 = v81 + v27;
    goto LABEL_40;
  }
  v27 = (unsigned int)v86;
  v12 = v86;
  if ( HIDWORD(v86) < (unsigned int)v86 )
    goto LABEL_81;
LABEL_40:
  LODWORD(v86) = v12;
  if ( v12 )
    goto LABEL_20;
LABEL_41:
  if ( (unsigned int)(HIDWORD(v94) - v95) <= 1 )
  {
    memset(a1, 0, 0x60u);
    goto LABEL_119;
  }
  v30 = 1;
  v98 = 0;
  v99 = &v103;
  LODWORD(k) = *(_DWORD *)(v77 + 8);
  v32 = v77;
  v100 = 4;
  v101 = 0;
  v102 = 1;
  v78 = (const void *)(v77 + 16);
  if ( !(_DWORD)k )
  {
LABEL_83:
    v52 = sub_D4A4C0(*(_QWORD *)v84);
    v108 &= v52;
    if ( !v108 )
      goto LABEL_102;
    v56 = *(_QWORD *)(v84 + 24);
    v57 = *(__int64 **)v56;
    v79 = *(_QWORD *)v56 + 8LL * *(unsigned int *)(v56 + 8);
    if ( v79 == *(_QWORD *)v56 )
      goto LABEL_102;
LABEL_85:
    while ( 2 )
    {
      v58 = *v57;
      if ( v96 )
      {
        v59 = v93;
        v53 = (unsigned __int64)&v93[HIDWORD(v94)];
        if ( v93 == (__int64 *)v53 )
        {
LABEL_101:
          if ( (__int64 *)v79 == ++v57 )
            goto LABEL_102;
          continue;
        }
        while ( v58 != *v59 )
        {
          if ( (__int64 *)v53 == ++v59 )
            goto LABEL_101;
        }
      }
      else
      {
        p_dest = *v57;
        if ( !sub_C8CA60((__int64)&v92, v58) )
        {
          if ( (__int64 *)v79 != ++v57 )
            continue;
LABEL_102:
          if ( !v109 )
            v108 = 0;
          v67 = *(_QWORD *)(v84 + 32);
          if ( (void **)v67 == &dest )
          {
            v68 = v105;
          }
          else
          {
            v53 = *(unsigned int *)(v67 + 8);
            v68 = *(_DWORD *)(v67 + 8);
            if ( v53 <= (unsigned int)v105 )
            {
              if ( *(_DWORD *)(v67 + 8) )
              {
                p_dest = *(_QWORD *)v67;
                memmove(dest, *(const void **)v67, 8 * v53);
              }
            }
            else
            {
              if ( v53 > HIDWORD(v105) )
              {
                v69 = 0;
                LODWORD(v105) = 0;
                sub_C8D5F0((__int64)&dest, v106, v53, 8u, v54, v55);
                v53 = *(unsigned int *)(v67 + 8);
              }
              else
              {
                v69 = 8LL * (unsigned int)v105;
                if ( (_DWORD)v105 )
                {
                  memmove(dest, *(const void **)v67, 8LL * (unsigned int)v105);
                  v53 = *(unsigned int *)(v67 + 8);
                }
              }
              v53 *= 8LL;
              p_dest = *(_QWORD *)v67 + v69;
              if ( p_dest != v53 + *(_QWORD *)v67 )
                memcpy((char *)dest + v69, (const void *)p_dest, v53 - v69);
            }
            LODWORD(v105) = v68;
          }
          *a1 = a1 + 2;
          a1[1] = 0x600000000LL;
          if ( v68 )
          {
            p_dest = (__int64)&dest;
            sub_F6B9F0((__int64)a1, (char **)&dest, v53, (__int64)a1, v54, v55);
          }
          a1[8] = v107;
          v70 = v108;
          *((_BYTE *)a1 + 88) = 1;
          *((_BYTE *)a1 + 72) = v70;
          a1[10] = v109;
          goto LABEL_115;
        }
      }
      break;
    }
    v60 = *(_QWORD *)(v58 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v60 == v58 + 48 )
      goto LABEL_101;
    if ( !v60 )
      goto LABEL_125;
    v61 = v60 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v60 - 24) - 30 > 0xA )
      goto LABEL_101;
    v83 = sub_B46E30(v61);
    if ( !v83 )
      goto LABEL_101;
    v62 = 0;
    while ( 1 )
    {
      p_dest = v62;
      v63 = sub_B46EC0(v61, v62);
      v64 = *(_QWORD *)v84;
      v65 = v63;
      if ( *(_BYTE *)(*(_QWORD *)v84 + 84LL) )
      {
        v66 = *(_QWORD **)(v64 + 64);
        v53 = (unsigned __int64)&v66[*(unsigned int *)(v64 + 76)];
        if ( v66 == (_QWORD *)v53 )
          goto LABEL_140;
        while ( v65 != *v66 )
        {
          if ( (_QWORD *)v53 == ++v66 )
            goto LABEL_140;
        }
LABEL_100:
        if ( v83 == ++v62 )
          goto LABEL_101;
      }
      else
      {
        p_dest = v63;
        if ( sub_C8CA60(v64 + 56, v63) )
          goto LABEL_100;
LABEL_140:
        v73 = sub_AA5930(v65);
        if ( v73 != v53 || v109 != v65 && v109 )
        {
          v108 = 0;
          if ( (__int64 *)v79 == ++v57 )
            goto LABEL_102;
          goto LABEL_85;
        }
        if ( !v108 )
          goto LABEL_101;
        v109 = v65;
        if ( v83 == ++v62 )
          goto LABEL_101;
      }
    }
  }
  while ( 2 )
  {
    v33 = *(_QWORD *)v32;
    v34 = *(_QWORD *)(*(_QWORD *)v32 + 8LL * (unsigned int)k - 8);
    *(_DWORD *)(v32 + 8) = k - 1;
    if ( !(_BYTE)v30 )
      goto LABEL_53;
    v35 = v99;
    p_dest = HIDWORD(v100);
    v30 = (__int64)&v99[8 * HIDWORD(v100)];
    if ( v99 != (char *)v30 )
    {
      while ( v34 != *(_QWORD *)v35 )
      {
        v35 += 8;
        if ( (char *)v30 == v35 )
          goto LABEL_76;
      }
LABEL_48:
      LODWORD(k) = *(_DWORD *)(v32 + 8);
LABEL_49:
      if ( !(_DWORD)k )
        goto LABEL_83;
      v30 = v102;
      continue;
    }
    break;
  }
LABEL_76:
  if ( HIDWORD(v100) >= (unsigned int)v100 )
  {
LABEL_53:
    p_dest = v34;
    sub_C8CC70((__int64)&v98, v34, v30, v33, a5, a6);
    if ( !v36 )
      goto LABEL_48;
    p_dest = *(_QWORD *)(v34 + 64);
    if ( !v96 )
      goto LABEL_78;
    goto LABEL_55;
  }
  ++HIDWORD(v100);
  *(_QWORD *)v30 = v34;
  p_dest = *(_QWORD *)(v34 + 64);
  ++v98;
  if ( v96 )
  {
LABEL_55:
    v37 = v93;
    v38 = &v93[HIDWORD(v94)];
    if ( v93 == v38 )
      goto LABEL_48;
    while ( p_dest != *v37 )
    {
      if ( v38 == ++v37 )
        goto LABEL_48;
    }
    goto LABEL_59;
  }
LABEL_78:
  if ( !sub_C8CA60((__int64)&v92, p_dest) )
    goto LABEL_48;
LABEL_59:
  if ( *(_DWORD *)(v84 + 40) <= (unsigned int)(HIDWORD(v100) - v101) )
    goto LABEL_138;
  if ( *(_BYTE *)v34 == 26 )
    goto LABEL_48;
  if ( *(_BYTE *)v34 != 27 )
    goto LABEL_71;
  v39 = *(_QWORD **)(v84 + 8);
  v40 = *(_QWORD *)(v84 + 16);
  v41 = *(const __m128i **)v40;
  v42 = 3LL * *(unsigned int *)(v40 + 8);
  v82 = &v41[v42];
  v43 = 0xAAAAAAAAAAAAAAABLL * ((v42 * 16) >> 4);
  if ( !(v43 >> 2) )
    goto LABEL_127;
  v76 = v32;
  v44 = v34;
  v45 = &v41[12 * (v43 >> 2)];
  do
  {
    v88 = _mm_loadu_si128(v41);
    v89 = _mm_loadu_si128(v41 + 1);
    v49 = _mm_loadu_si128(v41 + 2);
    v91 = 1;
    v90 = v49;
    p_dest = *(_QWORD *)(v44 + 72);
    if ( (sub_CF6520(v39, (unsigned __int8 *)p_dest, &v88) & 2) != 0 )
    {
      v34 = v44;
      v32 = v76;
      goto LABEL_70;
    }
    v88 = _mm_loadu_si128(v41 + 3);
    v89 = _mm_loadu_si128(v41 + 4);
    v46 = _mm_loadu_si128(v41 + 5);
    v91 = 1;
    v90 = v46;
    p_dest = *(_QWORD *)(v44 + 72);
    if ( (sub_CF6520(v39, (unsigned __int8 *)p_dest, &v88) & 2) != 0 )
    {
      v34 = v44;
      v41 += 3;
      v32 = v76;
      goto LABEL_70;
    }
    v88 = _mm_loadu_si128(v41 + 6);
    v89 = _mm_loadu_si128(v41 + 7);
    v47 = _mm_loadu_si128(v41 + 8);
    v91 = 1;
    v90 = v47;
    p_dest = *(_QWORD *)(v44 + 72);
    if ( (sub_CF6520(v39, (unsigned __int8 *)p_dest, &v88) & 2) != 0 )
    {
      v34 = v44;
      v41 += 6;
      v32 = v76;
      goto LABEL_70;
    }
    v88 = _mm_loadu_si128(v41 + 9);
    v89 = _mm_loadu_si128(v41 + 10);
    v48 = _mm_loadu_si128(v41 + 11);
    v91 = 1;
    v90 = v48;
    p_dest = *(_QWORD *)(v44 + 72);
    if ( (sub_CF6520(v39, (unsigned __int8 *)p_dest, &v88) & 2) != 0 )
    {
      v34 = v44;
      v41 += 9;
      v32 = v76;
      goto LABEL_70;
    }
    v41 += 12;
  }
  while ( v41 != v45 );
  v34 = v44;
  v32 = v76;
  v43 = 0xAAAAAAAAAAAAAAABLL * (v82 - v41);
LABEL_127:
  if ( v43 == 2 )
    goto LABEL_155;
  if ( v43 != 3 )
  {
    if ( v43 == 1 )
      goto LABEL_130;
    goto LABEL_71;
  }
  v88 = _mm_loadu_si128(v41);
  v89 = _mm_loadu_si128(v41 + 1);
  v74 = _mm_loadu_si128(v41 + 2);
  v91 = 1;
  v90 = v74;
  p_dest = *(_QWORD *)(v34 + 72);
  if ( (sub_CF6520(v39, (unsigned __int8 *)p_dest, &v88) & 2) == 0 )
  {
    v41 += 3;
LABEL_155:
    v88 = _mm_loadu_si128(v41);
    v89 = _mm_loadu_si128(v41 + 1);
    v75 = _mm_loadu_si128(v41 + 2);
    v91 = 1;
    v90 = v75;
    p_dest = *(_QWORD *)(v34 + 72);
    if ( (sub_CF6520(v39, (unsigned __int8 *)p_dest, &v88) & 2) == 0 )
    {
      v41 += 3;
LABEL_130:
      v88 = _mm_loadu_si128(v41);
      v89 = _mm_loadu_si128(v41 + 1);
      v72 = _mm_loadu_si128(v41 + 2);
      v91 = 1;
      v90 = v72;
      p_dest = *(_QWORD *)(v34 + 72);
      if ( (sub_CF6520(v39, (unsigned __int8 *)p_dest, &v88) & 2) == 0 )
      {
LABEL_71:
        v50 = *(_QWORD *)(v34 + 16);
        for ( k = *(unsigned int *)(v32 + 8); v50; v50 = *(_QWORD *)(v50 + 8) )
        {
          v51 = *(_QWORD *)(v50 + 24);
          if ( k + 1 > (unsigned __int64)*(unsigned int *)(v32 + 12) )
          {
            p_dest = (__int64)v78;
            sub_C8D5F0(v32, v78, k + 1, 8u, a5, a6);
            k = *(unsigned int *)(v32 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v32 + 8 * k) = v51;
          k = (unsigned int)(*(_DWORD *)(v32 + 8) + 1);
          *(_DWORD *)(v32 + 8) = k;
        }
        goto LABEL_49;
      }
    }
  }
LABEL_70:
  if ( v82 == v41 )
    goto LABEL_71;
LABEL_138:
  memset(a1, 0, 0x60u);
LABEL_115:
  if ( !v102 )
  {
    _libc_free(v99, p_dest);
    if ( !v96 )
      goto LABEL_117;
    goto LABEL_120;
  }
LABEL_119:
  if ( !v96 )
LABEL_117:
    _libc_free(v93, p_dest);
LABEL_120:
  if ( v85 != v87 )
    _libc_free(v85, p_dest);
  if ( dest != v106 )
    _libc_free(dest, p_dest);
  return a1;
}
