// Function: sub_37E1AE0
// Address: 0x37e1ae0
//
void __fastcall sub_37E1AE0(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4, __int64 a5, _QWORD *a6, __m128i a7)
{
  __int64 v8; // r12
  __int64 v9; // rbx
  int v10; // eax
  __int64 v11; // r15
  unsigned int v12; // edx
  __int64 v13; // rbx
  __int64 v14; // r15
  unsigned __int64 v15; // r14
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  int v18; // ecx
  __int64 v19; // r14
  unsigned int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // r15
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rcx
  __int64 v25; // r14
  unsigned int v26; // edx
  __int64 v27; // rsi
  __int64 v28; // rbx
  __int64 v29; // r14
  unsigned __int64 *v30; // r12
  unsigned __int64 *v31; // r15
  int v32; // edx
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // r8
  __int64 v36; // r14
  __int64 v37; // r13
  unsigned int v38; // edx
  __int64 v39; // r12
  unsigned int *v40; // rbx
  unsigned int *v41; // r15
  unsigned int v42; // eax
  __int64 v43; // rax
  __int64 v44; // r10
  unsigned __int64 v45; // rdx
  __m128i *v46; // rax
  __int64 v47; // rax
  const __m128i *v48; // r14
  unsigned __int64 v49; // rcx
  __m128i *v50; // rdi
  unsigned int v51; // eax
  int v52; // eax
  unsigned int v53; // eax
  __int64 v54; // r8
  __int64 v55; // r15
  _QWORD *v56; // r12
  __int64 v57; // r13
  unsigned int v58; // r10d
  unsigned __int64 *v59; // r14
  unsigned __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 v63; // r14
  __int8 *v64; // rsi
  __int64 i; // rax
  unsigned __int64 *v66; // rdx
  __int64 v67; // r15
  __int64 v68; // r13
  unsigned __int16 v69; // ax
  const __m128i *v70; // rax
  const __m128i *v71; // rbx
  const __m128i *v72; // rdi
  unsigned int v73; // edx
  int v74; // r15d
  unsigned int v75; // eax
  __int64 v76; // rbx
  unsigned __int64 *v77; // r13
  unsigned __int64 *v78; // r12
  __int64 v79; // r14
  __int64 v80; // rbx
  __int64 v81; // r15
  unsigned __int64 v82; // r12
  unsigned __int64 v83; // rdi
  unsigned __int64 v84; // rdi
  _QWORD *v85; // rax
  __int64 v86; // r13
  __int64 v87; // rbx
  __int64 v88; // r12
  _QWORD *v89; // r14
  unsigned __int64 v90; // rdi
  int v91; // edx
  int v92; // r14d
  unsigned int v93; // eax
  unsigned int v94; // eax
  int v95; // edx
  int v96; // r14d
  unsigned int v97; // eax
  unsigned int v98; // eax
  int v99; // edx
  int v100; // r15d
  unsigned int v101; // eax
  unsigned int v102; // eax
  _QWORD *v103; // [rsp+0h] [rbp-1B0h]
  _QWORD *v104; // [rsp+0h] [rbp-1B0h]
  _QWORD *v105; // [rsp+0h] [rbp-1B0h]
  unsigned __int64 v106; // [rsp+8h] [rbp-1A8h]
  __int64 v107; // [rsp+8h] [rbp-1A8h]
  __int64 v108; // [rsp+8h] [rbp-1A8h]
  __int64 v109; // [rsp+8h] [rbp-1A8h]
  __int64 v110; // [rsp+10h] [rbp-1A0h]
  __int64 v112; // [rsp+18h] [rbp-198h]
  __int64 v113; // [rsp+18h] [rbp-198h]
  __int64 v115; // [rsp+20h] [rbp-190h]
  __int64 v116; // [rsp+20h] [rbp-190h]
  int v117; // [rsp+20h] [rbp-190h]
  int v118; // [rsp+20h] [rbp-190h]
  unsigned int v119; // [rsp+28h] [rbp-188h]
  __int64 v120; // [rsp+28h] [rbp-188h]
  __m128i *v121; // [rsp+28h] [rbp-188h]
  _QWORD *v122; // [rsp+28h] [rbp-188h]
  unsigned __int64 v123; // [rsp+28h] [rbp-188h]
  const __m128i *v124; // [rsp+28h] [rbp-188h]
  __int64 v125; // [rsp+28h] [rbp-188h]
  int v126; // [rsp+28h] [rbp-188h]
  __int64 v128; // [rsp+38h] [rbp-178h]
  __int64 v129; // [rsp+38h] [rbp-178h]
  __int64 v130; // [rsp+40h] [rbp-170h]
  __m128i *v131; // [rsp+70h] [rbp-140h] BYREF
  __int64 v132; // [rsp+78h] [rbp-138h]
  _BYTE v133[304]; // [rsp+80h] [rbp-130h] BYREF

  v8 = a2;
  v9 = a1;
  v112 = a1 + 3408;
  v10 = *(_DWORD *)(a1 + 3424);
  ++*(_QWORD *)(a1 + 3408);
  v128 = a5;
  v119 = (unsigned int)a6;
  if ( v10 || *(_DWORD *)(a1 + 3428) )
  {
    v11 = *(_QWORD *)(a1 + 3416);
    v12 = 4 * v10;
    a5 = 88LL * *(unsigned int *)(a1 + 3432);
    if ( (unsigned int)(4 * v10) < 0x40 )
      v12 = 64;
    if ( *(_DWORD *)(a1 + 3432) <= v12 )
    {
      if ( v11 + a5 != v11 )
      {
        v13 = *(_QWORD *)(a1 + 3416);
        v14 = v11 + a5;
        do
        {
          if ( *(_DWORD *)v13 != -1 )
          {
            if ( *(_DWORD *)v13 != -2 )
            {
              v15 = *(_QWORD *)(v13 + 56);
              while ( v15 )
              {
                sub_37B80B0(*(_QWORD *)(v15 + 24));
                v16 = v15;
                v15 = *(_QWORD *)(v15 + 16);
                j_j___libc_free_0(v16);
              }
              v17 = *(_QWORD *)(v13 + 8);
              if ( v17 != v13 + 24 )
                _libc_free(v17);
            }
            *(_DWORD *)v13 = -1;
          }
          v13 += 88;
        }
        while ( v13 != v14 );
        v9 = a1;
      }
      goto LABEL_17;
    }
    v79 = v11 + a5;
    v80 = *(_QWORD *)(a1 + 3416);
    v81 = 88LL * *(unsigned int *)(a1 + 3432);
    v117 = v10;
    do
    {
      if ( *(_DWORD *)v80 <= 0xFFFFFFFD )
      {
        v82 = *(_QWORD *)(v80 + 56);
        while ( v82 )
        {
          sub_37B80B0(*(_QWORD *)(v82 + 24));
          v83 = v82;
          v82 = *(_QWORD *)(v82 + 16);
          j_j___libc_free_0(v83);
        }
        v84 = *(_QWORD *)(v80 + 8);
        if ( v84 != v80 + 24 )
          _libc_free(v84);
      }
      v80 += 88;
    }
    while ( v79 != v80 );
    v9 = a1;
    v8 = a2;
    v95 = *(_DWORD *)(a1 + 3432);
    if ( v117 )
    {
      v96 = 64;
      if ( v117 != 1 )
      {
        _BitScanReverse(&v97, v117 - 1);
        v96 = 1 << (33 - (v97 ^ 0x1F));
        if ( v96 < 64 )
          v96 = 64;
      }
      if ( v95 == v96 )
        goto LABEL_158;
      sub_C7D6A0(*(_QWORD *)(a1 + 3416), v81, 8);
      v98 = sub_37B8280(v96);
      *(_DWORD *)(a1 + 3432) = v98;
      if ( v98 )
      {
        *(_QWORD *)(a1 + 3416) = sub_C7D670(88LL * v98, 8);
LABEL_158:
        sub_37BEAB0(v112);
        goto LABEL_18;
      }
    }
    else
    {
      if ( !v95 )
        goto LABEL_158;
      sub_C7D6A0(*(_QWORD *)(a1 + 3416), v81, 8);
      *(_DWORD *)(a1 + 3432) = 0;
    }
    *(_QWORD *)(a1 + 3416) = 0;
LABEL_17:
    *(_QWORD *)(v9 + 3424) = 0;
  }
LABEL_18:
  v18 = *(_DWORD *)(v9 + 3456);
  ++*(_QWORD *)(v9 + 3440);
  v110 = v9 + 3440;
  if ( v18 || *(_DWORD *)(v9 + 3460) )
  {
    a5 = 64;
    v19 = *(_QWORD *)(v9 + 3448);
    v20 = 4 * v18;
    v21 = 88LL * *(unsigned int *)(v9 + 3464);
    if ( (unsigned int)(4 * v18) < 0x40 )
      v20 = 64;
    v22 = v19 + v21;
    if ( *(_DWORD *)(v9 + 3464) <= v20 )
    {
      while ( v22 != v19 )
      {
        if ( *(_DWORD *)v19 != -1 )
        {
          if ( *(_DWORD *)v19 != -2 )
          {
            v23 = *(_QWORD *)(v19 + 8);
            if ( v23 != v19 + 24 )
              _libc_free(v23);
          }
          *(_DWORD *)v19 = -1;
        }
        v19 += 88;
      }
    }
    else
    {
      v85 = a3;
      v109 = v9;
      v86 = v8;
      v87 = v19 + v21;
      v88 = v19;
      v118 = v18;
      v89 = v85;
      do
      {
        if ( *(_DWORD *)v88 <= 0xFFFFFFFD )
        {
          v90 = *(_QWORD *)(v88 + 8);
          if ( v90 != v88 + 24 )
            _libc_free(v90);
        }
        v88 += 88;
      }
      while ( v87 != v88 );
      v9 = v109;
      v8 = v86;
      a3 = v89;
      v91 = *(_DWORD *)(v109 + 3464);
      if ( v118 )
      {
        v92 = 64;
        if ( v118 != 1 )
        {
          _BitScanReverse(&v93, v118 - 1);
          v92 = 1 << (33 - (v93 ^ 0x1F));
          if ( v92 < 64 )
            v92 = 64;
        }
        if ( v91 == v92 )
          goto LABEL_150;
        sub_C7D6A0(*(_QWORD *)(v109 + 3448), v21, 8);
        v94 = sub_37B8280(v92);
        *(_DWORD *)(v109 + 3464) = v94;
        if ( v94 )
        {
          *(_QWORD *)(v109 + 3448) = sub_C7D670(88LL * v94, 8);
LABEL_150:
          sub_37BEAF0(v110);
          goto LABEL_32;
        }
      }
      else
      {
        if ( !v91 )
          goto LABEL_150;
        sub_C7D6A0(*(_QWORD *)(v109 + 3448), v21, 8);
        *(_DWORD *)(v109 + 3464) = 0;
      }
      *(_QWORD *)(v109 + 3448) = 0;
    }
    *(_QWORD *)(v9 + 3456) = 0;
  }
LABEL_32:
  *(_DWORD *)(v9 + 3144) = 0;
  if ( *(_DWORD *)(v9 + 3148) < v119 )
    sub_C8D5F0(v9 + 3136, (const void *)(v9 + 3152), v119, 8u, a5, (__int64)a6);
  v24 = *(unsigned int *)(v9 + 3568);
  ++*(_QWORD *)(v9 + 3552);
  if ( __PAIR64__(*(_DWORD *)(v9 + 3572), v24) )
  {
    v25 = *(_QWORD *)(v9 + 3560);
    v26 = 4 * v24;
    v27 = 112LL * *(unsigned int *)(v9 + 3576);
    if ( (unsigned int)(4 * v24) < 0x40 )
      v26 = 64;
    if ( *(_DWORD *)(v9 + 3576) <= v26 )
    {
      if ( v25 != v25 + v27 )
      {
        v115 = v8;
        v120 = v9;
        v28 = *(_QWORD *)(v9 + 3560);
        v29 = v25 + v27;
        do
        {
          if ( *(_DWORD *)v28 != -1 )
          {
            if ( *(_DWORD *)v28 != -2 )
            {
              v30 = *(unsigned __int64 **)(v28 + 8);
              v31 = &v30[11 * *(unsigned int *)(v28 + 16)];
              if ( v30 != v31 )
              {
                do
                {
                  v31 -= 11;
                  if ( (unsigned __int64 *)*v31 != v31 + 2 )
                    _libc_free(*v31);
                }
                while ( v30 != v31 );
                v31 = *(unsigned __int64 **)(v28 + 8);
              }
              if ( v31 != (unsigned __int64 *)(v28 + 24) )
                _libc_free((unsigned __int64)v31);
            }
            *(_DWORD *)v28 = -1;
          }
          v28 += 112;
        }
        while ( v28 != v29 );
        v9 = v120;
        v8 = v115;
      }
      goto LABEL_52;
    }
    v116 = v9;
    v76 = *(_QWORD *)(v9 + 3560);
    v126 = v24;
    v108 = v8;
    v105 = a3;
    do
    {
      if ( *(_DWORD *)v76 <= 0xFFFFFFFD )
      {
        v77 = *(unsigned __int64 **)(v76 + 8);
        v78 = &v77[11 * *(unsigned int *)(v76 + 16)];
        if ( v77 != v78 )
        {
          do
          {
            v78 -= 11;
            if ( (unsigned __int64 *)*v78 != v78 + 2 )
              _libc_free(*v78);
          }
          while ( v77 != v78 );
          v78 = *(unsigned __int64 **)(v76 + 8);
        }
        if ( v78 != (unsigned __int64 *)(v76 + 24) )
          _libc_free((unsigned __int64)v78);
      }
      v76 += 112;
    }
    while ( v76 != v25 + v27 );
    v9 = v116;
    v8 = v108;
    a3 = v105;
    v99 = *(_DWORD *)(v116 + 3576);
    if ( v126 )
    {
      v100 = 64;
      if ( v126 != 1 )
      {
        _BitScanReverse(&v101, v126 - 1);
        v100 = 1 << (33 - (v101 ^ 0x1F));
        if ( v100 < 64 )
          v100 = 64;
      }
      if ( v99 == v100 )
        goto LABEL_166;
      sub_C7D6A0(*(_QWORD *)(v116 + 3560), v27, 8);
      v102 = sub_37B8280(v100);
      *(_DWORD *)(v116 + 3576) = v102;
      if ( v102 )
      {
        *(_QWORD *)(v116 + 3560) = sub_C7D670(112LL * v102, 8);
LABEL_166:
        sub_37BEB30(v116 + 3552);
        goto LABEL_53;
      }
    }
    else
    {
      if ( !v99 )
        goto LABEL_166;
      sub_C7D6A0(*(_QWORD *)(v116 + 3560), v27, 8);
      *(_DWORD *)(v116 + 3576) = 0;
    }
    *(_QWORD *)(v116 + 3560) = 0;
LABEL_52:
    *(_QWORD *)(v9 + 3568) = 0;
  }
LABEL_53:
  v32 = *(_DWORD *)(v9 + 3600);
  ++*(_QWORD *)(v9 + 3584);
  if ( !v32 )
  {
    if ( !*(_DWORD *)(v9 + 3604) )
      goto LABEL_59;
    v33 = *(unsigned int *)(v9 + 3608);
    if ( (unsigned int)v33 <= 0x40 )
      goto LABEL_56;
    sub_C7D6A0(*(_QWORD *)(v9 + 3592), 4 * v33, 4);
    *(_DWORD *)(v9 + 3608) = 0;
LABEL_177:
    *(_QWORD *)(v9 + 3592) = 0;
LABEL_58:
    *(_QWORD *)(v9 + 3600) = 0;
    goto LABEL_59;
  }
  v24 = (unsigned int)(4 * v32);
  v33 = *(unsigned int *)(v9 + 3608);
  if ( (unsigned int)v24 < 0x40 )
    v24 = 64;
  if ( (unsigned int)v24 >= (unsigned int)v33 )
  {
LABEL_56:
    if ( 4LL * (unsigned int)v33 )
      memset(*(void **)(v9 + 3592), 255, 4LL * (unsigned int)v33);
    goto LABEL_58;
  }
  v73 = v32 - 1;
  if ( v73 )
  {
    _BitScanReverse(&v73, v73);
    v74 = 1 << (33 - (v73 ^ 0x1F));
    if ( v74 < 64 )
      v74 = 64;
    if ( v74 == (_DWORD)v33 )
      goto LABEL_120;
  }
  else
  {
    v74 = 64;
  }
  sub_C7D6A0(*(_QWORD *)(v9 + 3592), 4 * v33, 4);
  v75 = sub_37B8280(v74);
  *(_DWORD *)(v9 + 3608) = v75;
  if ( !v75 )
    goto LABEL_177;
  *(_QWORD *)(v9 + 3592) = sub_C7D670(4LL * v75, 4);
LABEL_120:
  sub_2C2BFC0(v9 + 3584);
LABEL_59:
  v131 = (__m128i *)v133;
  v132 = 0x1000000000LL;
  v34 = *(unsigned int *)(v128 + 8);
  v35 = *(_QWORD *)v128 + 72 * v34;
  if ( *(_QWORD *)v128 == v35 )
    goto LABEL_78;
  a6 = a3;
  v36 = *(_QWORD *)v128 + 8LL;
  v37 = v8;
  v38 = 0;
  v39 = v9;
  while ( 1 )
  {
    if ( *(_DWORD *)(v36 + 56) == 1 )
    {
      v40 = (unsigned int *)(v36 + 4LL * *(unsigned int *)(v36 + 32));
      if ( v40 != (unsigned int *)v36 )
      {
        v41 = (unsigned int *)v36;
        do
        {
          v42 = *v41;
          if ( (*v41 & 1) == 0 )
          {
            if ( v42 == dword_5051178[0] )
              v130 = qword_5051170;
            else
              v130 = *(_QWORD *)(*a4 + 8LL * (v42 >> 1));
            v43 = v38;
            v24 = HIDWORD(v132);
            v44 = v130;
            v45 = v38 + 1LL;
            if ( v45 > HIDWORD(v132) )
            {
              v104 = a6;
              v107 = v35;
              sub_C8D5F0((__int64)&v131, v133, v45, 0x10u, v35, (__int64)a6);
              v43 = (unsigned int)v132;
              a6 = v104;
              v35 = v107;
              v44 = v130;
            }
            v46 = &v131[v43];
            v46->m128i_i64[0] = v44;
            v46->m128i_i64[1] = 0;
            v38 = v132 + 1;
            LODWORD(v132) = v132 + 1;
          }
          ++v41;
        }
        while ( v40 != v41 );
      }
    }
    if ( v35 == v36 + 64 )
      break;
    v36 += 72;
  }
  v9 = v39;
  v8 = v37;
  v47 = v38;
  a3 = a6;
  v48 = &v131[v47];
  if ( v131 == &v131[v47] )
    goto LABEL_77;
  v106 = 16LL * v38;
  _BitScanReverse64(&v49, (v47 * 16) >> 4);
  v121 = v131;
  sub_37E1980(
    (__int64)v131,
    v131[v47].m128i_i64,
    2LL * (int)(63 - (v49 ^ 0x3F)),
    (unsigned __int8 (__fastcall *)(__int64 *, __int64 *))sub_37B5D10);
  v50 = v121;
  if ( v106 <= 0x100 )
  {
    sub_37DD4D0(v121, v48, (unsigned __int8 (__fastcall *)(__m128i *, const __m128i *))sub_37B5D10);
LABEL_77:
    LODWORD(v34) = *(_DWORD *)(v128 + 8);
    goto LABEL_78;
  }
  v124 = v121 + 16;
  sub_37DD4D0(v50, v124, (unsigned __int8 (__fastcall *)(__m128i *, const __m128i *))sub_37B5D10);
  v70 = v124;
  if ( v48 == v124 )
    goto LABEL_77;
  v125 = v9;
  v71 = v70;
  do
  {
    v72 = v71++;
    sub_37DD460(v72, (unsigned __int8 (__fastcall *)(__m128i *, const __m128i *))sub_37B5D10);
  }
  while ( v48 != v71 );
  v9 = v125;
  LODWORD(v34) = *(_DWORD *)(v128 + 8);
LABEL_78:
  if ( !(_DWORD)v34 )
  {
    ++*(_QWORD *)(v9 + 3408);
LABEL_80:
    v52 = *(_DWORD *)(v128 + 8);
    if ( v52 )
      goto LABEL_81;
LABEL_106:
    ++*(_QWORD *)(v9 + 3440);
    goto LABEL_83;
  }
  v51 = sub_AF1560(4 * (int)v34 / 3u + 1);
  ++*(_QWORD *)(v9 + 3408);
  if ( *(_DWORD *)(v9 + 3432) >= v51 )
    goto LABEL_80;
  sub_37BEC40(v112, v51);
  v52 = *(_DWORD *)(v128 + 8);
  if ( !v52 )
    goto LABEL_106;
LABEL_81:
  v53 = sub_AF1560(4 * v52 / 3u + 1);
  ++*(_QWORD *)(v9 + 3440);
  if ( *(_DWORD *)(v9 + 3464) < v53 )
    sub_37BF080(v110, v53);
LABEL_83:
  v54 = *(unsigned int *)(*(_QWORD *)(v9 + 16) + 40LL);
  if ( (_DWORD)v54 )
  {
    a6 = &qword_5051170;
    v113 = v8;
    v55 = 0;
    v56 = a3;
    v57 = *(unsigned int *)(*(_QWORD *)(v9 + 16) + 40LL);
    do
    {
      v58 = v55;
      v59 = (unsigned __int64 *)(*v56 + 8 * v55);
      v60 = *v59;
      if ( *v59 != *a6 )
      {
        v61 = *(unsigned int *)(v9 + 3144);
        if ( v61 + 1 > (unsigned __int64)*(unsigned int *)(v9 + 3148) )
        {
          v103 = a6;
          v123 = *v59;
          sub_C8D5F0(v9 + 3136, (const void *)(v9 + 3152), v61 + 1, 8u, v54, (__int64)a6);
          v61 = *(unsigned int *)(v9 + 3144);
          a6 = v103;
          v58 = v55;
          v60 = v123;
        }
        *(_QWORD *)(*(_QWORD *)(v9 + 3136) + 8 * v61) = v60;
        v62 = (unsigned int)v132;
        ++*(_DWORD *)(v9 + 3144);
        v24 = *v59;
        v63 = (__int64)v131;
        v62 *= 16;
        v64 = &v131->m128i_i8[v62];
        for ( i = v62 >> 4; i > 0; i >>= 1 )
        {
          while ( 1 )
          {
            v66 = (unsigned __int64 *)(v63 + 16 * (i >> 1));
            if ( *v66 >= v24 )
              break;
            v63 = (__int64)(v66 + 2);
            i = i - (i >> 1) - 1;
            if ( i <= 0 )
              goto LABEL_92;
          }
        }
LABEL_92:
        if ( v64 != (__int8 *)v63 && *(_QWORD *)v63 == v24 )
        {
          v122 = a6;
          v69 = sub_37B9C80(v9, v58, *(_BYTE *)(v63 + 11));
          a6 = v122;
          if ( HIBYTE(v69) )
          {
            v24 = v55 & 0xFFFFFF;
            *(_DWORD *)(v63 + 8) = v24 | *(_DWORD *)(v63 + 8) & 0xFF000000;
            *(_BYTE *)(v63 + 11) = v69;
          }
        }
      }
      ++v55;
    }
    while ( v57 != v55 );
    v8 = v113;
  }
  v67 = *(_QWORD *)v128;
  if ( *(_QWORD *)v128 + 72LL * *(unsigned int *)(v128 + 8) != *(_QWORD *)v128 )
  {
    v129 = *(_QWORD *)v128 + 72LL * *(unsigned int *)(v128 + 8);
    v68 = v67;
    do
    {
      v68 += 72;
      sub_37CF670(
        v9,
        v8,
        a4,
        (__int64)&v131,
        *(_DWORD *)(v68 - 72),
        (__int64)a6,
        *(_QWORD *)(v68 - 64),
        *(_QWORD *)(v68 - 56),
        *(_QWORD *)(v68 - 48),
        *(_QWORD *)(v68 - 40),
        *(_QWORD *)(v68 - 32),
        a7);
    }
    while ( v129 != v68 );
  }
  sub_37C43E0(v9, *(_QWORD *)(v8 + 56), v8, v24, v54, (__int64)a6);
  if ( v131 != (__m128i *)v133 )
    _libc_free((unsigned __int64)v131);
}
