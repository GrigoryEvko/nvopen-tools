// Function: sub_136E2F0
// Address: 0x136e2f0
//
void __fastcall sub_136E2F0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r15
  __int64 v7; // rdx
  __int64 v8; // rcx
  _BYTE *v9; // rsi
  char *v10; // rdi
  const __m128i *v11; // rcx
  const __m128i *v12; // r8
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __m128i *v16; // rdi
  __m128i *v17; // rdx
  const __m128i *v18; // rax
  __m128i *v19; // rax
  __m128i *v20; // rax
  __int8 *v21; // rax
  const __m128i *v22; // rcx
  const __m128i *v23; // r8
  unsigned __int64 v24; // r13
  __int64 v25; // rax
  __m128i *v26; // rdi
  __m128i *v27; // rdx
  const __m128i *v28; // rax
  __m128i *v29; // rax
  __m128i *v30; // rax
  __int8 *v31; // rax
  char *v32; // r15
  char *v33; // rdx
  char *v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rsi
  char *v37; // r12
  __int64 v38; // r8
  unsigned int v39; // edx
  __int64 *v40; // rax
  __int64 v41; // rdi
  unsigned int v42; // esi
  __int64 v43; // r13
  int v44; // ecx
  int v45; // ecx
  __int64 v46; // r9
  int v47; // edx
  unsigned int v48; // esi
  __int64 v49; // r8
  __int64 v50; // rdx
  const __m128i **v51; // r12
  unsigned __int64 v52; // rsi
  const __m128i *v53; // rdi
  const __m128i *v54; // rax
  const __m128i *v55; // rcx
  __m128i *v56; // r8
  signed __int64 v57; // r13
  __int64 v58; // rax
  __m128i *v59; // rdx
  unsigned __int64 v60; // rax
  __m128i *v61; // rsi
  unsigned __int64 v62; // rsi
  unsigned __int64 v63; // rdx
  __int64 v64; // rdi
  __int64 v65; // rcx
  unsigned __int64 v66; // rax
  unsigned __int64 v67; // r8
  __int64 v68; // rax
  int v69; // r11d
  __int64 *v70; // r10
  int v71; // edi
  int v72; // esi
  int v73; // esi
  __int64 *v74; // r10
  __int64 v75; // r9
  int v76; // r11d
  unsigned int v77; // ecx
  __int64 v78; // r8
  __int64 v79; // r13
  __int64 v80; // r14
  char *v81; // rax
  const void *v82; // r8
  char *v83; // rcx
  char *v84; // rax
  __int64 v85; // rsi
  int v86; // r11d
  __int64 v87; // [rsp+8h] [rbp-338h]
  __int64 v88; // [rsp+8h] [rbp-338h]
  __m128i *v89; // [rsp+8h] [rbp-338h]
  const void *v90; // [rsp+8h] [rbp-338h]
  char *v91; // [rsp+8h] [rbp-338h]
  __int64 v92; // [rsp+10h] [rbp-330h] BYREF
  _QWORD *v93; // [rsp+18h] [rbp-328h]
  _QWORD *v94; // [rsp+20h] [rbp-320h]
  __int64 v95; // [rsp+28h] [rbp-318h]
  int v96; // [rsp+30h] [rbp-310h]
  _QWORD v97[8]; // [rsp+38h] [rbp-308h] BYREF
  const __m128i *v98; // [rsp+78h] [rbp-2C8h] BYREF
  const __m128i *v99; // [rsp+80h] [rbp-2C0h]
  __int64 v100; // [rsp+88h] [rbp-2B8h]
  _QWORD v101[16]; // [rsp+90h] [rbp-2B0h] BYREF
  char v102[8]; // [rsp+110h] [rbp-230h] BYREF
  __int64 v103; // [rsp+118h] [rbp-228h]
  unsigned __int64 v104; // [rsp+120h] [rbp-220h]
  _BYTE v105[64]; // [rsp+138h] [rbp-208h] BYREF
  __m128i *v106; // [rsp+178h] [rbp-1C8h]
  __m128i *v107; // [rsp+180h] [rbp-1C0h]
  __int8 *v108; // [rsp+188h] [rbp-1B8h]
  char v109[8]; // [rsp+190h] [rbp-1B0h] BYREF
  __int64 v110; // [rsp+198h] [rbp-1A8h]
  unsigned __int64 v111; // [rsp+1A0h] [rbp-1A0h]
  char v112[64]; // [rsp+1B8h] [rbp-188h] BYREF
  __m128i *v113; // [rsp+1F8h] [rbp-148h]
  __m128i *v114; // [rsp+200h] [rbp-140h]
  __int8 *v115; // [rsp+208h] [rbp-138h]
  char v116[8]; // [rsp+210h] [rbp-130h] BYREF
  __int64 v117; // [rsp+218h] [rbp-128h]
  unsigned __int64 v118; // [rsp+220h] [rbp-120h]
  _BYTE v119[64]; // [rsp+238h] [rbp-108h] BYREF
  __m128i *v120; // [rsp+278h] [rbp-C8h]
  __m128i *v121; // [rsp+280h] [rbp-C0h]
  __int8 *v122; // [rsp+288h] [rbp-B8h]
  __m128i v123; // [rsp+290h] [rbp-B0h] BYREF
  unsigned __int64 v124; // [rsp+2A0h] [rbp-A0h]
  char v125[64]; // [rsp+2B8h] [rbp-88h] BYREF
  __m128i *v126; // [rsp+2F8h] [rbp-48h]
  __m128i *v127; // [rsp+300h] [rbp-40h]
  __int8 *v128; // [rsp+308h] [rbp-38h]

  v1 = a1 + 136;
  v3 = *(_QWORD *)(a1 + 128);
  v4 = *(_QWORD *)(v3 + 80);
  v5 = v3 + 72;
  if ( v4 )
  {
    v6 = v4 - 24;
    if ( v5 == v4 )
      goto LABEL_7;
  }
  else
  {
    v6 = 0;
  }
  v7 = 0;
  do
  {
    v4 = *(_QWORD *)(v4 + 8);
    ++v7;
  }
  while ( v5 != v4 );
  if ( v7 > 0xFFFFFFFFFFFFFFFLL )
    goto LABEL_129;
  v8 = *(_QWORD *)(a1 + 136);
  if ( v7 > (unsigned __int64)((*(_QWORD *)(a1 + 152) - v8) >> 3) )
  {
    v79 = 8 * v7;
    v80 = *(_QWORD *)(a1 + 144) - v8;
    v81 = (char *)sub_22077B0(8 * v7);
    v82 = *(const void **)(a1 + 136);
    v83 = v81;
    if ( (__int64)(*(_QWORD *)(a1 + 144) - (_QWORD)v82) > 0 )
    {
      v90 = *(const void **)(a1 + 136);
      v84 = (char *)memmove(v81, v90, *(_QWORD *)(a1 + 144) - (_QWORD)v82);
      v82 = v90;
      v83 = v84;
      v85 = *(_QWORD *)(a1 + 152) - (_QWORD)v90;
    }
    else
    {
      if ( !v82 )
      {
LABEL_103:
        *(_QWORD *)(a1 + 136) = v83;
        *(_QWORD *)(a1 + 144) = &v83[v80];
        *(_QWORD *)(a1 + 152) = &v83[v79];
        goto LABEL_7;
      }
      v85 = *(_QWORD *)(a1 + 152) - (_QWORD)v82;
    }
    v91 = v83;
    j_j___libc_free_0(v82, v85);
    v83 = v91;
    goto LABEL_103;
  }
LABEL_7:
  v97[0] = v6;
  memset(v101, 0, sizeof(v101));
  v101[1] = &v101[5];
  v101[2] = &v101[5];
  v93 = v97;
  v94 = v97;
  LODWORD(v101[3]) = 8;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v95 = 0x100000008LL;
  v96 = 0;
  v92 = 1;
  v123.m128i_i64[1] = sub_157EBA0(v6);
  v123.m128i_i64[0] = v6;
  LODWORD(v124) = 0;
  sub_136D560(&v98, 0, &v123);
  sub_136D710((__int64)&v92);
  v9 = v119;
  v10 = v116;
  sub_16CCCB0(v116, v119, v101);
  v11 = (const __m128i *)v101[14];
  v12 = (const __m128i *)v101[13];
  v120 = 0;
  v121 = 0;
  v122 = 0;
  v13 = v101[14] - v101[13];
  if ( v101[14] == v101[13] )
  {
    v15 = 0;
    v16 = 0;
  }
  else
  {
    if ( v13 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_132;
    v87 = v101[14] - v101[13];
    v14 = sub_22077B0(v101[14] - v101[13]);
    v11 = (const __m128i *)v101[14];
    v12 = (const __m128i *)v101[13];
    v15 = v87;
    v16 = (__m128i *)v14;
  }
  v120 = v16;
  v121 = v16;
  v122 = &v16->m128i_i8[v15];
  if ( v11 != v12 )
  {
    v17 = v16;
    v18 = v12;
    do
    {
      if ( v17 )
      {
        *v17 = _mm_loadu_si128(v18);
        v17[1].m128i_i64[0] = v18[1].m128i_i64[0];
      }
      v18 = (const __m128i *)((char *)v18 + 24);
      v17 = (__m128i *)((char *)v17 + 24);
    }
    while ( v11 != v18 );
    v16 = (__m128i *)((char *)v16 + 8 * ((unsigned __int64)((char *)&v11[-2].m128i_u64[1] - (char *)v12) >> 3) + 24);
  }
  v121 = v16;
  sub_16CCEE0(&v123, v125, 8, v116);
  v19 = v120;
  v10 = v102;
  v9 = v105;
  v120 = 0;
  v126 = v19;
  v20 = v121;
  v121 = 0;
  v127 = v20;
  v21 = v122;
  v122 = 0;
  v128 = v21;
  sub_16CCCB0(v102, v105, &v92);
  v22 = v99;
  v23 = v98;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v24 = (char *)v99 - (char *)v98;
  if ( v99 != v98 )
  {
    if ( v24 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v25 = sub_22077B0((char *)v99 - (char *)v98);
      v22 = v99;
      v23 = v98;
      v26 = (__m128i *)v25;
      goto LABEL_19;
    }
LABEL_132:
    sub_4261EA(v10, v9, v13);
  }
  v26 = 0;
LABEL_19:
  v106 = v26;
  v107 = v26;
  v108 = &v26->m128i_i8[v24];
  if ( v22 != v23 )
  {
    v27 = v26;
    v28 = v23;
    do
    {
      if ( v27 )
      {
        *v27 = _mm_loadu_si128(v28);
        v27[1].m128i_i64[0] = v28[1].m128i_i64[0];
      }
      v28 = (const __m128i *)((char *)v28 + 24);
      v27 = (__m128i *)((char *)v27 + 24);
    }
    while ( v22 != v28 );
    v26 = (__m128i *)((char *)v26 + 8 * ((unsigned __int64)((char *)&v22[-2].m128i_u64[1] - (char *)v23) >> 3) + 24);
  }
  v107 = v26;
  sub_16CCEE0(v109, v112, 8, v102);
  v29 = v106;
  v106 = 0;
  v113 = v29;
  v30 = v107;
  v107 = 0;
  v114 = v30;
  v31 = v108;
  v108 = 0;
  v115 = v31;
  sub_136DA30((__int64)v109, (__int64)&v123, v1);
  if ( v113 )
    j_j___libc_free_0(v113, v115 - (__int8 *)v113);
  if ( v111 != v110 )
    _libc_free(v111);
  if ( v106 )
    j_j___libc_free_0(v106, v108 - (__int8 *)v106);
  if ( v104 != v103 )
    _libc_free(v104);
  if ( v126 )
    j_j___libc_free_0(v126, v128 - (__int8 *)v126);
  if ( v124 != v123.m128i_i64[1] )
    _libc_free(v124);
  if ( v120 )
    j_j___libc_free_0(v120, v122 - (__int8 *)v120);
  if ( v118 != v117 )
    _libc_free(v118);
  if ( v98 )
    j_j___libc_free_0(v98, v100 - (_QWORD)v98);
  if ( v94 != v93 )
    _libc_free((unsigned __int64)v94);
  if ( v101[13] )
    j_j___libc_free_0(v101[13], v101[15] - v101[13]);
  if ( v101[2] != v101[1] )
    _libc_free(v101[2]);
  v32 = *(char **)(a1 + 144);
  v33 = *(char **)(a1 + 136);
  if ( v33 != v32 )
  {
    v34 = v32 - 8;
    if ( v33 >= v32 - 8 )
      goto LABEL_53;
    do
    {
      v35 = *(_QWORD *)v33;
      v36 = *(_QWORD *)v34;
      v33 += 8;
      v34 -= 8;
      *((_QWORD *)v33 - 1) = v36;
      *((_QWORD *)v34 + 1) = v35;
    }
    while ( v34 > v33 );
    v33 = *(char **)(a1 + 136);
    v32 = *(char **)(a1 + 144);
    if ( v33 != v32 )
    {
LABEL_53:
      v37 = v33;
      v88 = a1 + 160;
      while ( 1 )
      {
        v42 = *(_DWORD *)(a1 + 184);
        v43 = (v37 - v33) >> 3;
        if ( !v42 )
          break;
        v38 = *(_QWORD *)(a1 + 168);
        v39 = (v42 - 1) & (((unsigned int)*(_QWORD *)v37 >> 9) ^ ((unsigned int)*(_QWORD *)v37 >> 4));
        v40 = (__int64 *)(v38 + 16LL * v39);
        v41 = *v40;
        if ( *(_QWORD *)v37 == *v40 )
        {
LABEL_55:
          v37 += 8;
          *((_DWORD *)v40 + 2) = v43;
          if ( v32 == v37 )
            goto LABEL_64;
          goto LABEL_56;
        }
        v69 = 1;
        v70 = 0;
        while ( v41 != -8 )
        {
          if ( !v70 && v41 == -16 )
            v70 = v40;
          v39 = (v42 - 1) & (v69 + v39);
          v40 = (__int64 *)(v38 + 16LL * v39);
          v41 = *v40;
          if ( *(_QWORD *)v37 == *v40 )
            goto LABEL_55;
          ++v69;
        }
        v71 = *(_DWORD *)(a1 + 176);
        if ( v70 )
          v40 = v70;
        ++*(_QWORD *)(a1 + 160);
        v47 = v71 + 1;
        if ( 4 * (v71 + 1) >= 3 * v42 )
          goto LABEL_59;
        if ( v42 - *(_DWORD *)(a1 + 180) - v47 <= v42 >> 3 )
        {
          sub_136BA80(v88, v42);
          v72 = *(_DWORD *)(a1 + 184);
          if ( !v72 )
          {
LABEL_133:
            ++*(_DWORD *)(a1 + 176);
            BUG();
          }
          v73 = v72 - 1;
          v74 = 0;
          v75 = *(_QWORD *)(a1 + 168);
          v76 = 1;
          v47 = *(_DWORD *)(a1 + 176) + 1;
          v77 = v73 & (((unsigned int)*(_QWORD *)v37 >> 9) ^ ((unsigned int)*(_QWORD *)v37 >> 4));
          v40 = (__int64 *)(v75 + 16LL * v77);
          v78 = *v40;
          if ( *v40 != *(_QWORD *)v37 )
          {
            while ( v78 != -8 )
            {
              if ( v78 == -16 && !v74 )
                v74 = v40;
              v77 = v73 & (v76 + v77);
              v40 = (__int64 *)(v75 + 16LL * v77);
              v78 = *v40;
              if ( *(_QWORD *)v37 == *v40 )
                goto LABEL_61;
              ++v76;
            }
            goto LABEL_98;
          }
        }
LABEL_61:
        *(_DWORD *)(a1 + 176) = v47;
        if ( *v40 != -8 )
          --*(_DWORD *)(a1 + 180);
        v50 = *(_QWORD *)v37;
        v37 += 8;
        *((_DWORD *)v40 + 2) = -1;
        *((_DWORD *)v40 + 2) = v43;
        *v40 = v50;
        if ( v32 == v37 )
        {
LABEL_64:
          v32 = *(char **)(a1 + 144);
          v33 = *(char **)(a1 + 136);
          v51 = (const __m128i **)(a1 + 64);
          v52 = (v32 - v33) >> 3;
          if ( (unsigned __int64)(v32 - v33) <= 0x2AAAAAAAAAAAAAA8LL )
            goto LABEL_65;
LABEL_129:
          sub_4262D8((__int64)"vector::reserve");
        }
LABEL_56:
        v33 = *(char **)(a1 + 136);
      }
      ++*(_QWORD *)(a1 + 160);
LABEL_59:
      sub_136BA80(v88, 2 * v42);
      v44 = *(_DWORD *)(a1 + 184);
      if ( !v44 )
        goto LABEL_133;
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a1 + 168);
      v47 = *(_DWORD *)(a1 + 176) + 1;
      v48 = v45 & (((unsigned int)*(_QWORD *)v37 >> 9) ^ ((unsigned int)*(_QWORD *)v37 >> 4));
      v40 = (__int64 *)(v46 + 16LL * v48);
      v49 = *v40;
      if ( *(_QWORD *)v37 != *v40 )
      {
        v86 = 1;
        v74 = 0;
        while ( v49 != -8 )
        {
          if ( v49 != -16 || v74 )
            v40 = v74;
          v48 = v45 & (v86 + v48);
          v49 = *(_QWORD *)(v46 + 16LL * v48);
          if ( *(_QWORD *)v37 == v49 )
          {
            v40 = (__int64 *)(v46 + 16LL * v48);
            goto LABEL_61;
          }
          ++v86;
          v74 = v40;
          v40 = (__int64 *)(v46 + 16LL * v48);
        }
LABEL_98:
        if ( v74 )
          v40 = v74;
        goto LABEL_61;
      }
      goto LABEL_61;
    }
  }
  v51 = (const __m128i **)(a1 + 64);
  v52 = (v32 - v33) >> 3;
LABEL_65:
  v53 = *(const __m128i **)(a1 + 64);
  v54 = v53;
  if ( v52 > 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 80) - (_QWORD)v53) >> 3) )
  {
    v55 = *(const __m128i **)(a1 + 72);
    v56 = 0;
    v57 = (char *)v55 - (char *)v53;
    if ( v52 )
    {
      v58 = sub_22077B0(24 * v52);
      v53 = *(const __m128i **)(a1 + 64);
      v55 = *(const __m128i **)(a1 + 72);
      v56 = (__m128i *)v58;
      v54 = v53;
    }
    if ( v55 != v53 )
    {
      v59 = v56;
      do
      {
        if ( v59 )
        {
          *v59 = _mm_loadu_si128(v54);
          v59[1].m128i_i64[0] = v54[1].m128i_i64[0];
        }
        v54 = (const __m128i *)((char *)v54 + 24);
        v59 = (__m128i *)((char *)v59 + 24);
      }
      while ( v55 != v54 );
    }
    if ( v53 )
    {
      v89 = v56;
      j_j___libc_free_0(v53, *(_QWORD *)(a1 + 80) - (_QWORD)v53);
      v56 = v89;
    }
    *(_QWORD *)(a1 + 64) = v56;
    v33 = *(char **)(a1 + 136);
    *(_QWORD *)(a1 + 72) = (char *)v56 + v57;
    v32 = *(char **)(a1 + 144);
    *(_QWORD *)(a1 + 80) = (char *)v56 + 24 * v52;
  }
  v123.m128i_i64[0] = 0;
  if ( v33 == v32 )
  {
    v64 = *(_QWORD *)(a1 + 16);
    v65 = *(_QWORD *)(a1 + 8);
    v63 = 0;
    v67 = 0xAAAAAAAAAAAAAAABLL * ((v64 - v65) >> 3);
  }
  else
  {
    LODWORD(v60) = 0;
    do
    {
      v61 = *(__m128i **)(a1 + 72);
      if ( v61 == *(__m128i **)(a1 + 80) )
      {
        sub_13699D0(v51, v61, &v123);
      }
      else
      {
        if ( v61 )
        {
          v61->m128i_i32[0] = v60;
          v61->m128i_i64[1] = 0;
          v61[1].m128i_i64[0] = 0;
          v61 = *(__m128i **)(a1 + 72);
        }
        *(_QWORD *)(a1 + 72) = (char *)v61 + 24;
      }
      v60 = v123.m128i_i64[0] + 1;
      v62 = (__int64)(*(_QWORD *)(a1 + 144) - *(_QWORD *)(a1 + 136)) >> 3;
      v123.m128i_i64[0] = v60;
      v63 = v62;
    }
    while ( v60 < v62 );
    v64 = *(_QWORD *)(a1 + 16);
    v65 = *(_QWORD *)(a1 + 8);
    v66 = 0xAAAAAAAAAAAAAAABLL * ((v64 - v65) >> 3);
    v67 = v66;
    if ( v62 > v66 )
    {
      sub_1369B80((const __m128i **)(a1 + 8), v62 - v66);
      return;
    }
  }
  if ( v63 < v67 )
  {
    v68 = v65 + 24 * v63;
    if ( v64 != v68 )
      *(_QWORD *)(a1 + 16) = v68;
  }
}
