// Function: sub_31CA2C0
// Address: 0x31ca2c0
//
int *__fastcall sub_31CA2C0(int *a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned __int64 v19; // rsi
  const __m128i *v20; // rdi
  unsigned __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // rcx
  __m128i *v24; // rdx
  const __m128i *v25; // rax
  __int64 v26; // r8
  __int64 v27; // r9
  const __m128i *v28; // rcx
  unsigned __int64 v29; // rbx
  __int64 v30; // rax
  unsigned __int64 v31; // rdi
  __m128i *v32; // rdx
  const __m128i *v33; // rax
  __int64 v34; // rbx
  unsigned __int64 v35; // rax
  __int64 v36; // r13
  __int64 v37; // r12
  __int64 v38; // r11
  unsigned int v39; // edi
  _QWORD *v40; // rax
  __int64 v41; // rcx
  int v42; // r14d
  __int64 v43; // rbx
  unsigned int v44; // esi
  int v45; // r8d
  unsigned int v46; // eax
  int v47; // ecx
  _QWORD *v48; // rdx
  __int64 v49; // r13
  __int64 *v50; // rax
  __int64 v51; // rcx
  __int64 *v52; // rdx
  __int64 v53; // r12
  __int64 *v54; // rax
  char v55; // dl
  unsigned __int64 v56; // rdx
  char v57; // cl
  int v59; // r10d
  int v60; // eax
  int v61; // r8d
  int v62; // esi
  unsigned int v63; // r13d
  _QWORD *v64; // rax
  __int64 v65; // rdi
  int v66; // edi
  _QWORD *v67; // rsi
  __int64 v68; // [rsp+0h] [rbp-340h]
  __int64 v69; // [rsp+0h] [rbp-340h]
  __int64 v70; // [rsp+0h] [rbp-340h]
  __int64 v71; // [rsp+8h] [rbp-338h]
  unsigned __int64 v72[16]; // [rsp+20h] [rbp-320h] BYREF
  __m128i v73; // [rsp+A0h] [rbp-2A0h] BYREF
  __int64 v74; // [rsp+B0h] [rbp-290h]
  int v75; // [rsp+B8h] [rbp-288h]
  char v76; // [rsp+BCh] [rbp-284h]
  _QWORD v77[8]; // [rsp+C0h] [rbp-280h] BYREF
  unsigned __int64 v78; // [rsp+100h] [rbp-240h] BYREF
  __int64 v79; // [rsp+108h] [rbp-238h]
  unsigned __int64 v80; // [rsp+110h] [rbp-230h]
  __int64 v81; // [rsp+120h] [rbp-220h] BYREF
  __int64 *v82; // [rsp+128h] [rbp-218h]
  unsigned int v83; // [rsp+130h] [rbp-210h]
  unsigned int v84; // [rsp+134h] [rbp-20Ch]
  char v85; // [rsp+13Ch] [rbp-204h]
  _BYTE v86[64]; // [rsp+140h] [rbp-200h] BYREF
  unsigned __int64 v87; // [rsp+180h] [rbp-1C0h] BYREF
  __int64 v88; // [rsp+188h] [rbp-1B8h]
  unsigned __int64 v89; // [rsp+190h] [rbp-1B0h]
  char v90[8]; // [rsp+1A0h] [rbp-1A0h] BYREF
  unsigned __int64 v91; // [rsp+1A8h] [rbp-198h]
  char v92; // [rsp+1BCh] [rbp-184h]
  _BYTE v93[64]; // [rsp+1C0h] [rbp-180h] BYREF
  unsigned __int64 v94; // [rsp+200h] [rbp-140h]
  unsigned __int64 v95; // [rsp+208h] [rbp-138h]
  unsigned __int64 v96; // [rsp+210h] [rbp-130h]
  __m128i v97; // [rsp+220h] [rbp-120h] BYREF
  char v98; // [rsp+230h] [rbp-110h]
  char v99; // [rsp+23Ch] [rbp-104h]
  char v100[64]; // [rsp+240h] [rbp-100h] BYREF
  const __m128i *v101; // [rsp+280h] [rbp-C0h]
  unsigned __int64 v102; // [rsp+288h] [rbp-B8h]
  unsigned __int64 v103; // [rsp+290h] [rbp-B0h]
  char v104[8]; // [rsp+298h] [rbp-A8h] BYREF
  unsigned __int64 v105; // [rsp+2A0h] [rbp-A0h]
  char v106; // [rsp+2B4h] [rbp-8Ch]
  char v107[64]; // [rsp+2B8h] [rbp-88h] BYREF
  const __m128i *v108; // [rsp+2F8h] [rbp-48h]
  const __m128i *v109; // [rsp+300h] [rbp-40h]
  unsigned __int64 v110; // [rsp+308h] [rbp-38h]

  *a1 = 1;
  *((_QWORD *)a1 + 1) = 0;
  *((_QWORD *)a1 + 2) = 0;
  *((_QWORD *)a1 + 3) = 0;
  a1[8] = 0;
  v71 = (__int64)(a1 + 2);
  memset(v72, 0, 0x78u);
  v77[0] = *(_QWORD *)(a2 + 96);
  v97.m128i_i64[0] = v77[0];
  v74 = 0x100000008LL;
  v72[1] = (unsigned __int64)&v72[4];
  v73.m128i_i64[1] = (__int64)v77;
  LODWORD(v72[2]) = 8;
  BYTE4(v72[3]) = 1;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v75 = 0;
  v76 = 1;
  v73.m128i_i64[0] = 1;
  v98 = 0;
  sub_31CA280((__int64)&v78, &v97);
  sub_C8CF70((__int64)v90, v93, 8, (__int64)&v72[4], (__int64)v72);
  v3 = v72[12];
  memset(&v72[12], 0, 24);
  v94 = v3;
  v95 = v72[13];
  v96 = v72[14];
  sub_C8CF70((__int64)&v81, v86, 8, (__int64)v77, (__int64)&v73);
  v4 = v78;
  v78 = 0;
  v87 = v4;
  v5 = v79;
  v79 = 0;
  v88 = v5;
  v6 = v80;
  v80 = 0;
  v89 = v6;
  sub_C8CF70((__int64)&v97, v100, 8, (__int64)v86, (__int64)&v81);
  v7 = v87;
  v87 = 0;
  v101 = (const __m128i *)v7;
  v8 = v88;
  v88 = 0;
  v102 = v8;
  v9 = v89;
  v89 = 0;
  v103 = v9;
  sub_C8CF70((__int64)v104, v107, 8, (__int64)v93, (__int64)v90);
  v13 = v94;
  v94 = 0;
  v108 = (const __m128i *)v13;
  v14 = v95;
  v95 = 0;
  v109 = (const __m128i *)v14;
  v15 = v96;
  v96 = 0;
  v110 = v15;
  if ( v87 )
    j_j___libc_free_0(v87);
  if ( !v85 )
    _libc_free((unsigned __int64)v82);
  if ( v94 )
    j_j___libc_free_0(v94);
  if ( !v92 )
    _libc_free(v91);
  if ( v78 )
    j_j___libc_free_0(v78);
  if ( !v76 )
    _libc_free(v73.m128i_u64[1]);
  if ( v72[12] )
    j_j___libc_free_0(v72[12]);
  if ( !BYTE4(v72[3]) )
    _libc_free(v72[1]);
  sub_C8CD80((__int64)&v81, (__int64)v86, (__int64)&v97, v10, v11, v12);
  v19 = v102;
  v20 = v101;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v21 = v102 - (_QWORD)v101;
  if ( (const __m128i *)v102 == v101 )
  {
    v21 = 0;
    v23 = 0;
  }
  else
  {
    if ( v21 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_119;
    v22 = sub_22077B0(v102 - (_QWORD)v101);
    v19 = v102;
    v20 = v101;
    v23 = v22;
  }
  v87 = v23;
  v88 = v23;
  v89 = v23 + v21;
  if ( (const __m128i *)v19 != v20 )
  {
    v24 = (__m128i *)v23;
    v25 = v20;
    do
    {
      if ( v24 )
      {
        *v24 = _mm_loadu_si128(v25);
        v17 = v25[1].m128i_i64[0];
        v24[1].m128i_i64[0] = v17;
      }
      v25 = (const __m128i *)((char *)v25 + 24);
      v24 = (__m128i *)((char *)v24 + 24);
    }
    while ( v25 != (const __m128i *)v19 );
    v23 += 8 * ((unsigned __int64)((char *)&v25[-2].m128i_u64[1] - (char *)v20) >> 3) + 24;
  }
  v20 = (const __m128i *)v90;
  v88 = v23;
  sub_C8CD80((__int64)v90, (__int64)v93, (__int64)v104, v23, v17, v18);
  v28 = v109;
  v19 = (unsigned __int64)v108;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v29 = (char *)v109 - (char *)v108;
  if ( v109 != v108 )
  {
    if ( v29 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v30 = sub_22077B0((char *)v109 - (char *)v108);
      v28 = v109;
      v19 = (unsigned __int64)v108;
      v31 = v30;
      goto LABEL_29;
    }
LABEL_119:
    sub_4261EA(v20, v19, v16);
  }
  v31 = 0;
LABEL_29:
  v94 = v31;
  v32 = (__m128i *)v31;
  v95 = v31;
  v96 = v31 + v29;
  if ( (const __m128i *)v19 != v28 )
  {
    v33 = (const __m128i *)v19;
    do
    {
      if ( v32 )
      {
        *v32 = _mm_loadu_si128(v33);
        v26 = v33[1].m128i_i64[0];
        v32[1].m128i_i64[0] = v26;
      }
      v33 = (const __m128i *)((char *)v33 + 24);
      v32 = (__m128i *)((char *)v32 + 24);
    }
    while ( v33 != v28 );
    v32 = (__m128i *)(v31 + 8 * (((unsigned __int64)&v33[-2].m128i_u64[1] - v19) >> 3) + 24);
  }
  v34 = v88;
  v35 = v87;
  v95 = (unsigned __int64)v32;
  if ( (__m128i *)(v88 - v87) == (__m128i *)((char *)v32 - v31) )
    goto LABEL_62;
  do
  {
LABEL_36:
    v36 = *(_QWORD *)(v34 - 24);
    v37 = *(_QWORD *)(*(_QWORD *)v36 + 56LL);
    v38 = *(_QWORD *)v36 + 48LL;
    if ( v38 == v37 )
      goto LABEL_50;
    do
    {
      while ( 1 )
      {
        v42 = *a1;
        v43 = v37 - 24;
        v44 = a1[8];
        if ( !v37 )
          v43 = 0;
        *a1 = v42 + 1;
        if ( !v44 )
        {
          ++*((_QWORD *)a1 + 1);
LABEL_44:
          v68 = v38;
          sub_A41E30(v71, 2 * v44);
          v45 = a1[8];
          if ( !v45 )
            goto LABEL_125;
          v26 = (unsigned int)(v45 - 1);
          v27 = *((_QWORD *)a1 + 2);
          v38 = v68;
          v46 = v26 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
          v47 = a1[6] + 1;
          v48 = (_QWORD *)(v27 + 16LL * v46);
          v49 = *v48;
          if ( v43 != *v48 )
          {
            v66 = 1;
            v67 = 0;
            while ( v49 != -4096 )
            {
              if ( !v67 && v49 == -8192 )
                v67 = v48;
              v46 = v26 & (v66 + v46);
              v48 = (_QWORD *)(v27 + 16LL * v46);
              v49 = *v48;
              if ( v43 == *v48 )
                goto LABEL_46;
              ++v66;
            }
            if ( v67 )
              v48 = v67;
          }
          goto LABEL_46;
        }
        v27 = v44 - 1;
        v26 = *((_QWORD *)a1 + 2);
        v39 = v27 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
        v40 = (_QWORD *)(v26 + 16LL * v39);
        v41 = *v40;
        if ( v43 != *v40 )
          break;
LABEL_39:
        v37 = *(_QWORD *)(v37 + 8);
        if ( v38 == v37 )
          goto LABEL_49;
      }
      v69 = *((_QWORD *)a1 + 2);
      v59 = 1;
      v48 = 0;
      while ( v41 != -4096 )
      {
        if ( v41 != -8192 || v48 )
          v40 = v48;
        v39 = v27 & (v59 + v39);
        v26 = v69 + 16LL * v39;
        v41 = *(_QWORD *)v26;
        if ( v43 == *(_QWORD *)v26 )
          goto LABEL_39;
        ++v59;
        v48 = v40;
        v40 = (_QWORD *)(v69 + 16LL * v39);
      }
      if ( !v48 )
        v48 = v40;
      v60 = a1[6];
      ++*((_QWORD *)a1 + 1);
      v47 = v60 + 1;
      if ( 4 * (v60 + 1) >= 3 * v44 )
        goto LABEL_44;
      if ( v44 - a1[7] - v47 <= v44 >> 3 )
      {
        v70 = v38;
        sub_A41E30(v71, v44);
        v61 = a1[8];
        if ( !v61 )
        {
LABEL_125:
          ++a1[6];
          BUG();
        }
        v26 = (unsigned int)(v61 - 1);
        v27 = *((_QWORD *)a1 + 2);
        v62 = 1;
        v63 = v26 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
        v38 = v70;
        v47 = a1[6] + 1;
        v64 = 0;
        v48 = (_QWORD *)(v27 + 16LL * v63);
        v65 = *v48;
        if ( v43 != *v48 )
        {
          while ( v65 != -4096 )
          {
            if ( !v64 && v65 == -8192 )
              v64 = v48;
            v63 = v26 & (v62 + v63);
            v48 = (_QWORD *)(v27 + 16LL * v63);
            v65 = *v48;
            if ( v43 == *v48 )
              goto LABEL_46;
            ++v62;
          }
          if ( v64 )
            v48 = v64;
        }
      }
LABEL_46:
      a1[6] = v47;
      if ( *v48 != -4096 )
        --a1[7];
      *v48 = v43;
      *((_DWORD *)v48 + 2) = v42;
      v37 = *(_QWORD *)(v37 + 8);
    }
    while ( v38 != v37 );
LABEL_49:
    v34 = v88;
    v36 = *(_QWORD *)(v88 - 24);
LABEL_50:
    while ( 2 )
    {
      if ( !*(_BYTE *)(v34 - 8) )
      {
        v50 = *(__int64 **)(v36 + 24);
        *(_BYTE *)(v34 - 8) = 1;
        *(_QWORD *)(v34 - 16) = v50;
        goto LABEL_52;
      }
      while ( 1 )
      {
        v50 = *(__int64 **)(v34 - 16);
LABEL_52:
        v51 = *(unsigned int *)(v36 + 32);
        if ( v50 == (__int64 *)(*(_QWORD *)(v36 + 24) + 8 * v51) )
          break;
        v52 = v50 + 1;
        *(_QWORD *)(v34 - 16) = v50 + 1;
        v53 = *v50;
        if ( v85 )
        {
          v54 = v82;
          v51 = v84;
          v52 = &v82[v84];
          if ( v82 != v52 )
          {
            while ( v53 != *v54 )
            {
              if ( v52 == ++v54 )
                goto LABEL_86;
            }
            continue;
          }
LABEL_86:
          if ( v84 < v83 )
          {
            ++v84;
            *v52 = v53;
            ++v81;
LABEL_60:
            v73.m128i_i64[0] = v53;
            LOBYTE(v74) = 0;
            sub_31CA280((__int64)&v87, &v73);
            v35 = v87;
            v34 = v88;
            goto LABEL_61;
          }
        }
        sub_C8CC70((__int64)&v81, v53, (__int64)v52, v51, v26, v27);
        if ( v55 )
          goto LABEL_60;
      }
      v88 -= 24;
      v35 = v87;
      v34 = v88;
      if ( v88 != v87 )
      {
        v36 = *(_QWORD *)(v88 - 24);
        continue;
      }
      break;
    }
LABEL_61:
    v31 = v94;
  }
  while ( v34 - v35 != v95 - v94 );
LABEL_62:
  if ( v34 != v35 )
  {
    v56 = v31;
    while ( *(_QWORD *)v35 == *(_QWORD *)v56 )
    {
      v57 = *(_BYTE *)(v35 + 16);
      if ( v57 != *(_BYTE *)(v56 + 16) || v57 && *(_QWORD *)(v35 + 8) != *(_QWORD *)(v56 + 8) )
        break;
      v35 += 24LL;
      v56 += 24LL;
      if ( v34 == v35 )
        goto LABEL_69;
    }
    goto LABEL_36;
  }
LABEL_69:
  if ( v31 )
    j_j___libc_free_0(v31);
  if ( !v92 )
    _libc_free(v91);
  if ( v87 )
    j_j___libc_free_0(v87);
  if ( !v85 )
    _libc_free((unsigned __int64)v82);
  if ( v108 )
    j_j___libc_free_0((unsigned __int64)v108);
  if ( !v106 )
    _libc_free(v105);
  if ( v101 )
    j_j___libc_free_0((unsigned __int64)v101);
  if ( !v99 )
    _libc_free(v97.m128i_u64[1]);
  return a1;
}
