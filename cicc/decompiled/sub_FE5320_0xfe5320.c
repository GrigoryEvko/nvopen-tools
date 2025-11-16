// Function: sub_FE5320
// Address: 0xfe5320
//
void __fastcall sub_FE5320(__int64 a1)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r15
  unsigned __int64 v6; // rsi
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  int v9; // eax
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  char *v22; // r15
  char *v23; // rdx
  char *v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rsi
  char *v27; // r12
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned int v32; // esi
  int v33; // edx
  __int64 v34; // rax
  _QWORD *v35; // rdi
  int v36; // ecx
  __int64 v37; // rdx
  _QWORD *v38; // rdx
  _QWORD *v39; // rdi
  __int64 v40; // rax
  __int64 v41; // r9
  unsigned int v42; // edx
  _QWORD *v43; // rcx
  __int64 v44; // r8
  __int64 v45; // rsi
  const __m128i **v46; // r12
  unsigned __int64 v47; // rsi
  const __m128i *v48; // rdi
  const __m128i *v49; // rax
  const __m128i *v50; // rcx
  __m128i *v51; // r15
  signed __int64 v52; // r13
  __int64 v53; // rax
  __m128i *v54; // rdx
  __int8 *v55; // r14
  unsigned __int64 v56; // rax
  __m128i *v57; // rsi
  unsigned __int64 v58; // rsi
  unsigned __int64 v59; // rdx
  __int64 v60; // rdi
  __int64 v61; // rcx
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // r8
  __int64 v64; // rax
  int v65; // r11d
  int v66; // ecx
  int v67; // edx
  int v68; // edx
  __int64 v69; // r8
  _QWORD *v70; // r9
  int v71; // r11d
  __int64 v72; // rcx
  __int64 v73; // rsi
  int v74; // edx
  __int64 v75; // r8
  int v76; // r11d
  __int64 v77; // rcx
  __int64 v78; // rsi
  __int64 v79; // [rsp+8h] [rbp-A88h]
  __int64 v80; // [rsp+18h] [rbp-A78h]
  _QWORD *v81; // [rsp+18h] [rbp-A78h]
  int v82; // [rsp+20h] [rbp-A70h]
  _QWORD *v83; // [rsp+20h] [rbp-A70h]
  unsigned __int64 v84; // [rsp+30h] [rbp-A60h]
  __int64 v85; // [rsp+40h] [rbp-A50h] BYREF
  __int64 *v86; // [rsp+48h] [rbp-A48h]
  int v87; // [rsp+50h] [rbp-A40h]
  int v88; // [rsp+54h] [rbp-A3Ch]
  int v89; // [rsp+58h] [rbp-A38h]
  char v90; // [rsp+5Ch] [rbp-A34h]
  __int64 v91; // [rsp+60h] [rbp-A30h] BYREF
  unsigned __int64 *v92; // [rsp+A0h] [rbp-9F0h]
  __int64 v93; // [rsp+A8h] [rbp-9E8h]
  unsigned __int64 v94; // [rsp+B0h] [rbp-9E0h] BYREF
  int v95; // [rsp+B8h] [rbp-9D8h]
  unsigned __int64 v96; // [rsp+C0h] [rbp-9D0h]
  int v97; // [rsp+C8h] [rbp-9C8h]
  __int64 v98; // [rsp+D0h] [rbp-9C0h]
  _QWORD v99[54]; // [rsp+1F0h] [rbp-8A0h] BYREF
  char v100[8]; // [rsp+3A0h] [rbp-6F0h] BYREF
  __int64 v101; // [rsp+3A8h] [rbp-6E8h]
  char v102; // [rsp+3BCh] [rbp-6D4h]
  char *v103; // [rsp+400h] [rbp-690h]
  char v104; // [rsp+410h] [rbp-680h] BYREF
  __int64 v105; // [rsp+550h] [rbp-540h] BYREF
  __int64 v106; // [rsp+558h] [rbp-538h]
  __int64 v107; // [rsp+560h] [rbp-530h]
  char v108; // [rsp+56Ch] [rbp-524h]
  char *v109; // [rsp+5B0h] [rbp-4E0h]
  char v110; // [rsp+5C0h] [rbp-4D0h] BYREF
  void *v111; // [rsp+700h] [rbp-390h] BYREF
  _QWORD v112[2]; // [rsp+708h] [rbp-388h] BYREF
  __int64 v113; // [rsp+718h] [rbp-378h]
  __int64 v114; // [rsp+720h] [rbp-370h]
  char *v115; // [rsp+760h] [rbp-330h]
  char v116; // [rsp+770h] [rbp-320h] BYREF
  unsigned __int64 v117; // [rsp+8B0h] [rbp-1E0h] BYREF
  void *v118; // [rsp+8B8h] [rbp-1D8h]
  unsigned __int64 v119; // [rsp+8C0h] [rbp-1D0h] BYREF
  __int64 v120; // [rsp+8C8h] [rbp-1C8h]
  __int64 v121; // [rsp+8D0h] [rbp-1C0h]
  __int64 v122; // [rsp+8D8h] [rbp-1B8h]
  char *v123; // [rsp+910h] [rbp-180h]
  char v124; // [rsp+920h] [rbp-170h] BYREF

  v2 = *(_QWORD *)(a1 + 128);
  v3 = *(_QWORD *)(v2 + 80);
  v4 = v2 + 72;
  if ( v3 )
  {
    v5 = v3 - 24;
    if ( v3 == v4 )
    {
      v6 = 0;
      goto LABEL_5;
    }
  }
  else
  {
    v5 = 0;
  }
  v6 = 0;
  do
  {
    v3 = *(_QWORD *)(v3 + 8);
    ++v6;
  }
  while ( v3 != v4 );
LABEL_5:
  sub_FDDB70(a1 + 136, v6);
  memset(v99, 0, sizeof(v99));
  HIDWORD(v99[13]) = 8;
  v99[1] = &v99[4];
  v99[12] = &v99[14];
  v86 = &v91;
  v92 = &v94;
  v93 = 0x800000000LL;
  v7 = *(_QWORD *)(v5 + 48);
  LODWORD(v99[2]) = 8;
  v8 = v7 & 0xFFFFFFFFFFFFFFF8LL;
  BYTE4(v99[3]) = 1;
  v87 = 8;
  v89 = 0;
  v90 = 1;
  v88 = 1;
  v91 = v5;
  v85 = 1;
  if ( v8 == v5 + 48 )
    goto LABEL_122;
  if ( !v8 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 > 0xA )
  {
LABEL_122:
    v9 = 0;
    v11 = 0;
    v10 = 0;
  }
  else
  {
    v84 = v8 - 24;
    v9 = sub_B46E30(v8 - 24);
    v10 = v84;
    v11 = v84;
  }
  v98 = v5;
  v94 = v11;
  v96 = v10;
  v95 = v9;
  v97 = 0;
  LODWORD(v93) = 1;
  sub_FDEBC0((__int64)&v85);
  sub_FDEF40((__int64)&v111, (__int64)v99, v12, v13, v14, v15);
  sub_FDEE20((__int64)&v117, (__int64)&v111);
  sub_FDEF40((__int64)v100, (__int64)&v85, v16, v17, (__int64)&v85, v18);
  sub_FDEE20((__int64)&v105, (__int64)v100);
  sub_FE4FD0((__int64)&v105, (__int64)&v117, a1 + 136, v19, v20, v21);
  if ( v109 != &v110 )
    _libc_free(v109, &v117);
  if ( !v108 )
    _libc_free(v106, &v117);
  if ( v103 != &v104 )
    _libc_free(v103, &v117);
  if ( !v102 )
    _libc_free(v101, &v117);
  if ( v123 != &v124 )
    _libc_free(v123, &v117);
  if ( !BYTE4(v120) )
    _libc_free(v118, &v117);
  if ( v115 != &v116 )
    _libc_free(v115, &v117);
  if ( !BYTE4(v113) )
    _libc_free(v112[0], &v117);
  if ( v92 != &v94 )
    _libc_free(v92, &v117);
  if ( !v90 )
    _libc_free(v86, &v117);
  if ( (_QWORD *)v99[12] != &v99[14] )
    _libc_free(v99[12], &v117);
  if ( !BYTE4(v99[3]) )
    _libc_free(v99[1], &v117);
  v22 = *(char **)(a1 + 144);
  v23 = *(char **)(a1 + 136);
  if ( v23 == v22 )
    goto LABEL_131;
  v24 = v22 - 8;
  if ( v23 < v22 - 8 )
  {
    do
    {
      v25 = *(_QWORD *)v23;
      v26 = *(_QWORD *)v24;
      v23 += 8;
      v24 -= 8;
      *((_QWORD *)v23 - 1) = v26;
      *((_QWORD *)v24 + 1) = v25;
    }
    while ( v24 > v23 );
    v23 = *(char **)(a1 + 136);
    v22 = *(char **)(a1 + 144);
    if ( v23 == v22 )
    {
LABEL_131:
      v46 = (const __m128i **)(a1 + 64);
      v47 = (v22 - v23) >> 3;
      goto LABEL_81;
    }
  }
  v27 = v23;
  v79 = a1 + 160;
  do
  {
    v28 = (v27 - v23) >> 3;
    v29 = *(_QWORD *)v27;
    v112[0] = 2;
    v112[1] = 0;
    v113 = v29;
    if ( v29 == -4096 || v29 == 0 || v29 == -8192 )
    {
      v121 = v29;
      v30 = a1;
      v111 = &unk_49E5548;
      v114 = a1;
      LODWORD(v117) = v28;
      v119 = 2;
      v120 = 0;
    }
    else
    {
      v82 = v28;
      sub_BD73F0((__int64)v112);
      v111 = &unk_49E5548;
      v114 = a1;
      LODWORD(v117) = v82;
      v120 = 0;
      v119 = v112[0] & 6;
      v121 = v113;
      if ( v113 == 0 || v113 == -4096 || v113 == -8192 )
      {
        v30 = a1;
      }
      else
      {
        sub_BD6050(&v119, v112[0] & 0xFFFFFFFFFFFFFFF8LL);
        v30 = v114;
      }
    }
    v118 = &unk_49E5548;
    v122 = v30;
    v31 = *(_QWORD *)v27;
    v105 = 0;
    v107 = v31;
    v106 = 0;
    if ( v31 != -4096 && v31 != 0 && v31 != -8192 )
      sub_BD73F0((__int64)&v105);
    v32 = *(_DWORD *)(a1 + 184);
    if ( !v32 )
    {
      ++*(_QWORD *)(a1 + 160);
      goto LABEL_48;
    }
    v34 = v107;
    v41 = *(_QWORD *)(a1 + 168);
    v42 = (v32 - 1) & (((unsigned int)v107 >> 9) ^ ((unsigned int)v107 >> 4));
    v43 = (_QWORD *)(v41 + 72LL * v42);
    v44 = v43[2];
    if ( v44 != v107 )
    {
      v65 = 1;
      v35 = 0;
      while ( v44 != -4096 )
      {
        if ( !v35 && v44 == -8192 )
          v35 = v43;
        v42 = (v32 - 1) & (v65 + v42);
        v43 = (_QWORD *)(v41 + 72LL * v42);
        v44 = v43[2];
        if ( v107 == v44 )
          goto LABEL_63;
        ++v65;
      }
      if ( !v35 )
        v35 = v43;
      v66 = *(_DWORD *)(a1 + 176);
      ++*(_QWORD *)(a1 + 160);
      v36 = v66 + 1;
      if ( 4 * v36 < 3 * v32 )
      {
        if ( v32 - *(_DWORD *)(a1 + 180) - v36 > v32 >> 3 )
        {
LABEL_51:
          *(_DWORD *)(a1 + 176) = v36;
          if ( v35[2] == -4096 )
          {
            if ( v34 == -4096 )
              goto LABEL_59;
          }
          else
          {
            --*(_DWORD *)(a1 + 180);
            v37 = v35[2];
            if ( v34 == v37 )
              goto LABEL_59;
            if ( v37 != -4096 && v37 != 0 && v37 != -8192 )
            {
              v80 = v34;
              sub_BD60C0(v35);
              v34 = v80;
            }
          }
          v35[2] = v34;
          if ( v34 != 0 && v34 != -4096 && v34 != -8192 )
            sub_BD73F0((__int64)v35);
LABEL_59:
          v35[4] = &unk_49E5548;
          v38 = v35 + 3;
          v39 = v35 + 5;
          *((_DWORD *)v39 - 4) = -1;
          *(_OWORD *)(v39 + 1) = 0;
          v39[3] = 0;
          *v39 = 2;
          *((_DWORD *)v39 - 4) = v117;
          v40 = v121;
          if ( !v121 )
            goto LABEL_70;
          goto LABEL_67;
        }
        sub_FE0650(v79, v32);
        v67 = *(_DWORD *)(a1 + 184);
        if ( !v67 )
          goto LABEL_49;
        v34 = v107;
        v68 = v67 - 1;
        v69 = *(_QWORD *)(a1 + 168);
        v70 = 0;
        v71 = 1;
        LODWORD(v72) = v68 & (((unsigned int)v107 >> 9) ^ ((unsigned int)v107 >> 4));
        v35 = (_QWORD *)(v69 + 72LL * (unsigned int)v72);
        v73 = v35[2];
        if ( v107 == v73 )
          goto LABEL_50;
        while ( v73 != -4096 )
        {
          if ( !v70 && v73 == -8192 )
            v70 = v35;
          v72 = v68 & (unsigned int)(v72 + v71);
          v35 = (_QWORD *)(v69 + 72 * v72);
          v73 = v35[2];
          if ( v107 == v73 )
            goto LABEL_50;
          ++v71;
        }
        goto LABEL_125;
      }
LABEL_48:
      sub_FE0650(v79, 2 * v32);
      v33 = *(_DWORD *)(a1 + 184);
      if ( !v33 )
      {
LABEL_49:
        v34 = v107;
        v35 = 0;
LABEL_50:
        v36 = *(_DWORD *)(a1 + 176) + 1;
        goto LABEL_51;
      }
      v34 = v107;
      v74 = v33 - 1;
      v75 = *(_QWORD *)(a1 + 168);
      v70 = 0;
      v76 = 1;
      LODWORD(v77) = v74 & (((unsigned int)v107 >> 9) ^ ((unsigned int)v107 >> 4));
      v35 = (_QWORD *)(v75 + 72LL * (unsigned int)v77);
      v78 = v35[2];
      if ( v78 == v107 )
        goto LABEL_50;
      while ( v78 != -4096 )
      {
        if ( !v70 && v78 == -8192 )
          v70 = v35;
        v77 = v74 & (unsigned int)(v77 + v76);
        v35 = (_QWORD *)(v75 + 72 * v77);
        v78 = v35[2];
        if ( v107 == v78 )
          goto LABEL_50;
        ++v76;
      }
LABEL_125:
      if ( v70 )
        v35 = v70;
      goto LABEL_50;
    }
LABEL_63:
    v45 = v43[7];
    v38 = v43 + 3;
    *((_DWORD *)v43 + 6) = v117;
    v40 = v121;
    if ( v121 == v45 )
      goto LABEL_70;
    v39 = v43 + 5;
    if ( v45 != -4096 && v45 != 0 && v45 != -8192 )
    {
      v81 = v43 + 3;
      sub_BD60C0(v39);
      v40 = v121;
      v38 = v81;
    }
LABEL_67:
    v38[4] = v40;
    if ( v40 != 0 && v40 != -4096 && v40 != -8192 )
    {
      v83 = v38;
      sub_BD6050(v39, v119 & 0xFFFFFFFFFFFFFFF8LL);
      v38 = v83;
    }
LABEL_70:
    v38[5] = v122;
    if ( v107 != -4096 && v107 != 0 && v107 != -8192 )
      sub_BD60C0(&v105);
    v118 = &unk_49DB368;
    if ( v121 != 0 && v121 != -4096 && v121 != -8192 )
      sub_BD60C0(&v119);
    v111 = &unk_49DB368;
    if ( v113 != -4096 && v113 != 0 && v113 != -8192 )
      sub_BD60C0(v112);
    v23 = *(char **)(a1 + 136);
    v27 += 8;
  }
  while ( v22 != v27 );
  v22 = *(char **)(a1 + 144);
  v46 = (const __m128i **)(a1 + 64);
  v47 = (v22 - v23) >> 3;
  if ( (unsigned __int64)(v22 - v23) > 0x2AAAAAAAAAAAAAA8LL )
    sub_4262D8((__int64)"vector::reserve");
LABEL_81:
  v48 = *(const __m128i **)(a1 + 64);
  v49 = v48;
  if ( 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 80) - (_QWORD)v48) >> 3) < v47 )
  {
    v50 = *(const __m128i **)(a1 + 72);
    v51 = 0;
    v52 = (char *)v50 - (char *)v48;
    if ( v47 )
    {
      v53 = sub_22077B0(24 * v47);
      v48 = *(const __m128i **)(a1 + 64);
      v50 = *(const __m128i **)(a1 + 72);
      v51 = (__m128i *)v53;
      v49 = v48;
    }
    if ( v50 != v48 )
    {
      v54 = v51;
      do
      {
        if ( v54 )
        {
          *v54 = _mm_loadu_si128(v49);
          v54[1].m128i_i64[0] = v49[1].m128i_i64[0];
        }
        v49 = (const __m128i *)((char *)v49 + 24);
        v54 = (__m128i *)((char *)v54 + 24);
      }
      while ( v50 != v49 );
    }
    if ( v48 )
      j_j___libc_free_0(v48, *(_QWORD *)(a1 + 80) - (_QWORD)v48);
    v55 = &v51->m128i_i8[24 * v47];
    *(_QWORD *)(a1 + 64) = v51;
    v23 = *(char **)(a1 + 136);
    *(_QWORD *)(a1 + 72) = (char *)v51 + v52;
    v22 = *(char **)(a1 + 144);
    *(_QWORD *)(a1 + 80) = v55;
  }
  v117 = 0;
  if ( v22 == v23 )
  {
    v60 = *(_QWORD *)(a1 + 16);
    v61 = *(_QWORD *)(a1 + 8);
    v59 = 0;
    v63 = 0xAAAAAAAAAAAAAAABLL * ((v60 - v61) >> 3);
    goto LABEL_101;
  }
  LODWORD(v56) = 0;
  do
  {
    v57 = *(__m128i **)(a1 + 72);
    if ( v57 == *(__m128i **)(a1 + 80) )
    {
      sub_FDDEB0(v46, v57, &v117);
    }
    else
    {
      if ( v57 )
      {
        v57->m128i_i32[0] = v56;
        v57->m128i_i64[1] = 0;
        v57[1].m128i_i64[0] = 0;
        v57 = *(__m128i **)(a1 + 72);
      }
      *(_QWORD *)(a1 + 72) = (char *)v57 + 24;
    }
    v56 = v117 + 1;
    v58 = (__int64)(*(_QWORD *)(a1 + 144) - *(_QWORD *)(a1 + 136)) >> 3;
    v117 = v56;
    v59 = v58;
  }
  while ( v56 < v58 );
  v60 = *(_QWORD *)(a1 + 16);
  v61 = *(_QWORD *)(a1 + 8);
  v62 = 0xAAAAAAAAAAAAAAABLL * ((v60 - v61) >> 3);
  v63 = v62;
  if ( v58 > v62 )
  {
    sub_FDE060((const __m128i **)(a1 + 8), v58 - v62);
  }
  else
  {
LABEL_101:
    if ( v59 < v63 )
    {
      v64 = v61 + 24 * v59;
      if ( v60 != v64 )
        *(_QWORD *)(a1 + 16) = v64;
    }
  }
}
