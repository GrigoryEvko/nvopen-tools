// Function: sub_35DD2C0
// Address: 0x35dd2c0
//
void __fastcall sub_35DD2C0(__int64 a1, __m128i **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __m128i **v8; // r12
  _BYTE *v9; // r13
  __int64 v10; // rdx
  bool (__fastcall *v11)(__int64, __int64); // rax
  unsigned int v12; // eax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rbx
  _DWORD *v16; // rax
  _BYTE *v17; // rdx
  _DWORD *i; // rdx
  unsigned int v19; // ebx
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 *v23; // r14
  unsigned __int32 v24; // ecx
  _DWORD *v25; // r14
  __int64 v26; // rsi
  int v27; // eax
  _DWORD *v28; // rdx
  __int64 v29; // r15
  unsigned int v30; // ebx
  __int64 v31; // r14
  __int64 v32; // rax
  __int64 v33; // r9
  __int64 v34; // rdx
  _DWORD *v35; // rax
  __int64 v36; // r8
  __int64 v37; // r14
  _DWORD *v38; // rdx
  _DWORD *v39; // rax
  _BYTE *v40; // rdx
  __m128i *v41; // rax
  __m128i *v42; // rdx
  __m128i *j; // rdx
  __int64 v44; // r12
  signed __int64 v45; // r13
  __int64 v46; // r14
  __int64 v47; // rax
  unsigned int v48; // edx
  int v49; // ecx
  unsigned int v50; // ecx
  unsigned int *v51; // rax
  __m128i **v52; // rax
  unsigned int v53; // ebx
  signed __int64 v54; // r14
  unsigned int v55; // r12d
  const __m128i **v56; // r15
  __int64 v57; // rax
  __m128i *v58; // rax
  unsigned int v59; // r13d
  __int64 v60; // rax
  char v61; // al
  __m128i *v62; // rax
  unsigned __int64 v63; // rcx
  __int64 v64; // rax
  unsigned __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // [rsp-8h] [rbp-1B8h]
  __int64 v68; // [rsp+20h] [rbp-190h]
  unsigned int v69; // [rsp+2Ch] [rbp-184h]
  signed __int64 v72; // [rsp+40h] [rbp-170h]
  __int64 v73; // [rsp+48h] [rbp-168h]
  signed __int64 v74; // [rsp+48h] [rbp-168h]
  __int64 v75; // [rsp+58h] [rbp-158h]
  __int64 v76; // [rsp+58h] [rbp-158h]
  unsigned __int32 v77; // [rsp+70h] [rbp-140h]
  __m128i **v78; // [rsp+70h] [rbp-140h]
  signed __int64 v79; // [rsp+80h] [rbp-130h]
  __m128i v81; // [rsp+90h] [rbp-120h] BYREF
  __m128i v82; // [rsp+A0h] [rbp-110h] BYREF
  int v83; // [rsp+B0h] [rbp-100h]
  _BYTE *v84; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v85; // [rsp+C8h] [rbp-E8h]
  _BYTE v86[32]; // [rsp+D0h] [rbp-E0h] BYREF
  _DWORD *v87; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v88; // [rsp+F8h] [rbp-B8h]
  _DWORD v89[8]; // [rsp+100h] [rbp-B0h] BYREF
  _BYTE *v90; // [rsp+120h] [rbp-90h] BYREF
  __int64 v91; // [rsp+128h] [rbp-88h]
  _BYTE v92[32]; // [rsp+130h] [rbp-80h] BYREF
  __m128i v93; // [rsp+150h] [rbp-60h] BYREF
  __m128i v94; // [rsp+160h] [rbp-50h] BYREF
  int v95; // [rsp+170h] [rbp-40h]

  v8 = a2;
  v9 = *(_BYTE **)(a1 + 80);
  v10 = *(_QWORD *)(a3 + 40);
  v11 = *(bool (__fastcall **)(__int64, __int64))(*(_QWORD *)v9 + 664LL);
  if ( v11 == sub_2FE3D70 )
  {
    v93.m128i_i64[0] = sub_B2D7E0(*(_QWORD *)(v10 + 72), "no-jump-tables", 0xEu);
    if ( (unsigned __int8)sub_A72A30(v93.m128i_i64) || (v9[7217] & 0xFB) != 0 && (v9[7216] & 0xFB) != 0 )
      return;
  }
  else if ( !v11((__int64)v9, *(_QWORD *)(v10 + 72)) )
  {
    return;
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 80) + 856LL))(*(_QWORD *)(a1 + 80));
  v69 = v12;
  v15 = (char *)a2[1] - (char *)*a2;
  v75 = v15;
  if ( v15 <= 40 )
    return;
  v68 = v12;
  v79 = 0xCCCCCCCCCCCCCCCDLL * (v15 >> 3);
  if ( v12 > v79 )
    return;
  v16 = v86;
  v17 = v86;
  v84 = v86;
  v85 = 0x800000000LL;
  if ( v15 > 320 )
  {
    sub_C8D5F0((__int64)&v84, v86, v79, 4u, v13, v14);
    v17 = v84;
    v16 = &v84[4 * (unsigned int)v85];
  }
  for ( i = &v17[0x3333333333333334LL * (v15 >> 3)]; i != v16; ++v16 )
  {
    if ( v16 )
      *v16 = 0;
  }
  v73 = a6;
  v19 = 0;
  v20 = 0;
  LODWORD(v85) = v79;
  do
  {
    v21 = (__int64)&(*v8)->m128i_i64[5 * v20];
    v22 = *(_QWORD *)(v21 + 16);
    v23 = (__int64 *)(*(_QWORD *)(v21 + 8) + 24LL);
    LODWORD(v91) = *(_DWORD *)(v22 + 32);
    if ( (unsigned int)v91 > 0x40 )
      sub_C43780((__int64)&v90, (const void **)(v22 + 24));
    else
      v90 = *(_BYTE **)(v22 + 24);
    sub_C46B40((__int64)&v90, v23);
    v24 = v91;
    v25 = v90;
    LODWORD(v91) = 0;
    v26 = 4 * v20;
    v93.m128i_i32[2] = v24;
    v93.m128i_i64[0] = (__int64)v90;
    v77 = v24;
    if ( v24 <= 0x40 )
    {
      *(_DWORD *)&v84[4 * v20] = (_DWORD)v90 + 1;
    }
    else
    {
      v27 = sub_C444A0((__int64)&v93);
      v26 = 4 * v20;
      v28 = &v84[4 * v20];
      if ( v77 - v27 <= 0x40 )
      {
        *v28 = *v25 + 1;
LABEL_31:
        j_j___libc_free_0_0((unsigned __int64)v25);
        goto LABEL_14;
      }
      *v28 = 0;
      if ( v25 )
        goto LABEL_31;
    }
LABEL_14:
    if ( (unsigned int)v91 > 0x40 && v90 )
      j_j___libc_free_0_0((unsigned __int64)v90);
    if ( v19 )
      *(_DWORD *)&v84[v26] += *(_DWORD *)&v84[4 * v19 - 4];
    v20 = ++v19;
  }
  while ( v19 < v79 );
  v29 = v73;
  v30 = v79 - 1;
  v31 = sub_35D8900(v8, 0, (int)v79 - 1);
  v32 = sub_35D8A40(&v84, 0, (int)v79 - 1);
  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64, __int64))(**(_QWORD **)(a1 + 80)
                                                                                              + 672LL))(
         *(_QWORD *)(a1 + 80),
         a3,
         v32,
         v31,
         v73,
         a7) )
  {
    v95 = -1;
    v61 = sub_35DBCC0(a1, v8, 0, v30, a3, a4, a5, (__int64)&v93);
    v33 = v67;
    if ( v61 )
    {
      v62 = *v8;
      *v62 = _mm_loadu_si128(&v93);
      v62[1] = _mm_loadu_si128(&v94);
      v62[2].m128i_i32[0] = v95;
      v63 = (char *)v8[1] - (char *)*v8;
      if ( v63 )
      {
        if ( v63 > 0x28 )
        {
          v64 = (__int64)&(*v8)[2].m128i_i64[1];
          if ( v8[1] != (__m128i *)v64 )
            v8[1] = (__m128i *)v64;
        }
      }
      else
      {
        sub_35D8D40((const __m128i **)v8, 1u);
      }
      goto LABEL_35;
    }
  }
  v34 = *(_QWORD *)(a1 + 88);
  if ( (unsigned int)(*(_DWORD *)(v34 + 544) - 42) <= 1 || !*(_DWORD *)(v34 + 648) )
    goto LABEL_35;
  v35 = v89;
  v36 = 0x800000000LL;
  v37 = 4 * v79;
  v87 = v89;
  v88 = 0x800000000LL;
  v38 = &v89[v79];
  v74 = v79 - 1;
  if ( v75 > 320 )
  {
    sub_C8D5F0((__int64)&v87, v89, v79, 4u, 0x800000000LL, v33);
    v36 = 0x800000000LL;
    v35 = &v87[(unsigned int)v88];
    v38 = &v87[(unsigned __int64)v37 / 4];
    if ( &v87[(unsigned __int64)v37 / 4] == v35 )
    {
      v91 = 0x800000000LL;
      LODWORD(v88) = v79;
      v90 = v92;
      goto LABEL_100;
    }
  }
  do
  {
    if ( v35 )
      *v35 = 0;
    ++v35;
  }
  while ( v38 != v35 );
  v91 = 0x800000000LL;
  LODWORD(v88) = v79;
  v39 = v92;
  v40 = &v92[v37];
  v90 = v92;
  if ( v75 > 320 )
  {
LABEL_100:
    sub_C8D5F0((__int64)&v90, v92, v79, 4u, 0x800000000LL, v33);
    v39 = &v90[4 * (unsigned int)v91];
    v40 = &v90[v37];
    if ( &v90[v37] != (_BYTE *)v39 )
      goto LABEL_43;
    LODWORD(v91) = v79;
    v93.m128i_i64[0] = (__int64)&v94;
    v93.m128i_i64[1] = 0x800000000LL;
  }
  else
  {
    do
    {
LABEL_43:
      if ( v39 )
        *v39 = 0;
      ++v39;
    }
    while ( v40 != (_BYTE *)v39 );
    v93.m128i_i64[1] = 0x800000000LL;
    LODWORD(v91) = v79;
    v41 = &v94;
    v42 = &v94;
    v93.m128i_i64[0] = (__int64)&v94;
    if ( v75 <= 320 )
      goto LABEL_47;
  }
  sub_C8D5F0((__int64)&v93, &v94, v79, 4u, v36, v33);
  v42 = (__m128i *)v93.m128i_i64[0];
  v41 = (__m128i *)(v93.m128i_i64[0] + 4LL * v93.m128i_u32[2]);
LABEL_47:
  for ( j = (__m128i *)((char *)v42 + v37); j != v41; v41 = (__m128i *)((char *)v41 + 4) )
  {
    if ( v41 )
      v41->m128i_i32[0] = 0;
  }
  v93.m128i_i32[2] = v79;
  v87[v74] = 1;
  *(_DWORD *)&v90[4 * v74] = v30;
  *(_DWORD *)(v93.m128i_i64[0] + 4 * v74) = 2;
  v72 = v79 - 2;
  if ( v79 != 1 )
  {
    v78 = v8;
    do
    {
      v44 = v79 - v72;
      v87[v72] = v87[v72 + 1] + 1;
      *(_DWORD *)&v90[4 * v72] = v72;
      *(_DWORD *)(v93.m128i_i64[0] + 4 * v72) = *(_DWORD *)(v93.m128i_i64[0] + 4 * v72 + 4) + 2;
      if ( v72 < v74 )
      {
        v76 = v72;
        v45 = v79 - 1;
        do
        {
          while ( 1 )
          {
            v46 = sub_35D8900(v78, v72, v45);
            v47 = sub_35D8A40(&v84, v72, v45);
            if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64, __int64))(**(_QWORD **)(a1 + 80) + 672LL))(
                   *(_QWORD *)(a1 + 80),
                   a3,
                   v47,
                   v46,
                   v29,
                   a7) )
            {
              if ( v45 == v74 )
              {
                v48 = 1;
                v49 = 0;
              }
              else
              {
                v48 = v87[v45 + 1] + 1;
                v49 = *(_DWORD *)(v93.m128i_i64[0] + 4 * (v45 + 1));
              }
              if ( v44 == 1 )
                v50 = v49 + 2;
              else
                v50 = v69 >> 1 < v44 ? (v68 <= v44) + v49 : v49 + 1;
              v51 = &v87[v76];
              if ( v87[v76] > v48 || *v51 == v48 && *(_DWORD *)(v93.m128i_i64[0] + 4 * v72) < v50 )
                break;
            }
            --v44;
            --v45;
            if ( v44 == 1 )
              goto LABEL_65;
          }
          *v51 = v48;
          --v44;
          *(_DWORD *)&v90[4 * v72] = v45--;
          *(_DWORD *)(v93.m128i_i64[0] + 4 * v72) = v50;
        }
        while ( v44 != 1 );
      }
LABEL_65:
      --v72;
    }
    while ( v72 != -1 );
    v8 = v78;
  }
  v52 = v8;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = (const __m128i **)v52;
  do
  {
    v59 = *(_DWORD *)&v90[4 * v54];
    v83 = -1;
    if ( v69 <= v59 + 1 - v53 && (unsigned __int8)sub_35DBCC0(a1, v56, v53, v59, a3, a4, a5, (__int64)&v81) )
    {
      v57 = v55++;
      v58 = (__m128i *)((char *)*v56 + 40 * v57);
      *v58 = _mm_loadu_si128(&v81);
      v58[1] = _mm_loadu_si128(&v82);
      v58[2].m128i_i32[0] = v83;
    }
    else if ( v53 <= v59 )
    {
      while ( 1 )
      {
        ++v53;
        v60 = 5LL * v55++;
        memmove((char *)*v56 + 8 * v60, (char *)*v56 + 40 * v54, 0x28u);
        if ( v59 < v53 )
          break;
        v54 = v53;
      }
    }
    v54 = v59 + 1;
    v53 = v59 + 1;
  }
  while ( v54 < v79 );
  v65 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v56[1] - (char *)*v56) >> 3);
  if ( v65 < v55 )
  {
    sub_35D8D40(v56, v55 - v65);
  }
  else if ( v65 > v55 )
  {
    v66 = (__int64)&(*v56)->m128i_i64[5 * v55];
    if ( v56[1] != (const __m128i *)v66 )
      v56[1] = (const __m128i *)v66;
  }
  if ( (__m128i *)v93.m128i_i64[0] != &v94 )
    _libc_free(v93.m128i_u64[0]);
  if ( v90 != v92 )
    _libc_free((unsigned __int64)v90);
  if ( v87 != v89 )
    _libc_free((unsigned __int64)v87);
LABEL_35:
  if ( v84 != v86 )
    _libc_free((unsigned __int64)v84);
}
