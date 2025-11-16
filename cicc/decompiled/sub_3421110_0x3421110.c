// Function: sub_3421110
// Address: 0x3421110
//
void __fastcall sub_3421110(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // rdi
  __int64 (*v5)(); // rcx
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 (*v8)(); // rcx
  __int64 v9; // rdx
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rbx
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __m128i v16; // xmm3
  __m128i v17; // xmm4
  unsigned int v18; // eax
  _QWORD **v19; // rbx
  __int64 v20; // rax
  _QWORD *v21; // r13
  unsigned __int64 v22; // r15
  __int64 v23; // rdi
  unsigned int v24; // eax
  _QWORD *v25; // rbx
  _QWORD *v26; // r13
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 *v29; // rax
  __int64 *v30; // rbx
  unsigned __int64 v31; // r13
  unsigned __int64 v32; // rbx
  __int64 *v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 *v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 *v40; // rbx
  __int64 v41; // r13
  __int64 v42; // r15
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // r9
  __int64 *v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __m128i v50; // xmm5
  __m128i v51; // xmm6
  __m128i v52; // xmm7
  __m128i v53; // xmm0
  __m128i v54; // xmm1
  __int64 *v55; // rdx
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 *v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 *v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // r14
  __int64 v66; // rbx
  _QWORD *v67; // rax
  _QWORD *v68; // rax
  unsigned __int64 v69; // rdi
  __int64 *v70; // rdx
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 *v73; // rdx
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 v78; // r11
  __int64 v79; // r10
  unsigned __int64 v80; // rdi
  __int64 v81; // [rsp+0h] [rbp-100h]
  __int64 v82; // [rsp+8h] [rbp-F8h]
  __int64 v83; // [rsp+8h] [rbp-F8h]
  __int64 v84; // [rsp+10h] [rbp-F0h]
  _QWORD **v85; // [rsp+18h] [rbp-E8h]
  __int64 v86; // [rsp+18h] [rbp-E8h]
  __int64 v87; // [rsp+18h] [rbp-E8h]
  __int64 v88; // [rsp+18h] [rbp-E8h]
  __m128i v89; // [rsp+20h] [rbp-E0h] BYREF
  __m128i v90; // [rsp+30h] [rbp-D0h] BYREF
  __m128i v91; // [rsp+40h] [rbp-C0h] BYREF
  __m128i v92; // [rsp+50h] [rbp-B0h] BYREF
  __m128i v93; // [rsp+60h] [rbp-A0h] BYREF
  _BYTE v94[8]; // [rsp+70h] [rbp-90h] BYREF
  _QWORD *v95; // [rsp+78h] [rbp-88h]
  unsigned int v96; // [rsp+88h] [rbp-78h]
  __int64 v97; // [rsp+98h] [rbp-68h]
  unsigned int v98; // [rsp+A8h] [rbp-58h]
  __int64 v99; // [rsp+B8h] [rbp-48h]
  unsigned int v100; // [rsp+C8h] [rbp-38h]

  v3 = *(__int64 **)(a1 + 40);
  v4 = v3[2];
  v84 = *v3;
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 128LL);
  v6 = 0;
  if ( v5 != sub_2DAC790 )
  {
    v6 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v5)(v4, a2, 0);
    v3 = *(__int64 **)(a1 + 40);
  }
  *(_QWORD *)(a1 + 800) = v6;
  v7 = v3[2];
  v8 = *(__int64 (**)())(*(_QWORD *)v7 + 144LL);
  v9 = 0;
  if ( v8 != sub_2C8F680 )
  {
    v9 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v8)(v7, a2, 0);
    v3 = *(__int64 **)(a1 + 40);
  }
  *(_QWORD *)(a1 + 808) = v9;
  *(_QWORD *)(a1 + 56) = v3[4];
  v10 = *(__int64 **)(a2 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
    goto LABEL_113;
  while ( *(_UNKNOWN **)v11 != &unk_4F6D3F0 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_113;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)v94);
  sub_983BD0((__int64)&v89, v13 + 176, v84);
  v82 = v13 + 408;
  if ( *(_BYTE *)(v13 + 488) )
  {
    v14 = _mm_loadu_si128(&v90);
    v15 = _mm_loadu_si128(&v91);
    v16 = _mm_loadu_si128(&v92);
    v17 = _mm_loadu_si128(&v93);
    *(__m128i *)(v13 + 408) = _mm_loadu_si128(&v89);
    *(__m128i *)(v13 + 424) = v14;
    *(__m128i *)(v13 + 440) = v15;
    *(__m128i *)(v13 + 456) = v16;
    *(__m128i *)(v13 + 472) = v17;
  }
  else
  {
    v50 = _mm_loadu_si128(&v89);
    v51 = _mm_loadu_si128(&v90);
    *(_BYTE *)(v13 + 488) = 1;
    v52 = _mm_loadu_si128(&v91);
    v53 = _mm_loadu_si128(&v92);
    v54 = _mm_loadu_si128(&v93);
    *(__m128i *)(v13 + 408) = v50;
    *(__m128i *)(v13 + 424) = v51;
    *(__m128i *)(v13 + 440) = v52;
    *(__m128i *)(v13 + 456) = v53;
    *(__m128i *)(v13 + 472) = v54;
  }
  sub_C7D6A0(v99, 24LL * v100, 8);
  v18 = v98;
  if ( v98 )
  {
    v19 = (_QWORD **)(v97 + 8);
    v85 = (_QWORD **)(v97 + 32LL * v98);
    while ( 1 )
    {
      v20 = (__int64)*(v19 - 1);
      if ( v20 != -4096 && v20 != -8192 )
      {
        v21 = *v19;
        while ( v21 != v19 )
        {
          v22 = (unsigned __int64)v21;
          v21 = (_QWORD *)*v21;
          v23 = *(_QWORD *)(v22 + 24);
          if ( v23 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v23 + 8LL))(v23);
          j_j___libc_free_0(v22);
        }
      }
      if ( v85 == v19 + 3 )
        break;
      v19 += 4;
    }
    v18 = v98;
  }
  sub_C7D6A0(v97, 32LL * v18, 8);
  v24 = v96;
  if ( v96 )
  {
    v25 = v95;
    v26 = &v95[2 * v96];
    do
    {
      if ( *v25 != -8192 && *v25 != -4096 )
      {
        v27 = v25[1];
        if ( v27 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v27 + 8LL))(v27);
      }
      v25 += 2;
    }
    while ( v26 != v25 );
    v24 = v96;
  }
  sub_C7D6A0((__int64)v95, 16LL * v24, 8);
  *(_QWORD *)(a1 + 16) = v82;
  v28 = 0;
  if ( (*(_BYTE *)(v84 + 3) & 0x40) != 0 )
  {
    v58 = *(__int64 **)(a2 + 8);
    v59 = *v58;
    v60 = v58[1];
    if ( v59 == v60 )
      goto LABEL_113;
    while ( *(_UNKNOWN **)v59 != &unk_501DA08 )
    {
      v59 += 16;
      if ( v60 == v59 )
        goto LABEL_113;
    }
    v61 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v59 + 8) + 104LL))(
            *(_QWORD *)(v59 + 8),
            &unk_501DA08);
    v28 = sub_2DD15F0(v61, v84);
  }
  *(_QWORD *)(a1 + 776) = v28;
  v29 = (__int64 *)sub_22077B0(0x18u);
  v30 = v29;
  if ( v29 )
    sub_1049690(v29, v84);
  v31 = *(_QWORD *)(a1 + 888);
  *(_QWORD *)(a1 + 888) = v30;
  if ( v31 )
  {
    v32 = *(_QWORD *)(v31 + 16);
    if ( v32 )
    {
      sub_FDC110(*(__int64 **)(v31 + 16));
      j_j___libc_free_0(v32);
    }
    j_j___libc_free_0(v31);
  }
  v33 = *(__int64 **)(a2 + 8);
  v34 = *v33;
  v35 = v33[1];
  if ( v34 == v35 )
    goto LABEL_113;
  while ( *(_UNKNOWN **)v34 != &unk_4F8662C )
  {
    v34 += 16;
    if ( v35 == v34 )
      goto LABEL_113;
  }
  v36 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v34 + 8) + 104LL))(*(_QWORD *)(v34 + 8), &unk_4F8662C);
  *(_QWORD *)(a1 + 768) = sub_CFFAC0(v36, v84);
  v37 = *(__int64 **)(a2 + 8);
  v38 = *v37;
  v39 = v37[1];
  if ( v38 == v39 )
    goto LABEL_113;
  while ( *(_UNKNOWN **)v38 != &unk_4F87C64 )
  {
    v38 += 16;
    if ( v39 == v38 )
      goto LABEL_113;
  }
  v40 = 0;
  v41 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v38 + 8) + 104LL))(
                      *(_QWORD *)(v38 + 8),
                      &unk_4F87C64)
                  + 176);
  if ( v41 )
  {
    v40 = *(__int64 **)(v41 + 8);
    if ( v40 )
    {
      v40 = 0;
      if ( *(_DWORD *)(a1 + 792) )
      {
        v73 = *(__int64 **)(a2 + 8);
        v74 = *v73;
        v75 = v73[1];
        if ( v74 == v75 )
          goto LABEL_113;
        while ( *(_UNKNOWN **)v74 != &unk_4F8EE48 )
        {
          v74 += 16;
          if ( v75 == v74 )
            goto LABEL_113;
        }
        v76 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v74 + 8) + 104LL))(
                *(_QWORD *)(v74 + 8),
                &unk_4F8EE48);
        v40 = (__int64 *)(v76 + 176);
        if ( !*(_BYTE *)(v76 + 184) )
        {
          v77 = *(_QWORD *)(v76 + 200);
          v78 = *(_QWORD *)(v76 + 208);
          v79 = *(_QWORD *)(v77 + 176);
          if ( !*(_BYTE *)(v79 + 280) )
          {
            v81 = v76;
            v83 = *(_QWORD *)(v76 + 208);
            v88 = *(_QWORD *)(v77 + 176);
            sub_FF9360((_QWORD *)v88, *(_QWORD *)(v79 + 288), *(_QWORD *)(v79 + 296), *(__int64 **)(v79 + 304), 0, 0);
            v79 = v88;
            v76 = v81;
            v78 = v83;
            *(_BYTE *)(v88 + 280) = 1;
          }
          v87 = v76;
          sub_FE7D70(v40, *(const char **)(v76 + 192), v79, v78);
          *(_BYTE *)(v87 + 184) = 1;
        }
      }
    }
  }
  v42 = 0;
  if ( (unsigned __int8)sub_AEA460(*(_QWORD *)(v84 + 40)) )
  {
    v55 = *(__int64 **)(a2 + 8);
    v56 = *v55;
    v57 = v55[1];
    if ( v56 == v57 )
      goto LABEL_113;
    while ( *(_UNKNOWN **)v56 != &unk_50165D0 )
    {
      v56 += 16;
      if ( v57 == v56 )
        goto LABEL_113;
    }
    v42 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v56 + 8) + 104LL))(
                        *(_QWORD *)(v56 + 8),
                        &unk_50165D0)
                    + 176);
  }
  v43 = sub_B82360(*(_QWORD *)(a2 + 8), (__int64)&unk_4F8FC84);
  if ( v43 && (v44 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v43 + 104LL))(v43, &unk_4F8FC84)) != 0 )
    v45 = v44 + 184;
  else
    v45 = 0;
  v46 = *(__int64 **)(a2 + 8);
  v47 = *v46;
  v48 = v46[1];
  if ( v47 == v48 )
    goto LABEL_113;
  while ( *(_UNKNOWN **)v47 != &unk_50208C0 )
  {
    v47 += 16;
    if ( v48 == v47 )
      goto LABEL_113;
  }
  v86 = v45;
  v49 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v47 + 8) + 104LL))(*(_QWORD *)(v47 + 8), &unk_50208C0);
  sub_33CC500(
    *(_QWORD **)(a1 + 64),
    *(_QWORD *)(a1 + 40),
    *(_QWORD *)(a1 + 888),
    a2,
    *(_QWORD *)(a1 + 16),
    v86,
    v41,
    (__int64)v40,
    v49 + 176,
    v42);
  if ( (_BYTE)qword_5039C48 && *(_DWORD *)(a1 + 792) )
  {
    v70 = *(__int64 **)(a2 + 8);
    v71 = *v70;
    v72 = v70[1];
    if ( v71 == v72 )
      goto LABEL_113;
    while ( *(_UNKNOWN **)v71 != &unk_4F8E808 )
    {
      v71 += 16;
      if ( v72 == v71 )
        goto LABEL_113;
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 24) + 32LL) = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v71 + 8) + 104LL))(
                                                 *(_QWORD *)(v71 + 8),
                                                 &unk_4F8E808)
                                             + 176;
  }
  else
  {
    *(_QWORD *)(*(_QWORD *)(a1 + 24) + 32LL) = 0;
  }
  if ( !*(_DWORD *)(a1 + 792) )
  {
    if ( *(_BYTE *)(a1 + 760) )
    {
      *(_BYTE *)(a1 + 760) = 0;
      *(_QWORD *)(a1 + 608) = &unk_49DDBE8;
      if ( (*(_BYTE *)(a1 + 624) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(a1 + 632), 16LL * *(unsigned int *)(a1 + 640), 8);
      nullsub_184();
      v69 = *(_QWORD *)(a1 + 456);
      if ( v69 != a1 + 472 )
        _libc_free(v69);
      if ( (*(_BYTE *)(a1 + 104) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(a1 + 112), 40LL * *(unsigned int *)(a1 + 120), 8);
    }
    return;
  }
  v62 = *(__int64 **)(a2 + 8);
  v63 = *v62;
  v64 = v62[1];
  if ( v63 == v64 )
LABEL_113:
    BUG();
  while ( *(_UNKNOWN **)v63 != &unk_4F86530 )
  {
    v63 += 16;
    if ( v64 == v63 )
      goto LABEL_113;
  }
  v65 = a1 + 472;
  v66 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v63 + 8) + 104LL))(
                      *(_QWORD *)(v63 + 8),
                      &unk_4F86530)
                  + 176);
  if ( *(_BYTE *)(a1 + 760) )
  {
    *(_BYTE *)(a1 + 760) = 0;
    *(_QWORD *)(a1 + 608) = &unk_49DDBE8;
    if ( (*(_BYTE *)(a1 + 624) & 1) == 0 )
      sub_C7D6A0(*(_QWORD *)(a1 + 632), 16LL * *(unsigned int *)(a1 + 640), 8);
    nullsub_184();
    v80 = *(_QWORD *)(a1 + 456);
    if ( v65 != v80 )
      _libc_free(v80);
    if ( (*(_BYTE *)(a1 + 104) & 1) == 0 )
      sub_C7D6A0(*(_QWORD *)(a1 + 112), 40LL * *(unsigned int *)(a1 + 120), 8);
  }
  *(_QWORD *)(a1 + 80) = v66;
  v67 = (_QWORD *)(a1 + 112);
  *(_QWORD *)(a1 + 88) = v66;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 1;
  do
  {
    if ( v67 )
    {
      *v67 = -4;
      v67[1] = -3;
      v67[2] = -4;
      v67[3] = -3;
    }
    v67 += 5;
  }
  while ( v67 != (_QWORD *)(a1 + 432) );
  *(_QWORD *)(a1 + 432) = a1 + 608;
  *(_QWORD *)(a1 + 464) = 0x400000000LL;
  *(_QWORD *)(a1 + 440) = 0;
  *(_BYTE *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = v65;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 624) = 1;
  *(_WORD *)(a1 + 600) = 256;
  *(_QWORD *)(a1 + 608) = &unk_49DDBE8;
  v68 = (_QWORD *)(a1 + 632);
  do
  {
    if ( v68 )
      *v68 = -4096;
    v68 += 2;
  }
  while ( v68 != (_QWORD *)(a1 + 760) );
  *(_BYTE *)(a1 + 760) = 1;
}
