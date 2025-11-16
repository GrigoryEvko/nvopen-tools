// Function: sub_2E01B80
// Address: 0x2e01b80
//
void __fastcall sub_2E01B80(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12)
{
  __int64 v13; // r14
  __int64 v14; // r13
  __int64 v15; // rbx
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // r11
  __int64 v19; // rax
  unsigned __int8 v20; // r11
  __int64 v21; // r15
  unsigned __int8 v22; // al
  _DWORD **v23; // rdi
  unsigned __int64 v24; // r12
  _DWORD *v25; // rax
  __int64 v26; // r8
  unsigned int *v27; // r9
  unsigned __int8 v28; // r10
  __int64 v29; // rax
  __int64 v30; // r9
  unsigned int *v31; // rbx
  const __m128i *v32; // r12
  __int64 *v33; // rdx
  __int64 v34; // rax
  unsigned int *i; // r13
  unsigned __int64 v36; // r11
  __m128i *v37; // rax
  bool v38; // r9
  _QWORD *v39; // r14
  bool v40; // r15
  __int64 v41; // rbx
  __int64 v42; // rsi
  __int64 v43; // rdx
  int v44; // r9d
  unsigned __int64 v45; // r8
  unsigned __int64 v46; // r14
  __int64 v47; // r9
  __int64 v48; // r12
  const __m128i *v49; // r13
  __int64 v50; // rax
  __int64 v51; // r8
  int v52; // eax
  __int64 v53; // rax
  __int64 v54; // rsi
  __int64 v55; // r8
  int v56; // r9d
  unsigned int v57; // eax
  __int64 v58; // rdx
  unsigned __int64 v59; // rcx
  unsigned __int64 j; // rax
  __int64 k; // r10
  __int16 v62; // dx
  unsigned int v63; // ecx
  __int64 *v64; // rdx
  __int64 v65; // r11
  unsigned int *v66; // r13
  __int64 v67; // r12
  __int64 v68; // rax
  __int64 v69; // r12
  unsigned int *v70; // r12
  _BYTE *v71; // rdi
  int v72; // ecx
  int v73; // edx
  signed __int64 v74; // r12
  _QWORD *v75; // rax
  __int64 v76; // rdx
  int v77; // edi
  __int64 v78; // rax
  __int64 *v79; // rsi
  __int64 v80; // rdx
  __int64 v81; // [rsp+18h] [rbp-218h]
  _BOOL4 v82; // [rsp+24h] [rbp-20Ch]
  int v83; // [rsp+28h] [rbp-208h]
  __int64 v84; // [rsp+30h] [rbp-200h]
  __int64 v87; // [rsp+48h] [rbp-1E8h]
  __int64 v88; // [rsp+48h] [rbp-1E8h]
  unsigned __int8 v90; // [rsp+70h] [rbp-1C0h]
  char v91; // [rsp+70h] [rbp-1C0h]
  __int64 v92; // [rsp+70h] [rbp-1C0h]
  char v94; // [rsp+78h] [rbp-1B8h]
  unsigned int *v95; // [rsp+78h] [rbp-1B8h]
  int v96; // [rsp+78h] [rbp-1B8h]
  _BYTE *v97; // [rsp+80h] [rbp-1B0h] BYREF
  __int64 v98; // [rsp+88h] [rbp-1A8h]
  _BYTE v99[32]; // [rsp+90h] [rbp-1A0h] BYREF
  const __m128i *v100; // [rsp+B0h] [rbp-180h] BYREF
  __int64 v101; // [rsp+B8h] [rbp-178h]
  _QWORD v102[46]; // [rsp+C0h] [rbp-170h] BYREF
  __int64 v103; // [rsp+250h] [rbp+20h]
  __int64 v104; // [rsp+250h] [rbp+20h]
  __int64 v105; // [rsp+260h] [rbp+30h]

  v13 = a9;
  v14 = a7;
  v15 = a11;
  v16 = *(_QWORD *)(a9 + 32);
  v17 = *(unsigned int *)(a2 + 24);
  v18 = *(_QWORD *)(*(_QWORD *)(v16 + 152) + 16 * v17 + 8);
  if ( (*(_DWORD *)((v18 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v18 >> 1) & 3) < (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                        | (unsigned int)(a4 >> 1) & 3) )
    a4 = *(_QWORD *)(*(_QWORD *)(v16 + 152) + 16 * v17 + 8);
  v81 = a4;
  v19 = sub_2E01820(a2, a3, *(_QWORD *)(a9 + 32), a12, v16);
  v20 = *(_BYTE *)(a5 + 8);
  v21 = v19;
  v100 = (const __m128i *)v102;
  v101 = 0x800000000LL;
  v22 = v20 & 0x3F;
  v90 = v20 & 0x3F;
  if ( (v20 & 0x3F) != 0 )
  {
    v23 = (_DWORD **)a5;
    v24 = v22;
    LODWORD(v97) = -1;
    v87 = 2LL * v22;
    v25 = sub_2DF4D60(*v23, (__int64)&(*v23)[(unsigned __int64)v87 / 2], (int *)&v97);
    if ( (_DWORD *)v26 == v25 )
    {
      if ( v27 != (unsigned int *)v26 )
      {
        v29 = *v27;
        v30 = (__int64)(v27 + 1);
        v31 = (unsigned int *)v26;
        v32 = (const __m128i *)(a1[7] + 40 * v29);
        v33 = v102;
        v34 = 0;
        for ( i = (unsigned int *)v30; ; ++i )
        {
          v37 = (__m128i *)&v33[5 * v34];
          *v37 = _mm_loadu_si128(v32);
          v37[1] = _mm_loadu_si128(v32 + 1);
          v37[2].m128i_i64[0] = v32[2].m128i_i64[0];
          v34 = (unsigned int)(v101 + 1);
          LODWORD(v101) = v101 + 1;
          if ( v31 == i )
            break;
          v36 = v34 + 1;
          v32 = (const __m128i *)(a1[7] + 40LL * *i);
          v33 = (__int64 *)v100;
          if ( v34 + 1 > (unsigned __int64)HIDWORD(v101) )
          {
            if ( v100 > v32 || v32 >= (const __m128i *)((char *)v100 + 40 * v34) )
            {
              sub_C8D5F0((__int64)&v100, v102, v36, 0x28u, v26, v30);
              v33 = (__int64 *)v100;
              v34 = (unsigned int)v101;
            }
            else
            {
              v74 = (char *)v32 - (char *)v100;
              sub_C8D5F0((__int64)&v100, v102, v36, 0x28u, v26, v30);
              v33 = (__int64 *)v100;
              v34 = (unsigned int)v101;
              v32 = (const __m128i *)((char *)v100 + v74);
            }
          }
        }
        v13 = a9;
        v15 = a11;
        v14 = a7;
        v20 = *(_BYTE *)(a5 + 8);
      }
      goto LABEL_11;
    }
    if ( v24 > 8 )
    {
      sub_C8D5F0((__int64)&v100, v102, v24, 0x28u, v26, (__int64)v27);
      v78 = (__int64)v100;
      v79 = &v100[v87].m128i_i64[v24];
      do
      {
        if ( v78 )
        {
          *(_BYTE *)v78 = 0;
          v80 = *(_QWORD *)v78;
          *(_DWORD *)(v78 + 8) = 0;
          *(_QWORD *)(v78 + 16) = 0;
          *(_QWORD *)(v78 + 24) = 0;
          *(_QWORD *)v78 = v80 & 0xFFFFFFF0000000FFLL | 0x800000000LL;
          *(_QWORD *)(v78 + 32) = 0;
        }
        v78 += 40;
      }
      while ( (__int64 *)v78 != v79 );
      LODWORD(v101) = v90;
      v20 = *(_BYTE *)(a5 + 8);
      goto LABEL_11;
    }
    if ( v24 )
    {
      v75 = v102;
      do
      {
        v76 = *v75;
        *((_DWORD *)v75 + 2) = 0;
        v75 += 5;
        *(v75 - 3) = 0;
        *(v75 - 2) = 0;
        *(v75 - 1) = 0;
        *(v75 - 5) = v76 & 0xFFFFFFF000000000LL | 0x800000000LL;
      }
      while ( v75 != &v102[v87 * 2 + v24] );
    }
  }
  else
  {
    v28 = 0;
  }
  LODWORD(v101) = v28;
LABEL_11:
  v91 = v20 >> 7;
  v38 = (v20 & 0x40) != 0;
  v88 = *(_QWORD *)(a5 + 16);
  if ( (_DWORD)a8 )
  {
    v103 = v13;
    v84 = v21;
    v39 = *(_QWORD **)(a5 + 16);
    v40 = (v20 & 0x40) != 0;
    v105 = v15;
    v41 = 0;
    while ( 1 )
    {
      v94 = *(_BYTE *)(v14 + v41);
      if ( !v94 )
        goto LABEL_14;
      if ( v91 )
      {
        v42 = *(unsigned int *)(a6 + 4 * v41);
        v97 = v99;
        v98 = 0x400000000LL;
        sub_AF6280((__int64)&v97, v42);
        v43 = (unsigned int)v98;
        v44 = v41;
        v45 = (unsigned int)v98 + 1LL;
        if ( v45 > HIDWORD(v98) )
        {
          sub_C8D5F0((__int64)&v97, v99, (unsigned int)v98 + 1LL, 8u, v45, (unsigned int)v41);
          v43 = (unsigned int)v98;
          v44 = v41;
        }
        *(_QWORD *)&v97[8 * v43] = 6;
        LODWORD(v98) = v98 + 1;
        v39 = (_QWORD *)sub_B0DBA0(v39, v97, (unsigned int)v98, v44, 0);
        if ( v97 == v99 )
          goto LABEL_14;
        _libc_free((unsigned __int64)v97);
        if ( (unsigned int)a8 == ++v41 )
        {
LABEL_21:
          v88 = (__int64)v39;
          v38 = v40;
          v13 = v103;
          v21 = v84;
          v15 = v105;
          break;
        }
      }
      else
      {
        v39 = (_QWORD *)sub_B0DAC0(v39, 2 * v40, *(unsigned int *)(a6 + 4 * v41));
        v40 = v94;
LABEL_14:
        if ( (unsigned int)a8 == ++v41 )
          goto LABEL_21;
      }
    }
  }
  v104 = v13;
  v46 = v21;
  v83 = v91 == 0 ? -560 : -600;
  v82 = v38;
  v92 = a2 + 48;
  do
  {
    sub_2E90D80(a2, v46, (_DWORD)a1 + 32, *(_DWORD *)(a10 + 8) + v83, v82, *a1, (__int64)v100, (unsigned int)v101, v88);
    v48 = (__int64)v100;
    v97 = v99;
    v49 = (const __m128i *)((char *)v100 + 40 * (unsigned int)v101);
    v98 = 0x400000000LL;
    if ( v100 == v49 )
      break;
    v50 = 0;
    do
    {
      while ( *(_BYTE *)v48 )
      {
        v48 += 40;
        if ( v49 == (const __m128i *)v48 )
          goto LABEL_30;
      }
      v51 = *(unsigned int *)(v48 + 8);
      if ( v50 + 1 > (unsigned __int64)HIDWORD(v98) )
      {
        v96 = *(_DWORD *)(v48 + 8);
        sub_C8D5F0((__int64)&v97, v99, v50 + 1, 4u, v51, v47);
        v50 = (unsigned int)v98;
        LODWORD(v51) = v96;
      }
      v48 += 40;
      *(_DWORD *)&v97[4 * v50] = v51;
      v50 = (unsigned int)(v98 + 1);
      LODWORD(v98) = v98 + 1;
    }
    while ( v49 != (const __m128i *)v48 );
LABEL_30:
    if ( !(_DWORD)v50 )
    {
      if ( v97 != v99 )
      {
        v46 = a2 + 48;
        _libc_free((unsigned __int64)v97);
        continue;
      }
      break;
    }
    if ( v92 == v46 )
    {
LABEL_57:
      v71 = v97;
      if ( v97 != v99 )
        goto LABEL_67;
      continue;
    }
    while ( 1 )
    {
      v52 = *(_DWORD *)(v46 + 44);
      if ( (v52 & 4) != 0 || (v52 & 8) == 0 )
      {
        if ( (*(_QWORD *)(*(_QWORD *)(v46 + 16) + 24LL) & 0x200LL) != 0 )
          goto LABEL_66;
      }
      else if ( (unsigned __int8)sub_2E88A90(v46, 512, 1) )
      {
        goto LABEL_66;
      }
      v53 = *(_QWORD *)(v104 + 32);
      v54 = *(unsigned int *)(v53 + 144);
      v55 = *(_QWORD *)(v53 + 128);
      if ( (_DWORD)v54 )
        break;
LABEL_47:
      v66 = (unsigned int *)v97;
      v67 = 4LL * (unsigned int)v98;
      v95 = (unsigned int *)&v97[v67];
      v68 = v67 >> 2;
      v69 = v67 >> 4;
      if ( v69 )
      {
        v70 = (unsigned int *)&v97[16 * v69];
        while ( (unsigned int)sub_2E8E710(v46, *v66, v15, 0, 0) == -1 )
        {
          if ( (unsigned int)sub_2E8E710(v46, v66[1], v15, 0, 0) != -1 )
          {
            ++v66;
            break;
          }
          if ( (unsigned int)sub_2E8E710(v46, v66[2], v15, 0, 0) != -1 )
          {
            v66 += 2;
            break;
          }
          if ( (unsigned int)sub_2E8E710(v46, v66[3], v15, 0, 0) != -1 )
          {
            v66 += 3;
            break;
          }
          v66 += 4;
          if ( v70 == v66 )
          {
            v68 = v95 - v66;
            goto LABEL_69;
          }
        }
LABEL_54:
        if ( v95 != v66 )
          goto LABEL_74;
        goto LABEL_55;
      }
LABEL_69:
      if ( v68 != 2 )
      {
        if ( v68 != 3 )
        {
          if ( v68 != 1 )
            goto LABEL_55;
          goto LABEL_72;
        }
        if ( (unsigned int)sub_2E8E710(v46, *v66, v15, 0, 0) != -1 )
          goto LABEL_54;
        ++v66;
      }
      if ( (unsigned int)sub_2E8E710(v46, *v66, v15, 0, 0) != -1 )
        goto LABEL_54;
      ++v66;
LABEL_72:
      if ( (unsigned int)sub_2E8E710(v46, *v66, v15, 0, 0) != -1 && v95 != v66 )
      {
LABEL_74:
        if ( (*(_BYTE *)v46 & 4) == 0 && (*(_BYTE *)(v46 + 44) & 8) != 0 )
        {
          do
            v46 = *(_QWORD *)(v46 + 8);
          while ( (*(_BYTE *)(v46 + 44) & 8) != 0 );
        }
        v46 = *(_QWORD *)(v46 + 8);
        goto LABEL_57;
      }
LABEL_55:
      if ( (*(_BYTE *)v46 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v46 + 44) & 8) != 0 )
          v46 = *(_QWORD *)(v46 + 8);
      }
      v46 = *(_QWORD *)(v46 + 8);
      if ( v46 == v92 )
        goto LABEL_57;
    }
    v56 = v54 - 1;
    v57 = (v54 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
    v58 = *(_QWORD *)(v55 + 16LL * v57);
    if ( v58 != v46 )
    {
      v72 = 1;
      while ( v58 != -4096 )
      {
        v57 = v56 & (v72 + v57);
        v58 = *(_QWORD *)(v55 + 16LL * v57);
        if ( v58 == v46 )
          goto LABEL_37;
        ++v72;
      }
      goto LABEL_47;
    }
LABEL_37:
    v59 = v46;
    for ( j = v46; (*(_BYTE *)(j + 44) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
      ;
    if ( (*(_DWORD *)(v46 + 44) & 8) != 0 )
    {
      do
        v59 = *(_QWORD *)(v59 + 8);
      while ( (*(_BYTE *)(v59 + 44) & 8) != 0 );
    }
    for ( k = *(_QWORD *)(v59 + 8); k != j; j = *(_QWORD *)(j + 8) )
    {
      v62 = *(_WORD *)(j + 68);
      if ( (unsigned __int16)(v62 - 14) > 4u && v62 != 24 )
        break;
    }
    v63 = v56 & (((unsigned int)j >> 9) ^ ((unsigned int)j >> 4));
    v64 = (__int64 *)(v55 + 16LL * v63);
    v65 = *v64;
    if ( *v64 != j )
    {
      v73 = 1;
      while ( v65 != -4096 )
      {
        v77 = v73 + 1;
        v63 = v56 & (v73 + v63);
        v64 = (__int64 *)(v55 + 16LL * v63);
        v65 = *v64;
        if ( j == *v64 )
          goto LABEL_46;
        v73 = v77;
      }
      v64 = (__int64 *)(v55 + 16 * v54);
    }
LABEL_46:
    if ( *(_DWORD *)((v64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) < *(_DWORD *)((v81 & 0xFFFFFFFFFFFFFFF8LL) + 24) )
      goto LABEL_47;
LABEL_66:
    v46 = a2 + 48;
    v71 = v97;
    if ( v97 == v99 )
      continue;
LABEL_67:
    _libc_free((unsigned __int64)v71);
  }
  while ( v46 != v92 );
  if ( v100 != (const __m128i *)v102 )
    _libc_free((unsigned __int64)v100);
}
