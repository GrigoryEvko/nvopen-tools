// Function: sub_1FDCFE0
// Address: 0x1fdcfe0
//
__int64 __fastcall sub_1FDCFE0(_QWORD *a1, __int64 a2)
{
  unsigned int v2; // r14d
  _QWORD *v3; // rbx
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r11
  __int64 v10; // rax
  __int64 v11; // r15
  unsigned int v12; // esi
  __int64 v13; // r8
  unsigned int v14; // r12d
  unsigned int v15; // edi
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r12
  __int64 *v19; // rax
  char v20; // dl
  __int64 v21; // rax
  __int64 v22; // rdx
  _QWORD *v23; // r12
  __int64 v24; // r14
  unsigned __int8 v25; // al
  __int64 v26; // rax
  __int64 v27; // rcx
  unsigned __int64 v28; // rsi
  unsigned __int64 v29; // rdx
  unsigned int v30; // r12d
  __int64 v32; // rax
  char v33; // di
  unsigned int v34; // esi
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // rbx
  __int64 v40; // rsi
  __int64 v41; // rsi
  __int64 v42; // rsi
  __int64 v43; // rsi
  __int32 v44; // edx
  __int64 v45; // rcx
  _BYTE *v46; // rax
  _BYTE *v47; // rbx
  __m128i *v48; // rsi
  __int64 v49; // rsi
  unsigned __int8 *v50; // rsi
  __int64 v51; // rax
  __int64 *v52; // rsi
  __int64 *v53; // rcx
  __int64 *v54; // rdx
  int v55; // eax
  int v56; // eax
  __int64 v57; // rsi
  int v58; // eax
  int v59; // r12d
  __int64 v60; // r10
  unsigned int v61; // ecx
  __int64 v62; // r8
  int v63; // edi
  __int64 *v64; // rsi
  __int64 v65; // rax
  __int64 v66; // rsi
  unsigned __int64 v67; // rdx
  unsigned __int64 v68; // rcx
  __int64 v69; // rdx
  int v70; // r10d
  int v71; // r10d
  __int64 v72; // r9
  __int64 *v73; // rcx
  unsigned int v74; // r12d
  int v75; // esi
  __int64 v76; // rdi
  __int64 v77; // [rsp+0h] [rbp-B0h]
  __int64 *v78; // [rsp+0h] [rbp-B0h]
  unsigned int v79; // [rsp+8h] [rbp-A8h]
  int v80; // [rsp+Ch] [rbp-A4h]
  __int64 v81; // [rsp+10h] [rbp-A0h]
  __int64 v82; // [rsp+18h] [rbp-98h]
  _BYTE *v83; // [rsp+18h] [rbp-98h]
  int v84; // [rsp+18h] [rbp-98h]
  __int64 v85; // [rsp+18h] [rbp-98h]
  __int64 v86; // [rsp+18h] [rbp-98h]
  __m128i v87; // [rsp+20h] [rbp-90h] BYREF
  __int64 v88; // [rsp+30h] [rbp-80h] BYREF
  __int64 *v89; // [rsp+38h] [rbp-78h]
  __int64 *v90; // [rsp+40h] [rbp-70h]
  __int64 v91; // [rsp+48h] [rbp-68h]
  int v92; // [rsp+50h] [rbp-60h]
  _BYTE v93[88]; // [rsp+58h] [rbp-58h] BYREF

  v2 = 0;
  v3 = a1;
  v5 = sub_157EBA0(a2);
  v6 = a1[5];
  v88 = 0;
  v81 = v5;
  v7 = v5;
  v89 = (__int64 *)v93;
  v90 = (__int64 *)v93;
  v8 = *(_QWORD *)(v6 + 912) - *(_QWORD *)(v6 + 904);
  v91 = 4;
  v92 = 0;
  *(_DWORD *)(v6 + 928) = v8 >> 4;
  v80 = sub_15F4D60(v7);
  if ( !v80 )
  {
LABEL_60:
    v30 = 1;
    goto LABEL_18;
  }
  while ( 1 )
  {
LABEL_4:
    v9 = sub_15F4DF0(v81, v2);
    v10 = *(_QWORD *)(v9 + 48);
    if ( !v10 )
      BUG();
    if ( *(_BYTE *)(v10 - 8) != 77 )
      goto LABEL_3;
    v11 = v3[5];
    v12 = *(_DWORD *)(v11 + 72);
    if ( v12 )
    {
      v13 = *(_QWORD *)(v11 + 56);
      v14 = ((unsigned int)v9 >> 4) ^ ((unsigned int)v9 >> 9);
      v15 = (v12 - 1) & v14;
      v16 = (__int64 *)(v13 + 16LL * v15);
      v17 = *v16;
      if ( v9 == *v16 )
      {
        v18 = v16[1];
        goto LABEL_9;
      }
      v84 = 1;
      v54 = 0;
      while ( v17 != -8 )
      {
        if ( v17 != -16 || v54 )
          v16 = v54;
        v15 = (v12 - 1) & (v84 + v15);
        v78 = (__int64 *)(v13 + 16LL * v15);
        v17 = *v78;
        if ( v9 == *v78 )
        {
          v18 = v78[1];
          goto LABEL_9;
        }
        ++v84;
        v54 = v16;
        v16 = (__int64 *)(v13 + 16LL * v15);
      }
      if ( !v54 )
        v54 = v16;
      v55 = *(_DWORD *)(v11 + 64);
      ++*(_QWORD *)(v11 + 48);
      v56 = v55 + 1;
      if ( 4 * v56 < 3 * v12 )
      {
        if ( v12 - *(_DWORD *)(v11 + 68) - v56 <= v12 >> 3 )
        {
          v86 = v9;
          sub_1D52F30(v11 + 48, v12);
          v70 = *(_DWORD *)(v11 + 72);
          if ( !v70 )
          {
LABEL_125:
            ++*(_DWORD *)(v11 + 64);
            BUG();
          }
          v71 = v70 - 1;
          v72 = *(_QWORD *)(v11 + 56);
          v73 = 0;
          v74 = v71 & v14;
          v9 = v86;
          v75 = 1;
          v56 = *(_DWORD *)(v11 + 64) + 1;
          v54 = (__int64 *)(v72 + 16LL * v74);
          v76 = *v54;
          if ( v86 != *v54 )
          {
            while ( v76 != -8 )
            {
              if ( !v73 && v76 == -16 )
                v73 = v54;
              v74 = v71 & (v75 + v74);
              v54 = (__int64 *)(v72 + 16LL * v74);
              v76 = *v54;
              if ( v86 == *v54 )
                goto LABEL_83;
              ++v75;
            }
            if ( v73 )
              v54 = v73;
          }
        }
        goto LABEL_83;
      }
    }
    else
    {
      ++*(_QWORD *)(v11 + 48);
    }
    v85 = v9;
    sub_1D52F30(v11 + 48, 2 * v12);
    v58 = *(_DWORD *)(v11 + 72);
    if ( !v58 )
      goto LABEL_125;
    v9 = v85;
    v59 = v58 - 1;
    v60 = *(_QWORD *)(v11 + 56);
    v61 = (v58 - 1) & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
    v56 = *(_DWORD *)(v11 + 64) + 1;
    v54 = (__int64 *)(v60 + 16LL * v61);
    v62 = *v54;
    if ( v85 != *v54 )
    {
      v63 = 1;
      v64 = 0;
      while ( v62 != -8 )
      {
        if ( v62 == -16 && !v64 )
          v64 = v54;
        v61 = v59 & (v63 + v61);
        v54 = (__int64 *)(v60 + 16LL * v61);
        v62 = *v54;
        if ( v85 == *v54 )
          goto LABEL_83;
        ++v63;
      }
      if ( v64 )
        v54 = v64;
    }
LABEL_83:
    *(_DWORD *)(v11 + 64) = v56;
    if ( *v54 != -8 )
      --*(_DWORD *)(v11 + 68);
    *v54 = v9;
    v18 = 0;
    v54[1] = 0;
LABEL_9:
    v19 = v89;
    if ( v90 == v89 )
    {
      v52 = &v89[HIDWORD(v91)];
      if ( v89 != v52 )
      {
        v53 = 0;
        do
        {
          if ( *v19 == v18 )
            goto LABEL_3;
          if ( *v19 == -2 )
            v53 = v19;
          ++v19;
        }
        while ( v52 != v19 );
        if ( v53 )
        {
          *v53 = v18;
          --v92;
          ++v88;
          goto LABEL_11;
        }
      }
      if ( HIDWORD(v91) < (unsigned int)v91 )
        break;
    }
    v82 = v9;
    sub_16CCBA0((__int64)&v88, v18);
    v9 = v82;
    if ( v20 )
      goto LABEL_11;
LABEL_3:
    if ( v80 == ++v2 )
      goto LABEL_60;
  }
  ++HIDWORD(v91);
  *v52 = v18;
  ++v88;
LABEL_11:
  v83 = *(_BYTE **)(v18 + 32);
  v21 = sub_157F280(v9);
  v77 = v22;
  if ( v21 == v22 )
    goto LABEL_3;
  v79 = v2;
  v23 = v3;
  v24 = v21;
  while ( !*(_QWORD *)(v24 + 8) )
  {
LABEL_55:
    v51 = *(_QWORD *)(v24 + 32);
    if ( !v51 )
      BUG();
    v24 = 0;
    if ( *(_BYTE *)(v51 - 8) == 77 )
      v24 = v51 - 24;
    if ( v77 == v24 )
    {
      v3 = v23;
      v2 = v79 + 1;
      if ( v80 == v79 + 1 )
        goto LABEL_60;
      goto LABEL_4;
    }
  }
  v25 = sub_1FD35E0(v23[12], *(_QWORD *)v24);
  if ( v25 <= 1u || !*(_QWORD *)(v23[14] + 8LL * v25 + 120) && v25 != 2 && (unsigned __int8)(v25 - 3) > 1u )
  {
    v26 = v23[5];
    v27 = *(_QWORD *)(v26 + 904);
    v28 = *(unsigned int *)(v26 + 928);
    v29 = (*(_QWORD *)(v26 + 912) - v27) >> 4;
    if ( v28 > v29 )
    {
      sub_1FD4090((const __m128i **)(v26 + 904), v28 - v29);
    }
    else if ( v28 < v29 )
    {
      v57 = v27 + 16 * v28;
      if ( *(_QWORD *)(v26 + 912) != v57 )
        *(_QWORD *)(v26 + 912) = v57;
    }
    goto LABEL_17;
  }
  v32 = 0x17FFFFFFE8LL;
  v33 = *(_BYTE *)(v24 + 23) & 0x40;
  v34 = *(_DWORD *)(v24 + 20) & 0xFFFFFFF;
  if ( v34 )
  {
    v35 = 24LL * *(unsigned int *)(v24 + 56) + 8;
    v36 = 0;
    do
    {
      v37 = v24 - 24LL * v34;
      if ( v33 )
        v37 = *(_QWORD *)(v24 - 8);
      if ( a2 == *(_QWORD *)(v37 + v35) )
      {
        v32 = 24 * v36;
        goto LABEL_29;
      }
      ++v36;
      v35 += 8;
    }
    while ( v34 != (_DWORD)v36 );
    v32 = 0x17FFFFFFE8LL;
  }
LABEL_29:
  if ( v33 )
    v38 = *(_QWORD *)(v24 - 8);
  else
    v38 = v24 - 24LL * v34;
  v39 = *(_QWORD *)(v38 + v32);
  if ( v23 + 10 != (_QWORD *)(v24 + 48) )
  {
    v40 = v23[10];
    if ( v40 )
      sub_161E7C0((__int64)(v23 + 10), v40);
    v41 = *(_QWORD *)(v24 + 48);
    v23[10] = v41;
    if ( v41 )
      sub_1623A60((__int64)(v23 + 10), v41, 2);
  }
  if ( *(_BYTE *)(v39 + 16) > 0x17u && v23 + 10 != (_QWORD *)(v39 + 48) )
  {
    v42 = v23[10];
    if ( v42 )
      sub_161E7C0((__int64)(v23 + 10), v42);
    v43 = *(_QWORD *)(v39 + 48);
    v23[10] = v43;
    if ( v43 )
      sub_1623A60((__int64)(v23 + 10), v43, 2);
  }
  v44 = sub_1FD8F60(v23, v39);
  if ( v44 )
  {
    v45 = v23[5];
    v46 = v83;
    if ( !v83 )
      BUG();
    if ( (*v83 & 4) == 0 && (v83[46] & 8) != 0 )
    {
      do
        v46 = (_BYTE *)*((_QWORD *)v46 + 1);
      while ( (v46[46] & 8) != 0 );
    }
    v47 = (_BYTE *)*((_QWORD *)v46 + 1);
    v87.m128i_i32[2] = v44;
    v87.m128i_i64[0] = (__int64)v83;
    v48 = *(__m128i **)(v45 + 912);
    if ( v48 == *(__m128i **)(v45 + 920) )
    {
      sub_1FD42F0((const __m128i **)(v45 + 904), v48, &v87);
    }
    else
    {
      if ( v48 )
      {
        *v48 = _mm_loadu_si128(&v87);
        v48 = *(__m128i **)(v45 + 912);
      }
      *(_QWORD *)(v45 + 912) = v48 + 1;
    }
    v87.m128i_i64[0] = 0;
    v49 = v23[10];
    if ( v49 )
    {
      sub_161E7C0((__int64)(v23 + 10), v49);
      v50 = (unsigned __int8 *)v87.m128i_i64[0];
      v23[10] = v87.m128i_i64[0];
      if ( v50 )
        sub_1623210((__int64)&v87, v50, (__int64)(v23 + 10));
    }
    v83 = v47;
    goto LABEL_55;
  }
  v65 = v23[5];
  v66 = *(_QWORD *)(v65 + 904);
  v67 = *(unsigned int *)(v65 + 928);
  v68 = (*(_QWORD *)(v65 + 912) - v66) >> 4;
  if ( v67 > v68 )
  {
    sub_1FD4090((const __m128i **)(v65 + 904), v67 - v68);
  }
  else if ( v67 < v68 )
  {
    v69 = v66 + 16 * v67;
    if ( *(_QWORD *)(v65 + 912) != v69 )
      *(_QWORD *)(v65 + 912) = v69;
  }
LABEL_17:
  v30 = 0;
LABEL_18:
  if ( v90 != v89 )
    _libc_free((unsigned __int64)v90);
  return v30;
}
