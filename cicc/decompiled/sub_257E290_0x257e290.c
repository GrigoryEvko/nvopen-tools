// Function: sub_257E290
// Address: 0x257e290
//
__int64 __fastcall sub_257E290(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdi
  char v6; // si
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned int v10; // r12d
  unsigned __int64 *v11; // r12
  __int64 v12; // rax
  unsigned __int64 *v13; // r14
  unsigned __int64 v14; // rbx
  int v15; // edx
  __int64 v16; // rdx
  __int64 v17; // rdx
  _QWORD *v18; // rax
  _QWORD *v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // rdx
  _QWORD *v22; // rdx
  unsigned int v23; // esi
  int v24; // r10d
  __int64 v25; // rdx
  unsigned int v26; // eax
  _QWORD *v27; // r13
  __int64 v28; // rcx
  unsigned __int64 v29; // r14
  unsigned __int64 v30; // r13
  void *v31; // rbx
  int v32; // ecx
  __int64 v33; // rdx
  __int64 v34; // rax
  int v35; // eax
  __int64 v37; // rax
  int v38; // eax
  __int64 v39; // rcx
  int v40; // edx
  unsigned int v41; // eax
  __int64 v42; // rsi
  int v43; // edi
  int v44; // eax
  unsigned __int64 v45; // r12
  unsigned __int64 v46; // rbx
  __int64 v47; // rax
  __int64 *v48; // rdx
  __int64 *v49; // rcx
  __int64 v50; // rsi
  __int64 v51; // rax
  const void *v52; // rsi
  __int64 v53; // r13
  unsigned __int64 v54; // rsi
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r13
  __int64 (__fastcall *v60)(__int64, __int64); // rax
  int v61; // eax
  __int64 v62; // rcx
  __int64 v63; // rdx
  __int64 v64; // r12
  unsigned int v65; // ebx
  char v66; // al
  __int64 v67; // rbx
  int v68; // ecx
  bool (__fastcall *v69)(__int64); // rax
  int v70; // eax
  int v71; // ecx
  __int64 v72; // [rsp-10h] [rbp-190h]
  __int64 v73; // [rsp-10h] [rbp-190h]
  unsigned __int64 v74; // [rsp-8h] [rbp-188h]
  unsigned __int64 *v75; // [rsp+0h] [rbp-180h]
  __int64 v76; // [rsp+8h] [rbp-178h]
  unsigned __int8 *v77; // [rsp+10h] [rbp-170h]
  _BYTE *v78; // [rsp+38h] [rbp-148h]
  int v80; // [rsp+40h] [rbp-140h]
  int v81; // [rsp+48h] [rbp-138h]
  char v82; // [rsp+4Fh] [rbp-131h]
  char v83; // [rsp+5Fh] [rbp-121h] BYREF
  unsigned __int64 v84; // [rsp+60h] [rbp-120h] BYREF
  unsigned __int64 v85; // [rsp+68h] [rbp-118h] BYREF
  __m128i v86; // [rsp+70h] [rbp-110h] BYREF
  __int64 v87; // [rsp+80h] [rbp-100h] BYREF
  __int64 v88; // [rsp+88h] [rbp-F8h]
  unsigned __int64 v89; // [rsp+90h] [rbp-F0h]
  _QWORD v90[4]; // [rsp+A0h] [rbp-E0h] BYREF
  unsigned __int64 *v91; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v92; // [rsp+C8h] [rbp-B8h]
  _BYTE v93[48]; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v94; // [rsp+100h] [rbp-80h] BYREF
  __int64 v95; // [rsp+108h] [rbp-78h]
  __int64 v96; // [rsp+110h] [rbp-70h]
  __int64 v97; // [rsp+118h] [rbp-68h]
  void *s1; // [rsp+120h] [rbp-60h] BYREF
  __int64 v99; // [rsp+128h] [rbp-58h]
  _BYTE v100[80]; // [rsp+130h] [rbp-50h] BYREF

  v4 = sub_2509740((_QWORD *)(a1 + 72));
  v5 = *(_QWORD *)(v4 - 32);
  v77 = (unsigned __int8 *)v4;
  s1 = v100;
  v99 = 0x400000000LL;
  v6 = *(_BYTE *)(a1 + 296);
  v89 = v4 - 32;
  v90[1] = &v87;
  v82 = v6;
  v91 = (unsigned __int64 *)v93;
  v90[2] = &v94;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v87 = a2;
  v88 = a1;
  v90[0] = a1;
  v83 = 0;
  v92 = 0x300000000LL;
  v86.m128i_i64[0] = sub_250D2C0(v5, 0);
  v86.m128i_i64[1] = v7;
  if ( !(unsigned __int8)sub_2526B50(a2, &v86, a1, (__int64)&v91, 3u, &v83, 1u) )
  {
    if ( !*(_DWORD *)(a1 + 176) )
    {
      v10 = 0;
      *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
      goto LABEL_32;
    }
    sub_257B840(v90);
  }
  v11 = v91;
  v12 = 2LL * (unsigned int)v92;
  v13 = &v91[v12];
  if ( v91 == &v91[v12] )
    goto LABEL_27;
  while ( 1 )
  {
    v14 = *v11;
    v15 = *(unsigned __int8 *)*v11;
    if ( (unsigned int)(v15 - 12) <= 1 )
      goto LABEL_7;
    if ( (_BYTE)v15 == 20 )
      break;
    if ( (_BYTE)v15 )
      goto LABEL_40;
    v16 = *(unsigned int *)(a1 + 176);
    v84 = *v11;
    if ( !(_DWORD)v16 )
      goto LABEL_21;
    if ( *(_DWORD *)(a1 + 152) )
    {
      v38 = *(_DWORD *)(a1 + 160);
      v39 = *(_QWORD *)(a1 + 144);
      if ( !v38 )
        goto LABEL_7;
      v40 = v38 - 1;
      v41 = (v38 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v42 = *(_QWORD *)(v39 + 8LL * v41);
      if ( v14 != v42 )
      {
        v43 = 1;
        while ( v42 != -4096 )
        {
          v8 = (unsigned int)(v43 + 1);
          v41 = v40 & (v43 + v41);
          v42 = *(_QWORD *)(v39 + 8LL * v41);
          if ( v14 == v42 )
            goto LABEL_21;
          ++v43;
        }
        goto LABEL_7;
      }
      goto LABEL_21;
    }
    v17 = 8 * v16;
    v18 = *(_QWORD **)(a1 + 168);
    v19 = &v18[(unsigned __int64)v17 / 8];
    v20 = v17 >> 3;
    v21 = v17 >> 5;
    if ( v21 )
    {
      v22 = &v18[4 * v21];
      while ( v14 != *v18 )
      {
        if ( v14 == v18[1] )
        {
          ++v18;
          goto LABEL_20;
        }
        if ( v14 == v18[2] )
        {
          v18 += 2;
          goto LABEL_20;
        }
        if ( v14 == v18[3] )
        {
          v18 += 3;
          goto LABEL_20;
        }
        v18 += 4;
        if ( v22 == v18 )
        {
          v20 = v19 - v18;
          goto LABEL_101;
        }
      }
      goto LABEL_20;
    }
LABEL_101:
    if ( v20 == 2 )
      goto LABEL_117;
    if ( v20 == 3 )
    {
      if ( v14 == *v18 )
        goto LABEL_20;
      ++v18;
LABEL_117:
      if ( v14 == *v18 )
        goto LABEL_20;
      ++v18;
      goto LABEL_104;
    }
    if ( v20 != 1 )
      goto LABEL_7;
LABEL_104:
    if ( v14 != *v18 )
      goto LABEL_7;
LABEL_20:
    if ( v19 == v18 )
      goto LABEL_7;
LABEL_21:
    v23 = *(_DWORD *)(a1 + 128);
    v85 = v14;
    if ( !v23 )
    {
      ++*(_QWORD *)(a1 + 104);
      v86.m128i_i64[0] = 0;
      goto LABEL_98;
    }
    v9 = v23 - 1;
    v8 = *(_QWORD *)(a1 + 112);
    v24 = 1;
    v25 = 0;
    v26 = v9 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v27 = (_QWORD *)(v8 + 16LL * v26);
    v28 = *v27;
    if ( v14 != *v27 )
    {
      while ( v28 != -4096 )
      {
        if ( v28 == -8192 && !v25 )
          v25 = (__int64)v27;
        v26 = v9 & (v24 + v26);
        v27 = (_QWORD *)(v8 + 16LL * v26);
        v28 = *v27;
        if ( v14 == *v27 )
          goto LABEL_23;
        ++v24;
      }
      v70 = *(_DWORD *)(a1 + 120);
      if ( !v25 )
        v25 = (__int64)v27;
      ++*(_QWORD *)(a1 + 104);
      v71 = v70 + 1;
      v86.m128i_i64[0] = v25;
      if ( 4 * (v70 + 1) < 3 * v23 )
      {
        v8 = v14;
        v9 = v23 >> 3;
        if ( v23 - *(_DWORD *)(a1 + 124) - v71 > (unsigned int)v9 )
        {
LABEL_93:
          *(_DWORD *)(a1 + 120) = v71;
          if ( *(_QWORD *)v25 != -4096 )
            --*(_DWORD *)(a1 + 124);
          *(_QWORD *)v25 = v8;
          *(_BYTE *)(v25 + 9) = 0;
          v78 = (_BYTE *)(v25 + 8);
          goto LABEL_24;
        }
LABEL_99:
        sub_256A8F0(a1 + 104, v23);
        sub_255F6D0(a1 + 104, (__int64 *)&v85, &v86);
        v8 = v85;
        v25 = v86.m128i_i64[0];
        v71 = *(_DWORD *)(a1 + 120) + 1;
        goto LABEL_93;
      }
LABEL_98:
      v23 *= 2;
      goto LABEL_99;
    }
LABEL_23:
    v78 = v27 + 1;
LABEL_24:
    if ( v78[1] )
    {
      if ( !*v78 )
        goto LABEL_7;
LABEL_26:
      v11 += 2;
      sub_2571A80((__int64)&v94, (__int64 *)&v84);
      if ( v13 == v11 )
        goto LABEL_27;
    }
    else
    {
      v53 = v87;
      v54 = sub_250D2C0(v14, 0);
      v56 = sub_257B470(v53, v54, v55, v88, 1, 0, 1);
      v59 = v56;
      if ( !v56 )
        goto LABEL_69;
      v54 = v89;
      v60 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v56 + 112LL);
      if ( v60 == sub_254A530 )
      {
        if ( !*(_BYTE *)(v59 + 97) || (unsigned __int8)sub_B19060(v59 + 104, v89, v57, v58) )
        {
LABEL_69:
          v81 = *(_QWORD *)(v14 + 104);
          v76 = *(_QWORD *)(v14 + 104);
          v61 = sub_A17190(v77);
          v63 = v76;
          if ( (int)v76 <= v61 )
          {
LABEL_109:
            *(_WORD *)v78 = 257;
            goto LABEL_26;
          }
          v75 = v11;
          v64 = v14;
          v65 = v61;
          while ( 1 )
          {
            LOBYTE(v85) = 0;
            if ( (*(_BYTE *)(v64 + 2) & 1) != 0 )
              sub_B2C6D0(v64, v54, v63, v62);
            sub_250D230((unsigned __int64 *)&v86, *(_QWORD *)(v64 + 96) + 40LL * v65, 6, 0);
            v66 = sub_257DDD0(a2, a1, &v86, 1, &v85, 0, 0);
            v62 = v73;
            v54 = v74;
            if ( v66 )
              break;
            if ( v81 == ++v65 )
            {
              v11 = v75;
              goto LABEL_109;
            }
          }
          v11 = v75;
          if ( (_BYTE)v85 )
            goto LABEL_82;
          goto LABEL_7;
        }
      }
      else if ( ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, __int64, __int64, __int64))v60)(
                  v59,
                  v89,
                  v57,
                  v58,
                  v72) )
      {
        goto LABEL_69;
      }
      v69 = *(bool (__fastcall **)(__int64))(*(_QWORD *)(v59 + 88) + 24LL);
      if ( v69 != sub_2534ED0 )
      {
        if ( !v69(v59 + 88) )
          goto LABEL_7;
LABEL_82:
        *(_WORD *)v78 = 256;
        goto LABEL_7;
      }
      if ( *(_BYTE *)(v59 + 97) == *(_BYTE *)(v59 + 96) )
        goto LABEL_82;
LABEL_7:
      v11 += 2;
      if ( v13 == v11 )
        goto LABEL_27;
    }
  }
  v37 = *(_QWORD *)(v14 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v37 + 8) - 17 <= 1 )
    v37 = **(_QWORD **)(v37 + 16);
  if ( !(*(_DWORD *)(v37 + 8) >> 8) )
    goto LABEL_7;
LABEL_40:
  if ( !*(_DWORD *)(a1 + 176) )
  {
    v82 = 0;
    goto LABEL_7;
  }
  sub_257B840(v90);
LABEL_27:
  v29 = (unsigned int)v99;
  v30 = *(unsigned int *)(a1 + 256);
  v31 = *(void **)(a1 + 248);
  v32 = v99;
  if ( (unsigned int)v99 != v30
    || 8LL * (unsigned int)v99
    && (v80 = v99, v44 = memcmp(s1, *(const void **)(a1 + 248), 8LL * (unsigned int)v99), v32 = v80, v44)
    || (v10 = 1, *(_BYTE *)(a1 + 296) != v82) )
  {
    v33 = v95;
    v34 = *(_QWORD *)(a1 + 224);
    ++*(_QWORD *)(a1 + 216);
    *(_QWORD *)(a1 + 224) = v33;
    v95 = v34;
    LODWORD(v34) = *(_DWORD *)(a1 + 232);
    *(_DWORD *)(a1 + 232) = v96;
    LODWORD(v96) = v34;
    LODWORD(v34) = *(_DWORD *)(a1 + 236);
    *(_DWORD *)(a1 + 236) = HIDWORD(v96);
    HIDWORD(v96) = v34;
    LODWORD(v34) = *(_DWORD *)(a1 + 240);
    ++v94;
    *(_DWORD *)(a1 + 240) = v97;
    LODWORD(v97) = v34;
    if ( v31 == (void *)(a1 + 264) || s1 == v100 )
    {
      if ( v29 > *(unsigned int *)(a1 + 260) )
      {
        sub_C8D5F0(a1 + 248, (const void *)(a1 + 264), v29, 8u, v8, v9);
        v30 = *(unsigned int *)(a1 + 256);
      }
      if ( v30 > HIDWORD(v99) )
      {
        sub_C8D5F0((__int64)&s1, v100, v30, 8u, v8, v9);
        v30 = *(unsigned int *)(a1 + 256);
      }
      v45 = (unsigned int)v99;
      v46 = v30;
      if ( (unsigned int)v99 <= v30 )
        v46 = (unsigned int)v99;
      if ( v46 )
      {
        v47 = 0;
        do
        {
          v48 = (__int64 *)((char *)s1 + v47);
          v49 = (__int64 *)(v47 + *(_QWORD *)(a1 + 248));
          v47 += 8;
          v50 = *v49;
          *v49 = *v48;
          *v48 = v50;
        }
        while ( 8 * v46 != v47 );
        v30 = *(unsigned int *)(a1 + 256);
        v45 = (unsigned int)v99;
      }
      if ( v45 >= v30 )
      {
        if ( v45 > v30 )
        {
          v67 = 8 * v46;
          v68 = v30;
          if ( (char *)s1 + v67 != (char *)s1 + 8 * v45 )
          {
            memcpy((void *)(*(_QWORD *)(a1 + 248) + 8 * v30), (char *)s1 + v67, 8 * v45 - v67);
            v68 = *(_DWORD *)(a1 + 256);
          }
          *(_DWORD *)(a1 + 256) = v68 + v45 - v30;
        }
      }
      else
      {
        v51 = *(_QWORD *)(a1 + 248);
        v52 = (const void *)(v51 + 8 * v46);
        if ( v52 != (const void *)(8 * v30 + v51) )
          memcpy((char *)s1 + 8 * v45, v52, 8 * v30 - 8 * v46);
        *(_DWORD *)(a1 + 256) = v46;
      }
    }
    else
    {
      *(_QWORD *)(a1 + 248) = s1;
      v35 = HIDWORD(v99);
      s1 = v31;
      *(_DWORD *)(a1 + 256) = v32;
      *(_DWORD *)(a1 + 260) = v35;
    }
    v10 = 0;
    *(_BYTE *)(a1 + 296) = v82;
  }
LABEL_32:
  if ( v91 != (unsigned __int64 *)v93 )
    _libc_free((unsigned __int64)v91);
  if ( s1 != v100 )
    _libc_free((unsigned __int64)s1);
  sub_C7D6A0(v95, 8LL * (unsigned int)v97, 8);
  return v10;
}
