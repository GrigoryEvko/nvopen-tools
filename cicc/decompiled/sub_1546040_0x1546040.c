// Function: sub_1546040
// Address: 0x1546040
//
void __fastcall sub_1546040(__int64 a1, __int64 a2)
{
  unsigned int v3; // esi
  __int64 v5; // r8
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r10
  int v10; // eax
  int v11; // r12d
  unsigned __int32 v12; // r14d
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int v15; // esi
  __int64 v16; // rcx
  int v17; // r10d
  __int32 *v18; // r8
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rdi
  unsigned int j; // eax
  int *v23; // rdi
  int v24; // r11d
  unsigned int v25; // eax
  int v26; // edi
  __int32 v27; // edx
  int v28; // edi
  int v29; // r10d
  __int64 v30; // rsi
  int *v31; // r9
  unsigned __int64 v32; // r8
  unsigned __int64 v33; // r8
  unsigned int i; // eax
  int v35; // r11d
  unsigned int v36; // eax
  int v37; // eax
  int v38; // ecx
  __int64 v39; // r8
  int v40; // edx
  unsigned int v41; // eax
  __int64 *v42; // r12
  __int64 v43; // rdi
  char *v44; // rsi
  char *v45; // rsi
  int v46; // ecx
  __int64 v47; // rax
  __m128i *v48; // rsi
  __m128i *v49; // rsi
  int v50; // r11d
  int v51; // eax
  int v52; // eax
  int v53; // ecx
  int v54; // r10d
  __int64 *v55; // r9
  __int64 v56; // r8
  unsigned int v57; // eax
  __int64 v58; // rdi
  int v59; // eax
  int v60; // edi
  int v61; // edi
  int v62; // r10d
  __int64 v63; // rsi
  unsigned __int64 v64; // r8
  unsigned __int64 v65; // r8
  unsigned int k; // eax
  int v67; // r11d
  unsigned int v68; // eax
  int v69; // r10d
  int *v70; // [rsp+0h] [rbp-50h]
  __int64 v71; // [rsp+8h] [rbp-48h] BYREF
  __m128i v72; // [rsp+10h] [rbp-40h] BYREF

  v71 = a2;
  if ( !a2 )
    return;
  v3 = *(_DWORD *)(a1 + 408);
  v5 = a1 + 384;
  if ( !v3 )
  {
    ++*(_QWORD *)(a1 + 384);
    goto LABEL_34;
  }
  v6 = *(_QWORD *)(a1 + 392);
  v7 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v50 = 1;
    v42 = 0;
    while ( v9 != -4 )
    {
      if ( v9 == -8 && !v42 )
        v42 = v8;
      v7 = (v3 - 1) & (v50 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_4;
      ++v50;
    }
    v51 = *(_DWORD *)(a1 + 400);
    if ( !v42 )
      v42 = v8;
    ++*(_QWORD *)(a1 + 384);
    v40 = v51 + 1;
    if ( 4 * (v51 + 1) < 3 * v3 )
    {
      if ( v3 - *(_DWORD *)(a1 + 404) - v40 > v3 >> 3 )
        goto LABEL_36;
      sub_1545BD0(v5, v3);
      v52 = *(_DWORD *)(a1 + 408);
      if ( v52 )
      {
        v53 = v52 - 1;
        v54 = 1;
        v55 = 0;
        v56 = *(_QWORD *)(a1 + 392);
        v40 = *(_DWORD *)(a1 + 400) + 1;
        v57 = (v52 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
        v42 = (__int64 *)(v56 + 16LL * v57);
        v58 = *v42;
        if ( v71 != *v42 )
        {
          while ( v58 != -4 )
          {
            if ( v58 == -8 && !v55 )
              v55 = v42;
            v57 = v53 & (v54 + v57);
            v42 = (__int64 *)(v56 + 16LL * v57);
            v58 = *v42;
            if ( v71 == *v42 )
              goto LABEL_36;
            ++v54;
          }
LABEL_69:
          if ( v55 )
            v42 = v55;
          goto LABEL_36;
        }
        goto LABEL_36;
      }
      goto LABEL_112;
    }
LABEL_34:
    sub_1545BD0(v5, 2 * v3);
    v37 = *(_DWORD *)(a1 + 408);
    if ( v37 )
    {
      v38 = v37 - 1;
      v39 = *(_QWORD *)(a1 + 392);
      v40 = *(_DWORD *)(a1 + 400) + 1;
      v41 = (v37 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
      v42 = (__int64 *)(v39 + 16LL * v41);
      v43 = *v42;
      if ( *v42 != v71 )
      {
        v69 = 1;
        v55 = 0;
        while ( v43 != -4 )
        {
          if ( !v55 && v43 == -8 )
            v55 = v42;
          v41 = v38 & (v69 + v41);
          v42 = (__int64 *)(v39 + 16LL * v41);
          v43 = *v42;
          if ( v71 == *v42 )
            goto LABEL_36;
          ++v69;
        }
        goto LABEL_69;
      }
LABEL_36:
      *(_DWORD *)(a1 + 400) = v40;
      if ( *v42 != -4 )
        --*(_DWORD *)(a1 + 404);
      *((_DWORD *)v42 + 2) = 0;
      *v42 = v71;
      goto LABEL_39;
    }
LABEL_112:
    ++*(_DWORD *)(a1 + 400);
    BUG();
  }
LABEL_4:
  if ( !*((_DWORD *)v8 + 2) )
  {
    v42 = v8;
LABEL_39:
    v44 = *(char **)(a1 + 424);
    if ( v44 == *(char **)(a1 + 432) )
    {
      sub_153F9A0((char **)(a1 + 416), v44, &v71);
      v45 = *(char **)(a1 + 424);
    }
    else
    {
      if ( v44 )
      {
        *(_QWORD *)v44 = v71;
        v44 = *(char **)(a1 + 424);
      }
      v45 = v44 + 8;
      *(_QWORD *)(a1 + 424) = v45;
    }
    *((_DWORD *)v42 + 2) = (__int64)&v45[-*(_QWORD *)(a1 + 416)] >> 3;
  }
  v10 = sub_15601D0(&v71);
  v11 = v10 - 1;
  if ( !v10 )
    return;
  v12 = -1;
  do
  {
    while ( 1 )
    {
      v13 = sub_15601E0(&v71, v12);
      v14 = v13;
      if ( v13 )
        break;
LABEL_19:
      if ( ++v12 == v11 )
        return;
    }
    v15 = *(_DWORD *)(a1 + 352);
    v72.m128i_i32[0] = v12;
    v72.m128i_i64[1] = v13;
    if ( !v15 )
    {
      ++*(_QWORD *)(a1 + 328);
LABEL_24:
      sub_1545DA0(a1 + 328, 2 * v15);
      v26 = *(_DWORD *)(a1 + 352);
      if ( !v26 )
        goto LABEL_111;
      v27 = v72.m128i_i32[0];
      v28 = v26 - 1;
      v29 = 1;
      v31 = 0;
      v32 = (((((unsigned __int32)v72.m128i_i32[2] >> 9) ^ ((unsigned __int32)v72.m128i_i32[2] >> 4)
             | ((unsigned __int64)(unsigned int)(37 * v72.m128i_i32[0]) << 32))
            - 1
            - ((unsigned __int64)(((unsigned __int32)v72.m128i_i32[2] >> 9) ^ ((unsigned __int32)v72.m128i_i32[2] >> 4)) << 32)) >> 22)
          ^ ((((unsigned __int32)v72.m128i_i32[2] >> 9) ^ ((unsigned __int32)v72.m128i_i32[2] >> 4)
            | ((unsigned __int64)(unsigned int)(37 * v72.m128i_i32[0]) << 32))
           - 1
           - ((unsigned __int64)(((unsigned __int32)v72.m128i_i32[2] >> 9) ^ ((unsigned __int32)v72.m128i_i32[2] >> 4)) << 32));
      v33 = ((9 * (((v32 - 1 - (v32 << 13)) >> 8) ^ (v32 - 1 - (v32 << 13)))) >> 15)
          ^ (9 * (((v32 - 1 - (v32 << 13)) >> 8) ^ (v32 - 1 - (v32 << 13))));
      for ( i = v28 & (((v33 - 1 - (v33 << 27)) >> 31) ^ (v33 - 1 - ((_DWORD)v33 << 27))); ; i = v28 & v36 )
      {
        v30 = *(_QWORD *)(a1 + 336);
        v18 = (__int32 *)(v30 + 24LL * i);
        v35 = *v18;
        if ( *v18 == v72.m128i_i32[0] && *((_QWORD *)v18 + 1) == v72.m128i_i64[1] )
          break;
        if ( v35 == -1 )
        {
          if ( *((_QWORD *)v18 + 1) == -4 )
          {
LABEL_45:
            if ( v31 )
              v18 = v31;
            v46 = *(_DWORD *)(a1 + 344) + 1;
            goto LABEL_48;
          }
        }
        else if ( v35 == -2 && *((_QWORD *)v18 + 1) == -8 && !v31 )
        {
          v31 = (int *)(v30 + 24LL * i);
        }
        v36 = v29 + i;
        ++v29;
      }
LABEL_73:
      v46 = *(_DWORD *)(a1 + 344) + 1;
      goto LABEL_48;
    }
    v16 = *(_QWORD *)(a1 + 336);
    v17 = 1;
    v18 = 0;
    v19 = (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4) | ((unsigned __int64)(37 * v12) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)) << 32);
    v20 = ((v19 >> 22) ^ v19) - 1 - (((v19 >> 22) ^ v19) << 13);
    v21 = ((9 * ((v20 >> 8) ^ v20)) >> 15) ^ (9 * ((v20 >> 8) ^ v20));
    for ( j = (v15 - 1) & (((v21 - 1 - (v21 << 27)) >> 31) ^ (v21 - 1 - ((_DWORD)v21 << 27))); ; j = (v15 - 1) & v25 )
    {
      v23 = (int *)(v16 + 24LL * j);
      v24 = *v23;
      if ( v12 == *v23 && v14 == *((_QWORD *)v23 + 1) )
      {
        if ( v23[4] )
          goto LABEL_19;
        v18 = (__int32 *)(v16 + 24LL * j);
        goto LABEL_51;
      }
      if ( v24 == -1 )
        break;
      if ( v24 == -2 && *((_QWORD *)v23 + 1) == -8 && !v18 )
        v18 = (__int32 *)(v16 + 24LL * j);
LABEL_22:
      v25 = v17 + j;
      ++v17;
    }
    if ( *((_QWORD *)v23 + 1) != -4 )
      goto LABEL_22;
    v59 = *(_DWORD *)(a1 + 344);
    if ( !v18 )
      v18 = v23;
    ++*(_QWORD *)(a1 + 328);
    v46 = v59 + 1;
    if ( 4 * (v59 + 1) >= 3 * v15 )
      goto LABEL_24;
    v27 = v12;
    if ( v15 - *(_DWORD *)(a1 + 348) - v46 <= v15 >> 3 )
    {
      sub_1545DA0(a1 + 328, v15);
      v60 = *(_DWORD *)(a1 + 352);
      if ( v60 )
      {
        v27 = v72.m128i_i32[0];
        v61 = v60 - 1;
        v62 = 1;
        v31 = 0;
        v64 = (((((unsigned __int32)v72.m128i_i32[2] >> 9) ^ ((unsigned __int32)v72.m128i_i32[2] >> 4)
               | ((unsigned __int64)(unsigned int)(37 * v72.m128i_i32[0]) << 32))
              - 1
              - ((unsigned __int64)(((unsigned __int32)v72.m128i_i32[2] >> 9) ^ ((unsigned __int32)v72.m128i_i32[2] >> 4)) << 32)) >> 22)
            ^ ((((unsigned __int32)v72.m128i_i32[2] >> 9) ^ ((unsigned __int32)v72.m128i_i32[2] >> 4)
              | ((unsigned __int64)(unsigned int)(37 * v72.m128i_i32[0]) << 32))
             - 1
             - ((unsigned __int64)(((unsigned __int32)v72.m128i_i32[2] >> 9) ^ ((unsigned __int32)v72.m128i_i32[2] >> 4)) << 32));
        v65 = ((9 * (((v64 - 1 - (v64 << 13)) >> 8) ^ (v64 - 1 - (v64 << 13)))) >> 15)
            ^ (9 * (((v64 - 1 - (v64 << 13)) >> 8) ^ (v64 - 1 - (v64 << 13))));
        for ( k = v61 & (((v65 - 1 - (v65 << 27)) >> 31) ^ (v65 - 1 - ((_DWORD)v65 << 27))); ; k = v61 & v68 )
        {
          v63 = *(_QWORD *)(a1 + 336);
          v18 = (__int32 *)(v63 + 24LL * k);
          v67 = *v18;
          if ( *v18 == v72.m128i_i32[0] && *((_QWORD *)v18 + 1) == v72.m128i_i64[1] )
            break;
          if ( v67 == -1 )
          {
            if ( *((_QWORD *)v18 + 1) == -4 )
              goto LABEL_45;
          }
          else if ( v67 == -2 && *((_QWORD *)v18 + 1) == -8 && !v31 )
          {
            v31 = (int *)(v63 + 24LL * k);
          }
          v68 = v62 + k;
          ++v62;
        }
        goto LABEL_73;
      }
LABEL_111:
      ++*(_DWORD *)(a1 + 344);
      BUG();
    }
LABEL_48:
    *(_DWORD *)(a1 + 344) = v46;
    if ( *v18 != -1 || *((_QWORD *)v18 + 1) != -4 )
      --*(_DWORD *)(a1 + 348);
    *v18 = v27;
    v47 = v72.m128i_i64[1];
    v18[4] = 0;
    *((_QWORD *)v18 + 1) = v47;
LABEL_51:
    v48 = *(__m128i **)(a1 + 368);
    if ( v48 == *(__m128i **)(a1 + 376) )
    {
      v70 = v18;
      sub_153FB20((const __m128i **)(a1 + 360), v48, &v72);
      v49 = *(__m128i **)(a1 + 368);
      v18 = v70;
    }
    else
    {
      if ( v48 )
      {
        *v48 = _mm_loadu_si128(&v72);
        v48 = *(__m128i **)(a1 + 368);
      }
      v49 = v48 + 1;
      *(_QWORD *)(a1 + 368) = v49;
    }
    ++v12;
    v18[4] = ((__int64)v49->m128i_i64 - *(_QWORD *)(a1 + 360)) >> 4;
  }
  while ( v12 != v11 );
}
