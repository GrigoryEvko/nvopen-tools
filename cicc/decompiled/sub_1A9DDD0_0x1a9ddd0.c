// Function: sub_1A9DDD0
// Address: 0x1a9ddd0
//
__int64 __fastcall sub_1A9DDD0(__int64 a1, __int64 a2, __int64 a3)
{
  char *v4; // r14
  char *v5; // r12
  __int64 v6; // r9
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rax
  unsigned int v11; // esi
  __int64 *v12; // r15
  int v13; // ecx
  const __m128i *v14; // r13
  const __m128i *v15; // r12
  __int64 v16; // r8
  _QWORD *v17; // rcx
  unsigned int v18; // edx
  __int64 v19; // rdi
  __int64 v20; // rax
  int v21; // edx
  __int64 *v22; // rsi
  __int64 v23; // rdi
  int v24; // r11d
  __int64 *v25; // r10
  int v26; // edx
  _QWORD *v27; // rdi
  __int64 v28; // rax
  int v30; // r11d
  int v31; // edi
  __m128i *v32; // rsi
  __m128i *v33; // rsi
  int v34; // r10d
  unsigned __int64 v35; // rdx
  _QWORD *v36; // rax
  __int64 v37; // rax
  int v38; // r8d
  __int64 v39; // rsi
  unsigned int v40; // r9d
  __int64 *v41; // rdx
  __int64 v42; // r11
  char *v43; // rax
  char *v44; // rdx
  unsigned int v45; // r10d
  __int64 *v46; // r9
  __int64 v47; // r12
  char *v48; // rsi
  __int64 v49; // r8
  __int64 v50; // rsi
  char *v51; // rax
  __int64 v52; // r8
  char *v53; // rax
  unsigned __int64 v54; // r9
  _QWORD *v55; // rdx
  _QWORD *v56; // rsi
  int i; // edx
  int v58; // r12d
  int v59; // r9d
  int v60; // r13d
  void *v61; // rax
  __int64 v62; // rdx
  void *v63; // rsi
  int v64; // esi
  __int64 *v65; // [rsp+18h] [rbp-A8h] BYREF
  __m128i v66; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v67; // [rsp+30h] [rbp-90h] BYREF
  _QWORD *v68; // [rsp+38h] [rbp-88h]
  __int64 v69; // [rsp+40h] [rbp-80h]
  __int64 v70; // [rsp+48h] [rbp-78h]
  __int64 v71; // [rsp+50h] [rbp-70h] BYREF
  void *src; // [rsp+58h] [rbp-68h]
  __int64 v73; // [rsp+60h] [rbp-60h]
  __int64 v74; // [rsp+68h] [rbp-58h]
  char *v75; // [rsp+70h] [rbp-50h] BYREF
  char *v76; // [rsp+78h] [rbp-48h]
  __int64 v77; // [rsp+80h] [rbp-40h]

  v71 = 0;
  src = 0;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  sub_1A9DA40(a2 & 0xFFFFFFFFFFFFFFF8LL, a1, (__int64)&v71);
  v4 = v76;
  v5 = v75;
  if ( v75 != v76 )
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)v5;
      v11 = *(_DWORD *)(a3 + 80);
      LODWORD(v68) = 0;
      v66.m128i_i64[0] = v10;
      v66.m128i_i64[1] = v10;
      v67 = v10;
      if ( !v11 )
        break;
      v6 = *(_QWORD *)(a3 + 64);
      v7 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( v10 != *v8 )
      {
        v30 = 1;
        v12 = 0;
        while ( v9 != -8 )
        {
          if ( v9 != -16 || v12 )
            v8 = v12;
          v7 = (v11 - 1) & (v30 + v7);
          v9 = *(_QWORD *)(v6 + 16LL * v7);
          if ( v10 == v9 )
            goto LABEL_4;
          ++v30;
          v12 = v8;
          v8 = (__int64 *)(v6 + 16LL * v7);
        }
        v31 = *(_DWORD *)(a3 + 72);
        if ( !v12 )
          v12 = v8;
        ++*(_QWORD *)(a3 + 56);
        v13 = v31 + 1;
        if ( 4 * (v31 + 1) < 3 * v11 )
        {
          if ( v11 - *(_DWORD *)(a3 + 76) - v13 > v11 >> 3 )
            goto LABEL_37;
          goto LABEL_8;
        }
LABEL_7:
        v11 *= 2;
LABEL_8:
        sub_177C7D0(a3 + 56, v11);
        sub_190E590(a3 + 56, &v67, &v65);
        v12 = v65;
        v10 = v67;
        v13 = *(_DWORD *)(a3 + 72) + 1;
LABEL_37:
        *(_DWORD *)(a3 + 72) = v13;
        if ( *v12 != -8 )
          --*(_DWORD *)(a3 + 76);
        *v12 = v10;
        *((_DWORD *)v12 + 2) = (_DWORD)v68;
        v32 = *(__m128i **)(a3 + 96);
        if ( v32 == *(__m128i **)(a3 + 104) )
        {
          sub_177C650((const __m128i **)(a3 + 88), v32, &v66);
          v33 = *(__m128i **)(a3 + 96);
        }
        else
        {
          if ( v32 )
          {
            *v32 = _mm_loadu_si128(&v66);
            v32 = *(__m128i **)(a3 + 96);
          }
          v33 = v32 + 1;
          *(_QWORD *)(a3 + 96) = v33;
        }
        *((_DWORD *)v12 + 2) = (((__int64)v33->m128i_i64 - *(_QWORD *)(a3 + 88)) >> 4) - 1;
      }
LABEL_4:
      v5 += 8;
      if ( v4 == v5 )
        goto LABEL_9;
    }
    ++*(_QWORD *)(a3 + 56);
    goto LABEL_7;
  }
LABEL_9:
  v14 = *(const __m128i **)(a3 + 96);
  v15 = *(const __m128i **)(a3 + 88);
  v16 = 0;
  v17 = 0;
  v69 = 0;
  v67 = 0;
  v68 = 0;
  v70 = 0;
  if ( v14 == v15 )
    goto LABEL_26;
  do
  {
    while ( 1 )
    {
      v20 = v15->m128i_i64[0];
      v66 = _mm_loadu_si128(v15);
      if ( (_DWORD)v74 )
      {
        v18 = (v74 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v19 = *((_QWORD *)src + v18);
        if ( v20 == v19 )
          goto LABEL_12;
        v34 = 1;
        while ( v19 != -8 )
        {
          v18 = (v74 - 1) & (v34 + v18);
          v19 = *((_QWORD *)src + v18);
          if ( v20 == v19 )
            goto LABEL_12;
          ++v34;
        }
      }
      if ( !(_DWORD)v16 )
      {
        ++v67;
LABEL_99:
        v64 = 2 * v16;
        goto LABEL_97;
      }
      v21 = (v16 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v22 = &v17[v21];
      v23 = *v22;
      if ( v20 != *v22 )
        break;
LABEL_12:
      if ( v14 == ++v15 )
        goto LABEL_25;
    }
    v24 = 1;
    v25 = 0;
    while ( v23 != -8 )
    {
      if ( !v25 && v23 == -16 )
        v25 = v22;
      v21 = (v16 - 1) & (v24 + v21);
      v22 = &v17[v21];
      v23 = *v22;
      if ( v20 == *v22 )
        goto LABEL_12;
      ++v24;
    }
    if ( v25 )
      v22 = v25;
    ++v67;
    v26 = v69 + 1;
    if ( 4 * ((int)v69 + 1) >= (unsigned int)(3 * v16) )
      goto LABEL_99;
    if ( (int)v16 - (v26 + HIDWORD(v69)) > (unsigned int)v16 >> 3 )
      goto LABEL_22;
    v64 = v16;
LABEL_97:
    sub_1353F00((__int64)&v67, v64);
    sub_1A97120((__int64)&v67, v66.m128i_i64, &v65);
    v22 = v65;
    v20 = v66.m128i_i64[0];
    v26 = v69 + 1;
LABEL_22:
    LODWORD(v69) = v26;
    if ( *v22 != -8 )
      --HIDWORD(v69);
    ++v15;
    *v22 = v20;
    v17 = v68;
    v16 = (unsigned int)v70;
  }
  while ( v14 != v15 );
LABEL_25:
  v27 = &v17[v16];
  if ( (_DWORD)v69 && v27 != v17 )
  {
    while ( *v17 == -16 || *v17 == -8 )
    {
      if ( ++v17 == v27 )
        goto LABEL_26;
    }
LABEL_63:
    if ( v17 != v27 )
    {
      v37 = *(unsigned int *)(a3 + 80);
      if ( !(_DWORD)v37 )
        goto LABEL_60;
      v38 = v37 - 1;
      v39 = *(_QWORD *)(a3 + 64);
      v40 = (v37 - 1) & (((unsigned int)*v17 >> 9) ^ ((unsigned int)*v17 >> 4));
      v41 = (__int64 *)(v39 + 16LL * v40);
      v42 = *v41;
      if ( *v17 != *v41 )
      {
        for ( i = 1; ; i = v58 )
        {
          if ( v42 == -8 )
            goto LABEL_60;
          v58 = i + 1;
          v40 = v38 & (i + v40);
          v41 = (__int64 *)(v39 + 16LL * v40);
          v42 = *v41;
          if ( *v17 == *v41 )
            break;
        }
      }
      if ( v41 == (__int64 *)(v39 + 16 * v37) )
        goto LABEL_60;
      v43 = *(char **)(a3 + 96);
      v44 = (char *)(*(_QWORD *)(a3 + 88) + 16LL * *((unsigned int *)v41 + 2));
      if ( v43 == v44 )
        goto LABEL_60;
      v45 = v38 & (((unsigned int)*(_QWORD *)v44 >> 9) ^ ((unsigned int)*(_QWORD *)v44 >> 4));
      v46 = (__int64 *)(v39 + 16LL * v45);
      v47 = *v46;
      if ( *(_QWORD *)v44 == *v46 )
      {
LABEL_69:
        *v46 = -16;
        v43 = *(char **)(a3 + 96);
        --*(_DWORD *)(a3 + 72);
        ++*(_DWORD *)(a3 + 76);
      }
      else
      {
        v59 = 1;
        while ( v47 != -8 )
        {
          v60 = v59 + 1;
          v45 = v38 & (v59 + v45);
          v46 = (__int64 *)(v39 + 16LL * v45);
          v47 = *v46;
          if ( *(_QWORD *)v44 == *v46 )
            goto LABEL_69;
          v59 = v60;
        }
      }
      v48 = v44 + 16;
      if ( v44 + 16 != v43 )
      {
        v49 = v43 - v48;
        v50 = (v43 - v48) >> 4;
        if ( v49 > 0 )
        {
          v51 = v44;
          do
          {
            v52 = *((_QWORD *)v51 + 2);
            v51 += 16;
            *((_QWORD *)v51 - 2) = v52;
            *((_QWORD *)v51 - 1) = *((_QWORD *)v51 + 1);
            --v50;
          }
          while ( v50 );
          v43 = *(char **)(a3 + 96);
        }
      }
      v53 = v43 - 16;
      *(_QWORD *)(a3 + 96) = v53;
      if ( v44 != v53 )
      {
        v54 = (__int64)&v44[-*(_QWORD *)(a3 + 88)] >> 4;
        if ( *(_DWORD *)(a3 + 72) )
        {
          v55 = *(_QWORD **)(a3 + 64);
          v56 = &v55[2 * *(unsigned int *)(a3 + 80)];
          if ( v55 != v56 )
          {
            while ( 1 )
            {
              v36 = v55;
              if ( *v55 != -16 && *v55 != -8 )
                break;
              v55 += 2;
              if ( v56 == v55 )
                goto LABEL_60;
            }
            while ( v36 != v56 )
            {
              v35 = *((unsigned int *)v36 + 2);
              if ( v54 < v35 )
                *((_DWORD *)v36 + 2) = v35 - 1;
              v36 += 2;
              if ( v36 == v56 )
                break;
              while ( *v36 == -16 || *v36 == -8 )
              {
                v36 += 2;
                if ( v56 == v36 )
                  goto LABEL_60;
              }
            }
          }
        }
      }
LABEL_60:
      while ( ++v17 != v27 )
      {
        if ( *v17 != -8 && *v17 != -16 )
          goto LABEL_63;
      }
    }
  }
LABEL_26:
  j___libc_free_0(*(_QWORD *)(a3 + 8));
  v28 = (unsigned int)v74;
  *(_DWORD *)(a3 + 24) = v74;
  if ( (_DWORD)v28 )
  {
    v61 = (void *)sub_22077B0(8 * v28);
    v62 = *(unsigned int *)(a3 + 24);
    v63 = src;
    *(_QWORD *)(a3 + 8) = v61;
    *(_QWORD *)(a3 + 16) = v73;
    memcpy(v61, v63, 8 * v62);
  }
  else
  {
    *(_QWORD *)(a3 + 8) = 0;
    *(_QWORD *)(a3 + 16) = 0;
  }
  sub_1A953D0(a3 + 32, &v75);
  j___libc_free_0(v68);
  if ( v75 )
    j_j___libc_free_0(v75, v77 - (_QWORD)v75);
  return j___libc_free_0(src);
}
