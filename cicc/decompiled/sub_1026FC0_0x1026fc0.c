// Function: sub_1026FC0
// Address: 0x1026fc0
//
__int64 __fastcall sub_1026FC0(__int64 a1, __int64 a2, char a3, __m128i *a4)
{
  __int64 v4; // r14
  _QWORD *v7; // rbx
  unsigned __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *v10; // rdx
  unsigned int v11; // edi
  _QWORD *v12; // rax
  __int64 v13; // rcx
  __m128i *v14; // rax
  __int64 v15; // rsi
  __m128i v16; // xmm0
  __m128i v17; // xmm1
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 (__fastcall *v20)(__m128i *, __m128i *, __int64, __int64, unsigned __int64, __int64); // rsi
  __int64 v21; // rcx
  __int64 result; // rax
  int v23; // ecx
  unsigned int v24; // eax
  __int64 v25; // r8
  _QWORD *v26; // r15
  void (__fastcall *v27)(_QWORD *, _QWORD *, __int64); // rax
  int v28; // esi
  int v29; // eax
  int v30; // eax
  unsigned int v31; // esi
  int v32; // ecx
  __int64 v33; // rdi
  int v34; // r10d
  void (__fastcall *v35)(_QWORD *, _QWORD *, __int64); // rax
  int v36; // eax
  int v37; // edx
  _QWORD *v38; // rdi
  __int64 v39; // rbx
  unsigned int v40; // ecx
  unsigned int v41; // eax
  unsigned __int64 v42; // rdx
  unsigned __int64 v43; // rax
  _QWORD *v44; // rax
  _QWORD *i; // rdx
  int v46; // eax
  int v47; // eax
  __int64 v48; // rdi
  unsigned int v49; // r15d
  __int64 v50; // rsi
  _QWORD *v51; // rax
  __int64 v52; // [rsp+0h] [rbp-60h]
  int v53; // [rsp+Ch] [rbp-54h]
  __m128i v54; // [rsp+10h] [rbp-50h] BYREF
  __int64 (__fastcall *v55)(__m128i *, __m128i *, __int64, __int64, unsigned __int64, __int64); // [rsp+20h] [rbp-40h]
  __int64 v56; // [rsp+28h] [rbp-38h]

  v4 = a2;
  LODWORD(a2) = *(_DWORD *)(a1 + 24);
  v7 = *(_QWORD **)(a1 + 8);
  if ( !a3 )
    goto LABEL_2;
  v23 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v23 && !*(_DWORD *)(a1 + 20) )
    goto LABEL_2;
  v24 = 4 * v23;
  v25 = 40LL * (unsigned int)a2;
  if ( (unsigned int)(4 * v23) < 0x40 )
    v24 = 64;
  v26 = (_QWORD *)((char *)v7 + v25);
  if ( v24 < (unsigned int)a2 )
  {
    do
    {
      if ( *v7 != -8192 && *v7 != -4096 )
      {
        v35 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v7[3];
        if ( v35 )
        {
          v52 = v25;
          v53 = v23;
          v35(v7 + 1, v7 + 1, 3);
          v25 = v52;
          v23 = v53;
        }
      }
      v7 += 5;
    }
    while ( v7 != v26 );
    v37 = *(_DWORD *)(a1 + 24);
    v38 = *(_QWORD **)(a1 + 8);
    if ( !v23 )
    {
      if ( v37 )
      {
        sub_C7D6A0((__int64)v38, v25, 8);
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
      }
      else
      {
        *(_QWORD *)(a1 + 16) = 0;
      }
      goto LABEL_21;
    }
    v39 = 64;
    v40 = v23 - 1;
    if ( v40 )
    {
      _BitScanReverse(&v41, v40);
      v39 = (unsigned int)(1 << (33 - (v41 ^ 0x1F)));
      if ( (int)v39 < 64 )
        v39 = 64;
    }
    if ( (_DWORD)v39 == v37 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      v51 = &v38[5 * v39];
      do
      {
        if ( v38 )
          *v38 = -4096;
        v38 += 5;
      }
      while ( v38 != v51 );
      v7 = *(_QWORD **)(a1 + 8);
      LODWORD(a2) = *(_DWORD *)(a1 + 24);
    }
    else
    {
      sub_C7D6A0((__int64)v38, v25, 8);
      v42 = ((((((((4 * (int)v39 / 3u + 1) | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v39 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v39 / 3u + 1) | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v39 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 8)
           | (((((4 * (int)v39 / 3u + 1) | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v39 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v39 / 3u + 1) | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v39 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 16;
      v43 = (v42
           | (((((((4 * (int)v39 / 3u + 1) | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v39 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v39 / 3u + 1) | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v39 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 8)
           | (((((4 * (int)v39 / 3u + 1) | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v39 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v39 / 3u + 1) | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v39 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 24) = v43;
      v44 = (_QWORD *)sub_C7D670(40 * v43, 8);
      *(_QWORD *)(a1 + 16) = 0;
      v7 = v44;
      *(_QWORD *)(a1 + 8) = v44;
      a2 = *(unsigned int *)(a1 + 24);
      for ( i = &v44[5 * a2]; i != v44; v44 += 5 )
      {
        if ( v44 )
          *v44 = -4096;
      }
    }
LABEL_2:
    if ( (_DWORD)a2 )
      goto LABEL_3;
LABEL_21:
    ++*(_QWORD *)a1;
    v28 = 0;
    goto LABEL_22;
  }
  if ( v26 != v7 )
  {
    do
    {
      if ( *v7 != -4096 )
      {
        if ( *v7 != -8192 )
        {
          v27 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v7[3];
          if ( v27 )
            v27(v7 + 1, v7 + 1, 3);
        }
        *v7 = -4096;
      }
      v7 += 5;
    }
    while ( v7 != v26 );
    v7 = *(_QWORD **)(a1 + 8);
    LODWORD(a2) = *(_DWORD *)(a1 + 24);
  }
  *(_QWORD *)(a1 + 16) = 0;
  if ( !(_DWORD)a2 )
    goto LABEL_21;
LABEL_3:
  v8 = (unsigned int)(a2 - 1);
  v9 = 1;
  v10 = 0;
  v11 = v8 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v12 = &v7[5 * v11];
  v13 = *v12;
  if ( *v12 == v4 )
  {
LABEL_4:
    v14 = (__m128i *)(v12 + 1);
    goto LABEL_5;
  }
  while ( v13 != -4096 )
  {
    if ( !v10 && v13 == -8192 )
      v10 = v12;
    v11 = v8 & (v9 + v11);
    v12 = &v7[5 * v11];
    v13 = *v12;
    if ( *v12 == v4 )
      goto LABEL_4;
    v9 = (unsigned int)(v9 + 1);
  }
  if ( !v10 )
    v10 = v12;
  v36 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v32 = v36 + 1;
  v8 = (unsigned int)(4 * (v36 + 1));
  if ( (unsigned int)v8 >= 3 * (int)a2 )
  {
    v28 = 2 * a2;
LABEL_22:
    sub_1026D60(a1, v28);
    v29 = *(_DWORD *)(a1 + 24);
    if ( v29 )
    {
      v30 = v29 - 1;
      v8 = *(_QWORD *)(a1 + 8);
      v31 = v30 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v32 = *(_DWORD *)(a1 + 16) + 1;
      v10 = (_QWORD *)(v8 + 40LL * v31);
      v33 = *v10;
      if ( *v10 != v4 )
      {
        v34 = 1;
        v9 = 0;
        while ( v33 != -4096 )
        {
          if ( v33 == -8192 && !v9 )
            v9 = (__int64)v10;
          v31 = v30 & (v34 + v31);
          v10 = (_QWORD *)(v8 + 40LL * v31);
          v33 = *v10;
          if ( *v10 == v4 )
            goto LABEL_47;
          ++v34;
        }
        if ( v9 )
          v10 = (_QWORD *)v9;
      }
      goto LABEL_47;
    }
    goto LABEL_85;
  }
  if ( (int)a2 - (v32 + *(_DWORD *)(a1 + 20)) <= (unsigned int)a2 >> 3 )
  {
    sub_1026D60(a1, a2);
    v46 = *(_DWORD *)(a1 + 24);
    if ( v46 )
    {
      v47 = v46 - 1;
      v48 = *(_QWORD *)(a1 + 8);
      v8 = 0;
      v49 = v47 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v9 = 1;
      v32 = *(_DWORD *)(a1 + 16) + 1;
      v10 = (_QWORD *)(v48 + 40LL * v49);
      v50 = *v10;
      if ( *v10 != v4 )
      {
        while ( v50 != -4096 )
        {
          if ( !v8 && v50 == -8192 )
            v8 = (unsigned __int64)v10;
          v49 = v47 & (v9 + v49);
          v10 = (_QWORD *)(v48 + 40LL * v49);
          v50 = *v10;
          if ( *v10 == v4 )
            goto LABEL_47;
          v9 = (unsigned int)(v9 + 1);
        }
        if ( v8 )
          v10 = (_QWORD *)v8;
      }
      goto LABEL_47;
    }
LABEL_85:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_47:
  *(_DWORD *)(a1 + 16) = v32;
  if ( *v10 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v10 = v4;
  v14 = (__m128i *)(v10 + 1);
  v10[3] = 0;
LABEL_5:
  v15 = v56;
  v16 = _mm_loadu_si128(a4);
  v17 = _mm_loadu_si128(&v54);
  v18 = a4[1].m128i_i64[0];
  a4[1].m128i_i64[0] = 0;
  v19 = a4[1].m128i_i64[1];
  v54 = v16;
  a4[1].m128i_i64[1] = v15;
  *a4 = v17;
  v20 = (__int64 (__fastcall *)(__m128i *, __m128i *, __int64, __int64, unsigned __int64, __int64))v14[1].m128i_i64[0];
  v54 = _mm_loadu_si128(v14);
  *v14 = v16;
  v55 = v20;
  v14[1].m128i_i64[0] = v18;
  v21 = v14[1].m128i_i64[1];
  v56 = v21;
  v14[1].m128i_i64[1] = v19;
  result = (__int64)v55;
  if ( v55 )
    return v55(&v54, &v54, 3, v21, v8, v9);
  return result;
}
