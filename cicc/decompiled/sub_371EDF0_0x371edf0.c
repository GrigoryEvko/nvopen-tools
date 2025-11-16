// Function: sub_371EDF0
// Address: 0x371edf0
//
_QWORD *__fastcall sub_371EDF0(
        __int64 a1,
        __int64 a2,
        char a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 v11; // rdi
  unsigned int v12; // esi
  __int64 v13; // rcx
  int v14; // r11d
  __int64 *v15; // r15
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // r10
  _QWORD *result; // rax
  int v20; // eax
  int v21; // edx
  __m128i *v22; // r12
  _QWORD *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rsi
  unsigned int v28; // eax
  __int64 v29; // rdx
  unsigned int v30; // esi
  __int64 v31; // rcx
  unsigned int v32; // edx
  __int64 *v33; // rax
  __int64 v34; // r8
  int v35; // eax
  int v36; // ecx
  __int64 v37; // rdi
  unsigned int v38; // eax
  __int64 v39; // rsi
  int v40; // r9d
  __int64 *v41; // r8
  int v42; // eax
  int v43; // eax
  __int64 v44; // rsi
  int v45; // r8d
  unsigned int v46; // r14d
  __int64 *v47; // rdi
  __int64 v48; // rcx
  int v49; // eax
  int v50; // r9d
  __m128i v51; // [rsp+0h] [rbp-50h] BYREF
  __int64 v52; // [rsp+10h] [rbp-40h]

  v11 = a1 + 56;
  v12 = *(_DWORD *)(a1 + 80);
  if ( v12 )
  {
    v13 = *(_QWORD *)(a1 + 64);
    v14 = 1;
    v15 = 0;
    v16 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v17 = (__int64 *)(v13 + 16LL * v16);
    v18 = *v17;
    if ( a2 == *v17 )
      return (_QWORD *)v17[1];
    while ( v18 != -4096 )
    {
      if ( v18 == -8192 && !v15 )
        v15 = v17;
      v16 = (v12 - 1) & (v14 + v16);
      v17 = (__int64 *)(v13 + 16LL * v16);
      v18 = *v17;
      if ( a2 == *v17 )
        return (_QWORD *)v17[1];
      ++v14;
    }
    if ( !v15 )
      v15 = v17;
    v20 = *(_DWORD *)(a1 + 72);
    ++*(_QWORD *)(a1 + 56);
    v21 = v20 + 1;
    if ( 4 * (v20 + 1) < 3 * v12 )
    {
      if ( v12 - *(_DWORD *)(a1 + 76) - v21 > v12 >> 3 )
        goto LABEL_14;
      sub_371EC10(v11, v12);
      v42 = *(_DWORD *)(a1 + 80);
      if ( v42 )
      {
        v43 = v42 - 1;
        v44 = *(_QWORD *)(a1 + 64);
        v45 = 1;
        v46 = v43 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v21 = *(_DWORD *)(a1 + 72) + 1;
        v47 = 0;
        v15 = (__int64 *)(v44 + 16LL * v46);
        v48 = *v15;
        if ( a2 != *v15 )
        {
          while ( v48 != -4096 )
          {
            if ( !v47 && v48 == -8192 )
              v47 = v15;
            v46 = v43 & (v45 + v46);
            v15 = (__int64 *)(v44 + 16LL * v46);
            v48 = *v15;
            if ( a2 == *v15 )
              goto LABEL_14;
            ++v45;
          }
          if ( v47 )
            v15 = v47;
        }
        goto LABEL_14;
      }
LABEL_56:
      ++*(_DWORD *)(a1 + 72);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 56);
  }
  sub_371EC10(v11, 2 * v12);
  v35 = *(_DWORD *)(a1 + 80);
  if ( !v35 )
    goto LABEL_56;
  v36 = v35 - 1;
  v37 = *(_QWORD *)(a1 + 64);
  v38 = (v35 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v21 = *(_DWORD *)(a1 + 72) + 1;
  v15 = (__int64 *)(v37 + 16LL * v38);
  v39 = *v15;
  if ( a2 != *v15 )
  {
    v40 = 1;
    v41 = 0;
    while ( v39 != -4096 )
    {
      if ( v39 == -8192 && !v41 )
        v41 = v15;
      v38 = v36 & (v40 + v38);
      v15 = (__int64 *)(v37 + 16LL * v38);
      v39 = *v15;
      if ( a2 == *v15 )
        goto LABEL_14;
      ++v40;
    }
    if ( v41 )
      v15 = v41;
  }
LABEL_14:
  *(_DWORD *)(a1 + 72) = v21;
  if ( *v15 != -4096 )
    --*(_DWORD *)(a1 + 76);
  *v15 = a2;
  v15[1] = 0;
  if ( !a3 )
  {
    v25 = *(_QWORD *)(a2 + 40);
    v26 = *(_QWORD *)(a1 + 8);
    v51.m128i_i64[0] = a2;
    if ( v25 )
    {
      v27 = (unsigned int)(*(_DWORD *)(v25 + 44) + 1);
      v28 = *(_DWORD *)(v25 + 44) + 1;
    }
    else
    {
      v27 = 0;
      v28 = 0;
    }
    v29 = 0;
    if ( v28 < *(_DWORD *)(v26 + 32) )
      v29 = *(_QWORD *)(*(_QWORD *)(v26 + 24) + 8 * v27);
    v30 = *(_DWORD *)(a1 + 112);
    v31 = *(_QWORD *)(a1 + 96);
    v51.m128i_i64[1] = v29;
    if ( v30 )
    {
      v32 = (v30 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v33 = (__int64 *)(v31 + 16LL * v32);
      v34 = *v33;
      if ( a2 == *v33 )
      {
LABEL_25:
        LODWORD(v52) = *((_DWORD *)v33 + 2);
        goto LABEL_18;
      }
      v49 = 1;
      while ( v34 != -4096 )
      {
        v50 = v49 + 1;
        v32 = (v30 - 1) & (v49 + v32);
        v33 = (__int64 *)(v31 + 16LL * v32);
        v34 = *v33;
        if ( a2 == *v33 )
          goto LABEL_25;
        v49 = v50;
      }
    }
    v33 = (__int64 *)(v31 + 16LL * v30);
    goto LABEL_25;
  }
  v52 = a8;
  v51 = _mm_loadu_si128((const __m128i *)&a7);
LABEL_18:
  v22 = (__m128i *)sub_22077B0(0x18u);
  *v22 = _mm_loadu_si128(&v51);
  v22[1].m128i_i64[0] = v52;
  v23 = (_QWORD *)sub_22077B0(0x20u);
  v24 = *(_QWORD *)(a1 + 48);
  v23[1] = v22;
  v23[2] = (char *)v22 + 24;
  v23[3] = (char *)v22 + 24;
  *v23 = v24;
  *(_QWORD *)(a1 + 48) = v23;
  result = v23 + 1;
  v15[1] = (__int64)result;
  return result;
}
