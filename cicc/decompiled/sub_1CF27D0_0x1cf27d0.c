// Function: sub_1CF27D0
// Address: 0x1cf27d0
//
_QWORD *__fastcall sub_1CF27D0(__int64 a1, __int64 a2, const __m128i *a3)
{
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 v8; // rcx
  int v9; // r11d
  __int64 *v10; // r15
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r10
  _QWORD *result; // rax
  int v15; // eax
  int v16; // edx
  __m128i *v17; // r12
  _QWORD *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rsi
  unsigned int v25; // ecx
  __int64 *v26; // rdx
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rsi
  unsigned int v30; // ecx
  __int64 *v31; // rax
  __int64 v32; // r8
  int v33; // eax
  int v34; // ecx
  __int64 v35; // rdi
  unsigned int v36; // eax
  __int64 v37; // rsi
  int v38; // r9d
  __int64 *v39; // r8
  int v40; // eax
  int v41; // eax
  __int64 v42; // rsi
  int v43; // r8d
  unsigned int v44; // r14d
  __int64 *v45; // rdi
  __int64 v46; // rcx
  int v47; // edx
  int v48; // r10d
  int v49; // eax
  int v50; // r9d
  __m128i v51; // [rsp+0h] [rbp-50h] BYREF
  __int64 v52; // [rsp+10h] [rbp-40h]

  v6 = a1 + 56;
  v7 = *(_DWORD *)(a1 + 80);
  if ( v7 )
  {
    v8 = *(_QWORD *)(a1 + 64);
    v9 = 1;
    v10 = 0;
    v11 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v12 = (__int64 *)(v8 + 16LL * v11);
    v13 = *v12;
    if ( a2 == *v12 )
      return (_QWORD *)v12[1];
    while ( v13 != -8 )
    {
      if ( !v10 && v13 == -16 )
        v10 = v12;
      v11 = (v7 - 1) & (v9 + v11);
      v12 = (__int64 *)(v8 + 16LL * v11);
      v13 = *v12;
      if ( a2 == *v12 )
        return (_QWORD *)v12[1];
      ++v9;
    }
    if ( !v10 )
      v10 = v12;
    v15 = *(_DWORD *)(a1 + 72);
    ++*(_QWORD *)(a1 + 56);
    v16 = v15 + 1;
    if ( 4 * (v15 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 76) - v16 > v7 >> 3 )
        goto LABEL_14;
      sub_1CF2610(v6, v7);
      v40 = *(_DWORD *)(a1 + 80);
      if ( v40 )
      {
        v41 = v40 - 1;
        v42 = *(_QWORD *)(a1 + 64);
        v43 = 1;
        v44 = v41 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v16 = *(_DWORD *)(a1 + 72) + 1;
        v45 = 0;
        v10 = (__int64 *)(v42 + 16LL * v44);
        v46 = *v10;
        if ( a2 != *v10 )
        {
          while ( v46 != -8 )
          {
            if ( v46 == -16 && !v45 )
              v45 = v10;
            v44 = v41 & (v43 + v44);
            v10 = (__int64 *)(v42 + 16LL * v44);
            v46 = *v10;
            if ( a2 == *v10 )
              goto LABEL_14;
            ++v43;
          }
          if ( v45 )
            v10 = v45;
        }
        goto LABEL_14;
      }
LABEL_60:
      ++*(_DWORD *)(a1 + 72);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 56);
  }
  sub_1CF2610(v6, 2 * v7);
  v33 = *(_DWORD *)(a1 + 80);
  if ( !v33 )
    goto LABEL_60;
  v34 = v33 - 1;
  v35 = *(_QWORD *)(a1 + 64);
  v36 = (v33 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v16 = *(_DWORD *)(a1 + 72) + 1;
  v10 = (__int64 *)(v35 + 16LL * v36);
  v37 = *v10;
  if ( a2 != *v10 )
  {
    v38 = 1;
    v39 = 0;
    while ( v37 != -8 )
    {
      if ( !v39 && v37 == -16 )
        v39 = v10;
      v36 = v34 & (v38 + v36);
      v10 = (__int64 *)(v35 + 16LL * v36);
      v37 = *v10;
      if ( a2 == *v10 )
        goto LABEL_14;
      ++v38;
    }
    if ( v39 )
      v10 = v39;
  }
LABEL_14:
  *(_DWORD *)(a1 + 72) = v16;
  if ( *v10 != -8 )
    --*(_DWORD *)(a1 + 76);
  *v10 = a2;
  v10[1] = 0;
  if ( !a3[1].m128i_i8[8] )
  {
    v20 = *(_QWORD *)(a1 + 8);
    v51.m128i_i64[0] = a2;
    v21 = 0;
    v22 = *(unsigned int *)(v20 + 48);
    if ( !(_DWORD)v22 )
      goto LABEL_23;
    v23 = *(_QWORD *)(a2 + 40);
    v24 = *(_QWORD *)(v20 + 32);
    v25 = (v22 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
    v26 = (__int64 *)(v24 + 16LL * v25);
    v27 = *v26;
    if ( v23 == *v26 )
    {
LABEL_21:
      if ( v26 != (__int64 *)(v24 + 16 * v22) )
      {
        v21 = v26[1];
        goto LABEL_23;
      }
    }
    else
    {
      v47 = 1;
      while ( v27 != -8 )
      {
        v48 = v47 + 1;
        v25 = (v22 - 1) & (v47 + v25);
        v26 = (__int64 *)(v24 + 16LL * v25);
        v27 = *v26;
        if ( v23 == *v26 )
          goto LABEL_21;
        v47 = v48;
      }
    }
    v21 = 0;
LABEL_23:
    v51.m128i_i64[1] = v21;
    v28 = *(unsigned int *)(a1 + 112);
    v29 = *(_QWORD *)(a1 + 96);
    if ( (_DWORD)v28 )
    {
      v30 = (v28 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v31 = (__int64 *)(v29 + 16LL * v30);
      v32 = *v31;
      if ( a2 == *v31 )
      {
LABEL_25:
        LODWORD(v52) = *((_DWORD *)v31 + 2);
        goto LABEL_18;
      }
      v49 = 1;
      while ( v32 != -8 )
      {
        v50 = v49 + 1;
        v30 = (v28 - 1) & (v49 + v30);
        v31 = (__int64 *)(v29 + 16LL * v30);
        v32 = *v31;
        if ( a2 == *v31 )
          goto LABEL_25;
        v49 = v50;
      }
    }
    v31 = (__int64 *)(v29 + 16 * v28);
    goto LABEL_25;
  }
  v52 = a3[1].m128i_i64[0];
  v51 = _mm_loadu_si128(a3);
LABEL_18:
  v17 = (__m128i *)sub_22077B0(24);
  *v17 = _mm_loadu_si128(&v51);
  v17[1].m128i_i64[0] = v52;
  v18 = (_QWORD *)sub_22077B0(32);
  v19 = *(_QWORD *)(a1 + 48);
  v18[1] = v17;
  v18[2] = (char *)v17 + 24;
  v18[3] = (char *)v17 + 24;
  *v18 = v19;
  *(_QWORD *)(a1 + 48) = v18;
  result = v18 + 1;
  v10[1] = (__int64)result;
  return result;
}
