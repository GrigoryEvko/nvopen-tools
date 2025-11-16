// Function: sub_30E4310
// Address: 0x30e4310
//
_QWORD *__fastcall sub_30E4310(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r13
  unsigned __int64 v9; // rcx
  int v10; // r12d
  __int64 v11; // rax
  __int64 v12; // rdi
  int v13; // eax
  unsigned int v14; // esi
  __int64 *v15; // r9
  int v16; // r13d
  __int64 v17; // rdi
  __int64 v18; // r8
  __int64 *v19; // r11
  int v20; // r15d
  __int64 v21; // rcx
  _QWORD *v22; // rax
  __int64 v23; // rdx
  _DWORD *v24; // rax
  void (__fastcall *v25)(__m128i *, __int64, __int64, __int64, __int64, __int64 *); // rax
  __int64 v26; // rdx
  __int64 v27; // rsi
  __m128i v28; // xmm1
  __int64 v29; // rdi
  __m128i v30; // xmm0
  __int64 v31; // rcx
  _QWORD *result; // rax
  int v33; // eax
  int v34; // edx
  int v35; // eax
  int v36; // esi
  __int64 v37; // rax
  int v38; // r10d
  int v39; // eax
  int v40; // eax
  __int64 v41; // rsi
  int v42; // r10d
  __int64 v43; // [rsp+8h] [rbp-98h] BYREF
  __m128i v44; // [rsp+10h] [rbp-90h] BYREF
  void (__fastcall *v45)(__m128i *, __m128i *, __int64); // [rsp+20h] [rbp-80h]
  __int64 v46; // [rsp+28h] [rbp-78h]
  __m128i v47; // [rsp+30h] [rbp-70h] BYREF
  void (__fastcall *v48)(__m128i *, __m128i *, __int64); // [rsp+40h] [rbp-60h]
  __int64 v49; // [rsp+48h] [rbp-58h]
  __m128i v50; // [rsp+50h] [rbp-50h] BYREF
  __int64 v51; // [rsp+68h] [rbp-38h]

  v7 = *(unsigned int *)(a1 + 16);
  v8 = *(_QWORD *)a2;
  v9 = *(unsigned int *)(a1 + 20);
  v10 = *(_DWORD *)(a2 + 8);
  v43 = *(_QWORD *)a2;
  if ( v7 + 1 > v9 )
  {
    sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), v7 + 1, 8u, a5, a6);
    v7 = *(unsigned int *)(a1 + 16);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8 * v7) = v8;
  v11 = v43;
  ++*(_DWORD *)(a1 + 16);
  v12 = *(_QWORD *)(v11 - 32);
  if ( v12 )
  {
    if ( *(_BYTE *)v12 )
    {
      v12 = 0;
    }
    else if ( *(_QWORD *)(v12 + 24) != *(_QWORD *)(v11 + 80) )
    {
      v12 = 0;
    }
  }
  v13 = sub_B2BED0(v12);
  v14 = *(_DWORD *)(a1 + 240);
  v15 = (__int64 *)(a1 + 216);
  v16 = v13;
  if ( !v14 )
  {
    ++*(_QWORD *)(a1 + 216);
    goto LABEL_32;
  }
  v17 = v43;
  v18 = *(_QWORD *)(a1 + 224);
  v19 = 0;
  v20 = 1;
  v21 = (v14 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
  v22 = (_QWORD *)(v18 + 16 * v21);
  v23 = *v22;
  if ( v43 == *v22 )
  {
LABEL_9:
    v24 = v22 + 1;
    goto LABEL_10;
  }
  while ( v23 != -4096 )
  {
    if ( !v19 && v23 == -8192 )
      v19 = v22;
    v21 = (v14 - 1) & (v20 + (_DWORD)v21);
    v22 = (_QWORD *)(v18 + 16LL * (unsigned int)v21);
    v23 = *v22;
    if ( v43 == *v22 )
      goto LABEL_9;
    ++v20;
  }
  if ( !v19 )
    v19 = v22;
  v33 = *(_DWORD *)(a1 + 232);
  ++*(_QWORD *)(a1 + 216);
  v34 = v33 + 1;
  if ( 4 * (v33 + 1) >= 3 * v14 )
  {
LABEL_32:
    sub_30E1D80(a1 + 216, 2 * v14);
    v35 = *(_DWORD *)(a1 + 240);
    if ( v35 )
    {
      v17 = v43;
      v36 = v35 - 1;
      v18 = *(_QWORD *)(a1 + 224);
      v34 = *(_DWORD *)(a1 + 232) + 1;
      v21 = (v35 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
      v19 = (__int64 *)(v18 + 16 * v21);
      v37 = *v19;
      if ( *v19 == v43 )
        goto LABEL_28;
      v38 = 1;
      v15 = 0;
      while ( v37 != -4096 )
      {
        if ( !v15 && v37 == -8192 )
          v15 = v19;
        v21 = v36 & (unsigned int)(v38 + v21);
        v19 = (__int64 *)(v18 + 16LL * (unsigned int)v21);
        v37 = *v19;
        if ( v43 == *v19 )
          goto LABEL_28;
        ++v38;
      }
LABEL_36:
      if ( v15 )
        v19 = v15;
      goto LABEL_28;
    }
LABEL_52:
    ++*(_DWORD *)(a1 + 232);
    BUG();
  }
  v21 = v14 >> 3;
  if ( v14 - *(_DWORD *)(a1 + 236) - v34 <= (unsigned int)v21 )
  {
    sub_30E1D80(a1 + 216, v14);
    v39 = *(_DWORD *)(a1 + 240);
    if ( v39 )
    {
      v17 = v43;
      v40 = v39 - 1;
      v41 = *(_QWORD *)(a1 + 224);
      v15 = 0;
      v42 = 1;
      v21 = v40 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
      v34 = *(_DWORD *)(a1 + 232) + 1;
      v19 = (__int64 *)(v41 + 16 * v21);
      v18 = *v19;
      if ( *v19 == v43 )
        goto LABEL_28;
      while ( v18 != -4096 )
      {
        if ( v18 == -8192 && !v15 )
          v15 = v19;
        v21 = v40 & (unsigned int)(v42 + v21);
        v19 = (__int64 *)(v41 + 16LL * (unsigned int)v21);
        v18 = *v19;
        if ( v43 == *v19 )
          goto LABEL_28;
        ++v42;
      }
      goto LABEL_36;
    }
    goto LABEL_52;
  }
LABEL_28:
  *(_DWORD *)(a1 + 232) = v34;
  if ( *v19 != -4096 )
    --*(_DWORD *)(a1 + 236);
  *v19 = v17;
  v24 = v19 + 1;
  *((_DWORD *)v19 + 2) = -1;
LABEL_10:
  *v24 = v16;
  v25 = *(void (__fastcall **)(__m128i *, __int64, __int64, __int64, __int64, __int64 *))(a1 + 168);
  v45 = 0;
  v26 = v46;
  if ( v25 )
  {
    v25(&v44, a1 + 152, 2, v21, v18, v15);
    v26 = *(_QWORD *)(a1 + 176);
    v25 = *(void (__fastcall **)(__m128i *, __int64, __int64, __int64, __int64, __int64 *))(a1 + 168);
  }
  v27 = *(unsigned int *)(a1 + 16);
  v28 = _mm_loadu_si128(&v50);
  v48 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v25;
  v29 = *(_QWORD *)(a1 + 8);
  v49 = v26;
  v30 = _mm_loadu_si128(&v44);
  v27 *= 8;
  v45 = 0;
  v46 = v51;
  v44 = v28;
  v47 = v30;
  v31 = *(_QWORD *)(v29 + v27 - 8);
  v50 = v30;
  sub_30E31D0(v29, (v27 >> 3) - 1, 0, v31, (__int64)&v47);
  if ( v48 )
    v48(&v47, &v47, 3);
  if ( v45 )
    v45(&v44, &v44, 3);
  result = sub_30E40C0(a1 + 184, &v43);
  *(_DWORD *)result = v10;
  return result;
}
