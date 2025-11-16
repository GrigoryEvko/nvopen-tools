// Function: sub_30E5220
// Address: 0x30e5220
//
_QWORD *__fastcall sub_30E5220(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r13
  unsigned __int64 v9; // rcx
  int v10; // r12d
  int *v11; // rcx
  __int32 v12; // r13d
  unsigned int v13; // esi
  __int64 *v14; // r9
  __int64 v15; // rdi
  __int64 v16; // r8
  __int64 *v17; // r11
  int v18; // r15d
  __int64 v19; // rcx
  _QWORD *v20; // rax
  __int64 v21; // rdx
  __int32 *v22; // rax
  void (__fastcall *v23)(__m128i *, __int64, __int64, __int64, __int64, __int64 *); // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // rsi
  __m128i v26; // xmm0
  __m128i v27; // xmm1
  __int64 v28; // rdi
  _QWORD *result; // rax
  int v30; // eax
  int v31; // edx
  int v32; // eax
  int v33; // eax
  __int64 *v34; // r10
  int v35; // eax
  int v36; // eax
  __int64 v37; // rsi
  int v38; // r10d
  __int64 v39; // [rsp+8h] [rbp-B8h] BYREF
  __m128i v40; // [rsp+10h] [rbp-B0h] BYREF
  void (__fastcall *v41)(__m128i *, __m128i *, __int64); // [rsp+20h] [rbp-A0h]
  unsigned __int64 v42; // [rsp+28h] [rbp-98h]
  __m128i v43; // [rsp+30h] [rbp-90h] BYREF
  void (__fastcall *v44)(__m128i *, __m128i *, __int64); // [rsp+40h] [rbp-80h]
  unsigned __int64 v45; // [rsp+48h] [rbp-78h]
  __m128i v46; // [rsp+50h] [rbp-70h] BYREF
  unsigned __int64 v47; // [rsp+68h] [rbp-58h]
  unsigned int v48; // [rsp+70h] [rbp-50h]
  unsigned __int64 v49; // [rsp+78h] [rbp-48h]
  unsigned int v50; // [rsp+80h] [rbp-40h]
  char v51; // [rsp+88h] [rbp-38h]

  v7 = *(unsigned int *)(a1 + 16);
  v8 = *(_QWORD *)a2;
  v9 = *(unsigned int *)(a1 + 20);
  v10 = *(_DWORD *)(a2 + 8);
  v39 = *(_QWORD *)a2;
  if ( v7 + 1 > v9 )
  {
    sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), v7 + 1, 8u, a5, a6);
    v7 = *(unsigned int *)(a1 + 16);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8 * v7) = v8;
  v11 = *(int **)(a1 + 256);
  ++*(_DWORD *)(a1 + 16);
  sub_30E1100((__int64)&v46, v39, *(_QWORD *)(a1 + 248), v11);
  v12 = v46.m128i_i32[0];
  if ( v51 )
  {
    v51 = 0;
    if ( v50 > 0x40 && v49 )
      j_j___libc_free_0_0(v49);
    if ( v48 > 0x40 && v47 )
      j_j___libc_free_0_0(v47);
  }
  v13 = *(_DWORD *)(a1 + 240);
  v14 = (__int64 *)(a1 + 216);
  if ( !v13 )
  {
    ++*(_QWORD *)(a1 + 216);
    goto LABEL_34;
  }
  v15 = v39;
  v16 = *(_QWORD *)(a1 + 224);
  v17 = 0;
  v18 = 1;
  v19 = (v13 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
  v20 = (_QWORD *)(v16 + 16 * v19);
  v21 = *v20;
  if ( v39 == *v20 )
  {
LABEL_6:
    v22 = (__int32 *)(v20 + 1);
    goto LABEL_7;
  }
  while ( v21 != -4096 )
  {
    if ( !v17 && v21 == -8192 )
      v17 = v20;
    v19 = (v13 - 1) & (v18 + (_DWORD)v19);
    v20 = (_QWORD *)(v16 + 16LL * (unsigned int)v19);
    v21 = *v20;
    if ( v39 == *v20 )
      goto LABEL_6;
    ++v18;
  }
  if ( !v17 )
    v17 = v20;
  v30 = *(_DWORD *)(a1 + 232);
  ++*(_QWORD *)(a1 + 216);
  v31 = v30 + 1;
  if ( 4 * (v30 + 1) >= 3 * v13 )
  {
LABEL_34:
    sub_30E1BA0(a1 + 216, 2 * v13);
    v32 = *(_DWORD *)(a1 + 240);
    if ( v32 )
    {
      v14 = (__int64 *)(unsigned int)(v32 - 1);
      v16 = *(_QWORD *)(a1 + 224);
      v19 = (unsigned int)v14 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
      v17 = (__int64 *)(v16 + 16 * v19);
      v15 = *v17;
      v31 = *(_DWORD *)(a1 + 232) + 1;
      if ( v39 != *v17 )
      {
        v33 = 1;
        v34 = 0;
        while ( v15 != -4096 )
        {
          if ( !v34 && v15 == -8192 )
            v34 = v17;
          v19 = (unsigned int)v14 & (v33 + (_DWORD)v19);
          v17 = (__int64 *)(v16 + 16LL * (unsigned int)v19);
          v15 = *v17;
          if ( v39 == *v17 )
            goto LABEL_30;
          ++v33;
        }
        v15 = v39;
        if ( v34 )
          v17 = v34;
      }
      goto LABEL_30;
    }
    goto LABEL_57;
  }
  v19 = v13 >> 3;
  if ( v13 - *(_DWORD *)(a1 + 236) - v31 <= (unsigned int)v19 )
  {
    sub_30E1BA0(a1 + 216, v13);
    v35 = *(_DWORD *)(a1 + 240);
    if ( v35 )
    {
      v15 = v39;
      v36 = v35 - 1;
      v37 = *(_QWORD *)(a1 + 224);
      v14 = 0;
      v38 = 1;
      v19 = v36 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
      v31 = *(_DWORD *)(a1 + 232) + 1;
      v17 = (__int64 *)(v37 + 16 * v19);
      v16 = *v17;
      if ( *v17 != v39 )
      {
        while ( v16 != -4096 )
        {
          if ( v16 == -8192 && !v14 )
            v14 = v17;
          v19 = v36 & (unsigned int)(v38 + v19);
          v17 = (__int64 *)(v37 + 16LL * (unsigned int)v19);
          v16 = *v17;
          if ( v39 == *v17 )
            goto LABEL_30;
          ++v38;
        }
        if ( v14 )
          v17 = v14;
      }
      goto LABEL_30;
    }
LABEL_57:
    ++*(_DWORD *)(a1 + 232);
    BUG();
  }
LABEL_30:
  *(_DWORD *)(a1 + 232) = v31;
  if ( *v17 != -4096 )
    --*(_DWORD *)(a1 + 236);
  *v17 = v15;
  v22 = (__int32 *)(v17 + 1);
  *((_DWORD *)v17 + 2) = 0x7FFFFFFF;
LABEL_7:
  *v22 = v12;
  v23 = *(void (__fastcall **)(__m128i *, __int64, __int64, __int64, __int64, __int64 *))(a1 + 168);
  v41 = 0;
  v24 = v42;
  if ( v23 )
  {
    v23(&v40, a1 + 152, 2, v19, v16, v14);
    v24 = *(_QWORD *)(a1 + 176);
    v23 = *(void (__fastcall **)(__m128i *, __int64, __int64, __int64, __int64, __int64 *))(a1 + 168);
  }
  v25 = *(unsigned int *)(a1 + 16);
  v26 = _mm_loadu_si128(&v40);
  v44 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v23;
  v27 = _mm_loadu_si128(&v46);
  v45 = v24;
  v28 = *(_QWORD *)(a1 + 8);
  v46 = v26;
  v41 = 0;
  v42 = v47;
  v40 = v27;
  v43 = v26;
  sub_30E31D0(v28, ((8 * v25) >> 3) - 1, 0, *(_QWORD *)(v28 + 8 * v25 - 8), (__int64)&v43);
  if ( v44 )
    v44(&v43, &v43, 3);
  if ( v41 )
    v41(&v40, &v40, 3);
  result = sub_30E40C0(a1 + 184, &v39);
  *(_DWORD *)result = v10;
  return result;
}
