// Function: sub_208C270
// Address: 0x208c270
//
void __fastcall sub_208C270(__int64 a1, __int64 *a2, int a3, __m128i a4, __m128i a5, __m128i a6)
{
  __int64 *v8; // r14
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rsi
  int v14; // r9d
  __int64 v15; // rcx
  unsigned int v16; // r10d
  unsigned int v17; // r8d
  __int64 v18; // rax
  __int64 *v19; // rdi
  unsigned int v20; // ecx
  __int64 *v21; // r11
  int v22; // edx
  __int64 v23; // rax
  __int64 v24; // rsi
  int v25; // r8d
  int v26; // r9d
  __int64 v27; // rax
  char *v28; // rdi
  __int64 v29; // r11
  __int64 v30; // rdi
  int v31; // eax
  int v32; // r9d
  int v33; // eax
  int v34; // eax
  __int64 v35; // rdi
  unsigned int v36; // ecx
  __int64 *v37; // rsi
  int v38; // r8d
  __int64 v39; // r10
  int v40; // eax
  int v41; // eax
  __int64 v42; // rdi
  int v43; // r8d
  unsigned int v44; // ecx
  __int64 *v45; // rsi
  int v47; // [rsp+Ch] [rbp-124h]
  __int64 v48; // [rsp+10h] [rbp-120h]
  __int64 v49; // [rsp+18h] [rbp-118h]
  int i; // [rsp+18h] [rbp-118h]
  __int64 **v51; // [rsp+18h] [rbp-118h]
  unsigned int v52; // [rsp+18h] [rbp-118h]
  unsigned int v53; // [rsp+18h] [rbp-118h]
  __int64 v54; // [rsp+18h] [rbp-118h]
  __int64 v55; // [rsp+20h] [rbp-110h]
  unsigned int v56; // [rsp+20h] [rbp-110h]
  unsigned int v57; // [rsp+20h] [rbp-110h]
  int v58; // [rsp+20h] [rbp-110h]
  __int64 v59; // [rsp+20h] [rbp-110h]
  __int64 v60; // [rsp+20h] [rbp-110h]
  unsigned __int64 v61; // [rsp+28h] [rbp-108h]
  __m128i v62; // [rsp+30h] [rbp-100h] BYREF
  __int64 v63; // [rsp+40h] [rbp-F0h] BYREF
  int v64; // [rsp+48h] [rbp-E8h]
  unsigned __int64 v65[2]; // [rsp+50h] [rbp-E0h] BYREF
  char v66; // [rsp+60h] [rbp-D0h] BYREF
  char *v67; // [rsp+A0h] [rbp-90h]
  char v68; // [rsp+B0h] [rbp-80h] BYREF
  char *v69; // [rsp+B8h] [rbp-78h]
  char v70; // [rsp+C8h] [rbp-68h] BYREF
  char *v71; // [rsp+D8h] [rbp-58h]
  char v72; // [rsp+E8h] [rbp-48h] BYREF

  v8 = sub_208C170(a1, (__int64)a2, a4, a5, a6);
  v61 = v9;
  v48 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL);
  sub_2043DE0((__int64)&v63, (__int64)a2);
  v49 = *a2;
  v55 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
  v10 = sub_16498A0((__int64)a2);
  sub_204E3C0((__int64)v65, v10, v48, v55, a3, v49, (unsigned int *)&v63);
  v11 = *(_QWORD *)(a1 + 712);
  v12 = *(_QWORD *)(a1 + 552);
  v62.m128i_i32[2] = 0;
  v13 = *(unsigned int *)(v11 + 824);
  v62.m128i_i64[0] = v12 + 88;
  if ( !(_DWORD)v13 )
    goto LABEL_7;
  v14 = v13 - 1;
  v15 = *(_QWORD *)(v11 + 808);
  v16 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v17 = (v13 - 1) & v16;
  v18 = v15 + 16LL * v17;
  v19 = *(__int64 **)v18;
  if ( *(__int64 **)v18 == a2 )
  {
    if ( v18 != v15 + 16 * v13 )
    {
LABEL_4:
      v20 = *(_DWORD *)(v18 + 8);
      goto LABEL_8;
    }
LABEL_7:
    v20 = 144;
    goto LABEL_8;
  }
  v56 = (v13 - 1) & v16;
  v21 = *(__int64 **)v18;
  for ( i = 1; ; i = v47 )
  {
    if ( v21 == (__int64 *)-8LL )
      goto LABEL_7;
    v47 = i + 1;
    v56 = v14 & (i + v56);
    v51 = (__int64 **)(v15 + 16LL * v56);
    v21 = *v51;
    if ( *v51 == a2 )
      break;
  }
  v18 = v15 + 16LL * (v14 & v16);
  if ( v51 == (__int64 **)(v15 + 16LL * (unsigned int)v13) )
    goto LABEL_7;
  if ( v19 == a2 )
    goto LABEL_4;
  v58 = 1;
  v29 = 0;
  while ( v19 != (__int64 *)-8LL )
  {
    if ( v29 || v19 != (__int64 *)-16LL )
      v18 = v29;
    v17 = v14 & (v58 + v17);
    v54 = v15 + 16LL * v17;
    v19 = *(__int64 **)v54;
    if ( *(__int64 **)v54 == a2 )
    {
      v20 = *(_DWORD *)(v54 + 8);
      goto LABEL_8;
    }
    ++v58;
    v29 = v18;
    v18 = v15 + 16LL * v17;
  }
  v30 = v11 + 800;
  if ( !v29 )
    v29 = v18;
  v31 = *(_DWORD *)(v11 + 816);
  ++*(_QWORD *)(v11 + 800);
  v32 = v31 + 1;
  if ( 4 * (v31 + 1) >= (unsigned int)(3 * v13) )
  {
    v59 = v11;
    v52 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    sub_1FE17D0(v30, 2 * v13);
    v11 = v59;
    v33 = *(_DWORD *)(v59 + 824);
    if ( v33 )
    {
      v34 = v33 - 1;
      v35 = *(_QWORD *)(v59 + 808);
      v36 = v34 & v52;
      v32 = *(_DWORD *)(v59 + 816) + 1;
      v29 = v35 + 16LL * (v34 & v52);
      v37 = *(__int64 **)v29;
      if ( *(__int64 **)v29 == a2 )
        goto LABEL_34;
      v38 = 1;
      v39 = 0;
      while ( v37 != (__int64 *)-8LL )
      {
        if ( v37 == (__int64 *)-16LL && !v39 )
          v39 = v29;
        v36 = v34 & (v38 + v36);
        v29 = v35 + 16LL * v36;
        v37 = *(__int64 **)v29;
        if ( *(__int64 **)v29 == a2 )
          goto LABEL_34;
        ++v38;
      }
LABEL_41:
      if ( v39 )
        v29 = v39;
      goto LABEL_34;
    }
LABEL_64:
    ++*(_DWORD *)(v11 + 816);
    BUG();
  }
  if ( (int)v13 - *(_DWORD *)(v11 + 820) - v32 <= (unsigned int)v13 >> 3 )
  {
    v60 = v11;
    v53 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    sub_1FE17D0(v30, v13);
    v11 = v60;
    v40 = *(_DWORD *)(v60 + 824);
    if ( v40 )
    {
      v41 = v40 - 1;
      v42 = *(_QWORD *)(v60 + 808);
      v43 = 1;
      v44 = v41 & v53;
      v32 = *(_DWORD *)(v60 + 816) + 1;
      v39 = 0;
      v29 = v42 + 16LL * (v41 & v53);
      v45 = *(__int64 **)v29;
      if ( *(__int64 **)v29 == a2 )
        goto LABEL_34;
      while ( v45 != (__int64 *)-8LL )
      {
        if ( !v39 && v45 == (__int64 *)-16LL )
          v39 = v29;
        v44 = v41 & (v43 + v44);
        v29 = v42 + 16LL * v44;
        v45 = *(__int64 **)v29;
        if ( *(__int64 **)v29 == a2 )
          goto LABEL_34;
        ++v43;
      }
      goto LABEL_41;
    }
    goto LABEL_64;
  }
LABEL_34:
  *(_DWORD *)(v11 + 816) = v32;
  if ( *(_QWORD *)v29 != -8 )
    --*(_DWORD *)(v11 + 820);
  *(_QWORD *)v29 = a2;
  v20 = 0;
  *(_DWORD *)(v29 + 8) = 0;
LABEL_8:
  v22 = *(_DWORD *)(a1 + 536);
  v23 = *(_QWORD *)a1;
  v63 = 0;
  v64 = v22;
  if ( v23 )
  {
    if ( &v63 != (__int64 *)(v23 + 48) )
    {
      v24 = *(_QWORD *)(v23 + 48);
      v63 = v24;
      if ( v24 )
      {
        v57 = v20;
        sub_1623A60((__int64)&v63, v24, 2);
        v20 = v57;
      }
    }
  }
  sub_204FAE0(
    (__int64)v65,
    (__int64)v8,
    v61,
    *(__int64 **)(a1 + 552),
    (__int64)&v63,
    &v62,
    a4,
    *(double *)a5.m128i_i64,
    a6,
    0,
    (__int64)a2,
    v20);
  if ( v63 )
    sub_161E7C0((__int64)&v63, v63);
  v27 = *(unsigned int *)(a1 + 400);
  if ( (unsigned int)v27 >= *(_DWORD *)(a1 + 404) )
  {
    sub_16CD150(a1 + 392, (const void *)(a1 + 408), 0, 16, v25, v26);
    v27 = *(unsigned int *)(a1 + 400);
  }
  *(__m128i *)(*(_QWORD *)(a1 + 392) + 16 * v27) = _mm_load_si128(&v62);
  v28 = v71;
  ++*(_DWORD *)(a1 + 400);
  if ( v28 != &v72 )
    _libc_free((unsigned __int64)v28);
  if ( v69 != &v70 )
    _libc_free((unsigned __int64)v69);
  if ( v67 != &v68 )
    _libc_free((unsigned __int64)v67);
  if ( (char *)v65[0] != &v66 )
    _libc_free(v65[0]);
}
