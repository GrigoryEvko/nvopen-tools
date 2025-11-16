// Function: sub_30EAB20
// Address: 0x30eab20
//
__int64 __fastcall sub_30EAB20(__int64 a1)
{
  void (__fastcall *v2)(__m128i *, __int64, __int64); // rax
  unsigned __int64 v3; // rdx
  __int64 v4; // rcx
  unsigned __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rsi
  unsigned int v8; // edx
  unsigned int v9; // eax
  __int64 *v10; // r12
  __int64 v11; // rdi
  __int32 v12; // r14d
  __int32 v13; // r15d
  void (__fastcall *v14)(__m128i *, __int64, __int64); // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rsi
  __m128i v17; // xmm0
  __m128i v18; // xmm1
  __int64 v19; // rdi
  unsigned __int64 v20; // rdx
  __m128i v21; // xmm0
  __int64 *v22; // r15
  __m128i v23; // xmm2
  unsigned __int64 v24; // rdx
  __int64 *v25; // r14
  __int64 v26; // r9
  unsigned int v27; // esi
  __int64 v28; // r12
  unsigned int v29; // edx
  __int64 v30; // r9
  unsigned int v31; // ecx
  __int64 *v32; // rax
  __int64 v33; // rdi
  int v35; // r9d
  __int64 *v36; // r14
  __m128i v37; // xmm0
  __m128i v38; // xmm3
  unsigned __int64 v39; // rdx
  __int64 *v40; // r12
  __int64 v41; // r15
  int v42; // eax
  int v43; // r8d
  __int64 v44; // r14
  __int64 *v45; // r15
  __int64 *v46; // r8
  int v47; // eax
  int v48; // edx
  int v49; // edx
  int v50; // eax
  int v51; // ecx
  __int64 v52; // rdi
  unsigned int v53; // eax
  __int64 v54; // rsi
  int v55; // r10d
  __int64 *v56; // r9
  int v57; // eax
  int v58; // eax
  __int64 v59; // rsi
  __int64 *v60; // rdi
  unsigned int v61; // r13d
  int v62; // r9d
  __int64 v63; // rcx
  int v64; // [rsp+4h] [rbp-BCh]
  __int64 v65; // [rsp+8h] [rbp-B8h]
  unsigned int v66; // [rsp+8h] [rbp-B8h]
  __m128i v67; // [rsp+10h] [rbp-B0h] BYREF
  void (__fastcall *v68)(__m128i *, __int64, __int64); // [rsp+20h] [rbp-A0h]
  unsigned __int64 v69; // [rsp+28h] [rbp-98h]
  __m128i v70; // [rsp+30h] [rbp-90h] BYREF
  void (__fastcall *v71)(__m128i *, __m128i *, __int64); // [rsp+40h] [rbp-80h]
  unsigned __int64 v72; // [rsp+48h] [rbp-78h]
  __m128i v73; // [rsp+50h] [rbp-70h] BYREF
  void (__fastcall *v74)(__m128i *, __m128i *, __int64); // [rsp+60h] [rbp-60h]
  unsigned __int64 v75; // [rsp+68h] [rbp-58h]
  unsigned int v76; // [rsp+70h] [rbp-50h]
  unsigned __int64 v77; // [rsp+78h] [rbp-48h]
  unsigned int v78; // [rsp+80h] [rbp-40h]
  char v79; // [rsp+88h] [rbp-38h]

  v2 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a1 + 168);
  v68 = 0;
  if ( !v2 )
  {
    v5 = *(unsigned int *)(a1 + 16);
    v4 = v5;
    if ( v5 <= 1 )
      goto LABEL_6;
    goto LABEL_42;
  }
  v2(&v67, a1 + 152, 2);
  v3 = *(unsigned int *)(a1 + 16);
  v69 = *(_QWORD *)(a1 + 176);
  v2 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a1 + 168);
  v4 = v3;
  v68 = v2;
  if ( v3 > 1 )
  {
LABEL_42:
    v36 = *(__int64 **)(a1 + 8);
    v71 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v2;
    v37 = _mm_loadu_si128(&v67);
    v38 = _mm_loadu_si128(&v73);
    v68 = 0;
    v39 = v69;
    v69 = v75;
    v67 = v38;
    v40 = &v36[v4 - 1];
    v72 = v39;
    v73 = v37;
    v70 = v37;
    v41 = *v40;
    *v40 = *v36;
    v74 = 0;
    if ( v71 )
    {
      v71(&v73, &v70, 2);
      v75 = v72;
      v74 = v71;
    }
    sub_30E32A0((__int64)v36, 0, v40 - v36, v41, (__int64)&v73);
    if ( v74 )
      v74(&v73, &v73, 3);
    if ( v71 )
LABEL_23:
      v71(&v70, &v70, 3);
LABEL_24:
    v2 = v68;
  }
LABEL_3:
  if ( v2 )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64, __int64))v2)(&v67, &v67, 3, v4 * 8);
  v5 = *(unsigned int *)(a1 + 16);
  while ( 1 )
  {
LABEL_6:
    v6 = *(_QWORD *)(a1 + 224);
    v7 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8 * v5 - 8);
    v8 = *(_DWORD *)(a1 + 240);
    if ( !v8 )
      goto LABEL_38;
    v9 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v7 != *v10 )
    {
      v35 = 1;
      while ( v11 != -4096 )
      {
        v9 = (v8 - 1) & (v35 + v9);
        v10 = (__int64 *)(v6 + 16LL * v9);
        v11 = *v10;
        if ( v7 == *v10 )
          goto LABEL_8;
        ++v35;
      }
LABEL_38:
      v10 = (__int64 *)(v6 + 16LL * v8);
    }
LABEL_8:
    v12 = *((_DWORD *)v10 + 2);
    sub_30E1100((__int64)&v73, v7, *(_QWORD *)(a1 + 248), *(int **)(a1 + 256));
    v13 = v73.m128i_i32[0];
    if ( v79 )
    {
      v79 = 0;
      if ( v78 > 0x40 && v77 )
        j_j___libc_free_0_0(v77);
      if ( v76 > 0x40 && v75 )
        break;
    }
    *((_DWORD *)v10 + 2) = v13;
    if ( v12 >= v13 )
      goto LABEL_31;
LABEL_10:
    v14 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a1 + 168);
    v15 = v69;
    v68 = 0;
    if ( v14 )
    {
      v14(&v67, a1 + 152, 2);
      v15 = *(_QWORD *)(a1 + 176);
      v14 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a1 + 168);
    }
    v16 = *(unsigned int *)(a1 + 16);
    v17 = _mm_loadu_si128(&v67);
    v71 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v14;
    v18 = _mm_loadu_si128(&v73);
    v72 = v15;
    v19 = *(_QWORD *)(a1 + 8);
    v73 = v17;
    v68 = 0;
    v69 = v75;
    v67 = v18;
    v70 = v17;
    sub_30E31D0(v19, ((8 * v16) >> 3) - 1, 0, *(_QWORD *)(v19 + 8 * v16 - 8), (__int64)&v70);
    if ( v71 )
      v71(&v70, &v70, 3);
    if ( v68 )
      v68(&v67, (__int64)&v67, 3);
    v2 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a1 + 168);
    v68 = 0;
    if ( v2 )
    {
      v2(&v67, a1 + 152, 2);
      v20 = *(unsigned int *)(a1 + 16);
      v69 = *(_QWORD *)(a1 + 176);
      v2 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a1 + 168);
      v4 = v20;
      v68 = v2;
      if ( v20 <= 1 )
        goto LABEL_3;
LABEL_18:
      v21 = _mm_loadu_si128(&v67);
      v22 = *(__int64 **)(a1 + 8);
      v71 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v2;
      v23 = _mm_loadu_si128(&v73);
      v68 = 0;
      v24 = v69;
      v73 = v21;
      v69 = v75;
      v25 = &v22[v4 - 1];
      v72 = v24;
      v67 = v23;
      v70 = v21;
      v26 = *v25;
      *v25 = *v22;
      v74 = 0;
      if ( v71 )
      {
        v65 = v26;
        v71(&v73, &v70, 2);
        v26 = v65;
        v75 = v72;
        v74 = v71;
      }
      sub_30E32A0((__int64)v22, 0, v25 - v22, v26, (__int64)&v73);
      if ( v74 )
        v74(&v73, &v73, 3);
      if ( !v71 )
        goto LABEL_24;
      goto LABEL_23;
    }
    v5 = *(unsigned int *)(a1 + 16);
    v4 = v5;
    if ( v5 > 1 )
      goto LABEL_18;
  }
  j_j___libc_free_0_0(v75);
  *((_DWORD *)v10 + 2) = v13;
  if ( v12 < v13 )
    goto LABEL_10;
LABEL_31:
  v27 = *(_DWORD *)(a1 + 208);
  v28 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL * (unsigned int)(*(_DWORD *)(a1 + 16))-- - 8);
  if ( v27 )
  {
    v29 = v27 - 1;
    v30 = *(_QWORD *)(a1 + 192);
    v31 = (v27 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
    v32 = (__int64 *)(v30 + 16LL * v31);
    v33 = *v32;
    if ( v28 == *v32 )
    {
LABEL_33:
      if ( v28 == v33 )
      {
LABEL_34:
        *v32 = -8192;
        --*(_DWORD *)(a1 + 200);
        ++*(_DWORD *)(a1 + 204);
      }
      else
      {
        v42 = 1;
        while ( v33 != -4096 )
        {
          v43 = v42 + 1;
          v31 = v29 & (v42 + v31);
          v32 = (__int64 *)(v30 + 16LL * v31);
          v33 = *v32;
          if ( v28 == *v32 )
            goto LABEL_34;
          v42 = v43;
        }
      }
      return v28;
    }
    v66 = (v27 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
    v44 = *v32;
    v45 = (__int64 *)(v30 + 16LL * (v29 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4))));
    v46 = 0;
    v64 = 1;
    while ( v44 != -4096 )
    {
      if ( v46 || v44 != -8192 )
        v45 = v46;
      v66 = v29 & (v64 + v66);
      v44 = *(_QWORD *)(v30 + 16LL * v66);
      if ( v28 == v44 )
        goto LABEL_33;
      ++v64;
      v46 = v45;
      v45 = (__int64 *)(v30 + 16LL * v66);
    }
    v47 = *(_DWORD *)(a1 + 200);
    if ( !v46 )
      v46 = v45;
    ++*(_QWORD *)(a1 + 184);
    v48 = v47 + 1;
    if ( 4 * (v47 + 1) < 3 * v27 )
    {
      if ( v27 - *(_DWORD *)(a1 + 204) - v48 > v27 >> 3 )
        goto LABEL_60;
      sub_30E3EE0(a1 + 184, v27);
      v57 = *(_DWORD *)(a1 + 208);
      if ( v57 )
      {
        v58 = v57 - 1;
        v59 = *(_QWORD *)(a1 + 192);
        v60 = 0;
        v61 = v58 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v62 = 1;
        v48 = *(_DWORD *)(a1 + 200) + 1;
        v46 = (__int64 *)(v59 + 16LL * v61);
        v63 = *v46;
        if ( v28 != *v46 )
        {
          while ( v63 != -4096 )
          {
            if ( !v60 && v63 == -8192 )
              v60 = v46;
            v61 = v58 & (v62 + v61);
            v46 = (__int64 *)(v59 + 16LL * v61);
            v63 = *v46;
            if ( v28 == *v46 )
              goto LABEL_60;
            ++v62;
          }
          if ( v60 )
            v46 = v60;
        }
        goto LABEL_60;
      }
LABEL_93:
      ++*(_DWORD *)(a1 + 200);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 184);
  }
  sub_30E3EE0(a1 + 184, 2 * v27);
  v50 = *(_DWORD *)(a1 + 208);
  if ( !v50 )
    goto LABEL_93;
  v51 = v50 - 1;
  v52 = *(_QWORD *)(a1 + 192);
  v53 = (v50 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
  v48 = *(_DWORD *)(a1 + 200) + 1;
  v46 = (__int64 *)(v52 + 16LL * v53);
  v54 = *v46;
  if ( v28 != *v46 )
  {
    v55 = 1;
    v56 = 0;
    while ( v54 != -4096 )
    {
      if ( v54 == -8192 && !v56 )
        v56 = v46;
      v53 = v51 & (v55 + v53);
      v46 = (__int64 *)(v52 + 16LL * v53);
      v54 = *v46;
      if ( v28 == *v46 )
        goto LABEL_60;
      ++v55;
    }
    if ( v56 )
      v46 = v56;
  }
LABEL_60:
  *(_DWORD *)(a1 + 200) = v48;
  if ( *v46 != -4096 )
    --*(_DWORD *)(a1 + 204);
  *v46 = v28;
  *((_DWORD *)v46 + 2) = 0;
  v49 = *(_DWORD *)(a1 + 208);
  if ( v49 )
  {
    v29 = v49 - 1;
    v30 = *(_QWORD *)(a1 + 192);
    v31 = v29 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
    v32 = (__int64 *)(v30 + 16LL * v31);
    v33 = *v32;
    goto LABEL_33;
  }
  return v28;
}
