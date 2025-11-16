// Function: sub_30EA360
// Address: 0x30ea360
//
__int64 __fastcall sub_30EA360(__int64 a1)
{
  void (__fastcall *v2)(__m128i *, __int64, __int64); // rax
  unsigned __int64 v3; // rdx
  __int64 v4; // rcx
  unsigned __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 *v10; // r12
  __int64 v11; // rdi
  __int64 v12; // rdi
  unsigned int v13; // r13d
  unsigned int v14; // eax
  void (__fastcall *v15)(__m128i *, __int64, __int64); // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  __m128i v18; // xmm1
  __int64 v19; // rdi
  __m128i v20; // xmm0
  unsigned __int64 v21; // rdx
  __int64 *v22; // r14
  __m128i v23; // xmm0
  __m128i v24; // xmm2
  __int64 *v25; // r13
  __int64 v26; // r9
  int v27; // r9d
  unsigned int v28; // eax
  unsigned int v29; // esi
  __int64 v30; // r12
  unsigned int v31; // edx
  __int64 v32; // r9
  unsigned int v33; // ecx
  __int64 *v34; // rax
  __int64 v35; // rdi
  __int64 *v37; // r13
  __m128i v38; // xmm0
  __m128i v39; // xmm3
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
  int v64; // [rsp+4h] [rbp-9Ch]
  __int64 v65; // [rsp+8h] [rbp-98h]
  unsigned int v66; // [rsp+8h] [rbp-98h]
  __m128i v67; // [rsp+10h] [rbp-90h] BYREF
  void (__fastcall *v68)(__m128i *, __int64, __int64); // [rsp+20h] [rbp-80h]
  __int64 v69; // [rsp+28h] [rbp-78h]
  __m128i v70; // [rsp+30h] [rbp-70h] BYREF
  void (__fastcall *v71)(__m128i *, __m128i *, __int64); // [rsp+40h] [rbp-60h]
  __int64 v72; // [rsp+48h] [rbp-58h]
  __m128i v73; // [rsp+50h] [rbp-50h] BYREF
  void (__fastcall *v74)(__m128i *, __m128i *, __int64); // [rsp+60h] [rbp-40h]
  __int64 v75; // [rsp+68h] [rbp-38h]

  v2 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a1 + 168);
  v68 = 0;
  if ( !v2 )
  {
    v5 = *(unsigned int *)(a1 + 16);
    v4 = v5;
    if ( v5 <= 1 )
      goto LABEL_6;
    goto LABEL_40;
  }
  v2(&v67, a1 + 152, 2);
  v3 = *(unsigned int *)(a1 + 16);
  v69 = *(_QWORD *)(a1 + 176);
  v2 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a1 + 168);
  v4 = v3;
  v68 = v2;
  if ( v3 > 1 )
  {
LABEL_40:
    v37 = *(__int64 **)(a1 + 8);
    v71 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v2;
    v38 = _mm_loadu_si128(&v67);
    v39 = _mm_loadu_si128(&v73);
    v68 = 0;
    v72 = v69;
    v67 = v39;
    v40 = &v37[v4 - 1];
    v69 = v75;
    v73 = v38;
    v70 = v38;
    v41 = *v40;
    *v40 = *v37;
    v74 = 0;
    if ( v71 )
    {
      v71(&v73, &v70, 2);
      v75 = v72;
      v74 = v71;
    }
    sub_30E32A0((__int64)v37, 0, v40 - v37, v41, (__int64)&v73);
    if ( v74 )
      v74(&v73, &v73, 3);
    if ( v71 )
LABEL_26:
      v71(&v70, &v70, 3);
LABEL_27:
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
    v8 = *(unsigned int *)(a1 + 240);
    if ( (_DWORD)v8 )
    {
      v9 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v10 = (__int64 *)(v6 + 16LL * v9);
      v11 = *v10;
      if ( v7 == *v10 )
        goto LABEL_8;
      v27 = 1;
      while ( v11 != -4096 )
      {
        v9 = (v8 - 1) & (v27 + v9);
        v10 = (__int64 *)(v6 + 16LL * v9);
        v11 = *v10;
        if ( v7 == *v10 )
          goto LABEL_8;
        ++v27;
      }
    }
    v10 = (__int64 *)(v6 + 16 * v8);
LABEL_8:
    v12 = *(_QWORD *)(v7 - 32);
    v13 = *((_DWORD *)v10 + 2);
    if ( !v12 )
      goto LABEL_12;
    if ( *(_BYTE *)v12 )
      break;
    if ( *(_QWORD *)(v12 + 24) != *(_QWORD *)(v7 + 80) )
      v12 = 0;
LABEL_12:
    v14 = sub_B2BED0(v12);
    *((_DWORD *)v10 + 2) = v14;
    if ( v13 >= v14 )
      goto LABEL_34;
LABEL_13:
    v15 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a1 + 168);
    v16 = v69;
    v68 = 0;
    if ( v15 )
    {
      v15(&v67, a1 + 152, 2);
      v16 = *(_QWORD *)(a1 + 176);
      v15 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a1 + 168);
    }
    v17 = *(unsigned int *)(a1 + 16);
    v18 = _mm_loadu_si128(&v73);
    v71 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v15;
    v19 = *(_QWORD *)(a1 + 8);
    v72 = v16;
    v20 = _mm_loadu_si128(&v67);
    v68 = 0;
    v69 = v75;
    v67 = v18;
    v73 = v20;
    v70 = v20;
    sub_30E31D0(v19, ((8 * v17) >> 3) - 1, 0, *(_QWORD *)(v19 + 8 * v17 - 8), (__int64)&v70);
    if ( v71 )
      v71(&v70, &v70, 3);
    if ( v68 )
      v68(&v67, (__int64)&v67, 3);
    v2 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a1 + 168);
    v68 = 0;
    if ( v2 )
    {
      v2(&v67, a1 + 152, 2);
      v21 = *(unsigned int *)(a1 + 16);
      v69 = *(_QWORD *)(a1 + 176);
      v2 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a1 + 168);
      v4 = v21;
      v68 = v2;
      if ( v21 <= 1 )
        goto LABEL_3;
LABEL_21:
      v22 = *(__int64 **)(a1 + 8);
      v71 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v2;
      v23 = _mm_loadu_si128(&v67);
      v24 = _mm_loadu_si128(&v73);
      v68 = 0;
      v72 = v69;
      v67 = v24;
      v25 = &v22[v4 - 1];
      v69 = v75;
      v73 = v23;
      v70 = v23;
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
        goto LABEL_27;
      goto LABEL_26;
    }
    v5 = *(unsigned int *)(a1 + 16);
    v4 = v5;
    if ( v5 > 1 )
      goto LABEL_21;
  }
  v28 = sub_B2BED0(0);
  *((_DWORD *)v10 + 2) = v28;
  if ( v13 < v28 )
    goto LABEL_13;
LABEL_34:
  v29 = *(_DWORD *)(a1 + 208);
  v30 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL * (unsigned int)(*(_DWORD *)(a1 + 16))-- - 8);
  if ( v29 )
  {
    v31 = v29 - 1;
    v32 = *(_QWORD *)(a1 + 192);
    v33 = (v29 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
    v34 = (__int64 *)(v32 + 16LL * v33);
    v35 = *v34;
    if ( v30 == *v34 )
    {
LABEL_36:
      if ( v30 == v35 )
      {
LABEL_37:
        *v34 = -8192;
        --*(_DWORD *)(a1 + 200);
        ++*(_DWORD *)(a1 + 204);
      }
      else
      {
        v42 = 1;
        while ( v35 != -4096 )
        {
          v43 = v42 + 1;
          v33 = v31 & (v42 + v33);
          v34 = (__int64 *)(v32 + 16LL * v33);
          v35 = *v34;
          if ( v30 == *v34 )
            goto LABEL_37;
          v42 = v43;
        }
      }
      return v30;
    }
    v66 = (v29 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
    v44 = *v34;
    v45 = (__int64 *)(v32 + 16LL * (v31 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4))));
    v46 = 0;
    v64 = 1;
    while ( v44 != -4096 )
    {
      if ( v44 != -8192 || v46 )
        v45 = v46;
      v66 = v31 & (v64 + v66);
      v44 = *(_QWORD *)(v32 + 16LL * v66);
      if ( v30 == v44 )
        goto LABEL_36;
      ++v64;
      v46 = v45;
      v45 = (__int64 *)(v32 + 16LL * v66);
    }
    v47 = *(_DWORD *)(a1 + 200);
    if ( !v46 )
      v46 = v45;
    ++*(_QWORD *)(a1 + 184);
    v48 = v47 + 1;
    if ( 4 * (v47 + 1) < 3 * v29 )
    {
      if ( v29 - *(_DWORD *)(a1 + 204) - v48 > v29 >> 3 )
        goto LABEL_58;
      sub_30E3EE0(a1 + 184, v29);
      v57 = *(_DWORD *)(a1 + 208);
      if ( v57 )
      {
        v58 = v57 - 1;
        v59 = *(_QWORD *)(a1 + 192);
        v60 = 0;
        v61 = v58 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
        v62 = 1;
        v48 = *(_DWORD *)(a1 + 200) + 1;
        v46 = (__int64 *)(v59 + 16LL * v61);
        v63 = *v46;
        if ( v30 != *v46 )
        {
          while ( v63 != -4096 )
          {
            if ( v63 == -8192 && !v60 )
              v60 = v46;
            v61 = v58 & (v62 + v61);
            v46 = (__int64 *)(v59 + 16LL * v61);
            v63 = *v46;
            if ( v30 == *v46 )
              goto LABEL_58;
            ++v62;
          }
          if ( v60 )
            v46 = v60;
        }
        goto LABEL_58;
      }
LABEL_91:
      ++*(_DWORD *)(a1 + 200);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 184);
  }
  sub_30E3EE0(a1 + 184, 2 * v29);
  v50 = *(_DWORD *)(a1 + 208);
  if ( !v50 )
    goto LABEL_91;
  v51 = v50 - 1;
  v52 = *(_QWORD *)(a1 + 192);
  v53 = (v50 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
  v48 = *(_DWORD *)(a1 + 200) + 1;
  v46 = (__int64 *)(v52 + 16LL * v53);
  v54 = *v46;
  if ( v30 != *v46 )
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
      if ( v30 == *v46 )
        goto LABEL_58;
      ++v55;
    }
    if ( v56 )
      v46 = v56;
  }
LABEL_58:
  *(_DWORD *)(a1 + 200) = v48;
  if ( *v46 != -4096 )
    --*(_DWORD *)(a1 + 204);
  *v46 = v30;
  *((_DWORD *)v46 + 2) = 0;
  v49 = *(_DWORD *)(a1 + 208);
  if ( v49 )
  {
    v31 = v49 - 1;
    v32 = *(_QWORD *)(a1 + 192);
    v33 = v31 & (((unsigned int)v30 >> 4) ^ ((unsigned int)v30 >> 9));
    v34 = (__int64 *)(v32 + 16LL * v33);
    v35 = *v34;
    goto LABEL_36;
  }
  return v30;
}
