// Function: sub_33F8870
// Address: 0x33f8870
//
__int64 __fastcall sub_33F8870(_QWORD *a1, const char *a2, unsigned int a3, __int64 a4, unsigned __int32 a5)
{
  size_t v5; // rax
  size_t v6; // r12
  __m128i *v7; // rdx
  size_t v8; // r13
  __m128i *v9; // r15
  size_t v10; // r14
  int v11; // eax
  size_t v12; // rdx
  signed __int64 v13; // rax
  signed __int64 v14; // rax
  size_t v15; // rbx
  const void *v16; // r12
  __int64 v17; // r12
  size_t v18; // rbx
  const void *v19; // r13
  size_t v20; // rdx
  int v21; // eax
  signed __int64 v22; // rax
  signed __int64 v23; // rax
  __int64 v24; // r15
  __m128i *v25; // rax
  __m128i *v26; // rbx
  size_t v27; // r14
  unsigned __int32 v28; // r13d
  __int64 v29; // rax
  _QWORD *v30; // rdx
  __int64 v31; // r15
  size_t v32; // r8
  char v33; // di
  __int64 result; // rax
  __m128i *v35; // rdi
  size_t v36; // rbx
  const void *v37; // r9
  const void *v38; // r15
  size_t v39; // rdx
  int v40; // eax
  int v41; // eax
  __int64 v42; // rbx
  unsigned __int64 v43; // rdi
  unsigned __int64 v44; // rdi
  __m128i *v45; // rax
  __int64 v46; // r13
  __m128i *v47; // r14
  int v48; // edx
  int v49; // ebx
  unsigned __int8 *v50; // rsi
  signed __int64 v51; // rax
  __int64 v52; // rcx
  unsigned __int64 v53; // rax
  size_t v54; // [rsp+8h] [rbp-C8h]
  _QWORD *v58; // [rsp+30h] [rbp-A0h]
  size_t v61; // [rsp+50h] [rbp-80h]
  size_t v62; // [rsp+50h] [rbp-80h]
  size_t na; // [rsp+58h] [rbp-78h]
  size_t n; // [rsp+58h] [rbp-78h]
  size_t nb; // [rsp+58h] [rbp-78h]
  size_t nc; // [rsp+58h] [rbp-78h]
  __int64 v67; // [rsp+68h] [rbp-68h] BYREF
  void *s2; // [rsp+70h] [rbp-60h] BYREF
  size_t v69; // [rsp+78h] [rbp-58h]
  __m128i v70; // [rsp+80h] [rbp-50h] BYREF
  unsigned __int32 v71; // [rsp+90h] [rbp-40h]

  s2 = &v70;
  if ( !a2 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v5 = strlen(a2);
  v67 = v5;
  v6 = v5;
  if ( v5 > 0xF )
  {
    s2 = (void *)sub_22409D0((__int64)&s2, (unsigned __int64 *)&v67, 0);
    v35 = (__m128i *)s2;
    v70.m128i_i64[0] = v67;
    goto LABEL_54;
  }
  if ( v5 != 1 )
  {
    if ( !v5 )
    {
      v7 = &v70;
      goto LABEL_5;
    }
    v35 = &v70;
LABEL_54:
    memcpy(v35, a2, v6);
    v5 = v67;
    v7 = (__m128i *)s2;
    goto LABEL_5;
  }
  v70.m128i_i8[0] = *a2;
  v7 = &v70;
LABEL_5:
  v69 = v5;
  v7->m128i_i8[v5] = 0;
  v71 = a5;
  v8 = a1[120];
  v58 = a1 + 119;
  if ( !v8 )
  {
    v17 = (__int64)(a1 + 119);
    goto LABEL_41;
  }
  v9 = (__m128i *)s2;
  v10 = v69;
  v61 = (size_t)(a1 + 119);
  do
  {
    while ( 1 )
    {
      v15 = *(_QWORD *)(v8 + 40);
      v12 = v10;
      v16 = *(const void **)(v8 + 32);
      if ( v15 <= v10 )
        v12 = *(_QWORD *)(v8 + 40);
      if ( !v12 )
      {
        v13 = v15 - v10;
        if ( (__int64)(v15 - v10) >= 0x80000000LL )
          goto LABEL_13;
LABEL_9:
        if ( v13 <= (__int64)0xFFFFFFFF7FFFFFFFLL || (int)v13 < 0 )
          goto LABEL_26;
        if ( !v12 )
          goto LABEL_13;
        goto LABEL_12;
      }
      na = v12;
      v11 = memcmp(*(const void **)(v8 + 32), v9, v12);
      v12 = na;
      if ( v11 )
      {
        if ( v11 < 0 )
          goto LABEL_26;
      }
      else
      {
        v13 = v15 - v10;
        if ( (__int64)(v15 - v10) < 0x80000000LL )
          goto LABEL_9;
      }
LABEL_12:
      LODWORD(v14) = memcmp(v9, v16, v12);
      if ( (_DWORD)v14 )
        break;
LABEL_13:
      v14 = v10 - v15;
      if ( (__int64)(v10 - v15) >= 0x80000000LL )
        goto LABEL_16;
      if ( v14 > (__int64)0xFFFFFFFF7FFFFFFFLL )
        break;
LABEL_17:
      v61 = v8;
      v8 = *(_QWORD *)(v8 + 16);
      if ( !v8 )
        goto LABEL_27;
    }
    if ( (int)v14 < 0 )
      goto LABEL_17;
LABEL_16:
    if ( *(_DWORD *)(v8 + 64) >= a5 )
      goto LABEL_17;
LABEL_26:
    v8 = *(_QWORD *)(v8 + 24);
  }
  while ( v8 );
LABEL_27:
  v17 = v61;
  if ( v58 == (_QWORD *)v61 )
    goto LABEL_41;
  v18 = *(_QWORD *)(v61 + 40);
  v19 = *(const void **)(v61 + 32);
  v20 = v18;
  if ( v10 <= v18 )
    v20 = v10;
  if ( !v20 )
  {
    v22 = v10 - v18;
    if ( (__int64)(v10 - v18) <= 0x7FFFFFFF )
      goto LABEL_33;
    goto LABEL_37;
  }
  n = v20;
  v21 = memcmp(v9, *(const void **)(v61 + 32), v20);
  v20 = n;
  if ( v21 )
  {
    if ( v21 < 0 )
      goto LABEL_41;
    LODWORD(v23) = memcmp(v19, v9, n);
    if ( !(_DWORD)v23 )
      goto LABEL_37;
LABEL_39:
    if ( (int)v23 < 0 )
      goto LABEL_49;
LABEL_40:
    if ( *(_DWORD *)(v61 + 64) <= a5 )
      goto LABEL_49;
LABEL_41:
    v24 = v17;
    v25 = (__m128i *)sub_22077B0(0x50u);
    v26 = v25 + 3;
    v17 = (__int64)v25;
    v25[2].m128i_i64[0] = (__int64)v25[3].m128i_i64;
    if ( s2 == &v70 )
    {
      v25[3] = _mm_load_si128(&v70);
    }
    else
    {
      v25[2].m128i_i64[0] = (__int64)s2;
      v25[3].m128i_i64[0] = v70.m128i_i64[0];
    }
    v27 = v69;
    v28 = v71;
    v70.m128i_i8[0] = 0;
    v69 = 0;
    v25[2].m128i_i64[1] = v27;
    s2 = &v70;
    v25[4].m128i_i32[0] = v28;
    v25[4].m128i_i64[1] = 0;
    v29 = sub_33F8310(a1 + 118, v24, (__int64)v25[2].m128i_i64);
    v31 = v29;
    v32 = (size_t)v30;
    if ( !v30 )
    {
      v43 = *(_QWORD *)(v17 + 32);
      if ( v26 != (__m128i *)v43 )
        j_j___libc_free_0(v43);
      v44 = v17;
      v17 = v31;
      j_j___libc_free_0(v44);
      goto LABEL_48;
    }
    if ( v58 == v30 || v29 )
    {
      v33 = 1;
LABEL_47:
      sub_220F040(v33, v17, (_QWORD *)v32, v58);
      ++a1[123];
LABEL_48:
      v9 = (__m128i *)s2;
      goto LABEL_49;
    }
    v37 = (const void *)v30[4];
    v38 = *(const void **)(v17 + 32);
    v39 = v30[5];
    v36 = v39;
    if ( v27 <= v39 )
      v39 = v27;
    if ( v39 )
    {
      v54 = v32;
      v62 = v39;
      nb = (size_t)v37;
      v40 = memcmp(*(const void **)(v17 + 32), v37, v39);
      v37 = (const void *)nb;
      v39 = v62;
      v32 = v54;
      if ( v40 )
      {
        v33 = 1;
        if ( v40 < 0 )
          goto LABEL_47;
LABEL_65:
        nc = v32;
        v41 = memcmp(v37, v38, v39);
        v32 = nc;
        if ( v41 )
          goto LABEL_69;
        goto LABEL_66;
      }
      v51 = v27 - v36;
      if ( (__int64)(v27 - v36) > 0x7FFFFFFF )
        goto LABEL_65;
    }
    else
    {
      v51 = v27 - v36;
      if ( (__int64)(v27 - v36) > 0x7FFFFFFF )
        goto LABEL_66;
    }
    v33 = 1;
    if ( v51 < (__int64)0xFFFFFFFF80000000LL || (int)v51 < 0 )
      goto LABEL_47;
    if ( v39 )
      goto LABEL_65;
LABEL_66:
    v42 = v36 - v27;
    if ( v42 > 0x7FFFFFFF )
      goto LABEL_70;
    if ( v42 < (__int64)0xFFFFFFFF80000000LL )
    {
      v33 = 0;
      goto LABEL_47;
    }
    v41 = v42;
LABEL_69:
    v33 = 0;
    if ( v41 < 0 )
      goto LABEL_47;
LABEL_70:
    v33 = v28 < *(_DWORD *)(v32 + 64);
    goto LABEL_47;
  }
  v22 = v10 - v18;
  if ( (__int64)(v10 - v18) > 0x7FFFFFFF )
    goto LABEL_36;
LABEL_33:
  if ( v22 < (__int64)0xFFFFFFFF80000000LL || (int)v22 < 0 )
    goto LABEL_41;
  if ( !v20 )
    goto LABEL_37;
LABEL_36:
  LODWORD(v23) = memcmp(v19, v9, v20);
  if ( (_DWORD)v23 )
    goto LABEL_39;
LABEL_37:
  v23 = v18 - v10;
  if ( (__int64)(v18 - v10) > 0x7FFFFFFF )
    goto LABEL_40;
  if ( v23 >= (__int64)0xFFFFFFFF80000000LL )
    goto LABEL_39;
LABEL_49:
  if ( v9 != &v70 )
    j_j___libc_free_0((unsigned __int64)v9);
  result = *(_QWORD *)(v17 + 72);
  if ( !result )
  {
    v45 = sub_33ED250((__int64)a1, a3, a4);
    v46 = a1[52];
    v47 = v45;
    v49 = v48;
    if ( v46 )
    {
      a1[52] = *(_QWORD *)v46;
    }
    else
    {
      v52 = a1[53];
      a1[63] += 120LL;
      v53 = (v52 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( a1[54] >= v53 + 120 && v52 )
      {
        a1[53] = v53 + 120;
        if ( !v53 )
        {
LABEL_79:
          *(_QWORD *)(v17 + 72) = v46;
          sub_33CC420((__int64)a1, v46);
          return *(_QWORD *)(v17 + 72);
        }
      }
      else
      {
        v53 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
      }
      v46 = v53;
    }
    s2 = 0;
    *(_QWORD *)v46 = 0;
    *(_QWORD *)(v46 + 8) = 0;
    *(_QWORD *)(v46 + 16) = 0;
    *(_QWORD *)(v46 + 24) = 42;
    *(_WORD *)(v46 + 34) = -1;
    *(_DWORD *)(v46 + 36) = -1;
    *(_QWORD *)(v46 + 40) = 0;
    *(_QWORD *)(v46 + 48) = v47;
    *(_QWORD *)(v46 + 56) = 0;
    *(_DWORD *)(v46 + 64) = 0;
    *(_DWORD *)(v46 + 68) = v49;
    *(_DWORD *)(v46 + 72) = 0;
    v50 = (unsigned __int8 *)s2;
    *(_QWORD *)(v46 + 80) = s2;
    if ( v50 )
      sub_B976B0((__int64)&s2, v50, v46 + 80);
    *(_QWORD *)(v46 + 88) = 0xFFFFFFFFLL;
    *(_WORD *)(v46 + 32) = 0;
    *(_QWORD *)(v46 + 96) = a2;
    *(_DWORD *)(v46 + 104) = a5;
    goto LABEL_79;
  }
  return result;
}
