// Function: sub_1E93EA0
// Address: 0x1e93ea0
//
__int64 __fastcall sub_1E93EA0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 result; // rax
  size_t v4; // rbx
  const void *v5; // r9
  __int64 v6; // r13
  size_t v7; // r12
  const void *v8; // r8
  size_t v9; // rdx
  signed __int64 v10; // rax
  size_t v11; // r15
  const void *v12; // rsi
  size_t v13; // rdx
  int v14; // eax
  __int64 v15; // r12
  size_t v16; // rdx
  int v17; // eax
  __int64 v18; // rdx
  _QWORD *v19; // rbx
  _BYTE *v20; // r13
  size_t v21; // r14
  _QWORD *v22; // r12
  size_t v23; // r15
  size_t v24; // rdx
  int v25; // eax
  size_t v26; // rbx
  unsigned __int64 i; // r15
  size_t v28; // r14
  size_t v29; // rdx
  signed __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rbx
  size_t v33; // r15
  const void *v34; // rsi
  size_t v35; // rdx
  int v36; // eax
  __int64 v37; // rbx
  size_t v38; // rdx
  int v39; // eax
  __int64 v40; // r12
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // r12
  const __m128i *v44; // r14
  __int64 v45; // rdi
  __int64 v46; // rsi
  __int64 v47; // rcx
  __m128i *v48; // rdx
  __m128i v49; // xmm2
  __m128i *v50; // r15
  const void *v51; // r13
  __int64 v52; // rax
  __int64 v53; // rax
  _BYTE *v54; // rax
  __int64 v55; // r12
  size_t v56; // rdx
  __int64 v57; // [rsp+8h] [rbp-B8h]
  _QWORD *v58; // [rsp+10h] [rbp-B0h]
  const void *v60; // [rsp+20h] [rbp-A0h]
  const void *v61; // [rsp+20h] [rbp-A0h]
  _QWORD *v62; // [rsp+20h] [rbp-A0h]
  const void *v63; // [rsp+20h] [rbp-A0h]
  void *s2b; // [rsp+28h] [rbp-98h]
  _QWORD *s2; // [rsp+28h] [rbp-98h]
  __int64 s2a; // [rsp+28h] [rbp-98h]
  __m128i *v67; // [rsp+30h] [rbp-90h]
  __int64 v68; // [rsp+38h] [rbp-88h]
  __m128i v69; // [rsp+40h] [rbp-80h] BYREF
  __int64 v70; // [rsp+50h] [rbp-70h]
  __m128i v71; // [rsp+60h] [rbp-60h] BYREF
  __m128i v72; // [rsp+70h] [rbp-50h] BYREF
  __int64 v73; // [rsp+80h] [rbp-40h]

  result = (__int64)a2 - a1;
  v57 = a3;
  v58 = a2;
  if ( (__int64)a2 - a1 <= 640 )
    return result;
  if ( !a3 )
  {
    v62 = a2;
    goto LABEL_71;
  }
  while ( 2 )
  {
    --v57;
    v4 = *(_QWORD *)(a1 + 48);
    v5 = *(const void **)(a1 + 40);
    v6 = a1 + 40 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * (((__int64)v58 - a1) >> 3)) / 2);
    v7 = *(_QWORD *)(v6 + 8);
    v8 = *(const void **)v6;
    v9 = v7;
    if ( v4 <= v7 )
      v9 = *(_QWORD *)(a1 + 48);
    if ( !v9
      || (v60 = *(const void **)v6,
          s2b = *(void **)(a1 + 40),
          LODWORD(v10) = memcmp(s2b, *(const void **)v6, v9),
          v5 = s2b,
          v8 = v60,
          !(_DWORD)v10) )
    {
      v10 = v4 - v7;
      if ( (__int64)(v4 - v7) >= 0x80000000LL )
        goto LABEL_51;
      if ( v10 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_10;
    }
    if ( (int)v10 >= 0 )
    {
LABEL_51:
      v33 = *(v58 - 4);
      v34 = (const void *)*(v58 - 5);
      v35 = v33;
      if ( v4 <= v33 )
        v35 = v4;
      if ( !v35 || (v63 = v8, v36 = memcmp(v5, v34, v35), v8 = v63, !v36) )
      {
        v37 = v4 - v33;
        if ( v37 >= 0x80000000LL )
        {
LABEL_59:
          v38 = v33;
          if ( v7 <= v33 )
            v38 = v7;
          if ( !v38 || (v39 = memcmp(v8, v34, v38)) == 0 )
          {
            v40 = v7 - v33;
            if ( v40 >= 0x80000000LL )
              goto LABEL_68;
            if ( v40 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
              goto LABEL_67;
            v39 = v40;
          }
          if ( v39 < 0 )
            goto LABEL_67;
          goto LABEL_68;
        }
        if ( v37 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_23;
        v36 = v37;
      }
      if ( v36 < 0 )
        goto LABEL_23;
      goto LABEL_59;
    }
LABEL_10:
    v11 = *(v58 - 4);
    v12 = (const void *)*(v58 - 5);
    v13 = v11;
    if ( v7 <= v11 )
      v13 = v7;
    if ( !v13 || (v61 = v5, v14 = memcmp(v8, v12, v13), v5 = v61, !v14) )
    {
      v15 = v7 - v11;
      if ( v15 >= 0x80000000LL )
        goto LABEL_18;
      if ( v15 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v14 = v15;
        goto LABEL_17;
      }
LABEL_68:
      sub_22415E0(a1, v6);
      v42 = *(_QWORD *)(a1 + 32);
      *(_QWORD *)(a1 + 32) = *(_QWORD *)(v6 + 32);
      *(_QWORD *)(v6 + 32) = v42;
      goto LABEL_24;
    }
LABEL_17:
    if ( v14 < 0 )
      goto LABEL_68;
LABEL_18:
    v16 = v11;
    if ( v4 <= v11 )
      v16 = v4;
    if ( !v16 || (v17 = memcmp(v5, v12, v16)) == 0 )
    {
      v32 = v4 - v11;
      if ( v32 >= 0x80000000LL )
        goto LABEL_23;
      if ( v32 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
LABEL_67:
        sub_22415E0(a1, v58 - 5);
        v41 = *(_QWORD *)(a1 + 32);
        *(_QWORD *)(a1 + 32) = *(v58 - 1);
        *(v58 - 1) = v41;
        goto LABEL_24;
      }
      v17 = v32;
    }
    if ( v17 < 0 )
      goto LABEL_67;
LABEL_23:
    sub_22415E0(a1, a1 + 40);
    v18 = *(_QWORD *)(a1 + 72);
    *(_QWORD *)(a1 + 72) = *(_QWORD *)(a1 + 32);
    *(_QWORD *)(a1 + 32) = v18;
LABEL_24:
    v19 = (_QWORD *)(a1 + 40);
    v20 = *(_BYTE **)a1;
    v21 = *(_QWORD *)(a1 + 8);
    v22 = v58;
    while ( 1 )
    {
      v23 = v19[1];
      v24 = v21;
      v62 = v19;
      if ( v23 <= v21 )
        v24 = v19[1];
      if ( v24 )
      {
        v25 = memcmp((const void *)*v19, v20, v24);
        if ( v25 )
          break;
      }
      if ( (__int64)(v23 - v21) >= 0x80000000LL )
        goto LABEL_33;
      if ( (__int64)(v23 - v21) > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v25 = v23 - v21;
        break;
      }
LABEL_44:
      v19 += 5;
    }
    if ( v25 < 0 )
      goto LABEL_44;
LABEL_33:
    s2 = v19;
    v26 = v21;
    for ( i = (unsigned __int64)(v22 - 5); ; i -= 40LL )
    {
      v28 = *(_QWORD *)(i + 8);
      v29 = v26;
      v22 = (_QWORD *)i;
      if ( v28 <= v26 )
        v29 = *(_QWORD *)(i + 8);
      if ( !v29 )
        break;
      LODWORD(v30) = memcmp(v20, *(const void **)i, v29);
      if ( !(_DWORD)v30 )
        break;
LABEL_35:
      if ( (int)v30 >= 0 )
        goto LABEL_42;
LABEL_36:
      ;
    }
    v30 = v26 - v28;
    if ( (__int64)(v26 - v28) < 0x80000000LL )
    {
      if ( v30 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_36;
      goto LABEL_35;
    }
LABEL_42:
    v19 = s2;
    if ( (unsigned __int64)s2 < i )
    {
      sub_22415E0(s2, i);
      v31 = s2[4];
      s2[4] = *(_QWORD *)(i + 32);
      *(_QWORD *)(i + 32) = v31;
      v21 = *(_QWORD *)(a1 + 8);
      v20 = *(_BYTE **)a1;
      goto LABEL_44;
    }
    sub_1E93EA0(s2, v58, v57);
    result = (__int64)s2 - a1;
    if ( (__int64)s2 - a1 <= 640 )
      return result;
    if ( v57 )
    {
      v58 = s2;
      continue;
    }
    break;
  }
LABEL_71:
  s2a = 0xCCCCCCCCCCCCCCCDLL * (result >> 3);
  v43 = (s2a - 2) >> 1;
  v44 = (const __m128i *)(a1 + 40 * v43 + 16);
  while ( 2 )
  {
    v48 = (__m128i *)v44[-1].m128i_i64[0];
    if ( v48 == v44 )
    {
      v47 = v44[1].m128i_i64[0];
      v49 = _mm_loadu_si128(v44);
      v71.m128i_i64[0] = (__int64)&v72;
      v46 = v44[-1].m128i_i64[1];
      v44->m128i_i8[0] = 0;
      v44[-1].m128i_i64[1] = 0;
      v70 = v47;
      v69 = v49;
LABEL_80:
      v72 = _mm_load_si128(&v69);
    }
    else
    {
      v45 = v44->m128i_i64[0];
      v46 = v44[-1].m128i_i64[1];
      v44[-1].m128i_i64[0] = (__int64)v44;
      v47 = v44[1].m128i_i64[0];
      v69.m128i_i64[0] = v45;
      v44[-1].m128i_i64[1] = 0;
      v44->m128i_i8[0] = 0;
      v70 = v47;
      v71.m128i_i64[0] = (__int64)&v72;
      if ( v48 == &v69 )
        goto LABEL_80;
      v71.m128i_i64[0] = (__int64)v48;
      v72.m128i_i64[0] = v45;
    }
    v71.m128i_i64[1] = v46;
    v73 = v47;
    v69.m128i_i8[0] = 0;
    sub_1E937F0(a1, v43, s2a, &v71);
    if ( (__m128i *)v71.m128i_i64[0] != &v72 )
      j_j___libc_free_0(v71.m128i_i64[0], v72.m128i_i64[0] + 1);
    if ( v43 )
    {
      --v43;
      v44 = (const __m128i *)((char *)v44 - 40);
      continue;
    }
    break;
  }
  v50 = (__m128i *)(v62 - 3);
  v51 = (const void *)(a1 + 16);
  do
  {
    v67 = &v69;
    if ( (__m128i *)v50[-1].m128i_i64[0] == v50 )
    {
      v69 = _mm_loadu_si128(v50);
    }
    else
    {
      v67 = (__m128i *)v50[-1].m128i_i64[0];
      v69.m128i_i64[0] = v50->m128i_i64[0];
    }
    v52 = v50[-1].m128i_i64[1];
    v50[-1].m128i_i64[0] = (__int64)v50;
    v50[-1].m128i_i64[1] = 0;
    v68 = v52;
    v53 = v50[1].m128i_i64[0];
    v50->m128i_i8[0] = 0;
    v70 = v53;
    if ( *(const void **)a1 == v51 )
    {
      v56 = *(_QWORD *)(a1 + 8);
      if ( v56 )
      {
        if ( v56 == 1 )
          v50->m128i_i8[0] = *(_BYTE *)(a1 + 16);
        else
          memcpy(v50, v51, v56);
        v56 = *(_QWORD *)(a1 + 8);
      }
      v50[-1].m128i_i64[1] = v56;
      v50->m128i_i8[v56] = 0;
      v54 = *(_BYTE **)a1;
    }
    else
    {
      v50[-1].m128i_i64[0] = *(_QWORD *)a1;
      v50[-1].m128i_i64[1] = *(_QWORD *)(a1 + 8);
      v50->m128i_i64[0] = *(_QWORD *)(a1 + 16);
      v54 = (_BYTE *)(a1 + 16);
      *(_QWORD *)a1 = v51;
    }
    *(_QWORD *)(a1 + 8) = 0;
    *v54 = 0;
    v50[1].m128i_i64[0] = *(_QWORD *)(a1 + 32);
    v71.m128i_i64[0] = (__int64)&v72;
    if ( v67 == &v69 )
    {
      v72 = _mm_load_si128(&v69);
    }
    else
    {
      v71.m128i_i64[0] = (__int64)v67;
      v72.m128i_i64[0] = v69.m128i_i64[0];
    }
    v55 = (__int64)v50[-1].m128i_i64 - a1;
    v69.m128i_i8[0] = 0;
    v71.m128i_i64[1] = v68;
    v73 = v70;
    result = sub_1E937F0(a1, 0, 0xCCCCCCCCCCCCCCCDLL * (v55 >> 3), &v71);
    if ( (__m128i *)v71.m128i_i64[0] != &v72 )
      result = j_j___libc_free_0(v71.m128i_i64[0], v72.m128i_i64[0] + 1);
    v50 = (__m128i *)((char *)v50 - 40);
  }
  while ( v55 > 40 );
  return result;
}
