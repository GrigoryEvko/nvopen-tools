// Function: sub_38B39D0
// Address: 0x38b39d0
//
__int64 __fastcall sub_38B39D0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rbx
  unsigned int v4; // r15d
  void *v6; // r15
  unsigned __int64 v7; // r14
  __m128i *v8; // rax
  __int64 v9; // r14
  __m128i *v10; // r15
  __int64 v11; // r12
  size_t v12; // rbx
  size_t v13; // r13
  size_t v14; // rdx
  int v15; // eax
  size_t v16; // r9
  size_t v17; // rcx
  size_t v18; // rdx
  int v19; // eax
  __int64 v20; // r9
  _QWORD *v21; // rsi
  size_t v22; // r15
  __int64 v23; // rax
  _QWORD *v24; // rdx
  _QWORD *v25; // r8
  unsigned int v26; // edi
  __int64 v27; // rax
  __int64 v28; // rsi
  __m128i *v29; // rdi
  unsigned __int64 v30; // rdi
  __int64 v31; // rax
  _QWORD **v32; // rbx
  _QWORD *v33; // r12
  size_t v34; // r14
  int *v35; // rax
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // r13
  size_t v38; // rcx
  size_t v39; // rdx
  int v40; // eax
  unsigned int v41; // edi
  __int64 v42; // [rsp+10h] [rbp-130h]
  size_t v43; // [rsp+18h] [rbp-128h]
  size_t v44; // [rsp+18h] [rbp-128h]
  __int64 v45; // [rsp+20h] [rbp-120h]
  size_t v46; // [rsp+20h] [rbp-120h]
  _QWORD *v47; // [rsp+20h] [rbp-120h]
  _QWORD *v48; // [rsp+28h] [rbp-118h]
  _QWORD *v49; // [rsp+28h] [rbp-118h]
  _QWORD *v50; // [rsp+30h] [rbp-110h]
  __int64 v51; // [rsp+30h] [rbp-110h]
  __int64 v52; // [rsp+30h] [rbp-110h]
  _QWORD **v53; // [rsp+30h] [rbp-110h]
  int *v54; // [rsp+38h] [rbp-108h]
  unsigned __int64 v55[2]; // [rsp+40h] [rbp-100h] BYREF
  void *src; // [rsp+50h] [rbp-F0h] BYREF
  size_t n; // [rsp+58h] [rbp-E8h]
  _BYTE v58[16]; // [rsp+60h] [rbp-E0h] BYREF
  void *s2; // [rsp+70h] [rbp-D0h] BYREF
  size_t v60; // [rsp+78h] [rbp-C8h]
  __m128i v61[12]; // [rsp+80h] [rbp-C0h] BYREF

  v2 = a2;
  v3 = a1;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  src = v58;
  n = 0;
  v58[0] = 0;
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here")
    || (unsigned __int8)sub_388AF10(a1, 307, "expected 'name' here")
    || (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388B0A0(a1, (unsigned __int64 *)&src) )
  {
LABEL_2:
    v4 = 1;
    goto LABEL_3;
  }
  v6 = src;
  v50 = *(_QWORD **)(a1 + 184);
  if ( !src )
  {
    v61[0].m128i_i8[0] = 0;
    s2 = v61;
    v60 = 0;
    goto LABEL_15;
  }
  v7 = n;
  s2 = v61;
  v55[0] = n;
  if ( n > 0xF )
  {
    s2 = (void *)sub_22409D0((__int64)&s2, v55, 0);
    v29 = (__m128i *)s2;
    v61[0].m128i_i64[0] = v55[0];
    goto LABEL_60;
  }
  if ( n != 1 )
  {
    if ( !n )
    {
      v8 = v61;
      goto LABEL_14;
    }
    v29 = v61;
LABEL_60:
    memcpy(v29, v6, v7);
    v7 = v55[0];
    v8 = (__m128i *)s2;
    goto LABEL_14;
  }
  v61[0].m128i_i8[0] = *(_BYTE *)src;
  v8 = v61;
LABEL_14:
  v60 = v7;
  v8->m128i_i8[v7] = 0;
LABEL_15:
  v9 = (__int64)(v50 + 11);
  v48 = v50 + 11;
  if ( !v50[12] )
  {
    v9 = (__int64)(v50 + 11);
    goto LABEL_37;
  }
  v45 = v3;
  v10 = (__m128i *)s2;
  v11 = v50[12];
  v12 = v60;
  do
  {
    v13 = *(_QWORD *)(v11 + 40);
    v14 = v12;
    if ( v13 <= v12 )
      v14 = *(_QWORD *)(v11 + 40);
    if ( !v14 || (v15 = memcmp(*(const void **)(v11 + 32), v10, v14)) == 0 )
    {
      if ( (__int64)(v13 - v12) >= 0x80000000LL )
        goto LABEL_27;
      if ( (__int64)(v13 - v12) <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_17;
      v15 = v13 - v12;
    }
    if ( v15 < 0 )
    {
LABEL_17:
      v11 = *(_QWORD *)(v11 + 24);
      continue;
    }
LABEL_27:
    v9 = v11;
    v11 = *(_QWORD *)(v11 + 16);
  }
  while ( v11 );
  v16 = v12;
  v2 = a2;
  v3 = v45;
  if ( v48 == (_QWORD *)v9 )
    goto LABEL_37;
  v17 = *(_QWORD *)(v9 + 40);
  v18 = v16;
  if ( v17 <= v16 )
    v18 = *(_QWORD *)(v9 + 40);
  if ( v18
    && (v43 = v16,
        v46 = *(_QWORD *)(v9 + 40),
        v19 = memcmp(v10, *(const void **)(v9 + 32), v18),
        v17 = v46,
        v16 = v43,
        v19) )
  {
LABEL_36:
    if ( v19 < 0 )
      goto LABEL_37;
  }
  else
  {
    v20 = v16 - v17;
    if ( v20 <= 0x7FFFFFFF )
    {
      if ( v20 >= (__int64)0xFFFFFFFF80000000LL )
      {
        v19 = v20;
        goto LABEL_36;
      }
LABEL_37:
      v21 = (_QWORD *)v9;
      v9 = sub_22077B0(0x98u);
      *(_QWORD *)(v9 + 32) = v9 + 48;
      if ( s2 == v61 )
      {
        *(__m128i *)(v9 + 48) = _mm_load_si128(v61);
      }
      else
      {
        *(_QWORD *)(v9 + 32) = s2;
        *(_QWORD *)(v9 + 48) = v61[0].m128i_i64[0];
      }
      v22 = v60;
      v60 = 0;
      s2 = v61;
      *(_QWORD *)(v9 + 40) = v22;
      memset((void *)(v9 + 64), 0, 0x58u);
      v61[0].m128i_i8[0] = 0;
      *(_QWORD *)(v9 + 128) = v9 + 112;
      *(_QWORD *)(v9 + 136) = v9 + 112;
      v23 = sub_14F61B0(v50 + 10, v21, v9 + 32);
      v25 = v24;
      if ( v24 )
      {
        if ( v23 || v48 == v24 )
        {
LABEL_42:
          LOBYTE(v26) = 1;
          goto LABEL_43;
        }
        v39 = v24[5];
        v38 = v39;
        if ( v22 <= v39 )
          v39 = v22;
        if ( v39
          && (v44 = v38,
              v47 = v25,
              v40 = memcmp(*(const void **)(v9 + 32), (const void *)v25[4], v39),
              v25 = v47,
              v38 = v44,
              (v41 = v40) != 0) )
        {
LABEL_82:
          v26 = v41 >> 31;
        }
        else
        {
          LOBYTE(v26) = 0;
          if ( (__int64)(v22 - v38) <= 0x7FFFFFFF )
          {
            if ( (__int64)(v22 - v38) < (__int64)0xFFFFFFFF80000000LL )
              goto LABEL_42;
            v41 = v22 - v38;
            goto LABEL_82;
          }
        }
LABEL_43:
        sub_220F040(v26, v9, v25, v48);
        ++v50[15];
      }
      else
      {
        v51 = v23;
        sub_38888B0(0);
        v30 = *(_QWORD *)(v9 + 32);
        v31 = v51;
        if ( v9 + 48 != v30 )
        {
          j_j___libc_free_0(v30);
          v31 = v51;
        }
        v52 = v31;
        j_j___libc_free_0(v9);
        v9 = v52;
      }
      v10 = (__m128i *)s2;
    }
  }
  if ( v10 != v61 )
    j_j___libc_free_0((unsigned __int64)v10);
  if ( (unsigned __int8)sub_388AF10(v3, 4, "expected ',' here") )
    goto LABEL_2;
  if ( (unsigned __int8)sub_38B3910(v3, v9 + 64) )
    goto LABEL_2;
  v4 = sub_388AF10(v3, 13, "expected ')' here");
  if ( (_BYTE)v4 )
    goto LABEL_2;
  v49 = (_QWORD *)(v3 + 1352);
  v27 = *(_QWORD *)(v3 + 1360);
  if ( v27 )
  {
    v28 = v3 + 1352;
    do
    {
      if ( *(_DWORD *)(v27 + 32) < v2 )
      {
        v27 = *(_QWORD *)(v27 + 24);
      }
      else
      {
        v28 = v27;
        v27 = *(_QWORD *)(v27 + 16);
      }
    }
    while ( v27 );
    if ( (_QWORD *)v28 != v49 && *(_DWORD *)(v28 + 32) <= v2 )
    {
      v53 = *(_QWORD ***)(v28 + 48);
      if ( v53 != *(_QWORD ***)(v28 + 40) )
      {
        v42 = v3;
        v32 = *(_QWORD ***)(v28 + 40);
        do
        {
          v33 = *v32;
          v32 += 2;
          v34 = n;
          v54 = (int *)src;
          sub_16C1840(&s2);
          sub_16C1A90((int *)&s2, v54, v34);
          sub_16C1AA0(&s2, v55);
          *v33 = v55[0];
        }
        while ( v53 != v32 );
        v4 = 0;
        v3 = v42;
      }
      v35 = sub_220F330((int *)v28, v49);
      v36 = *((_QWORD *)v35 + 5);
      v37 = (unsigned __int64)v35;
      if ( v36 )
        j_j___libc_free_0(v36);
      j_j___libc_free_0(v37);
      --*(_QWORD *)(v3 + 1384);
    }
  }
LABEL_3:
  if ( src != v58 )
    j_j___libc_free_0((unsigned __int64)src);
  return v4;
}
