// Function: sub_1E94C70
// Address: 0x1e94c70
//
__int64 __fastcall sub_1E94C70(__int64 **a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 *v4; // rax
  __int64 *v5; // rbx
  __int8 *v7; // r9
  size_t v8; // rcx
  __m128i *v9; // rax
  __m128i *v10; // rsi
  __m128i *v11; // rdi
  __int64 v12; // r13
  unsigned __int64 v13; // rax
  const void *v14; // r10
  size_t v15; // r9
  __m128i *v16; // rax
  __int64 v17; // rax
  __m128i *v18; // rdi
  __m128i *v19; // rbx
  __m128i *v20; // r14
  signed __int64 v21; // r13
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rsi
  __int64 v24; // rdi
  unsigned __int64 *v25; // rdx
  __m128i *v26; // r14
  __int64 v27; // r13
  __int64 v28; // rax
  unsigned __int64 *v29; // rax
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // rax
  __m128i *v32; // r12
  __int64 v34; // rax
  __m128i *v35; // rdi
  __m128i *v36; // r13
  size_t n; // [rsp+8h] [rbp-118h]
  size_t na; // [rsp+8h] [rbp-118h]
  __int8 *src; // [rsp+10h] [rbp-110h]
  void *srca; // [rsp+10h] [rbp-110h]
  __int64 *v41; // [rsp+20h] [rbp-100h]
  __m128i *v42; // [rsp+30h] [rbp-F0h] BYREF
  __m128i *v43; // [rsp+38h] [rbp-E8h]
  __m128i *v44; // [rsp+40h] [rbp-E0h]
  _QWORD *v45; // [rsp+50h] [rbp-D0h] BYREF
  unsigned __int64 v46; // [rsp+58h] [rbp-C8h]
  _QWORD v47[2]; // [rsp+60h] [rbp-C0h] BYREF
  __m128i *v48; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v49; // [rsp+78h] [rbp-A8h]
  __m128i v50; // [rsp+80h] [rbp-A0h] BYREF
  __m128i v51; // [rsp+90h] [rbp-90h] BYREF
  __m128i v52; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v53; // [rsp+B0h] [rbp-70h]
  void *v54; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v55; // [rsp+C8h] [rbp-58h]
  __int64 v56; // [rsp+D0h] [rbp-50h]
  __int64 v57; // [rsp+D8h] [rbp-48h]
  int v58; // [rsp+E0h] [rbp-40h]
  _QWORD *v59; // [rsp+E8h] [rbp-38h]

  v3 = 0;
  v4 = a1[1];
  v5 = *a1;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v41 = v4;
  if ( v5 == v4 )
    return v3;
  do
  {
    v12 = *v5;
    v45 = v47;
    v46 = 0;
    v54 = &unk_49EFBE0;
    LOBYTE(v47[0]) = 0;
    v58 = 1;
    v57 = 0;
    v56 = 0;
    v55 = 0;
    v59 = &v45;
    sub_1E1A330(v12, (__int64)&v54, 1, 0, 0, 1, 0);
    if ( v57 != v55 )
      sub_16E7BA0((__int64 *)&v54);
    v13 = sub_22416F0(&v45, "=", 0, 1);
    if ( v13 != -1 )
    {
      if ( v13 > v46 )
        sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
      v48 = &v50;
      v7 = (char *)v45 + v13;
      if ( (_QWORD *)((char *)v45 + v46) && !v7 )
LABEL_75:
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v8 = v46 - v13;
      v51.m128i_i64[0] = v46 - v13;
      if ( v46 - v13 > 0xF )
      {
        n = v46 - v13;
        src = (char *)v45 + v13;
        v17 = sub_22409D0(&v48, &v51, 0);
        v7 = src;
        v8 = n;
        v48 = (__m128i *)v17;
        v18 = (__m128i *)v17;
        v50.m128i_i64[0] = v51.m128i_i64[0];
      }
      else
      {
        if ( v8 == 1 )
        {
          v50.m128i_i8[0] = *v7;
          v9 = &v50;
LABEL_9:
          v49 = v8;
          v9->m128i_i8[v8] = 0;
          goto LABEL_10;
        }
        if ( !v8 )
        {
          v9 = &v50;
          goto LABEL_9;
        }
        v18 = &v50;
      }
      memcpy(v18, v7, v8);
      v8 = v51.m128i_i64[0];
      v9 = v48;
      goto LABEL_9;
    }
    v14 = v45;
    v15 = v46;
    v48 = &v50;
    if ( (_QWORD *)((char *)v45 + v46) && !v45 )
      goto LABEL_75;
    v51.m128i_i64[0] = v46;
    if ( v46 > 0xF )
    {
      na = v46;
      srca = v45;
      v34 = sub_22409D0(&v48, &v51, 0);
      v14 = srca;
      v15 = na;
      v48 = (__m128i *)v34;
      v35 = (__m128i *)v34;
      v50.m128i_i64[0] = v51.m128i_i64[0];
      goto LABEL_67;
    }
    if ( v46 != 1 )
    {
      if ( !v46 )
      {
        v16 = &v50;
        goto LABEL_33;
      }
      v35 = &v50;
LABEL_67:
      memcpy(v35, v14, v15);
      v15 = v51.m128i_i64[0];
      v16 = v48;
      goto LABEL_33;
    }
    v50.m128i_i8[0] = *(_BYTE *)v45;
    v16 = &v50;
LABEL_33:
    v49 = v15;
    v16->m128i_i8[v15] = 0;
LABEL_10:
    v51.m128i_i64[0] = (__int64)&v52;
    if ( v48 == &v50 )
    {
      v52 = _mm_load_si128(&v50);
    }
    else
    {
      v51.m128i_i64[0] = (__int64)v48;
      v52.m128i_i64[0] = v50.m128i_i64[0];
    }
    v10 = v43;
    v53 = v12;
    v48 = &v50;
    v51.m128i_i64[1] = v49;
    v49 = 0;
    v50.m128i_i8[0] = 0;
    if ( v43 == v44 )
    {
      sub_1E949D0(&v42, v43, &v51);
      v11 = (__m128i *)v51.m128i_i64[0];
    }
    else
    {
      v11 = (__m128i *)v51.m128i_i64[0];
      if ( v43 )
      {
        v43->m128i_i64[0] = (__int64)v43[1].m128i_i64;
        if ( (__m128i *)v51.m128i_i64[0] == &v52 )
        {
          v10[1] = _mm_load_si128(&v52);
        }
        else
        {
          v10->m128i_i64[0] = v51.m128i_i64[0];
          v10[1].m128i_i64[0] = v52.m128i_i64[0];
        }
        v10->m128i_i64[1] = v51.m128i_i64[1];
        v51.m128i_i64[0] = (__int64)&v52;
        v51.m128i_i64[1] = 0;
        v11 = &v52;
        v52.m128i_i8[0] = 0;
        v10[2].m128i_i64[0] = v53;
        v10 = v43;
      }
      v43 = (__m128i *)((char *)v10 + 40);
    }
    if ( v11 != &v52 )
      j_j___libc_free_0(v11, v52.m128i_i64[0] + 1);
    if ( v48 != &v50 )
      j_j___libc_free_0(v48, v50.m128i_i64[0] + 1);
    sub_16E7BC0((__int64 *)&v54);
    if ( v45 != v47 )
      j_j___libc_free_0(v45, v47[0] + 1LL);
    ++v5;
  }
  while ( v41 != v5 );
  v19 = v43;
  v20 = v42;
  if ( v42 == v43 )
    goto LABEL_71;
  v21 = (char *)v43 - (char *)v42;
  _BitScanReverse64(&v22, 0xCCCCCCCCCCCCCCCDLL * (((char *)v43 - (char *)v42) >> 3));
  sub_1E93EA0((__int64)v42, v43, 2LL * (int)(63 - (v22 ^ 0x3F)));
  if ( v21 > 640 )
  {
    v36 = v20 + 40;
    v24 = (__int64)v20;
    v23 = (unsigned __int64)&v20[40];
    sub_1E934D0((__int64)v20, v20[40].m128i_i8);
    if ( v19 != &v20[40] )
    {
      do
      {
        v24 = (__int64)v36;
        v36 = (__m128i *)((char *)v36 + 40);
        sub_1E93240((__m128i *)v24);
      }
      while ( v19 != v36 );
    }
  }
  else
  {
    v23 = (unsigned __int64)v19;
    v24 = (__int64)v20;
    sub_1E934D0((__int64)v20, v19->m128i_i8);
  }
  v19 = v43;
  v26 = v42;
  if ( v42 == v43 )
  {
LABEL_71:
    v3 = 0;
  }
  else
  {
    do
    {
      v27 = v26[2].m128i_i64[0];
      if ( !*(_QWORD *)(a3 + 16) )
        sub_4263D6(v24, v23, v25);
      v24 = a3;
      v25 = (unsigned __int64 *)(*(__int64 (__fastcall **)(__int64))(a3 + 24))(a3);
      if ( (unsigned __int64 *)v27 != v25 )
      {
        if ( !v27 )
          BUG();
        v28 = v27;
        if ( (*(_QWORD *)v27 & 4) == 0 && (*(_BYTE *)(v27 + 46) & 8) != 0 )
        {
          do
            v28 = *(_QWORD *)(v28 + 8);
          while ( (*(_BYTE *)(v28 + 46) & 8) != 0 );
        }
        v29 = *(unsigned __int64 **)(v28 + 8);
        if ( (unsigned __int64 *)v27 != v29 && v25 != v29 && v29 != (unsigned __int64 *)v27 )
        {
          v30 = *v29;
          *(_QWORD *)((*(_QWORD *)v27 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v29;
          v23 = v30 & 0xFFFFFFFFFFFFFFF8LL;
          v24 = *v29 & 7;
          *v29 = v24 | *(_QWORD *)v27 & 0xFFFFFFFFFFFFFFF8LL;
          v31 = *v25;
          *(_QWORD *)(v23 + 8) = v25;
          v31 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)v27 = v31 | *(_QWORD *)v27 & 7LL;
          *(_QWORD *)(v31 + 8) = v27;
          *v25 = v23 | *v25 & 7;
        }
      }
      v26 = (__m128i *)((char *)v26 + 40);
    }
    while ( v26 != v19 );
    v32 = v43;
    v19 = v42;
    if ( v43 == v42 )
    {
      v3 = 1;
    }
    else
    {
      do
      {
        if ( (__m128i *)v19->m128i_i64[0] != &v19[1] )
          j_j___libc_free_0(v19->m128i_i64[0], v19[1].m128i_i64[0] + 1);
        v19 = (__m128i *)((char *)v19 + 40);
      }
      while ( v32 != v19 );
      v19 = v42;
      v3 = 1;
    }
  }
  if ( v19 )
    j_j___libc_free_0(v19, (char *)v44 - (char *)v19);
  return v3;
}
