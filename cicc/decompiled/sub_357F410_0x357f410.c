// Function: sub_357F410
// Address: 0x357f410
//
__int64 __fastcall sub_357F410(__int64 **a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r13d
  __int64 *v4; // rax
  __int64 *v5; // rbx
  __int8 *v6; // r9
  size_t v7; // rcx
  __m128i *v8; // rax
  __m128i *v9; // rsi
  __m128i *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r13
  char *v15; // rax
  const void *v16; // r10
  size_t v17; // r9
  __m128i *v18; // rax
  __int64 v19; // rax
  __m128i *v20; // rdi
  __m128i *v21; // r14
  __m128i *v22; // r12
  __int64 v23; // r13
  signed __int64 v24; // rbx
  unsigned __int64 v25; // rax
  __m128i *v26; // rsi
  __m128i *v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // r12
  __m128i *v30; // r13
  unsigned __int64 *v31; // r15
  unsigned __int64 *v32; // rbx
  unsigned __int64 *v33; // rax
  unsigned __int64 *v34; // r14
  unsigned __int64 v35; // rcx
  unsigned __int64 v36; // rax
  const __m128i *v37; // rbx
  __int64 v39; // rax
  __m128i *v40; // rdi
  __m128i *v41; // rbx
  size_t n; // [rsp+10h] [rbp-120h]
  size_t na; // [rsp+10h] [rbp-120h]
  __int8 *src; // [rsp+18h] [rbp-118h]
  void *srca; // [rsp+18h] [rbp-118h]
  __int64 *v48; // [rsp+20h] [rbp-110h]
  const __m128i *v49; // [rsp+28h] [rbp-108h]
  __m128i *v50; // [rsp+30h] [rbp-100h] BYREF
  __m128i *v51; // [rsp+38h] [rbp-F8h]
  const __m128i *v52; // [rsp+40h] [rbp-F0h]
  unsigned __int64 v53; // [rsp+50h] [rbp-E0h] BYREF
  size_t v54; // [rsp+58h] [rbp-D8h]
  _BYTE v55[16]; // [rsp+60h] [rbp-D0h] BYREF
  __m128i *v56; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v57; // [rsp+78h] [rbp-B8h]
  __m128i v58; // [rsp+80h] [rbp-B0h] BYREF
  __m128i v59; // [rsp+90h] [rbp-A0h] BYREF
  __m128i v60; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v61; // [rsp+B0h] [rbp-80h]
  __int64 v62[2]; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v63; // [rsp+D0h] [rbp-60h]
  __int64 v64; // [rsp+D8h] [rbp-58h]
  __int64 v65; // [rsp+E0h] [rbp-50h]
  __int64 v66; // [rsp+E8h] [rbp-48h]
  unsigned __int64 *v67; // [rsp+F0h] [rbp-40h]

  v3 = 0;
  v4 = a1[1];
  v5 = *a1;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v48 = v4;
  if ( v5 == v4 )
    return v3;
  do
  {
    v14 = *v5;
    v54 = 0;
    v53 = (unsigned __int64)v55;
    v66 = 0x100000000LL;
    v55[0] = 0;
    v62[1] = 0;
    v62[0] = (__int64)&unk_49DD210;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v67 = &v53;
    sub_CB5980((__int64)v62, 0, 0, 0);
    sub_2E91850(v14, (__int64)v62, 1u, 0, 0, 1, 0);
    if ( v65 != v63 )
      sub_CB5AE0(v62);
    v15 = sub_22417D0((__int64 *)&v53, 61, 0);
    if ( v15 != (char *)-1LL )
    {
      if ( (unsigned __int64)v15 > v54 )
        sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::substr", (size_t)v15, v54);
      v56 = &v58;
      v6 = &v15[v53];
      if ( v54 + v53 && !v6 )
LABEL_76:
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v7 = v54 - (_QWORD)v15;
      v59.m128i_i64[0] = v54 - (_QWORD)v15;
      if ( v54 - (unsigned __int64)v15 > 0xF )
      {
        n = v54 - (_QWORD)v15;
        src = &v15[v53];
        v19 = sub_22409D0((__int64)&v56, (unsigned __int64 *)&v59, 0);
        v6 = src;
        v7 = n;
        v56 = (__m128i *)v19;
        v20 = (__m128i *)v19;
        v58.m128i_i64[0] = v59.m128i_i64[0];
      }
      else
      {
        if ( v7 == 1 )
        {
          v58.m128i_i8[0] = *v6;
          v8 = &v58;
LABEL_9:
          v57 = v7;
          v8->m128i_i8[v7] = 0;
          goto LABEL_10;
        }
        if ( !v7 )
        {
          v8 = &v58;
          goto LABEL_9;
        }
        v20 = &v58;
      }
      memcpy(v20, v6, v7);
      v7 = v59.m128i_i64[0];
      v8 = v56;
      goto LABEL_9;
    }
    v16 = (const void *)v53;
    v17 = v54;
    v56 = &v58;
    if ( v54 + v53 && !v53 )
      goto LABEL_76;
    v59.m128i_i64[0] = v54;
    if ( v54 > 0xF )
    {
      na = v54;
      srca = (void *)v53;
      v39 = sub_22409D0((__int64)&v56, (unsigned __int64 *)&v59, 0);
      v16 = srca;
      v17 = na;
      v56 = (__m128i *)v39;
      v40 = (__m128i *)v39;
      v58.m128i_i64[0] = v59.m128i_i64[0];
      goto LABEL_68;
    }
    if ( v54 != 1 )
    {
      if ( !v54 )
      {
        v18 = &v58;
        goto LABEL_33;
      }
      v40 = &v58;
LABEL_68:
      memcpy(v40, v16, v17);
      v17 = v59.m128i_i64[0];
      v18 = v56;
      goto LABEL_33;
    }
    v58.m128i_i8[0] = *(_BYTE *)v53;
    v18 = &v58;
LABEL_33:
    v57 = v17;
    v18->m128i_i8[v17] = 0;
LABEL_10:
    v59.m128i_i64[0] = (__int64)&v60;
    if ( v56 == &v58 )
    {
      v60 = _mm_load_si128(&v58);
    }
    else
    {
      v59.m128i_i64[0] = (__int64)v56;
      v60.m128i_i64[0] = v58.m128i_i64[0];
    }
    v9 = v51;
    v61 = v14;
    v56 = &v58;
    v59.m128i_i64[1] = v57;
    v57 = 0;
    v58.m128i_i8[0] = 0;
    if ( v51 == v52 )
    {
      sub_357D870((unsigned __int64 *)&v50, v51, &v59);
      v10 = (__m128i *)v59.m128i_i64[0];
    }
    else
    {
      v10 = (__m128i *)v59.m128i_i64[0];
      if ( v51 )
      {
        v51->m128i_i64[0] = (__int64)v51[1].m128i_i64;
        if ( (__m128i *)v59.m128i_i64[0] == &v60 )
        {
          v9[1] = _mm_load_si128(&v60);
        }
        else
        {
          v9->m128i_i64[0] = v59.m128i_i64[0];
          v9[1].m128i_i64[0] = v60.m128i_i64[0];
        }
        v9->m128i_i64[1] = v59.m128i_i64[1];
        v59.m128i_i64[0] = (__int64)&v60;
        v59.m128i_i64[1] = 0;
        v60.m128i_i8[0] = 0;
        v9[2].m128i_i64[0] = v61;
        v9 = v51;
        v10 = &v60;
      }
      v51 = (__m128i *)((char *)v9 + 40);
    }
    if ( v10 != &v60 )
      j_j___libc_free_0((unsigned __int64)v10);
    if ( v56 != &v58 )
      j_j___libc_free_0((unsigned __int64)v56);
    v62[0] = (__int64)&unk_49DD210;
    sub_CB5840((__int64)v62);
    if ( (_BYTE *)v53 != v55 )
      j_j___libc_free_0(v53);
    ++v5;
  }
  while ( v48 != v5 );
  v21 = v51;
  v22 = v50;
  v23 = a3;
  if ( v51 == v50 )
    goto LABEL_72;
  v24 = (char *)v51 - (char *)v50;
  _BitScanReverse64(&v25, 0xCCCCCCCCCCCCCCCDLL * (((char *)v51 - (char *)v50) >> 3));
  sub_357EDB0(v50, (__int64)v51, 2LL * (int)(63 - (v25 ^ 0x3F)), v11, v12, v13, a3);
  if ( v24 > 640 )
  {
    v41 = v22 + 40;
    v27 = v22;
    v26 = v22 + 40;
    sub_357CE20((__int64)v22, v22[40].m128i_i8);
    if ( v21 != &v22[40] )
    {
      do
      {
        v27 = v41;
        v41 = (__m128i *)((char *)v41 + 40);
        sub_357CB90(v27);
      }
      while ( v21 != v41 );
    }
  }
  else
  {
    v26 = v21;
    v27 = v22;
    sub_357CE20((__int64)v22, v21->m128i_i8);
  }
  v22 = v51;
  if ( v50 == v51 )
  {
LABEL_72:
    v3 = 0;
  }
  else
  {
    v49 = v51;
    v29 = v23;
    v30 = v50;
    do
    {
      v31 = (unsigned __int64 *)v30[2].m128i_i64[0];
      if ( !*(_QWORD *)(v29 + 16) )
        sub_4263D6(v27, v26, v28);
      v27 = (__m128i *)v29;
      v32 = (unsigned __int64 *)(*(__int64 (__fastcall **)(__int64))(v29 + 24))(v29);
      if ( v31 != v32 )
      {
        if ( !v31 )
          BUG();
        v33 = v31;
        if ( (*(_BYTE *)v31 & 4) == 0 && (*((_BYTE *)v31 + 44) & 8) != 0 )
        {
          do
            v33 = (unsigned __int64 *)v33[1];
          while ( (*((_BYTE *)v33 + 44) & 8) != 0 );
        }
        v34 = (unsigned __int64 *)v33[1];
        if ( v31 != v34 && v32 != v34 )
        {
          v26 = (__m128i *)(a2 + 40);
          v27 = (__m128i *)(a2 + 40);
          sub_2E310C0((__int64 *)(a2 + 40), (__int64 *)(a2 + 40), (__int64)v31, v33[1]);
          if ( v34 != v31 )
          {
            v35 = *v34 & 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)((*v31 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v34;
            *v34 = *v34 & 7 | *v31 & 0xFFFFFFFFFFFFFFF8LL;
            v36 = *v32;
            *(_QWORD *)(v35 + 8) = v32;
            v36 &= 0xFFFFFFFFFFFFFFF8LL;
            v28 = v36 | *v31 & 7;
            *v31 = v28;
            *(_QWORD *)(v36 + 8) = v31;
            *v32 = v35 | *v32 & 7;
          }
        }
      }
      v30 = (__m128i *)((char *)v30 + 40);
    }
    while ( v30 != v49 );
    v37 = v51;
    v22 = v50;
    if ( v51 == v50 )
    {
      v3 = 1;
    }
    else
    {
      do
      {
        if ( (__m128i *)v22->m128i_i64[0] != &v22[1] )
          j_j___libc_free_0(v22->m128i_i64[0]);
        v22 = (__m128i *)((char *)v22 + 40);
      }
      while ( v37 != v22 );
      v22 = v50;
      v3 = 1;
    }
  }
  if ( v22 )
    j_j___libc_free_0((unsigned __int64)v22);
  return v3;
}
