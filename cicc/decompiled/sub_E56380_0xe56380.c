// Function: sub_E56380
// Address: 0xe56380
//
void __fastcall sub_E56380(
        __int64 a1,
        char *a2,
        __int64 a3,
        char *a4,
        size_t a5,
        unsigned __int32 a6,
        __int128 a7,
        char a8,
        __int128 a9,
        __int64 a10)
{
  __int64 v11; // rdi
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rdx
  __m128i v18; // xmm2
  __m128i v19; // xmm3
  __m128i *v20; // rdi
  __int64 v21; // rsi
  size_t v22; // r9
  __int64 v23; // rdx
  __m128i *v24; // rdi
  __int64 v25; // rsi
  size_t v26; // r9
  __int64 v27; // rdx
  char v28; // al
  __m128i v29; // xmm4
  __m128i v30; // xmm6
  __int64 v31; // rdi
  __m128i *v32; // rsi
  size_t v33; // rdx
  size_t v34; // rdx
  __m128i v35; // rax
  char v36; // [rsp+4Eh] [rbp-192h]
  char v37; // [rsp+4Fh] [rbp-191h]
  __m128i v40; // [rsp+60h] [rbp-180h] BYREF
  char v41; // [rsp+70h] [rbp-170h]
  __m128i v42; // [rsp+80h] [rbp-160h] BYREF
  char v43; // [rsp+90h] [rbp-150h]
  __m128i v44; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v45; // [rsp+B0h] [rbp-130h]
  __int16 v46; // [rsp+C0h] [rbp-120h]
  __m128i v47; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v48; // [rsp+E0h] [rbp-100h]
  __int64 v49; // [rsp+E8h] [rbp-F8h]
  __int64 v50; // [rsp+F0h] [rbp-F0h]
  __int64 v51; // [rsp+F8h] [rbp-E8h]
  __m128i **v52; // [rsp+100h] [rbp-E0h]
  __m128i *v53; // [rsp+110h] [rbp-D0h] BYREF
  size_t n; // [rsp+118h] [rbp-C8h]
  __int64 v55; // [rsp+120h] [rbp-C0h] BYREF
  _BYTE v56[184]; // [rsp+128h] [rbp-B8h] BYREF

  v11 = *(_QWORD *)(a1 + 8);
  if ( *(_WORD *)(v11 + 1904) <= 4u )
    return;
  v47.m128i_i32[0] = a6;
  v36 = a8;
  v14 = v11 + 1736;
  v41 = a8;
  v44 = _mm_loadu_si128((const __m128i *)&a9);
  v45 = a10;
  v40 = _mm_loadu_si128((const __m128i *)&a7);
  v37 = a10;
  v15 = *(_QWORD *)(v11 + 1744);
  if ( !v15 )
    goto LABEL_10;
  do
  {
    while ( 1 )
    {
      v16 = *(_QWORD *)(v15 + 16);
      v17 = *(_QWORD *)(v15 + 24);
      if ( a6 <= *(_DWORD *)(v15 + 32) )
        break;
      v15 = *(_QWORD *)(v15 + 24);
      if ( !v17 )
        goto LABEL_8;
    }
    v14 = v15;
    v15 = *(_QWORD *)(v15 + 16);
  }
  while ( v16 );
LABEL_8:
  if ( v11 + 1736 == v14 || a6 < *(_DWORD *)(v14 + 32) )
  {
LABEL_10:
    v53 = &v47;
    v14 = sub_E56230((_QWORD *)(v11 + 1728), v14, (unsigned int **)&v53);
  }
  v18 = _mm_loadu_si128(&v40);
  v19 = _mm_loadu_si128(&v44);
  v43 = v41;
  v53 = (__m128i *)&v55;
  v48 = v45;
  v42 = v18;
  v47 = v19;
  sub_E4CC80((__int64 *)&v53, a2, (__int64)&a2[a3]);
  v20 = *(__m128i **)(v14 + 440);
  if ( v53 == (__m128i *)&v55 )
  {
    v34 = n;
    if ( n )
    {
      if ( n == 1 )
        v20->m128i_i8[0] = v55;
      else
        memcpy(v20, &v55, n);
      v34 = n;
      v20 = *(__m128i **)(v14 + 440);
    }
    *(_QWORD *)(v14 + 448) = v34;
    v20->m128i_i8[v34] = 0;
    v20 = v53;
  }
  else
  {
    v21 = v55;
    v22 = n;
    if ( v20 == (__m128i *)(v14 + 456) )
    {
      *(_QWORD *)(v14 + 440) = v53;
      *(_QWORD *)(v14 + 448) = v22;
      *(_QWORD *)(v14 + 456) = v21;
    }
    else
    {
      v23 = *(_QWORD *)(v14 + 456);
      *(_QWORD *)(v14 + 440) = v53;
      *(_QWORD *)(v14 + 448) = v22;
      *(_QWORD *)(v14 + 456) = v21;
      if ( v20 )
      {
        v53 = v20;
        v55 = v23;
        goto LABEL_15;
      }
    }
    v53 = (__m128i *)&v55;
    v20 = (__m128i *)&v55;
  }
LABEL_15:
  n = 0;
  v20->m128i_i8[0] = 0;
  if ( v53 != (__m128i *)&v55 )
    j_j___libc_free_0(v53, v55 + 1);
  v53 = (__m128i *)&v55;
  sub_E4CC80((__int64 *)&v53, a4, (__int64)&a4[a5]);
  v24 = *(__m128i **)(v14 + 472);
  if ( v53 == (__m128i *)&v55 )
  {
    v33 = n;
    if ( n )
    {
      if ( n == 1 )
        v24->m128i_i8[0] = v55;
      else
        memcpy(v24, &v55, n);
      v33 = n;
      v24 = *(__m128i **)(v14 + 472);
    }
    *(_QWORD *)(v14 + 480) = v33;
    v24->m128i_i8[v33] = 0;
    v24 = v53;
    goto LABEL_21;
  }
  v25 = v55;
  v26 = n;
  if ( v24 == (__m128i *)(v14 + 488) )
  {
    *(_QWORD *)(v14 + 472) = v53;
    *(_QWORD *)(v14 + 480) = v26;
    *(_QWORD *)(v14 + 488) = v25;
    goto LABEL_39;
  }
  v27 = *(_QWORD *)(v14 + 488);
  *(_QWORD *)(v14 + 472) = v53;
  *(_QWORD *)(v14 + 480) = v26;
  *(_QWORD *)(v14 + 488) = v25;
  if ( !v24 )
  {
LABEL_39:
    v53 = (__m128i *)&v55;
    v24 = (__m128i *)&v55;
    goto LABEL_21;
  }
  v53 = v24;
  v55 = v27;
LABEL_21:
  n = 0;
  v24->m128i_i8[0] = 0;
  if ( v53 != (__m128i *)&v55 )
    j_j___libc_free_0(v53, v55 + 1);
  v28 = v43;
  v29 = _mm_loadu_si128(&v42);
  *(_DWORD *)(v14 + 504) = 0;
  *(_BYTE *)(v14 + 524) = v28;
  *(__m128i *)(v14 + 508) = v29;
  *(__m128i *)(v14 + 528) = _mm_loadu_si128(&v47);
  *(_QWORD *)(v14 + 544) = v48;
  *(_BYTE *)(v14 + 553) &= v36;
  *(_BYTE *)(v14 + 554) |= v36;
  *(_BYTE *)(v14 + 552) |= v37;
  if ( !*(_BYTE *)(*(_QWORD *)(a1 + 312) + 21LL) )
  {
    v51 = 0x100000000LL;
    v52 = &v53;
    v53 = (__m128i *)v56;
    v47.m128i_i64[0] = (__int64)&unk_49DD288;
    n = 0;
    v55 = 128;
    v47.m128i_i64[1] = 2;
    v48 = 0;
    v49 = 0;
    v50 = 0;
    sub_CB5980((__int64)&v47, 0, 0, 0);
    v30 = _mm_loadu_si128((const __m128i *)&a7);
    sub_E55510(
      a1,
      0,
      a2,
      a3,
      a4,
      a5,
      v30.m128i_i8[0],
      v30.m128i_i32[2],
      a8,
      (char *)a9,
      *((__int64 *)&a9 + 1),
      a10,
      *(_BYTE *)(a1 + 747),
      (__int64)&v47);
    v31 = *(_QWORD *)(a1 + 16);
    if ( v31 )
    {
      v32 = *v52;
      (*(void (__fastcall **)(__int64, __m128i *, __m128i *))(*(_QWORD *)v31 + 40LL))(v31, *v52, v52[1]);
    }
    else
    {
      v32 = &v44;
      v35 = *(__m128i *)v52;
      v46 = 261;
      v44 = v35;
      sub_E99A90(a1, &v44);
    }
    v47.m128i_i64[0] = (__int64)&unk_49DD388;
    sub_CB5840((__int64)&v47);
    if ( v53 != (__m128i *)v56 )
      _libc_free(v53, v32);
  }
}
