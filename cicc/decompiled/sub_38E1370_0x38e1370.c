// Function: sub_38E1370
// Address: 0x38e1370
//
bool __fastcall sub_38E1370(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        _BYTE *a4,
        __int64 a5,
        __int64 a6,
        const __m128i *a7,
        unsigned __int32 a8)
{
  __int64 v10; // rdi
  __int8 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rcx
  __int64 v15; // rdx
  __m128i *v16; // rdi
  __m128i *v17; // rax
  __int64 v18; // rdi
  size_t v19; // r8
  __int64 v20; // rsi
  __m128i *v21; // rdi
  __m128i *v22; // rax
  __int64 v23; // rdi
  size_t v24; // r8
  __int64 v25; // rsi
  char v26; // al
  __int64 v28; // rax
  size_t v29; // rdx
  size_t v30; // rdx
  __m128i v31; // xmm3
  __int64 v32; // [rsp+8h] [rbp-A8h]
  __m128i v35[2]; // [rsp+20h] [rbp-90h] BYREF
  __m128i v36[2]; // [rsp+40h] [rbp-70h] BYREF
  __m128i *v37; // [rsp+60h] [rbp-50h] BYREF
  size_t n; // [rsp+68h] [rbp-48h]
  _QWORD src[8]; // [rsp+70h] [rbp-40h] BYREF

  v10 = *(_QWORD *)(a1 + 8);
  v11 = a7[1].m128i_i8[0];
  if ( v11 )
    v35[0] = _mm_loadu_si128(a7);
  v36[0].m128i_i32[0] = a8;
  v12 = *(_QWORD *)(v10 + 992);
  v13 = v10 + 984;
  if ( !v12 )
    goto LABEL_30;
  do
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(v12 + 16);
      v15 = *(_QWORD *)(v12 + 24);
      if ( a8 <= *(_DWORD *)(v12 + 32) )
        break;
      v12 = *(_QWORD *)(v12 + 24);
      if ( !v15 )
        goto LABEL_8;
    }
    v13 = v12;
    v12 = *(_QWORD *)(v12 + 16);
  }
  while ( v14 );
LABEL_8:
  if ( v10 + 984 == v13 || a8 < *(_DWORD *)(v13 + 32) )
  {
LABEL_30:
    v32 = a3;
    v37 = v36;
    v28 = sub_38E1100((_QWORD *)(v10 + 976), v13, (unsigned int **)&v37);
    a3 = v32;
    v13 = v28;
  }
  if ( v11 )
    v36[0] = _mm_loadu_si128(v35);
  if ( !a2 )
  {
    LOBYTE(src[0]) = 0;
    v29 = 0;
    v37 = (__m128i *)src;
    v16 = *(__m128i **)(v13 + 424);
LABEL_36:
    *(_QWORD *)(v13 + 432) = v29;
    v16->m128i_i8[v29] = 0;
    v17 = v37;
    goto LABEL_17;
  }
  v37 = (__m128i *)src;
  sub_38DC1A0((__int64 *)&v37, a2, (__int64)&a2[a3]);
  v16 = *(__m128i **)(v13 + 424);
  v17 = v16;
  if ( v37 == (__m128i *)src )
  {
    v29 = n;
    if ( n )
    {
      if ( n == 1 )
        v16->m128i_i8[0] = src[0];
      else
        memcpy(v16, src, n);
      v29 = n;
      v16 = *(__m128i **)(v13 + 424);
    }
    goto LABEL_36;
  }
  v18 = src[0];
  v19 = n;
  if ( v17 == (__m128i *)(v13 + 440) )
  {
    *(_QWORD *)(v13 + 424) = v37;
    *(_QWORD *)(v13 + 432) = v19;
    *(_QWORD *)(v13 + 440) = v18;
  }
  else
  {
    v20 = *(_QWORD *)(v13 + 440);
    *(_QWORD *)(v13 + 424) = v37;
    *(_QWORD *)(v13 + 432) = v19;
    *(_QWORD *)(v13 + 440) = v18;
    if ( v17 )
    {
      v37 = v17;
      src[0] = v20;
      goto LABEL_17;
    }
  }
  v37 = (__m128i *)src;
  v17 = (__m128i *)src;
LABEL_17:
  n = 0;
  v17->m128i_i8[0] = 0;
  if ( v37 != (__m128i *)src )
    j_j___libc_free_0((unsigned __int64)v37);
  if ( !a4 )
  {
    v37 = (__m128i *)src;
    v30 = 0;
    LOBYTE(src[0]) = 0;
    v21 = *(__m128i **)(v13 + 456);
LABEL_38:
    *(_QWORD *)(v13 + 464) = v30;
    v21->m128i_i8[v30] = 0;
    v22 = v37;
    goto LABEL_24;
  }
  v37 = (__m128i *)src;
  sub_38DC1A0((__int64 *)&v37, a4, (__int64)&a4[a5]);
  v21 = *(__m128i **)(v13 + 456);
  v22 = v21;
  if ( v37 == (__m128i *)src )
  {
    v30 = n;
    if ( n )
    {
      if ( n == 1 )
        v21->m128i_i8[0] = src[0];
      else
        memcpy(v21, src, n);
      v30 = n;
      v21 = *(__m128i **)(v13 + 456);
    }
    goto LABEL_38;
  }
  v23 = src[0];
  v24 = n;
  if ( v22 == (__m128i *)(v13 + 472) )
  {
    *(_QWORD *)(v13 + 456) = v37;
    *(_QWORD *)(v13 + 464) = v24;
    *(_QWORD *)(v13 + 472) = v23;
  }
  else
  {
    v25 = *(_QWORD *)(v13 + 472);
    *(_QWORD *)(v13 + 456) = v37;
    *(_QWORD *)(v13 + 464) = v24;
    *(_QWORD *)(v13 + 472) = v23;
    if ( v22 )
    {
      v37 = v22;
      src[0] = v25;
      goto LABEL_24;
    }
  }
  v37 = (__m128i *)src;
  v22 = (__m128i *)src;
LABEL_24:
  n = 0;
  v22->m128i_i8[0] = 0;
  if ( v37 != (__m128i *)src )
    j_j___libc_free_0((unsigned __int64)v37);
  *(_DWORD *)(v13 + 488) = 0;
  *(_QWORD *)(v13 + 496) = a6;
  v26 = *(_BYTE *)(v13 + 520);
  if ( v11 )
  {
    if ( v26 )
    {
      *(__m128i *)(v13 + 504) = _mm_loadu_si128(v36);
    }
    else
    {
      v31 = _mm_loadu_si128(v36);
      *(_BYTE *)(v13 + 520) = 1;
      *(__m128i *)(v13 + 504) = v31;
    }
  }
  else if ( v26 )
  {
    *(_BYTE *)(v13 + 520) = 0;
  }
  *(_BYTE *)(v13 + 528) = v11;
  *(_BYTE *)(v13 + 529) &= a6 != 0;
  *(_BYTE *)(v13 + 530) |= a6 != 0;
  return a6 != 0;
}
