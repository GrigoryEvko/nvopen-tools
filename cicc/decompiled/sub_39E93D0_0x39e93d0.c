// Function: sub_39E93D0
// Address: 0x39e93d0
//
void __fastcall sub_39E93D0(
        __int64 a1,
        unsigned __int8 *a2,
        __int64 a3,
        unsigned __int8 *a4,
        __int64 a5,
        char *a6,
        const __m128i *a7,
        unsigned __int32 a8)
{
  __int64 v10; // rdi
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // rcx
  __int64 v15; // rdx
  __m128i *v16; // rdi
  __m128i *v17; // rax
  __int64 v18; // rsi
  size_t v19; // rdi
  __int64 v20; // rcx
  __m128i *v21; // rdi
  __m128i *v22; // rax
  size_t v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // rcx
  char v26; // al
  char v27; // al
  __int64 v28; // rdi
  size_t v29; // rdx
  size_t v30; // rdx
  __m128i v31; // xmm5
  __m128i *v32; // rdx
  _QWORD *v33; // [rsp+8h] [rbp-158h]
  __int64 v34; // [rsp+8h] [rbp-158h]
  _QWORD *v35; // [rsp+8h] [rbp-158h]
  __int64 v36; // [rsp+8h] [rbp-158h]
  __int8 v39; // [rsp+30h] [rbp-130h]
  _QWORD v41[2]; // [rsp+40h] [rbp-120h] BYREF
  __m128i v42; // [rsp+50h] [rbp-110h] BYREF
  __int16 v43; // [rsp+60h] [rbp-100h]
  __m128i v44; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v45; // [rsp+80h] [rbp-E0h]
  __int64 v46; // [rsp+88h] [rbp-D8h]
  int v47; // [rsp+90h] [rbp-D0h]
  __m128i **v48; // [rsp+98h] [rbp-C8h]
  __m128i *v49; // [rsp+A0h] [rbp-C0h] BYREF
  size_t n; // [rsp+A8h] [rbp-B8h]
  _QWORD src[22]; // [rsp+B0h] [rbp-B0h] BYREF

  v10 = *(_QWORD *)(a1 + 8);
  if ( *(_WORD *)(v10 + 1160) <= 4u )
    return;
  v39 = a7[1].m128i_i8[0];
  if ( v39 )
    v42 = _mm_loadu_si128(a7);
  v44.m128i_i32[0] = a8;
  v12 = *(_QWORD *)(v10 + 992);
  v13 = v10 + 984;
  if ( !v12 )
    goto LABEL_11;
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
        goto LABEL_9;
    }
    v13 = v12;
    v12 = *(_QWORD *)(v12 + 16);
  }
  while ( v14 );
LABEL_9:
  if ( v10 + 984 == v13 || a8 < *(_DWORD *)(v13 + 32) )
  {
LABEL_11:
    v49 = &v44;
    v13 = sub_39E9160((_QWORD *)(v10 + 976), v13, (unsigned int **)&v49);
  }
  if ( v39 )
    v44 = _mm_loadu_si128(&v42);
  if ( !a2 )
  {
    LOBYTE(src[0]) = 0;
    v30 = 0;
    v49 = (__m128i *)src;
    v16 = *(__m128i **)(v13 + 424);
LABEL_44:
    *(_QWORD *)(v13 + 432) = v30;
    v16->m128i_i8[v30] = 0;
    v17 = v49;
    goto LABEL_19;
  }
  v33 = (_QWORD *)v13;
  v49 = (__m128i *)src;
  sub_39DFBE0((__int64 *)&v49, a2, (__int64)&a2[a3]);
  v13 = (__int64)v33;
  v16 = (__m128i *)v33[53];
  v17 = v16;
  if ( v49 == (__m128i *)src )
  {
    v30 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        v16->m128i_i8[0] = src[0];
      }
      else
      {
        memcpy(v16, src, n);
        v13 = (__int64)v33;
      }
      v30 = n;
      v16 = (__m128i *)v33[53];
    }
    goto LABEL_44;
  }
  v18 = src[0];
  v19 = n;
  if ( v17 == (__m128i *)(v33 + 55) )
  {
    v33[53] = v49;
    v33[54] = v19;
    v33[55] = v18;
  }
  else
  {
    v20 = v33[55];
    v33[53] = v49;
    v33[54] = v19;
    v33[55] = v18;
    if ( v17 )
    {
      v49 = v17;
      src[0] = v20;
      goto LABEL_19;
    }
  }
  v49 = (__m128i *)src;
  v17 = (__m128i *)src;
LABEL_19:
  n = 0;
  v17->m128i_i8[0] = 0;
  if ( v49 != (__m128i *)src )
  {
    v34 = v13;
    j_j___libc_free_0((unsigned __int64)v49);
    v13 = v34;
  }
  if ( !a4 )
  {
    v49 = (__m128i *)src;
    v29 = 0;
    LOBYTE(src[0]) = 0;
    v21 = *(__m128i **)(v13 + 456);
LABEL_42:
    *(_QWORD *)(v13 + 464) = v29;
    v21->m128i_i8[v29] = 0;
    v22 = v49;
    goto LABEL_26;
  }
  v35 = (_QWORD *)v13;
  v49 = (__m128i *)src;
  sub_39DFBE0((__int64 *)&v49, a4, (__int64)&a4[a5]);
  v13 = (__int64)v35;
  v21 = (__m128i *)v35[57];
  v22 = v21;
  if ( v49 == (__m128i *)src )
  {
    v29 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        v21->m128i_i8[0] = src[0];
      }
      else
      {
        memcpy(v21, src, n);
        v13 = (__int64)v35;
      }
      v29 = n;
      v21 = (__m128i *)v35[57];
    }
    goto LABEL_42;
  }
  v23 = n;
  v24 = src[0];
  if ( v22 == (__m128i *)(v35 + 59) )
  {
    v35[57] = v49;
    v35[58] = v23;
    v35[59] = v24;
    goto LABEL_47;
  }
  v25 = v35[59];
  v35[57] = v49;
  v35[58] = v23;
  v35[59] = v24;
  if ( !v22 )
  {
LABEL_47:
    v49 = (__m128i *)src;
    v22 = (__m128i *)src;
    goto LABEL_26;
  }
  v49 = v22;
  src[0] = v25;
LABEL_26:
  n = 0;
  v22->m128i_i8[0] = 0;
  if ( v49 != (__m128i *)src )
  {
    v36 = v13;
    j_j___libc_free_0((unsigned __int64)v49);
    v13 = v36;
  }
  *(_DWORD *)(v13 + 488) = 0;
  *(_QWORD *)(v13 + 496) = a6;
  v26 = *(_BYTE *)(v13 + 520);
  if ( v39 )
  {
    if ( v26 )
    {
      *(__m128i *)(v13 + 504) = _mm_loadu_si128(&v44);
    }
    else
    {
      v31 = _mm_loadu_si128(&v44);
      *(_BYTE *)(v13 + 520) = 1;
      *(__m128i *)(v13 + 504) = v31;
    }
  }
  else if ( v26 )
  {
    *(_BYTE *)(v13 + 520) = 0;
  }
  *(_BYTE *)(v13 + 529) &= a6 != 0;
  *(_BYTE *)(v13 + 530) |= a6 != 0;
  *(_BYTE *)(v13 + 528) = v39;
  n = 0x8000000000LL;
  v49 = (__m128i *)src;
  v47 = 1;
  v44.m128i_i64[0] = (__int64)&unk_49EFC48;
  v46 = 0;
  v48 = &v49;
  v45 = 0;
  v44.m128i_i64[1] = 0;
  sub_16E7A40((__int64)&v44, 0, 0, 0);
  v27 = (*(_BYTE *)(a1 + 680) & 4) != 0;
  LOBYTE(v43) = a7[1].m128i_i8[0];
  if ( (_BYTE)v43 )
    v42 = _mm_loadu_si128(a7);
  sub_39E85A0(0, a2, a3, a4, a5, a6, (__int64)&v42, v27, (__int64)&v44);
  v28 = *(_QWORD *)(a1 + 16);
  if ( v28 )
  {
    (*(void (__fastcall **)(__int64, __m128i *, _QWORD))(*(_QWORD *)v28 + 40LL))(v28, *v48, *((unsigned int *)v48 + 2));
  }
  else
  {
    v32 = *v48;
    v41[1] = *((unsigned int *)v48 + 2);
    v43 = 261;
    v41[0] = v32;
    v42.m128i_i64[0] = (__int64)v41;
    sub_38DD5A0((__int64 *)a1, (__int64)&v42);
  }
  v44.m128i_i64[0] = (__int64)&unk_49EFD28;
  sub_16E7960((__int64)&v44);
  if ( v49 != (__m128i *)src )
    _libc_free((unsigned __int64)v49);
}
