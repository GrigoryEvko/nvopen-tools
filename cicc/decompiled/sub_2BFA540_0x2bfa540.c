// Function: sub_2BFA540
// Address: 0x2bfa540
//
__int64 __fastcall sub_2BFA540(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r15
  _BYTE *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rax
  const __m128i *v13; // rsi
  _BYTE *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned __int64 v18; // rdx
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // r8
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  __m128i *v24; // rdi
  __m128i *v25; // rdx
  const __m128i *v26; // rax
  __int64 v27; // r9
  __int64 v28; // r8
  __int64 v29; // rax
  __int8 *v30; // rdx
  __int64 v31; // rcx
  __m128i *v32; // rdx
  const __m128i *v33; // rax
  unsigned __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rsi
  char v37; // al
  char v39; // [rsp+8h] [rbp-228h]
  __int8 *v40; // [rsp+8h] [rbp-228h]
  _BYTE v41[32]; // [rsp+10h] [rbp-220h] BYREF
  _BYTE v42[64]; // [rsp+30h] [rbp-200h] BYREF
  __int64 v43; // [rsp+70h] [rbp-1C0h]
  __int64 v44; // [rsp+78h] [rbp-1B8h]
  __int8 *v45; // [rsp+80h] [rbp-1B0h]
  _QWORD v46[4]; // [rsp+90h] [rbp-1A0h] BYREF
  char v47[64]; // [rsp+B0h] [rbp-180h] BYREF
  __int64 v48; // [rsp+F0h] [rbp-140h]
  unsigned __int64 v49; // [rsp+F8h] [rbp-138h]
  __int8 *v50; // [rsp+100h] [rbp-130h]
  _QWORD v51[12]; // [rsp+110h] [rbp-120h] BYREF
  __int64 v52; // [rsp+170h] [rbp-C0h]
  __int64 v53; // [rsp+178h] [rbp-B8h]
  _BYTE v54[96]; // [rsp+188h] [rbp-A8h] BYREF
  __int64 v55; // [rsp+1E8h] [rbp-48h]
  const __m128i *v56; // [rsp+1F0h] [rbp-40h]

  v1 = sub_2BF4E60(*(_QWORD *)(a1 + 112));
  v3 = v2;
  v4 = v1;
  v5 = sub_2BF9BD0(a1);
  v39 = *(_BYTE *)(a1 + 128);
  v6 = sub_22077B0(0x88u);
  v9 = v6;
  if ( v6 )
  {
    v10 = *(_BYTE **)(a1 + 16);
    *(_BYTE *)(v6 + 8) = 0;
    v11 = *(_QWORD *)(a1 + 24);
    *(_QWORD *)v6 = &unk_4A23970;
    *(_QWORD *)(v6 + 16) = v6 + 32;
    sub_2BEF590((__int64 *)(v6 + 16), v10, (__int64)&v10[v11]);
    *(_QWORD *)(v9 + 48) = 0;
    *(_QWORD *)(v9 + 56) = v9 + 72;
    *(_QWORD *)(v9 + 64) = 0x100000000LL;
    *(_QWORD *)(v9 + 88) = 0x100000000LL;
    *(_QWORD *)(v9 + 80) = v9 + 96;
    *(_QWORD *)(v9 + 104) = 0;
    *(_QWORD *)v9 = &unk_4A23A38;
    *(_QWORD *)(v4 + 48) = v9;
    *(_QWORD *)(v9 + 112) = v4;
    *(_QWORD *)(v9 + 120) = v3;
    *(_BYTE *)(v9 + 128) = v39;
    *(_QWORD *)(v3 + 48) = v9;
  }
  v12 = *(unsigned int *)(v5 + 600);
  if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(v5 + 604) )
  {
    sub_C8D5F0(v5 + 592, (const void *)(v5 + 608), v12 + 1, 8u, v7, v8);
    v12 = *(unsigned int *)(v5 + 600);
  }
  *(_QWORD *)(*(_QWORD *)(v5 + 592) + 8 * v12) = v9;
  ++*(_DWORD *)(v5 + 600);
  v46[0] = v4;
  sub_2BF3840(v51, v46);
  v13 = (const __m128i *)v42;
  v14 = v41;
  sub_C8CD80((__int64)v41, (__int64)v42, (__int64)v51, v15, v16, v17);
  v20 = v53;
  v21 = v52;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v22 = v53 - v52;
  if ( v53 == v52 )
  {
    v22 = 0;
    v24 = 0;
  }
  else
  {
    if ( v22 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_36;
    v23 = sub_22077B0(v53 - v52);
    v20 = v53;
    v21 = v52;
    v24 = (__m128i *)v23;
  }
  v43 = (__int64)v24;
  v44 = (__int64)v24;
  v45 = &v24->m128i_i8[v22];
  if ( v20 != v21 )
  {
    v25 = v24;
    v26 = (const __m128i *)v21;
    do
    {
      if ( v25 )
      {
        *v25 = _mm_loadu_si128(v26);
        v25[1].m128i_i64[0] = v26[1].m128i_i64[0];
      }
      v26 = (const __m128i *)((char *)v26 + 24);
      v25 = (__m128i *)((char *)v25 + 24);
    }
    while ( v26 != (const __m128i *)v20 );
    v24 = (__m128i *)((char *)v24 + 8 * (((unsigned __int64)&v26[-2].m128i_u64[1] - v21) >> 3) + 24);
  }
  v44 = (__int64)v24;
  v14 = v46;
  sub_C8CD80((__int64)v46, (__int64)v47, (__int64)v54, v20, v21, v19);
  v13 = v56;
  v28 = v55;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v18 = (unsigned __int64)v56 - v55;
  if ( v56 == (const __m128i *)v55 )
  {
    v30 = 0;
    v31 = 0;
    goto LABEL_17;
  }
  if ( v18 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_36:
    sub_4261EA(v14, v13, v18);
  v40 = &v56->m128i_i8[-v55];
  v29 = sub_22077B0((unsigned __int64)v56 - v55);
  v13 = v56;
  v28 = v55;
  v30 = v40;
  v31 = v29;
LABEL_17:
  v48 = v31;
  v49 = v31;
  v50 = &v30[v31];
  if ( (const __m128i *)v28 == v13 )
  {
    v34 = v31;
  }
  else
  {
    v32 = (__m128i *)v31;
    v33 = (const __m128i *)v28;
    do
    {
      if ( v32 )
      {
        *v32 = _mm_loadu_si128(v33);
        v32[1].m128i_i64[0] = v33[1].m128i_i64[0];
      }
      v33 = (const __m128i *)((char *)v33 + 24);
      v32 = (__m128i *)((char *)v32 + 24);
    }
    while ( v13 != v33 );
    v34 = v31 + 8 * (((unsigned __int64)&v13[-2].m128i_u64[1] - v28) >> 3) + 24;
  }
  v49 = v34;
  while ( 1 )
  {
    v35 = v44;
    v36 = v43;
    if ( v44 - v43 != v34 - v31 )
      goto LABEL_24;
    if ( v44 == v43 )
      break;
    while ( *(_QWORD *)v36 == *(_QWORD *)v31 )
    {
      v37 = *(_BYTE *)(v36 + 16);
      if ( v37 != *(_BYTE *)(v31 + 16) || v37 && *(_QWORD *)(v36 + 8) != *(_QWORD *)(v31 + 8) )
        break;
      v36 += 24;
      v31 += 24;
      if ( v44 == v36 )
        goto LABEL_32;
    }
LABEL_24:
    *(_QWORD *)(*(_QWORD *)(v44 - 24) + 48LL) = v9;
    sub_2ADA290((__int64)v41, v36, v35, v31, v28, v27);
    v34 = v49;
    v31 = v48;
  }
LABEL_32:
  sub_2AB1B10((__int64)v46);
  sub_2AB1B10((__int64)v41);
  sub_2AB1B10((__int64)v54);
  sub_2AB1B10((__int64)v51);
  return v9;
}
