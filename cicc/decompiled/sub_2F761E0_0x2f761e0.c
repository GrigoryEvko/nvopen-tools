// Function: sub_2F761E0
// Address: 0x2f761e0
//
__int64 __fastcall sub_2F761E0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // rax
  __int64 i; // rdi
  __int64 j; // rsi
  __int16 v10; // dx
  __int64 v11; // rsi
  unsigned int v12; // edi
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  __int64 v15; // r9
  const __m128i *v16; // r13
  __int64 result; // rax
  __int64 v18; // r12
  unsigned __int64 v19; // r15
  __int64 *v20; // rcx
  __int64 v21; // rsi
  unsigned int v22; // edi
  unsigned int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r9
  unsigned __int64 v28; // rcx
  unsigned int v29; // eax
  __int64 v30; // r15
  unsigned int v31; // eax
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rax
  const __m128i *v35; // r15
  unsigned __int64 v36; // rdx
  unsigned __int64 v37; // r9
  __m128i *v38; // rax
  __int64 v39; // rcx
  int v40; // eax
  __int64 v41; // rsi
  unsigned __int64 v42; // rcx
  const __m128i *v43; // rax
  __int32 v44; // edx
  __m128i v45; // xmm0
  __int64 v46; // rdx
  unsigned __int64 v47; // rdx
  int v48; // edx
  __int64 v49; // rax
  unsigned __int64 v50; // r12
  _QWORD *v51; // rdx
  _QWORD *v52; // rdi
  int v53; // r10d
  __int64 v54; // rdi
  const void *v55; // rsi
  __int8 *v56; // r15
  __int64 v57; // [rsp+8h] [rbp-48h]
  __int32 v58; // [rsp+14h] [rbp-3Ch]
  __int64 v59; // [rsp+18h] [rbp-38h]

  v5 = a2;
  for ( i = *(_QWORD *)(a3 + 32); (*(_BYTE *)(v5 + 44) & 4) != 0; v5 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL )
    ;
  for ( ; (*(_BYTE *)(a2 + 44) & 8) != 0; a2 = *(_QWORD *)(a2 + 8) )
    ;
  for ( j = *(_QWORD *)(a2 + 8); j != v5; v5 = *(_QWORD *)(v5 + 8) )
  {
    v10 = *(_WORD *)(v5 + 68);
    if ( (unsigned __int16)(v10 - 14) > 4u && v10 != 24 )
      break;
  }
  v11 = *(_QWORD *)(i + 128);
  v12 = *(_DWORD *)(i + 144);
  if ( v12 )
  {
    a5 = v12 - 1;
    v13 = a5 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v14 = (__int64 *)(v11 + 16LL * v13);
    v15 = *v14;
    if ( *v14 == v5 )
      goto LABEL_11;
    v48 = 1;
    while ( v15 != -4096 )
    {
      v53 = v48 + 1;
      v13 = a5 & (v48 + v13);
      v14 = (__int64 *)(v11 + 16LL * v13);
      v15 = *v14;
      if ( *v14 == v5 )
        goto LABEL_11;
      v48 = v53;
    }
  }
  v14 = (__int64 *)(v11 + 16LL * v12);
LABEL_11:
  v16 = *(const __m128i **)(a1 + 208);
  v59 = v14[1];
  result = 3LL * *(unsigned int *)(a1 + 216);
  if ( result )
  {
    do
    {
      v27 = v16->m128i_u32[0];
      if ( (int)v27 >= 0 )
      {
        v18 = *(_QWORD *)(*(_QWORD *)(a3 + 424) + 8 * v27);
        goto LABEL_14;
      }
      v28 = *(unsigned int *)(a3 + 160);
      v29 = v27 & 0x7FFFFFFF;
      v30 = 8 * (v27 & 0x7FFFFFFF);
      if ( ((unsigned int)v27 & 0x7FFFFFFF) < (unsigned int)v28 )
      {
        v18 = *(_QWORD *)(*(_QWORD *)(a3 + 152) + 8LL * v29);
        if ( v18 )
          goto LABEL_15;
      }
      v31 = v29 + 1;
      if ( (unsigned int)v28 < v31 )
      {
        v47 = v31;
        if ( v31 != v28 )
        {
          if ( v31 >= v28 )
          {
            v49 = *(_QWORD *)(a3 + 168);
            v50 = v47 - v28;
            if ( v47 > *(unsigned int *)(a3 + 164) )
            {
              v57 = *(_QWORD *)(a3 + 168);
              v58 = v16->m128i_i32[0];
              sub_C8D5F0(a3 + 152, (const void *)(a3 + 168), v47, 8u, a5, v27);
              v28 = *(unsigned int *)(a3 + 160);
              v49 = v57;
              LODWORD(v27) = v58;
            }
            v32 = *(_QWORD *)(a3 + 152);
            v51 = (_QWORD *)(v32 + 8 * v28);
            v52 = &v51[v50];
            if ( v51 != v52 )
            {
              do
                *v51++ = v49;
              while ( v52 != v51 );
              LODWORD(v28) = *(_DWORD *)(a3 + 160);
              v32 = *(_QWORD *)(a3 + 152);
            }
            *(_DWORD *)(a3 + 160) = v50 + v28;
            goto LABEL_28;
          }
          *(_DWORD *)(a3 + 160) = v31;
        }
      }
      v32 = *(_QWORD *)(a3 + 152);
LABEL_28:
      v33 = sub_2E10F30(v27);
      *(_QWORD *)(v32 + v30) = v33;
      v18 = v33;
      sub_2E11E80((_QWORD *)a3, v33);
LABEL_14:
      if ( !v18 )
        goto LABEL_21;
LABEL_15:
      v19 = v59 & 0xFFFFFFFFFFFFFFF8LL;
      v20 = (__int64 *)sub_2E09D00((__int64 *)v18, v59 & 0xFFFFFFFFFFFFFFF8LL);
      v21 = *(_QWORD *)v18 + 24LL * *(unsigned int *)(v18 + 8);
      if ( v20 == (__int64 *)v21 )
        goto LABEL_21;
      v22 = *(_DWORD *)(v19 + 24);
      v23 = *(_DWORD *)((*v20 & 0xFFFFFFFFFFFFFFF8LL) + 24);
      if ( (unsigned __int64)(v23 | (*v20 >> 1) & 3) > v22 )
      {
        LOBYTE(v24) = 0;
      }
      else
      {
        v24 = v20[1];
        if ( v19 == (v24 & 0xFFFFFFFFFFFFFFF8LL) )
        {
          if ( (__int64 *)v21 == v20 + 3 )
            goto LABEL_20;
          v46 = v20[3];
          v20 += 3;
          v23 = *(_DWORD *)((v46 & 0xFFFFFFFFFFFFFFF8LL) + 24);
        }
      }
      if ( v23 <= v22 )
        v24 = v20[1];
LABEL_20:
      if ( (((unsigned __int8)v24 ^ 6) & 6) == 0 )
      {
        v34 = *(unsigned int *)(a1 + 424);
        v35 = v16;
        v36 = *(_QWORD *)(a1 + 416);
        v37 = v34 + 1;
        if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 428) )
        {
          v54 = a1 + 416;
          v55 = (const void *)(a1 + 432);
          if ( v36 > (unsigned __int64)v16 || (unsigned __int64)v16 >= v36 + 24 * v34 )
          {
            v35 = v16;
            sub_C8D5F0(v54, v55, v37, 0x18u, a5, v37);
            v36 = *(_QWORD *)(a1 + 416);
            v34 = *(unsigned int *)(a1 + 424);
          }
          else
          {
            v56 = &v16->m128i_i8[-v36];
            sub_C8D5F0(v54, v55, v37, 0x18u, a5, v37);
            v36 = *(_QWORD *)(a1 + 416);
            v34 = *(unsigned int *)(a1 + 424);
            v35 = (const __m128i *)&v56[v36];
          }
        }
        v38 = (__m128i *)(v36 + 24 * v34);
        *v38 = _mm_loadu_si128(v35);
        v38[1].m128i_i64[0] = v35[1].m128i_i64[0];
        v39 = *(unsigned int *)(a1 + 216);
        v26 = *(_QWORD *)(a1 + 208);
        ++*(_DWORD *)(a1 + 424);
        v40 = v39;
        v41 = v26 + 24 * v39 - ((_QWORD)v16 + 24);
        v42 = 0xAAAAAAAAAAAAAAABLL * (v41 >> 3);
        if ( v41 > 0 )
        {
          v43 = v16;
          do
          {
            v44 = v43[1].m128i_i32[2];
            v45 = _mm_loadu_si128(v43 + 2);
            v43 = (const __m128i *)((char *)v43 + 24);
            v43[-2].m128i_i32[2] = v44;
            v43[-1] = v45;
            --v42;
          }
          while ( v42 );
          v40 = *(_DWORD *)(a1 + 216);
          v26 = *(_QWORD *)(a1 + 208);
        }
        v25 = (unsigned int)(v40 - 1);
        *(_DWORD *)(a1 + 216) = v25;
        goto LABEL_22;
      }
LABEL_21:
      v25 = *(unsigned int *)(a1 + 216);
      v26 = *(_QWORD *)(a1 + 208);
      v16 = (const __m128i *)((char *)v16 + 24);
LABEL_22:
      result = v26 + 24 * v25;
    }
    while ( v16 != (const __m128i *)result );
  }
  return result;
}
