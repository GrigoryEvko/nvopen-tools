// Function: sub_2AE08E0
// Address: 0x2ae08e0
//
__int64 __fastcall sub_2AE08E0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rbx
  __m128i v11; // xmm6
  __m128i v12; // xmm7
  __m128i v13; // xmm6
  int v14; // eax
  __int64 *v15; // r12
  __int64 v16; // r15
  __int64 v17; // r14
  __int64 v18; // r8
  __int64 v19; // r13
  __int32 v20; // r15d
  __int8 v21; // bl
  unsigned __int64 v22; // rax
  __int32 v23; // edx
  __m128i v24; // xmm1
  __m128i v25; // xmm2
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  __m128i *v29; // rcx
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // r10
  __m128i *v32; // rax
  __m128i v33; // xmm6
  __m128i v34; // xmm7
  __int64 v35; // rdi
  const void *v36; // rsi
  __int8 *v37; // rbx
  int *v38; // rax
  int v39; // edx
  __int64 *i; // [rsp+8h] [rbp-F8h]
  __int64 v42; // [rsp+18h] [rbp-E8h]
  __int32 v43; // [rsp+20h] [rbp-E0h]
  int v44; // [rsp+24h] [rbp-DCh]
  __int64 v45; // [rsp+28h] [rbp-D8h]
  __int64 v46; // [rsp+30h] [rbp-D0h]
  __int64 v47; // [rsp+38h] [rbp-C8h]
  __m128i v48; // [rsp+40h] [rbp-C0h] BYREF
  __m128i v49; // [rsp+50h] [rbp-B0h] BYREF
  __m128i v50; // [rsp+60h] [rbp-A0h] BYREF
  __m128i v51; // [rsp+70h] [rbp-90h] BYREF
  __m128i v52; // [rsp+80h] [rbp-80h] BYREF
  __m128i v53; // [rsp+90h] [rbp-70h] BYREF
  __m128i v54; // [rsp+A0h] [rbp-60h] BYREF
  __m128i v55; // [rsp+B0h] [rbp-50h] BYREF
  __m128i v56; // [rsp+C0h] [rbp-40h] BYREF

  v2 = *(_DWORD *)(a2 + 96);
  if ( !v2 )
  {
    *(_QWORD *)a1 = 1;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 40) = 0;
    return a1;
  }
  if ( v2 == 1 )
  {
    v4 = **(_QWORD **)(a2 + 88);
    if ( *(_DWORD *)(v4 + 88) == 1 )
    {
      v38 = *(int **)(v4 + 80);
      v39 = *v38;
      LOBYTE(v38) = *((_BYTE *)v38 + 4);
      *(_QWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      *(_DWORD *)a1 = v39;
      *(_BYTE *)(a1 + 4) = (_BYTE)v38;
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      *(_DWORD *)(a1 + 40) = 0;
      *(_BYTE *)(a1 + 44) = 0;
      return a1;
    }
  }
  LODWORD(v46) = 1;
  BYTE4(v46) = 0;
  v5 = sub_2AD1E10(*(_QWORD *)(a2 + 48), v46);
  v50.m128i_i8[12] = 0;
  v10 = *(_QWORD *)(a2 + 72);
  v48.m128i_i64[1] = v5;
  v48.m128i_i64[0] = 1;
  v11 = _mm_loadu_si128(&v48);
  v49.m128i_i64[1] = v5;
  v50.m128i_i32[2] = 0;
  v51 = v11;
  v49.m128i_i32[0] = v6;
  v12 = _mm_loadu_si128(&v49);
  v50.m128i_i32[0] = v6;
  v13 = _mm_loadu_si128(&v50);
  v42 = v5;
  v52 = v12;
  v53 = v13;
  v14 = *(_DWORD *)(v10 + 40);
  v43 = v6;
  if ( v14 != -1 )
    goto LABEL_7;
  v44 = 0;
  if ( !(unsigned __int8)sub_F6E590(*(_QWORD *)(v10 + 104), v46, v6, v7, v8, v9) )
  {
    v14 = *(_DWORD *)(v10 + 40);
LABEL_7:
    v44 = v14;
    if ( v14 == 1 )
    {
      v52.m128i_i32[0] = 0;
      v51.m128i_i64[1] = 0x7FFFFFFFFFFFFFFFLL;
    }
  }
  v15 = *(__int64 **)(a2 + 88);
  v16 = a2;
  for ( i = &v15[*(unsigned int *)(a2 + 96)]; i != v15; ++v15 )
  {
    v17 = v16;
    v18 = *(_QWORD *)(*v15 + 80);
    v19 = v18;
    v45 = v18 + 8LL * *(unsigned int *)(*v15 + 88);
    if ( v18 != v45 )
    {
      do
      {
        while ( 1 )
        {
          v21 = *(_BYTE *)(v19 + 4);
          v47 = *(_QWORD *)v19;
          v20 = *(_QWORD *)v19;
          if ( (v20 != 1 || v21 == 1) && (v44 == 1 || (unsigned __int8)sub_2ADA390(*v15, v47, *(_QWORD *)(v17 + 32))) )
          {
            v22 = sub_2AE0750((__int64 *)v17, *v15, v47);
            v54.m128i_i8[4] = v21;
            v54.m128i_i64[1] = v22;
            v55.m128i_i32[0] = v23;
            v55.m128i_i64[1] = v42;
            v54.m128i_i32[0] = v20;
            v56.m128i_i32[0] = v43;
            v56.m128i_i32[2] = 0;
            v56.m128i_i8[12] = 0;
            if ( sub_2AB3FE0(v17, (__int64)&v54, (__int64)&v51) )
            {
              v24 = _mm_loadu_si128(&v55);
              v25 = _mm_loadu_si128(&v56);
              v51 = _mm_loadu_si128(&v54);
              v52 = v24;
              v53 = v25;
            }
            if ( sub_2AB3FE0(v17, (__int64)&v54, (__int64)&v48) )
              break;
          }
          v19 += 8;
          if ( v45 == v19 )
            goto LABEL_22;
        }
        v28 = *(unsigned int *)(v17 + 144);
        v29 = &v54;
        v30 = *(_QWORD *)(v17 + 136);
        v31 = v28 + 1;
        if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(v17 + 148) )
        {
          v35 = v17 + 136;
          v36 = (const void *)(v17 + 152);
          if ( v30 > (unsigned __int64)&v54 || (unsigned __int64)&v54 >= v30 + 48 * v28 )
          {
            sub_C8D5F0(v35, v36, v31, 0x30u, v26, v27);
            v30 = *(_QWORD *)(v17 + 136);
            v28 = *(unsigned int *)(v17 + 144);
            v29 = &v54;
          }
          else
          {
            v37 = &v54.m128i_i8[-v30];
            sub_C8D5F0(v35, v36, v31, 0x30u, v26, v27);
            v30 = *(_QWORD *)(v17 + 136);
            v28 = *(unsigned int *)(v17 + 144);
            v29 = (__m128i *)&v37[v30];
          }
        }
        v19 += 8;
        v32 = (__m128i *)(v30 + 48 * v28);
        *v32 = _mm_loadu_si128(v29);
        v32[1] = _mm_loadu_si128(v29 + 1);
        v32[2] = _mm_loadu_si128(v29 + 2);
        ++*(_DWORD *)(v17 + 144);
      }
      while ( v45 != v19 );
LABEL_22:
      v16 = v17;
    }
  }
  v33 = _mm_loadu_si128(&v52);
  *(__m128i *)a1 = _mm_loadu_si128(&v51);
  v34 = _mm_loadu_si128(&v53);
  *(__m128i *)(a1 + 16) = v33;
  *(__m128i *)(a1 + 32) = v34;
  return a1;
}
