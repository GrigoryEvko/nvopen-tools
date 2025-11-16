// Function: sub_C16570
// Address: 0xc16570
//
__int64 *__fastcall sub_C16570(__int64 *a1, __int64 a2, unsigned __int64 *a3, __int64 a4)
{
  unsigned __int64 *v5; // r9
  unsigned __int64 v6; // r13
  int *v7; // r14
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rcx
  const __m128i *v11; // rax
  __m128i *v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // r13
  __int64 *v16; // rbx
  __int64 *v17; // rdi
  __int64 v18; // rax
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // r14
  __int64 v21; // r13
  unsigned __int64 *v23; // r9
  unsigned __int64 v24; // r13
  int *v25; // r14
  __int64 v26; // rdx
  __int64 v27; // r8
  __int64 v28; // rcx
  const __m128i *v29; // rax
  __m128i *v30; // rdx
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // r12
  unsigned int *v34; // rbx
  __int64 *v35; // r14
  __int64 *v36; // rdi
  __int64 v37; // rax
  unsigned __int64 v38; // r14
  unsigned int *v39; // r13
  __int64 v40; // rbx
  __int64 *v41; // [rsp+8h] [rbp-108h]
  __int64 *v42; // [rsp+8h] [rbp-108h]
  __int64 *v43; // [rsp+10h] [rbp-100h]
  __int64 *v44; // [rsp+10h] [rbp-100h]
  unsigned __int64 v45; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v46; // [rsp+18h] [rbp-F8h]
  __int64 v47; // [rsp+28h] [rbp-E8h]
  __int64 v48; // [rsp+28h] [rbp-E8h]
  unsigned __int64 *v49; // [rsp+28h] [rbp-E8h]
  unsigned __int64 *v50; // [rsp+28h] [rbp-E8h]
  __int64 v51; // [rsp+30h] [rbp-E0h] BYREF
  _BYTE v52[216]; // [rsp+38h] [rbp-D8h] BYREF

  if ( a4 != 2 )
  {
    if ( a4 != 3 )
      BUG();
    v23 = a3 + 1;
    *a1 = (__int64)(a1 + 2);
    v44 = a1 + 2;
    a1[1] = 0x100000000LL;
    a1[23] = (__int64)(a1 + 25);
    v42 = a1 + 25;
    a1[24] = 0x600000000LL;
    v46 = *a3;
    if ( *a3 > 1 )
    {
      v50 = a3 + 1;
      sub_C8D5F0(a1, v44, *a3, 168);
      v23 = v50;
    }
    else if ( !v46 )
    {
      v31 = 6;
      v32 = 6;
LABEL_26:
      v33 = *v23;
      v34 = (unsigned int *)(v23 + 1);
      if ( *v23 > v31 )
      {
        v35 = a1 + 23;
        sub_C8D5F0(a1 + 23, v42, *v23, 8);
        v32 = *((unsigned int *)a1 + 49);
      }
      else
      {
        if ( !v33 )
          return a1;
        v35 = a1 + 23;
      }
      v36 = v35;
      v37 = *((unsigned int *)a1 + 48);
      v38 = 0;
      v39 = v34;
      while ( 1 )
      {
        v40 = *v39++;
        if ( v37 + 1 > v32 )
        {
          sub_C8D5F0(v36, v42, v37 + 1, 8);
          v37 = *((unsigned int *)a1 + 48);
        }
        ++v38;
        *(_QWORD *)(a1[23] + 8 * v37) = v40;
        v37 = (unsigned int)(*((_DWORD *)a1 + 48) + 1);
        *((_DWORD *)a1 + 48) = v37;
        if ( v33 <= v38 )
          break;
        v32 = *((unsigned int *)a1 + 49);
      }
      return a1;
    }
    v24 = 0;
    v25 = (int *)v23;
    do
    {
      memset(v52, 0, 0xA0u);
      v51 = (unsigned int)*v25;
      sub_C162E0((__int64)v52, a2, v25 + 1);
      v25 = (int *)((char *)v25 + sub_C16510(a2) + 4);
      v26 = *((unsigned int *)a1 + 2);
      v27 = v26 + 1;
      if ( v26 + 1 > (unsigned __int64)*((unsigned int *)a1 + 3) )
      {
        if ( *a1 > (unsigned __int64)&v51 || (v48 = *a1, (unsigned __int64)&v51 >= *a1 + 168 * v26) )
        {
          sub_C8D5F0(a1, v44, v27, 168);
          v28 = *a1;
          v26 = *((unsigned int *)a1 + 2);
          v29 = (const __m128i *)&v51;
        }
        else
        {
          sub_C8D5F0(a1, v44, v27, 168);
          v28 = *a1;
          v26 = *((unsigned int *)a1 + 2);
          v29 = (const __m128i *)&v52[*a1 - 8 - v48];
        }
      }
      else
      {
        v28 = *a1;
        v29 = (const __m128i *)&v51;
      }
      ++v24;
      v30 = (__m128i *)(v28 + 168 * v26);
      *v30 = _mm_loadu_si128(v29);
      v30[1] = _mm_loadu_si128(v29 + 1);
      v30[2] = _mm_loadu_si128(v29 + 2);
      v30[3] = _mm_loadu_si128(v29 + 3);
      v30[4] = _mm_loadu_si128(v29 + 4);
      v30[5] = _mm_loadu_si128(v29 + 5);
      v30[6] = _mm_loadu_si128(v29 + 6);
      v30[7] = _mm_loadu_si128(v29 + 7);
      v30[8] = _mm_loadu_si128(v29 + 8);
      v30[9] = _mm_loadu_si128(v29 + 9);
      v30[10].m128i_i64[0] = v29[10].m128i_i64[0];
      ++*((_DWORD *)a1 + 2);
    }
    while ( v46 > v24 );
    v31 = *((unsigned int *)a1 + 49);
    v23 = (unsigned __int64 *)v25;
    v32 = v31;
    goto LABEL_26;
  }
  v5 = a3 + 1;
  *a1 = (__int64)(a1 + 2);
  v43 = a1 + 2;
  a1[1] = 0x100000000LL;
  a1[23] = (__int64)(a1 + 25);
  v41 = a1 + 25;
  a1[24] = 0x600000000LL;
  v45 = *a3;
  if ( *a3 > 1 )
  {
    v49 = a3 + 1;
    sub_C8D5F0(a1, v43, *a3, 168);
    v5 = v49;
LABEL_4:
    v6 = 0;
    v7 = (int *)v5;
    do
    {
      memset(v52, 0, 0xA0u);
      v51 = *(_QWORD *)v7;
      sub_C162E0((__int64)v52, a2, v7 + 2);
      v7 = (int *)((char *)v7 + sub_C16510(a2) + 8);
      v8 = *((unsigned int *)a1 + 2);
      v9 = v8 + 1;
      if ( v8 + 1 > (unsigned __int64)*((unsigned int *)a1 + 3) )
      {
        if ( *a1 > (unsigned __int64)&v51 || (unsigned __int64)&v51 >= *a1 + 168 * v8 )
        {
          sub_C8D5F0(a1, v43, v9, 168);
          v10 = *a1;
          v8 = *((unsigned int *)a1 + 2);
          v11 = (const __m128i *)&v51;
        }
        else
        {
          v47 = *a1;
          sub_C8D5F0(a1, v43, v9, 168);
          v10 = *a1;
          v8 = *((unsigned int *)a1 + 2);
          v11 = (const __m128i *)&v52[*a1 - 8 - v47];
        }
      }
      else
      {
        v10 = *a1;
        v11 = (const __m128i *)&v51;
      }
      ++v6;
      v12 = (__m128i *)(v10 + 168 * v8);
      *v12 = _mm_loadu_si128(v11);
      v12[1] = _mm_loadu_si128(v11 + 1);
      v12[2] = _mm_loadu_si128(v11 + 2);
      v12[3] = _mm_loadu_si128(v11 + 3);
      v12[4] = _mm_loadu_si128(v11 + 4);
      v12[5] = _mm_loadu_si128(v11 + 5);
      v12[6] = _mm_loadu_si128(v11 + 6);
      v12[7] = _mm_loadu_si128(v11 + 7);
      v12[8] = _mm_loadu_si128(v11 + 8);
      v12[9] = _mm_loadu_si128(v11 + 9);
      v12[10].m128i_i64[0] = v11[10].m128i_i64[0];
      ++*((_DWORD *)a1 + 2);
    }
    while ( v45 > v6 );
    v13 = *((unsigned int *)a1 + 49);
    v5 = (unsigned __int64 *)v7;
    v14 = v13;
    goto LABEL_9;
  }
  if ( v45 )
    goto LABEL_4;
  v13 = 6;
  v14 = 6;
LABEL_9:
  v15 = *v5;
  v16 = (__int64 *)(v5 + 1);
  if ( *v5 > v13 )
  {
    sub_C8D5F0(a1 + 23, v41, *v5, 8);
    v14 = *((unsigned int *)a1 + 49);
    v17 = a1 + 23;
  }
  else
  {
    if ( !v15 )
      return a1;
    v17 = a1 + 23;
  }
  v18 = *((unsigned int *)a1 + 48);
  v19 = 0;
  v20 = v15;
  while ( 1 )
  {
    v21 = *v16++;
    if ( v18 + 1 > v14 )
    {
      sub_C8D5F0(v17, v41, v18 + 1, 8);
      v18 = *((unsigned int *)a1 + 48);
    }
    ++v19;
    *(_QWORD *)(a1[23] + 8 * v18) = v21;
    v18 = (unsigned int)(*((_DWORD *)a1 + 48) + 1);
    *((_DWORD *)a1 + 48) = v18;
    if ( v20 <= v19 )
      break;
    v14 = *((unsigned int *)a1 + 49);
  }
  return a1;
}
