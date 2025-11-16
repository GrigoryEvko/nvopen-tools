// Function: sub_2E0F380
// Address: 0x2e0f380
//
void __fastcall sub_2E0F380(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        _QWORD *a6,
        __int128 a7,
        __int64 a8)
{
  __m128i **v9; // rdi
  __int64 v10; // r13
  const __m128i *v11; // rcx
  __int64 v12; // r12
  unsigned __int64 v13; // r14
  __int64 v14; // r13
  unsigned int v15; // esi
  __m128i *v16; // rax
  __m128i *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 *v20; // rcx
  unsigned int v21; // eax
  unsigned int v22; // r8d
  __int64 v23; // rsi
  unsigned int v24; // esi
  __m128i *v25; // r8
  __int64 v26; // rsi
  unsigned int v27; // esi
  __int64 v28; // rdx
  __m128i v29; // xmm1
  __int64 v30; // rdx
  __int64 v31; // rax
  const __m128i *v32; // rdx
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // r8
  __m128i *v35; // rax
  __int64 v36; // rax
  const __m128i *v37; // rdx
  unsigned __int64 v38; // r9
  __m128i *v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdi
  const void *v43; // rsi
  char *v44; // r12
  const void *v45; // rsi
  char *v46; // r12

  v9 = *(__m128i ***)a1;
  if ( v9[12] )
  {
    sub_2E0ED20((__int64)v9, a2, a3, a4, a5, (__int64)a6, a7, a8);
    return;
  }
  v10 = a7;
  if ( (*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_3;
  v28 = *(_DWORD *)((a7 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)a7 >> 1) & 3;
  if ( (*(_DWORD *)((*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)(a1 + 8) >> 1) & 3) > (unsigned int)v28 )
  {
    sub_2E0B930(a1, a2, v28, a7 & 0xFFFFFFFFFFFFFFF8LL, a5);
    v9 = *(__m128i ***)a1;
LABEL_3:
    v11 = *v9;
    *(_QWORD *)(a1 + 24) = *v9;
    *(_QWORD *)(a1 + 16) = v11;
    goto LABEL_4;
  }
  v11 = *(const __m128i **)(a1 + 24);
LABEL_4:
  *(_QWORD *)(a1 + 8) = a7;
  v12 = (__int64)&(*v9)->m128i_i64[3 * *((unsigned int *)v9 + 2)];
  if ( (const __m128i *)v12 == v11 )
  {
    v16 = *(__m128i **)(a1 + 16);
LABEL_51:
    v20 = (__int64 *)v12;
    goto LABEL_25;
  }
  v13 = v10 & 0xFFFFFFFFFFFFFFF8LL;
  v14 = (v10 >> 1) & 3;
  v15 = v14 | *(_DWORD *)(v13 + 24);
  if ( (*(_DWORD *)((v11->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v11->m128i_i64[1] >> 1) & 3) > v15 )
    goto LABEL_14;
  if ( *(const __m128i **)(a1 + 16) != v11 )
  {
    sub_2E0B850((__m128i ***)a1);
    v11 = *(const __m128i **)(a1 + 24);
    v16 = *(__m128i **)(a1 + 16);
    if ( v11 != v16 )
    {
      if ( v11 == (const __m128i *)v12 )
        goto LABEL_53;
      while ( (*(_DWORD *)((v11->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v11->m128i_i64[1] >> 1) & 3) <= ((unsigned int)v14 | *(_DWORD *)(v13 + 24)) )
      {
        *(_QWORD *)(a1 + 24) = (char *)v11 + 24;
        v17 = *(__m128i **)(a1 + 16);
        *(_QWORD *)(a1 + 16) = (char *)v17 + 24;
        *v17 = _mm_loadu_si128(v11);
        v17[1].m128i_i64[0] = v11[1].m128i_i64[0];
        v11 = *(const __m128i **)(a1 + 24);
        if ( v11 == (const __m128i *)v12 )
          goto LABEL_52;
      }
      goto LABEL_12;
    }
    v9 = *(__m128i ***)a1;
  }
  v41 = sub_2E09D00((__int64 *)v9, a7);
  *(_QWORD *)(a1 + 16) = v41;
  v11 = (const __m128i *)v41;
  *(_QWORD *)(a1 + 24) = v41;
LABEL_12:
  if ( (const __m128i *)v12 == v11 )
  {
LABEL_52:
    v16 = *(__m128i **)(a1 + 16);
LABEL_53:
    v9 = *(__m128i ***)a1;
    goto LABEL_51;
  }
  v15 = v14 | *(_DWORD *)(v13 + 24);
LABEL_14:
  v18 = v11->m128i_i64[0];
  v19 = *((_QWORD *)&a7 + 1);
  if ( (*(_DWORD *)((v11->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v11->m128i_i64[0] >> 1) & 3) > v15 )
  {
    v20 = *(__int64 **)(a1 + 24);
    while ( 1 )
    {
LABEL_22:
      if ( v19 == *v20 )
      {
        if ( a8 != v20[2] )
          break;
        v21 = *(_DWORD *)((v19 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v19 >> 1) & 3;
      }
      else
      {
        v21 = *(_DWORD *)((v19 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v19 >> 1) & 3;
        if ( v21 < (*(_DWORD *)((*v20 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v20 >> 1) & 3) )
          break;
      }
      if ( (*(_DWORD *)((v20[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v20[1] >> 1) & 3) > v21 )
        *((_QWORD *)&a7 + 1) = v20[1];
      v20 += 3;
      *(_QWORD *)(a1 + 24) = v20;
      if ( v20 == (__int64 *)v12 )
        break;
      v19 = *((_QWORD *)&a7 + 1);
    }
  }
  else
  {
    if ( (*(_DWORD *)((v11->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v11->m128i_i64[1] >> 1) & 3) >= (*(_DWORD *)((*((_QWORD *)&a7 + 1) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*((__int64 *)&a7 + 1) >> 1) & 3) )
      return;
    v20 = &v11[1].m128i_i64[1];
    *(_QWORD *)&a7 = v18;
    *(_QWORD *)(a1 + 24) = v20;
    if ( v20 != (__int64 *)v12 )
      goto LABEL_22;
  }
  v16 = *(__m128i **)(a1 + 16);
  v9 = *(__m128i ***)a1;
LABEL_25:
  v22 = *(_DWORD *)(a1 + 40);
  if ( !v22 )
    goto LABEL_31;
  a6 = (_QWORD *)(*(_QWORD *)(a1 + 32) + 24LL * v22 - 24);
  v23 = a6[1];
  if ( (_QWORD)a7 == v23 )
  {
    if ( a6[2] != a8 )
      goto LABEL_31;
    v24 = *(_DWORD *)((a7 & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)a7 >> 1) & 3;
  }
  else
  {
    v24 = *(_DWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v23 >> 1) & 3;
    if ( v24 < (*(_DWORD *)((a7 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)a7 >> 1) & 3) )
      goto LABEL_31;
  }
  *(_QWORD *)&a7 = *a6;
  if ( (*(_DWORD *)((*((_QWORD *)&a7 + 1) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*((__int64 *)&a7 + 1) >> 1) & 3) <= v24 )
    *((_QWORD *)&a7 + 1) = a6[1];
  *(_DWORD *)(a1 + 40) = v22 - 1;
LABEL_31:
  v25 = *v9;
  if ( v16 != *v9 )
  {
    v26 = v16[-1].m128i_i64[0];
    if ( (_QWORD)a7 == v26 )
    {
      if ( v16[-1].m128i_i64[1] == a8 )
      {
        v27 = ((__int64)a7 >> 1) & 3 | *(_DWORD *)((a7 & 0xFFFFFFFFFFFFFFF8LL) + 24);
LABEL_34:
        if ( (*(_DWORD *)((*((_QWORD *)&a7 + 1) & 0xFFFFFFFFFFFFFFF8LL) + 24)
            | (unsigned int)(*((__int64 *)&a7 + 1) >> 1) & 3) > v27 )
          v16[-1].m128i_i64[0] = *((_QWORD *)&a7 + 1);
        return;
      }
    }
    else
    {
      v27 = *(_DWORD *)((v26 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v26 >> 1) & 3;
      a6 = (_QWORD *)(a7 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v27 >= (*(_DWORD *)((a7 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)a7 >> 1) & 3) )
        goto LABEL_34;
    }
  }
  if ( v20 == (__int64 *)v16 )
  {
    if ( (__int64 *)v12 == v20 )
    {
      v36 = *((unsigned int *)v9 + 2);
      v37 = (const __m128i *)&a7;
      v38 = v36 + 1;
      if ( v36 + 1 > (unsigned __int64)*((unsigned int *)v9 + 3) )
      {
        v45 = v9 + 2;
        if ( v25 > (__m128i *)&a7 || &a7 >= (__int128 *)&v25->m128i_i8[24 * v36] )
        {
          sub_C8D5F0((__int64)v9, v45, v38, 0x18u, (__int64)v25, v38);
          v37 = (const __m128i *)&a7;
          v25 = *v9;
          v36 = *((unsigned int *)v9 + 2);
        }
        else
        {
          v46 = (char *)((char *)&a7 - (char *)v25);
          sub_C8D5F0((__int64)v9, v45, v38, 0x18u, (__int64)v25, v38);
          v25 = *v9;
          v36 = *((unsigned int *)v9 + 2);
          v37 = (const __m128i *)&v46[(_QWORD)*v9];
        }
      }
      v39 = (__m128i *)((char *)v25 + 24 * v36);
      *v39 = _mm_loadu_si128(v37);
      v39[1].m128i_i64[0] = v37[1].m128i_i64[0];
      ++*((_DWORD *)v9 + 2);
      v40 = **(_QWORD **)a1 + 24LL * *(unsigned int *)(*(_QWORD *)a1 + 8LL);
      *(_QWORD *)(a1 + 24) = v40;
      *(_QWORD *)(a1 + 16) = v40;
    }
    else
    {
      v31 = *(unsigned int *)(a1 + 40);
      v32 = (const __m128i *)&a7;
      v33 = *(_QWORD *)(a1 + 32);
      v34 = v31 + 1;
      if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
      {
        v42 = a1 + 32;
        v43 = (const void *)(a1 + 48);
        if ( v33 > (unsigned __int64)&a7 || (unsigned __int64)&a7 >= v33 + 24 * v31 )
        {
          sub_C8D5F0(v42, v43, v34, 0x18u, v34, (__int64)a6);
          v33 = *(_QWORD *)(a1 + 32);
          v31 = *(unsigned int *)(a1 + 40);
          v32 = (const __m128i *)&a7;
        }
        else
        {
          v44 = (char *)&a7 - v33;
          sub_C8D5F0(v42, v43, v34, 0x18u, v34, (__int64)a6);
          v33 = *(_QWORD *)(a1 + 32);
          v31 = *(unsigned int *)(a1 + 40);
          v32 = (const __m128i *)&v44[v33];
        }
      }
      v35 = (__m128i *)(v33 + 24 * v31);
      *v35 = _mm_loadu_si128(v32);
      v35[1].m128i_i64[0] = v32[1].m128i_i64[0];
      ++*(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v29 = _mm_loadu_si128((const __m128i *)&a7);
    *(_QWORD *)(a1 + 16) = (char *)v16 + 24;
    v30 = a8;
    *v16 = v29;
    v16[1].m128i_i64[0] = v30;
  }
}
