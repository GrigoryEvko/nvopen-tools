// Function: sub_1DB8AC0
// Address: 0x1db8ac0
//
void __fastcall sub_1DB8AC0(
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
  __int64 v25; // r8
  __int64 v26; // rsi
  unsigned int v27; // esi
  __m128i v28; // xmm1
  __int64 v29; // rdx
  __int64 v30; // rax
  __m128i *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  __m128i *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax

  v9 = *(__m128i ***)a1;
  if ( v9[12] )
  {
    sub_1DB82B0((__int64)v9, a2, a3, a4, a5, (__int64)a6, a7, a8);
    return;
  }
  v10 = a7;
  if ( (*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_3;
  if ( (*(_DWORD *)((*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)(a1 + 8) >> 1) & 3) > (*(_DWORD *)((a7 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)a7 >> 1) & 3) )
  {
    sub_1DB55F0(a1);
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
  if ( v11 == (const __m128i *)v12 )
  {
    v16 = *(__m128i **)(a1 + 16);
LABEL_52:
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
    sub_1DB5510((__m128i ***)a1);
    v11 = *(const __m128i **)(a1 + 24);
    v16 = *(__m128i **)(a1 + 16);
    if ( v11 != v16 )
    {
      if ( v11 == (const __m128i *)v12 )
        goto LABEL_54;
      while ( (*(_DWORD *)((v11->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v11->m128i_i64[1] >> 1) & 3) <= ((unsigned int)v14 | *(_DWORD *)(v13 + 24)) )
      {
        *(_QWORD *)(a1 + 24) = (char *)v11 + 24;
        v17 = *(__m128i **)(a1 + 16);
        *(_QWORD *)(a1 + 16) = (char *)v17 + 24;
        *v17 = _mm_loadu_si128(v11);
        v17[1].m128i_i64[0] = v11[1].m128i_i64[0];
        v11 = *(const __m128i **)(a1 + 24);
        if ( v11 == (const __m128i *)v12 )
          goto LABEL_53;
      }
      goto LABEL_12;
    }
    v9 = *(__m128i ***)a1;
  }
  v37 = sub_1DB3C70((__int64 *)v9, a7);
  *(_QWORD *)(a1 + 16) = v37;
  v11 = (const __m128i *)v37;
  *(_QWORD *)(a1 + 24) = v37;
LABEL_12:
  if ( (const __m128i *)v12 == v11 )
  {
LABEL_53:
    v16 = *(__m128i **)(a1 + 16);
LABEL_54:
    v9 = *(__m128i ***)a1;
    goto LABEL_52;
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
  v25 = (__int64)*v9;
  if ( *v9 != v16 )
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
      LODWORD(a6) = a7 & 0xFFFFFFF8;
      if ( v27 >= (*(_DWORD *)((a7 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)a7 >> 1) & 3) )
        goto LABEL_34;
    }
  }
  if ( v20 == (__int64 *)v16 )
  {
    if ( (__int64 *)v12 == v20 )
    {
      v33 = *((unsigned int *)v9 + 2);
      if ( (unsigned int)v33 >= *((_DWORD *)v9 + 3) )
      {
        sub_16CD150((__int64)v9, v9 + 2, 0, 24, v25, (int)a6);
        v25 = (__int64)*v9;
        v33 = *((unsigned int *)v9 + 2);
      }
      v34 = a8;
      v35 = (__m128i *)(v25 + 24 * v33);
      *v35 = _mm_loadu_si128((const __m128i *)&a7);
      v35[1].m128i_i64[0] = v34;
      ++*((_DWORD *)v9 + 2);
      v36 = **(_QWORD **)a1 + 24LL * *(unsigned int *)(*(_QWORD *)a1 + 8LL);
      *(_QWORD *)(a1 + 24) = v36;
      *(_QWORD *)(a1 + 16) = v36;
    }
    else
    {
      v30 = *(unsigned int *)(a1 + 40);
      if ( (unsigned int)v30 >= *(_DWORD *)(a1 + 44) )
      {
        sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 24, v25, (int)a6);
        v30 = *(unsigned int *)(a1 + 40);
      }
      v31 = (__m128i *)(*(_QWORD *)(a1 + 32) + 24 * v30);
      v32 = a8;
      *v31 = _mm_loadu_si128((const __m128i *)&a7);
      v31[1].m128i_i64[0] = v32;
      ++*(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v28 = _mm_loadu_si128((const __m128i *)&a7);
    *(_QWORD *)(a1 + 16) = (char *)v16 + 24;
    v29 = a8;
    *v16 = v28;
    v16[1].m128i_i64[0] = v29;
  }
}
