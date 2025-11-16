// Function: sub_18DF0D0
// Address: 0x18df0d0
//
__int64 __fastcall sub_18DF0D0(__int64 a1, __int64 *a2)
{
  __int64 v4; // r14
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r10d
  __int64 *v8; // r13
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // rax
  int v14; // eax
  int v15; // edx
  __int64 v16; // rcx
  __m128i *v17; // r12
  __int64 m128i_i64; // r14
  __m128i *v19; // r15
  const __m128i *v20; // r8
  signed __int64 v21; // rsi
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rdx
  bool v24; // cf
  unsigned __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rdx
  __int8 *v29; // rsi
  __m128i *v30; // rcx
  const __m128i *v31; // rax
  int v32; // eax
  int v33; // ecx
  __int64 v34; // rdi
  unsigned int v35; // eax
  __int64 v36; // rsi
  int v37; // r9d
  __int64 *v38; // r8
  int v39; // eax
  int v40; // eax
  __int64 v41; // rsi
  int v42; // r8d
  unsigned int v43; // r15d
  __int64 *v44; // rdi
  __int64 v45; // rcx
  signed __int64 v46; // [rsp+8h] [rbp-48h]
  const __m128i *v47; // [rsp+10h] [rbp-40h]
  __int64 v48; // [rsp+18h] [rbp-38h]
  __int64 v49; // [rsp+18h] [rbp-38h]

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_44;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( v4 != *v10 )
  {
    while ( v11 != -8 )
    {
      if ( v11 == -16 && !v8 )
        v8 = v10;
      v9 = (v5 - 1) & (v7 + v9);
      v10 = (__int64 *)(v6 + 16LL * v9);
      v11 = *v10;
      if ( v4 == *v10 )
        goto LABEL_3;
      ++v7;
    }
    if ( !v8 )
      v8 = v10;
    v14 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 20) - v15 > v5 >> 3 )
        goto LABEL_15;
      sub_13FEAC0(a1, v5);
      v39 = *(_DWORD *)(a1 + 24);
      if ( v39 )
      {
        v40 = v39 - 1;
        v41 = *(_QWORD *)(a1 + 8);
        v42 = 1;
        v43 = v40 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v15 = *(_DWORD *)(a1 + 16) + 1;
        v44 = 0;
        v8 = (__int64 *)(v41 + 16LL * v43);
        v45 = *v8;
        if ( v4 != *v8 )
        {
          while ( v45 != -8 )
          {
            if ( !v44 && v45 == -16 )
              v44 = v8;
            v43 = v40 & (v42 + v43);
            v8 = (__int64 *)(v41 + 16LL * v43);
            v45 = *v8;
            if ( v4 == *v8 )
              goto LABEL_15;
            ++v42;
          }
          if ( v44 )
            v8 = v44;
        }
        goto LABEL_15;
      }
      goto LABEL_69;
    }
LABEL_44:
    sub_13FEAC0(a1, 2 * v5);
    v32 = *(_DWORD *)(a1 + 24);
    if ( v32 )
    {
      v33 = v32 - 1;
      v34 = *(_QWORD *)(a1 + 8);
      v35 = (v32 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v15 = *(_DWORD *)(a1 + 16) + 1;
      v8 = (__int64 *)(v34 + 16LL * v35);
      v36 = *v8;
      if ( v4 != *v8 )
      {
        v37 = 1;
        v38 = 0;
        while ( v36 != -8 )
        {
          if ( !v38 && v36 == -16 )
            v38 = v8;
          v35 = v33 & (v37 + v35);
          v8 = (__int64 *)(v34 + 16LL * v35);
          v36 = *v8;
          if ( v4 == *v8 )
            goto LABEL_15;
          ++v37;
        }
        if ( v38 )
          v8 = v38;
      }
LABEL_15:
      *(_DWORD *)(a1 + 16) = v15;
      if ( *v8 != -8 )
        --*(_DWORD *)(a1 + 20);
      *v8 = v4;
      *((_DWORD *)v8 + 2) = 0;
      v16 = *a2;
      v17 = *(__m128i **)(a1 + 40);
      if ( v17 != *(__m128i **)(a1 + 48) )
      {
        if ( v17 )
        {
          v17->m128i_i64[0] = v16;
          v17->m128i_i32[2] = 0;
          v17[1].m128i_i64[0] = 0;
          v17[1].m128i_i64[1] = 0;
          v17[2].m128i_i64[0] = 0;
          v17[2].m128i_i32[2] = 0;
          v17 = *(__m128i **)(a1 + 40);
        }
        m128i_i64 = (__int64)v17[3].m128i_i64;
        v19 = *(__m128i **)(a1 + 32);
        *(_QWORD *)(a1 + 40) = v17 + 3;
LABEL_21:
        v12 = -1431655765 * (unsigned int)((m128i_i64 - (__int64)v19) >> 4) - 1;
        *((_DWORD *)v8 + 2) = v12;
        return *(_QWORD *)(a1 + 32) + 48 * v12 + 8;
      }
      v20 = *(const __m128i **)(a1 + 32);
      v21 = (char *)v17 - (char *)v20;
      v22 = 0xAAAAAAAAAAAAAAABLL * (v17 - v20);
      if ( v22 == 0x2AAAAAAAAAAAAAALL )
        sub_4262D8((__int64)"vector::_M_realloc_insert");
      v23 = 1;
      if ( v22 )
        v23 = 0xAAAAAAAAAAAAAAABLL * (v17 - v20);
      v24 = __CFADD__(v23, v22);
      v25 = v23 - 0x5555555555555555LL * (v17 - v20);
      if ( v24 )
      {
        v26 = 0x7FFFFFFFFFFFFFE0LL;
      }
      else
      {
        if ( !v25 )
        {
          m128i_i64 = 48;
          v28 = 0;
          v19 = 0;
          goto LABEL_31;
        }
        if ( v25 > 0x2AAAAAAAAAAAAAALL )
          v25 = 0x2AAAAAAAAAAAAAALL;
        v26 = 48 * v25;
      }
      v46 = (char *)v17 - (char *)v20;
      v47 = *(const __m128i **)(a1 + 32);
      v48 = v16;
      v27 = sub_22077B0(v26);
      v16 = v48;
      v20 = v47;
      v21 = v46;
      v28 = v27 + v26;
      v19 = (__m128i *)v27;
      m128i_i64 = v27 + 48;
LABEL_31:
      v29 = &v19->m128i_i8[v21];
      if ( v29 )
      {
        *(_QWORD *)v29 = v16;
        *((_DWORD *)v29 + 2) = 0;
        *((_QWORD *)v29 + 2) = 0;
        *((_QWORD *)v29 + 3) = 0;
        *((_QWORD *)v29 + 4) = 0;
        *((_DWORD *)v29 + 10) = 0;
      }
      if ( v17 != v20 )
      {
        v30 = v19;
        v31 = v20;
        do
        {
          if ( v30 )
          {
            *v30 = _mm_loadu_si128(v31);
            v30[1] = _mm_loadu_si128(v31 + 1);
            v30[2] = _mm_loadu_si128(v31 + 2);
          }
          v31 += 3;
          v30 += 3;
        }
        while ( v17 != v31 );
        m128i_i64 = (__int64)v19[3
                               * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)&v17[-3] - (char *)v20) >> 4))
                                & 0xFFFFFFFFFFFFFFFLL)
                               + 6].m128i_i64;
      }
      if ( v20 )
      {
        v49 = v28;
        j_j___libc_free_0(v20, *(_QWORD *)(a1 + 48) - (_QWORD)v20);
        v28 = v49;
      }
      *(_QWORD *)(a1 + 32) = v19;
      *(_QWORD *)(a1 + 40) = m128i_i64;
      *(_QWORD *)(a1 + 48) = v28;
      goto LABEL_21;
    }
LABEL_69:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_3:
  v12 = *((unsigned int *)v10 + 2);
  return *(_QWORD *)(a1 + 32) + 48 * v12 + 8;
}
