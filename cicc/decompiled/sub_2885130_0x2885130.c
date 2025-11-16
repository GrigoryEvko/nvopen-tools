// Function: sub_2885130
// Address: 0x2885130
//
void __fastcall sub_2885130(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v8; // edx
  __int64 v9; // rcx
  char *v10; // rax
  __int64 v11; // rbx
  __int64 v12; // r13
  char *v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rdx
  char *v16; // rdx
  unsigned int v17; // eax
  unsigned int v18; // esi
  __int64 v19; // r8
  int v20; // ebx
  __int64 v21; // r11
  __int64 v22; // rdi
  unsigned int i; // eax
  __int64 *v24; // r9
  __int64 v25; // r14
  unsigned int v26; // eax
  int v27; // r8d
  __int64 v28; // rcx
  int v29; // r10d
  __int64 v30; // rsi
  __int64 v31; // rdi
  unsigned int v32; // eax
  unsigned int v33; // eax
  int v34; // edx
  __int64 v35; // rax
  __m128i si128; // xmm0
  int v37; // r8d
  __int64 v38; // rcx
  __int64 v39; // rsi
  int v40; // r10d
  unsigned int j; // eax
  __int64 v42; // rdi
  unsigned int v43; // eax
  __m128i v44[3]; // [rsp+0h] [rbp-30h] BYREF

  v8 = *(_DWORD *)(a1 + 16);
  if ( !v8 )
  {
    v9 = *(unsigned int *)(a1 + 40);
    v10 = *(char **)(a1 + 32);
    v11 = a2->m128i_i64[0];
    v12 = a2->m128i_i64[1];
    v13 = &v10[16 * v9];
    v14 = (16 * v9) >> 4;
    v15 = (16 * v9) >> 6;
    if ( v15 )
    {
      v16 = &v10[64 * v15];
      while ( *(_QWORD *)v10 != v11 || *((_QWORD *)v10 + 1) != v12 )
      {
        if ( *((_QWORD *)v10 + 2) == v11 && *((_QWORD *)v10 + 3) == v12 )
        {
          if ( v13 != v10 + 16 )
            return;
          goto LABEL_15;
        }
        if ( *((_QWORD *)v10 + 4) == v11 && *((_QWORD *)v10 + 5) == v12 )
        {
          if ( v13 != v10 + 32 )
            return;
          goto LABEL_15;
        }
        if ( *((_QWORD *)v10 + 6) == v11 && *((_QWORD *)v10 + 7) == v12 )
        {
          if ( v13 != v10 + 48 )
            return;
          goto LABEL_15;
        }
        v10 += 64;
        if ( v16 == v10 )
        {
          v14 = (v13 - v10) >> 4;
          goto LABEL_10;
        }
      }
LABEL_20:
      if ( v13 != v10 )
        return;
      goto LABEL_15;
    }
LABEL_10:
    if ( v14 != 2 )
    {
      if ( v14 != 3 )
      {
        if ( v14 != 1 )
          goto LABEL_15;
        goto LABEL_13;
      }
      if ( *(_QWORD *)v10 == v11 && *((_QWORD *)v10 + 1) == v12 )
        goto LABEL_20;
      v10 += 16;
    }
    if ( *(_QWORD *)v10 == v11 && *((_QWORD *)v10 + 1) == v12 )
      goto LABEL_20;
    v10 += 16;
LABEL_13:
    if ( *(_QWORD *)v10 == v11 && *((_QWORD *)v10 + 1) == v12 )
      goto LABEL_20;
LABEL_15:
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v9 + 1, 0x10u, a5, a6);
      v13 = (char *)(*(_QWORD *)(a1 + 32) + 16LL * *(unsigned int *)(a1 + 40));
    }
    *(_QWORD *)v13 = v11;
    *((_QWORD *)v13 + 1) = v12;
    v17 = *(_DWORD *)(a1 + 40) + 1;
    *(_DWORD *)(a1 + 40) = v17;
    if ( v17 > 4 )
      sub_2884DD0(a1);
    return;
  }
  v18 = *(_DWORD *)(a1 + 24);
  if ( !v18 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_48;
  }
  v19 = *(_QWORD *)(a1 + 8);
  v20 = 1;
  v21 = 0;
  v22 = a2->m128i_i64[1];
  for ( i = (v18 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)
              | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)))); ; i = (v18 - 1) & v26 )
  {
    v24 = (__int64 *)(v19 + 16LL * i);
    v25 = *v24;
    if ( *v24 == a2->m128i_i64[0] && v24[1] == v22 )
      return;
    if ( v25 == -4096 )
      break;
    if ( v25 == -8192 && v24[1] == -8192 && !v21 )
      v21 = v19 + 16LL * i;
LABEL_39:
    v26 = v20 + i;
    ++v20;
  }
  if ( v24[1] != -4096 )
    goto LABEL_39;
  if ( !v21 )
    v21 = v19 + 16LL * i;
  v34 = v8 + 1;
  ++*(_QWORD *)a1;
  if ( 4 * v34 < 3 * v18 )
  {
    if ( v18 - *(_DWORD *)(a1 + 20) - v34 > v18 >> 3 )
      goto LABEL_64;
    sub_2884B10(a1, v18);
    v37 = *(_DWORD *)(a1 + 24);
    if ( v37 )
    {
      v38 = a2->m128i_i64[1];
      v19 = (unsigned int)(v37 - 1);
      v21 = 0;
      v40 = 1;
      for ( j = v19
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4)
                  | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4)))); ; j = v19 & v43 )
      {
        v39 = *(_QWORD *)(a1 + 8);
        v42 = v39 + 16LL * j;
        v24 = *(__int64 **)v42;
        if ( *(_QWORD *)v42 == a2->m128i_i64[0] && *(_QWORD *)(v42 + 8) == v38 )
        {
          v21 = v39 + 16LL * j;
          v34 = *(_DWORD *)(a1 + 16) + 1;
          goto LABEL_64;
        }
        if ( v24 == (__int64 *)-4096LL )
        {
          if ( *(_QWORD *)(v42 + 8) == -4096 )
          {
            if ( !v21 )
              v21 = v39 + 16LL * j;
            v34 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_64;
          }
        }
        else if ( v24 == (__int64 *)-8192LL && *(_QWORD *)(v42 + 8) == -8192 && !v21 )
        {
          v21 = v39 + 16LL * j;
        }
        v43 = v40 + j;
        ++v40;
      }
    }
LABEL_94:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_48:
  sub_2884B10(a1, 2 * v18);
  v27 = *(_DWORD *)(a1 + 24);
  if ( !v27 )
    goto LABEL_94;
  v28 = a2->m128i_i64[1];
  v19 = (unsigned int)(v27 - 1);
  v29 = 1;
  v31 = 0;
  v32 = v19
      & (((0xBF58476D1CE4E5B9LL
         * (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4)
          | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))) >> 31)
       ^ (484763065 * (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4))));
  while ( 2 )
  {
    v30 = *(_QWORD *)(a1 + 8);
    v21 = v30 + 16LL * v32;
    v24 = *(__int64 **)v21;
    if ( *(_QWORD *)v21 == a2->m128i_i64[0] && *(_QWORD *)(v21 + 8) == v28 )
    {
      v34 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_64;
    }
    if ( v24 != (__int64 *)-4096LL )
    {
      if ( v24 == (__int64 *)-8192LL && *(_QWORD *)(v21 + 8) == -8192 && !v31 )
        v31 = v30 + 16LL * v32;
      goto LABEL_56;
    }
    if ( *(_QWORD *)(v21 + 8) != -4096 )
    {
LABEL_56:
      v33 = v29 + v32;
      ++v29;
      v32 = v19 & v33;
      continue;
    }
    break;
  }
  if ( v31 )
    v21 = v31;
  v34 = *(_DWORD *)(a1 + 16) + 1;
LABEL_64:
  *(_DWORD *)(a1 + 16) = v34;
  if ( *(_QWORD *)v21 != -4096 || *(_QWORD *)(v21 + 8) != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v21 = a2->m128i_i64[0];
  *(_QWORD *)(v21 + 8) = a2->m128i_i64[1];
  v35 = *(unsigned int *)(a1 + 40);
  si128 = _mm_loadu_si128(a2);
  if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    v44[0] = si128;
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v35 + 1, 0x10u, v19, (__int64)v24);
    v35 = *(unsigned int *)(a1 + 40);
    si128 = _mm_load_si128(v44);
  }
  *(__m128i *)(*(_QWORD *)(a1 + 32) + 16 * v35) = si128;
  ++*(_DWORD *)(a1 + 40);
}
