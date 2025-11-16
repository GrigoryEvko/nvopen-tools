// Function: sub_14090C0
// Address: 0x14090c0
//
__int64 __fastcall sub_14090C0(const __m128i *a1, const __m128i *a2)
{
  char v4; // dl
  const __m128i *v5; // r8
  int v6; // esi
  __int64 v7; // rdi
  int v8; // r11d
  __int64 v9; // r9
  __m128i *v10; // r10
  unsigned __int64 v11; // r9
  unsigned __int64 v12; // r9
  int i; // eax
  const __m128i *v14; // r9
  __int64 v15; // r13
  int v16; // eax
  unsigned int v17; // esi
  unsigned __int32 v18; // eax
  int v19; // ecx
  unsigned int v20; // edi
  __int64 v21; // rax
  const __m128i *v23; // rdi
  int v24; // edx
  __int64 v25; // rsi
  int v26; // r11d
  __int64 v27; // r8
  const __m128i *v28; // r9
  unsigned __int64 v29; // r8
  unsigned __int64 v30; // r8
  int j; // eax
  __int64 v32; // r8
  int v33; // eax
  const __m128i *v34; // rdi
  int v35; // edx
  __int64 v36; // rsi
  int v37; // r11d
  __int64 v38; // r8
  unsigned __int64 v39; // r8
  unsigned __int64 v40; // r8
  int k; // eax
  __int64 v42; // r8
  int v43; // eax
  __int32 v44; // edx
  __int32 v45; // edx

  v4 = a1->m128i_i8[8] & 1;
  if ( v4 )
  {
    v5 = a1 + 1;
    v6 = 3;
  }
  else
  {
    v17 = a1[1].m128i_u32[2];
    v5 = (const __m128i *)a1[1].m128i_i64[0];
    if ( !v17 )
    {
      v18 = a1->m128i_u32[2];
      ++a1->m128i_i64[0];
      v10 = 0;
      v19 = (v18 >> 1) + 1;
LABEL_14:
      v20 = 3 * v17;
      goto LABEL_15;
    }
    v6 = v17 - 1;
  }
  v7 = a2->m128i_i64[1];
  v8 = 1;
  v9 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
  v10 = 0;
  v11 = (((v9
         | ((unsigned __int64)((unsigned int)a2->m128i_i64[0] ^ (unsigned int)((unsigned __int64)a2->m128i_i64[0] >> 9)) << 32))
        - 1
        - (v9 << 32)) >> 22)
      ^ ((v9
        | ((unsigned __int64)((unsigned int)a2->m128i_i64[0] ^ (unsigned int)((unsigned __int64)a2->m128i_i64[0] >> 9)) << 32))
       - 1
       - (v9 << 32));
  v12 = ((9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13)))) >> 15)
      ^ (9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13))));
  for ( i = v6 & (((v12 - 1 - (v12 << 27)) >> 31) ^ (v12 - 1 - ((_DWORD)v12 << 27))); ; i = v6 & v16 )
  {
    v14 = &v5[i];
    v15 = v14->m128i_i64[0];
    if ( v14->m128i_i64[0] == a2->m128i_i64[0] && v14->m128i_i64[1] == v7 )
      return 0;
    if ( v15 == -2 )
      break;
    if ( v15 == -16 && v14->m128i_i64[1] == -16 && !v10 )
      v10 = (__m128i *)&v5[i];
LABEL_10:
    v16 = v8 + i;
    ++v8;
  }
  if ( v14->m128i_i64[1] != -8 )
    goto LABEL_10;
  v18 = a1->m128i_u32[2];
  if ( !v10 )
    v10 = (__m128i *)v14;
  ++a1->m128i_i64[0];
  v19 = (v18 >> 1) + 1;
  if ( !v4 )
  {
    v17 = a1[1].m128i_u32[2];
    goto LABEL_14;
  }
  v20 = 12;
  v17 = 4;
LABEL_15:
  if ( 4 * v19 >= v20 )
  {
    sub_14088C0(a1, 2 * v17);
    if ( (a1->m128i_i8[8] & 1) != 0 )
    {
      v23 = a1 + 1;
      v24 = 3;
    }
    else
    {
      v44 = a1[1].m128i_i32[2];
      v23 = (const __m128i *)a1[1].m128i_i64[0];
      if ( !v44 )
        goto LABEL_65;
      v24 = v44 - 1;
    }
    v25 = a2->m128i_i64[1];
    v26 = 1;
    v27 = ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4);
    v28 = 0;
    v29 = (((v27
           | ((unsigned __int64)((unsigned int)a2->m128i_i64[0] ^ (unsigned int)((unsigned __int64)a2->m128i_i64[0] >> 9)) << 32))
          - 1
          - (v27 << 32)) >> 22)
        ^ ((v27
          | ((unsigned __int64)((unsigned int)a2->m128i_i64[0] ^ (unsigned int)((unsigned __int64)a2->m128i_i64[0] >> 9)) << 32))
         - 1
         - (v27 << 32));
    v30 = ((9 * (((v29 - 1 - (v29 << 13)) >> 8) ^ (v29 - 1 - (v29 << 13)))) >> 15)
        ^ (9 * (((v29 - 1 - (v29 << 13)) >> 8) ^ (v29 - 1 - (v29 << 13))));
    for ( j = v24 & (((v30 - 1 - (v30 << 27)) >> 31) ^ (v30 - 1 - ((_DWORD)v30 << 27))); ; j = v24 & v33 )
    {
      v10 = (__m128i *)&v23[j];
      v32 = v10->m128i_i64[0];
      if ( v10->m128i_i64[0] == a2->m128i_i64[0] && v10->m128i_i64[1] == v25 )
        break;
      if ( v32 == -2 )
      {
        if ( v10->m128i_i64[1] == -8 )
        {
LABEL_60:
          if ( v28 )
            v10 = (__m128i *)v28;
          goto LABEL_56;
        }
      }
      else if ( v32 == -16 && v10->m128i_i64[1] == -16 && !v28 )
      {
        v28 = &v23[j];
      }
      v33 = v26 + j;
      ++v26;
    }
    goto LABEL_56;
  }
  if ( v17 - a1->m128i_i32[3] - v19 <= v17 >> 3 )
  {
    sub_14088C0(a1, v17);
    if ( (a1->m128i_i8[8] & 1) != 0 )
    {
      v34 = a1 + 1;
      v35 = 3;
      goto LABEL_43;
    }
    v45 = a1[1].m128i_i32[2];
    v34 = (const __m128i *)a1[1].m128i_i64[0];
    if ( v45 )
    {
      v35 = v45 - 1;
LABEL_43:
      v36 = a2->m128i_i64[1];
      v37 = 1;
      v38 = ((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4);
      v28 = 0;
      v39 = (((v38
             | ((unsigned __int64)((unsigned int)a2->m128i_i64[0]
                                 ^ (unsigned int)((unsigned __int64)a2->m128i_i64[0] >> 9)) << 32))
            - 1
            - (v38 << 32)) >> 22)
          ^ ((v38
            | ((unsigned __int64)((unsigned int)a2->m128i_i64[0]
                                ^ (unsigned int)((unsigned __int64)a2->m128i_i64[0] >> 9)) << 32))
           - 1
           - (v38 << 32));
      v40 = ((9 * (((v39 - 1 - (v39 << 13)) >> 8) ^ (v39 - 1 - (v39 << 13)))) >> 15)
          ^ (9 * (((v39 - 1 - (v39 << 13)) >> 8) ^ (v39 - 1 - (v39 << 13))));
      for ( k = v35 & (((v40 - 1 - (v40 << 27)) >> 31) ^ (v40 - 1 - ((_DWORD)v40 << 27))); ; k = v35 & v43 )
      {
        v10 = (__m128i *)&v34[k];
        v42 = v10->m128i_i64[0];
        if ( v10->m128i_i64[0] == a2->m128i_i64[0] && v10->m128i_i64[1] == v36 )
          break;
        if ( v42 == -2 )
        {
          if ( v10->m128i_i64[1] == -8 )
            goto LABEL_60;
        }
        else if ( v42 == -16 && v10->m128i_i64[1] == -16 && !v28 )
        {
          v28 = &v34[k];
        }
        v43 = v37 + k;
        ++v37;
      }
LABEL_56:
      v18 = a1->m128i_u32[2];
      goto LABEL_17;
    }
LABEL_65:
    a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
    BUG();
  }
LABEL_17:
  a1->m128i_i32[2] = (2 * (v18 >> 1) + 2) | v18 & 1;
  if ( v10->m128i_i64[0] != -2 || v10->m128i_i64[1] != -8 )
    --a1->m128i_i32[3];
  v10->m128i_i64[0] = a2->m128i_i64[0];
  v10->m128i_i64[1] = a2->m128i_i64[1];
  v21 = a1[5].m128i_u32[2];
  if ( (unsigned int)v21 >= a1[5].m128i_i32[3] )
  {
    sub_16CD150(&a1[5], &a1[6], 0, 16);
    v21 = a1[5].m128i_u32[2];
  }
  *(__m128i *)(a1[5].m128i_i64[0] + 16 * v21) = _mm_loadu_si128(a2);
  ++a1[5].m128i_i32[2];
  return 1;
}
