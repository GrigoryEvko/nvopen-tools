// Function: sub_26134F0
// Address: 0x26134f0
//
__int64 __fastcall sub_26134F0(__int64 a1, __m128i *a2, __int64 *a3)
{
  __int64 v6; // r8
  char v7; // cl
  __m128i *v8; // rdx
  int v9; // esi
  __int64 v10; // r9
  __m128i *v11; // r14
  int v12; // r11d
  int i; // eax
  __m128i *v14; // r10
  __int64 v15; // r15
  int v16; // eax
  unsigned int v17; // esi
  unsigned __int32 v18; // eax
  int v19; // edx
  unsigned int v20; // edi
  __m128i *v21; // rax
  __int64 v22; // rdx
  __m128i *v23; // rdx
  char v24; // al
  __m128i *v26; // rdi
  int v27; // edx
  __int64 v28; // rsi
  int v29; // r9d
  __m128i *v30; // r8
  int j; // eax
  __int64 v32; // r11
  int v33; // eax
  __m128i *v34; // rdi
  int v35; // edx
  __int64 v36; // rsi
  int v37; // r9d
  int k; // eax
  __int64 v39; // r11
  int v40; // eax
  __int32 v41; // edx
  __int32 v42; // edx

  v6 = a2->m128i_i64[0];
  v7 = a2->m128i_i8[8] & 1;
  if ( v7 )
  {
    v8 = a2 + 1;
    v9 = 3;
  }
  else
  {
    v8 = (__m128i *)a2[1].m128i_i64[0];
    v17 = a2[1].m128i_u32[2];
    if ( !v17 )
    {
      v18 = a2->m128i_u32[2];
      v14 = 0;
      a2->m128i_i64[0] = v6 + 1;
      v19 = (v18 >> 1) + 1;
LABEL_14:
      v20 = 3 * v17;
      goto LABEL_15;
    }
    v9 = v17 - 1;
  }
  v10 = a3[1];
  v11 = 0;
  v12 = 1;
  for ( i = v9
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)
              | ((unsigned __int64)(((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)))); ; i = v9 & v16 )
  {
    v14 = &v8[i];
    v15 = v14->m128i_i64[0];
    if ( v14->m128i_i64[0] == *a3 && v14->m128i_i64[1] == v10 )
    {
      if ( v7 )
        v23 = v8 + 4;
      else
        v23 = &v8[a2[1].m128i_u32[2]];
      v24 = 0;
      goto LABEL_22;
    }
    if ( v15 == -4096 )
      break;
    if ( v15 == -8192 && v14->m128i_i64[1] == -8192 && !v11 )
      v11 = &v8[i];
LABEL_10:
    v16 = v12 + i;
    ++v12;
  }
  if ( v14->m128i_i64[1] != -4096 )
    goto LABEL_10;
  v18 = a2->m128i_u32[2];
  if ( v11 )
    v14 = v11;
  a2->m128i_i64[0] = v6 + 1;
  v19 = (v18 >> 1) + 1;
  if ( !v7 )
  {
    v17 = a2[1].m128i_u32[2];
    goto LABEL_14;
  }
  v20 = 12;
  v17 = 4;
LABEL_15:
  if ( 4 * v19 >= v20 )
  {
    sub_2612F80(a2, 2 * v17);
    if ( (a2->m128i_i8[8] & 1) != 0 )
    {
      v26 = a2 + 1;
      v27 = 3;
    }
    else
    {
      v41 = a2[1].m128i_i32[2];
      v26 = (__m128i *)a2[1].m128i_i64[0];
      if ( !v41 )
        goto LABEL_70;
      v27 = v41 - 1;
    }
    v28 = a3[1];
    v29 = 1;
    v30 = 0;
    for ( j = v27
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4)
                | ((unsigned __int64)(((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4)))); ; j = v27 & v33 )
    {
      v14 = &v26[j];
      v32 = v14->m128i_i64[0];
      if ( v14->m128i_i64[0] == *a3 && v14->m128i_i64[1] == v28 )
        break;
      if ( v32 == -4096 )
      {
        if ( v14->m128i_i64[1] == -4096 )
        {
LABEL_65:
          if ( v30 )
            v14 = v30;
          goto LABEL_61;
        }
      }
      else if ( v32 == -8192 && v14->m128i_i64[1] == -8192 && !v30 )
      {
        v30 = &v26[j];
      }
      v33 = v29 + j;
      ++v29;
    }
    goto LABEL_61;
  }
  if ( v17 - a2->m128i_i32[3] - v19 <= v17 >> 3 )
  {
    sub_2612F80(a2, v17);
    if ( (a2->m128i_i8[8] & 1) != 0 )
    {
      v34 = a2 + 1;
      v35 = 3;
      goto LABEL_48;
    }
    v42 = a2[1].m128i_i32[2];
    v34 = (__m128i *)a2[1].m128i_i64[0];
    if ( v42 )
    {
      v35 = v42 - 1;
LABEL_48:
      v36 = a3[1];
      v37 = 1;
      v30 = 0;
      for ( k = v35
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4)
                  | ((unsigned __int64)(((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4)))); ; k = v35 & v40 )
      {
        v14 = &v34[k];
        v39 = v14->m128i_i64[0];
        if ( v14->m128i_i64[0] == *a3 && v14->m128i_i64[1] == v36 )
          break;
        if ( v39 == -4096 )
        {
          if ( v14->m128i_i64[1] == -4096 )
            goto LABEL_65;
        }
        else if ( v39 == -8192 && v14->m128i_i64[1] == -8192 && !v30 )
        {
          v30 = &v34[k];
        }
        v40 = v37 + k;
        ++v37;
      }
LABEL_61:
      v18 = a2->m128i_u32[2];
      goto LABEL_17;
    }
LABEL_70:
    a2->m128i_i32[2] = (2 * ((unsigned __int32)a2->m128i_i32[2] >> 1) + 2) | a2->m128i_i32[2] & 1;
    BUG();
  }
LABEL_17:
  a2->m128i_i32[2] = (2 * (v18 >> 1) + 2) | v18 & 1;
  if ( v14->m128i_i64[0] != -4096 || v14->m128i_i64[1] != -4096 )
    --a2->m128i_i32[3];
  v14->m128i_i64[0] = *a3;
  v14->m128i_i64[1] = a3[1];
  if ( (a2->m128i_i8[8] & 1) != 0 )
  {
    v21 = a2 + 1;
    v22 = 4;
  }
  else
  {
    v21 = (__m128i *)a2[1].m128i_i64[0];
    v22 = a2[1].m128i_u32[2];
  }
  v6 = a2->m128i_i64[0];
  v23 = &v21[v22];
  v24 = 1;
LABEL_22:
  *(_QWORD *)a1 = a2;
  *(_BYTE *)(a1 + 32) = v24;
  *(_QWORD *)(a1 + 8) = v6;
  *(_QWORD *)(a1 + 16) = v14;
  *(_QWORD *)(a1 + 24) = v23;
  return a1;
}
