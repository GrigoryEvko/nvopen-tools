// Function: sub_1AA79E0
// Address: 0x1aa79e0
//
__int64 __fastcall sub_1AA79E0(__int64 a1, __m128i *a2, __int64 *a3)
{
  __int64 v6; // r8
  char v7; // cl
  __m128i *v8; // rdx
  int v9; // esi
  __int64 v10; // r9
  __m128i *v11; // r14
  __int64 v12; // r10
  int v13; // r11d
  unsigned __int64 v14; // r10
  unsigned __int64 v15; // r10
  int i; // eax
  __m128i *v17; // r10
  __int64 v18; // r15
  int v19; // eax
  unsigned int v20; // esi
  unsigned __int32 v21; // eax
  int v22; // edx
  unsigned int v23; // edi
  __m128i *v24; // rax
  __int64 v25; // rdx
  __m128i *v26; // rdx
  char v27; // al
  __m128i *v29; // rdi
  int v30; // edx
  __int64 v31; // rsi
  __int64 v32; // r8
  int v33; // r9d
  unsigned __int64 v34; // r8
  unsigned __int64 v35; // r8
  int v36; // eax
  __m128i *v37; // r8
  unsigned int j; // eax
  __int64 v39; // r11
  unsigned int v40; // eax
  __m128i *v41; // rdi
  int v42; // edx
  __int64 v43; // rsi
  __int64 v44; // r8
  int v45; // r9d
  unsigned __int64 v46; // r8
  unsigned __int64 v47; // r8
  int v48; // eax
  unsigned int k; // eax
  __int64 v50; // r11
  unsigned int v51; // eax
  __int32 v52; // edx
  __int32 v53; // edx

  v6 = a2->m128i_i64[0];
  v7 = a2->m128i_i8[8] & 1;
  if ( v7 )
  {
    v8 = a2 + 1;
    v9 = 1;
  }
  else
  {
    v8 = (__m128i *)a2[1].m128i_i64[0];
    v20 = a2[1].m128i_u32[2];
    if ( !v20 )
    {
      v21 = a2->m128i_u32[2];
      v17 = 0;
      a2->m128i_i64[0] = v6 + 1;
      v22 = (v21 >> 1) + 1;
LABEL_14:
      v23 = 3 * v20;
      goto LABEL_15;
    }
    v9 = v20 - 1;
  }
  v10 = a3[1];
  v11 = 0;
  v12 = ((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4);
  v13 = 1;
  v14 = (((v12 | ((unsigned __int64)(((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4)) << 32)) - 1 - (v12 << 32)) >> 22)
      ^ ((v12 | ((unsigned __int64)(((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4)) << 32)) - 1 - (v12 << 32));
  v15 = ((9 * (((v14 - 1 - (v14 << 13)) >> 8) ^ (v14 - 1 - (v14 << 13)))) >> 15)
      ^ (9 * (((v14 - 1 - (v14 << 13)) >> 8) ^ (v14 - 1 - (v14 << 13))));
  for ( i = v9 & (((v15 - 1 - (v15 << 27)) >> 31) ^ (v15 - 1 - ((_DWORD)v15 << 27))); ; i = v9 & v19 )
  {
    v17 = &v8[i];
    v18 = v17->m128i_i64[0];
    if ( v17->m128i_i64[0] == *a3 && v17->m128i_i64[1] == v10 )
    {
      if ( v7 )
        v26 = v8 + 2;
      else
        v26 = &v8[a2[1].m128i_u32[2]];
      v27 = 0;
      goto LABEL_22;
    }
    if ( v18 == -8 )
      break;
    if ( v18 == -16 && v17->m128i_i64[1] == -16 && !v11 )
      v11 = &v8[i];
LABEL_10:
    v19 = v13 + i;
    ++v13;
  }
  if ( v17->m128i_i64[1] != -8 )
    goto LABEL_10;
  v21 = a2->m128i_u32[2];
  if ( v11 )
    v17 = v11;
  a2->m128i_i64[0] = v6 + 1;
  v22 = (v21 >> 1) + 1;
  if ( !v7 )
  {
    v20 = a2[1].m128i_u32[2];
    goto LABEL_14;
  }
  v23 = 6;
  v20 = 2;
LABEL_15:
  if ( 4 * v22 >= v23 )
  {
    sub_1AA7460(a2, 2 * v20);
    if ( (a2->m128i_i8[8] & 1) != 0 )
    {
      v29 = a2 + 1;
      v30 = 1;
    }
    else
    {
      v52 = a2[1].m128i_i32[2];
      v29 = (__m128i *)a2[1].m128i_i64[0];
      if ( !v52 )
        goto LABEL_70;
      v30 = v52 - 1;
    }
    v31 = a3[1];
    v32 = ((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4);
    v33 = 1;
    v34 = (((v32 | ((unsigned __int64)(((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4)) << 32)) - 1 - (v32 << 32)) >> 22)
        ^ ((v32 | ((unsigned __int64)(((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4)) << 32)) - 1 - (v32 << 32));
    v35 = ((9 * (((v34 - 1 - (v34 << 13)) >> 8) ^ (v34 - 1 - (v34 << 13)))) >> 15)
        ^ (9 * (((v34 - 1 - (v34 << 13)) >> 8) ^ (v34 - 1 - (v34 << 13))));
    v36 = ((v35 - 1 - (v35 << 27)) >> 31) ^ (v35 - 1 - ((_DWORD)v35 << 27));
    v37 = 0;
    for ( j = v30 & v36; ; j = v30 & v40 )
    {
      v17 = &v29[j];
      v39 = v17->m128i_i64[0];
      if ( v17->m128i_i64[0] == *a3 && v17->m128i_i64[1] == v31 )
        break;
      if ( v39 == -8 )
      {
        if ( v17->m128i_i64[1] == -8 )
        {
LABEL_65:
          if ( v37 )
            v17 = v37;
          goto LABEL_61;
        }
      }
      else if ( v39 == -16 && v17->m128i_i64[1] == -16 && !v37 )
      {
        v37 = &v29[j];
      }
      v40 = v33 + j;
      ++v33;
    }
    goto LABEL_61;
  }
  if ( v20 - a2->m128i_i32[3] - v22 <= v20 >> 3 )
  {
    sub_1AA7460(a2, v20);
    if ( (a2->m128i_i8[8] & 1) != 0 )
    {
      v41 = a2 + 1;
      v42 = 1;
      goto LABEL_48;
    }
    v53 = a2[1].m128i_i32[2];
    v41 = (__m128i *)a2[1].m128i_i64[0];
    if ( v53 )
    {
      v42 = v53 - 1;
LABEL_48:
      v43 = a3[1];
      v44 = ((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4);
      v45 = 1;
      v46 = (((v44 | ((unsigned __int64)(((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4)) << 32)) - 1 - (v44 << 32)) >> 22)
          ^ ((v44 | ((unsigned __int64)(((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4)) << 32)) - 1 - (v44 << 32));
      v47 = ((9 * (((v46 - 1 - (v46 << 13)) >> 8) ^ (v46 - 1 - (v46 << 13)))) >> 15)
          ^ (9 * (((v46 - 1 - (v46 << 13)) >> 8) ^ (v46 - 1 - (v46 << 13))));
      v48 = ((v47 - 1 - (v47 << 27)) >> 31) ^ (v47 - 1 - ((_DWORD)v47 << 27));
      v37 = 0;
      for ( k = v42 & v48; ; k = v42 & v51 )
      {
        v17 = &v41[k];
        v50 = v17->m128i_i64[0];
        if ( v17->m128i_i64[0] == *a3 && v17->m128i_i64[1] == v43 )
          break;
        if ( v50 == -8 )
        {
          if ( v17->m128i_i64[1] == -8 )
            goto LABEL_65;
        }
        else if ( v50 == -16 && v17->m128i_i64[1] == -16 && !v37 )
        {
          v37 = &v41[k];
        }
        v51 = v45 + k;
        ++v45;
      }
LABEL_61:
      v21 = a2->m128i_u32[2];
      goto LABEL_17;
    }
LABEL_70:
    a2->m128i_i32[2] = (2 * ((unsigned __int32)a2->m128i_i32[2] >> 1) + 2) | a2->m128i_i32[2] & 1;
    BUG();
  }
LABEL_17:
  a2->m128i_i32[2] = (2 * (v21 >> 1) + 2) | v21 & 1;
  if ( v17->m128i_i64[0] != -8 || v17->m128i_i64[1] != -8 )
    --a2->m128i_i32[3];
  v17->m128i_i64[0] = *a3;
  v17->m128i_i64[1] = a3[1];
  if ( (a2->m128i_i8[8] & 1) != 0 )
  {
    v24 = a2 + 1;
    v25 = 2;
  }
  else
  {
    v24 = (__m128i *)a2[1].m128i_i64[0];
    v25 = a2[1].m128i_u32[2];
  }
  v6 = a2->m128i_i64[0];
  v26 = &v24[v25];
  v27 = 1;
LABEL_22:
  *(_QWORD *)a1 = a2;
  *(_BYTE *)(a1 + 32) = v27;
  *(_QWORD *)(a1 + 8) = v6;
  *(_QWORD *)(a1 + 16) = v17;
  *(_QWORD *)(a1 + 24) = v26;
  return a1;
}
