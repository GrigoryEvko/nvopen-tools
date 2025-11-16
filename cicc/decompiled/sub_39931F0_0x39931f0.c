// Function: sub_39931F0
// Address: 0x39931f0
//
__int64 __fastcall sub_39931F0(__int64 a1, __m128i *a2, __int64 *a3, __int64 *a4)
{
  __int64 v8; // r11
  char v9; // dl
  __m128i *v10; // rcx
  int v11; // esi
  __int64 v12; // rdi
  __int64 v13; // r9
  unsigned int v14; // r10d
  unsigned __int64 v15; // r8
  unsigned __int64 v16; // r8
  int v17; // eax
  __int64 *v18; // r8
  unsigned int i; // eax
  __int64 *v20; // r10
  __int64 v21; // r15
  unsigned int v22; // eax
  unsigned int v23; // esi
  unsigned __int32 v24; // eax
  int v25; // ecx
  unsigned int v26; // edi
  __m128i *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v31; // rax
  __m128i *v32; // rdi
  int v33; // edx
  __int64 v34; // rsi
  int v35; // r11d
  __int64 *v36; // r10
  __int64 v37; // r8
  unsigned __int64 v38; // r8
  unsigned __int64 v39; // r8
  unsigned int j; // eax
  __int64 v41; // r9
  unsigned int v42; // eax
  __m128i *v43; // rdi
  int v44; // edx
  __int64 v45; // rsi
  int v46; // r11d
  __int64 v47; // r8
  unsigned __int64 v48; // r8
  unsigned __int64 v49; // r8
  unsigned int k; // eax
  __int64 v51; // r9
  unsigned int v52; // eax
  __int32 v53; // edx
  __int32 v54; // edx
  int v55; // [rsp+Ch] [rbp-34h]

  v8 = a2->m128i_i64[0];
  v9 = a2->m128i_i8[8] & 1;
  if ( v9 )
  {
    v10 = a2 + 1;
    v11 = 3;
  }
  else
  {
    v10 = (__m128i *)a2[1].m128i_i64[0];
    v23 = a2[1].m128i_u32[2];
    if ( !v23 )
    {
      v24 = a2->m128i_u32[2];
      v18 = 0;
      a2->m128i_i64[0] = v8 + 1;
      v25 = (v24 >> 1) + 1;
LABEL_14:
      v26 = 3 * v23;
      goto LABEL_15;
    }
    v11 = v23 - 1;
  }
  v12 = *a3;
  v13 = a3[1];
  v55 = 1;
  v14 = (unsigned int)v13 >> 9;
  v15 = (((v14 ^ ((unsigned int)v13 >> 4)
         | ((unsigned __int64)(((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(v14 ^ ((unsigned int)v13 >> 4)) << 32)) >> 22)
      ^ ((v14 ^ ((unsigned int)v13 >> 4)
        | ((unsigned __int64)(((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(v14 ^ ((unsigned int)v13 >> 4)) << 32));
  v16 = ((9 * (((v15 - 1 - (v15 << 13)) >> 8) ^ (v15 - 1 - (v15 << 13)))) >> 15)
      ^ (9 * (((v15 - 1 - (v15 << 13)) >> 8) ^ (v15 - 1 - (v15 << 13))));
  v17 = ((v16 - 1 - (v16 << 27)) >> 31) ^ (v16 - 1 - ((_DWORD)v16 << 27));
  v18 = 0;
  for ( i = v11 & v17; ; i = v11 & v22 )
  {
    v20 = &v10->m128i_i64[3 * i];
    v21 = *v20;
    if ( *v20 == v12 && v20[1] == v13 )
    {
      v31 = 96;
      if ( !v9 )
        v31 = 24LL * a2[1].m128i_u32[2];
      *(_QWORD *)a1 = a2;
      *(_QWORD *)(a1 + 8) = v8;
      *(_QWORD *)(a1 + 16) = v20;
      *(_QWORD *)(a1 + 24) = (char *)v10 + v31;
      *(_BYTE *)(a1 + 32) = 0;
      return a1;
    }
    if ( v21 == -8 )
      break;
    if ( v21 == -16 && v20[1] == -16 && !v18 )
      v18 = &v10->m128i_i64[3 * i];
LABEL_10:
    v22 = v55 + i;
    ++v55;
  }
  if ( v20[1] != -8 )
    goto LABEL_10;
  v24 = a2->m128i_u32[2];
  if ( !v18 )
    v18 = v20;
  a2->m128i_i64[0] = v8 + 1;
  v25 = (v24 >> 1) + 1;
  if ( !v9 )
  {
    v23 = a2[1].m128i_u32[2];
    goto LABEL_14;
  }
  v26 = 12;
  v23 = 4;
LABEL_15:
  if ( 4 * v25 >= v26 )
  {
    sub_3992C50(a2, 2 * v23);
    if ( (a2->m128i_i8[8] & 1) != 0 )
    {
      v32 = a2 + 1;
      v33 = 3;
    }
    else
    {
      v53 = a2[1].m128i_i32[2];
      v32 = (__m128i *)a2[1].m128i_i64[0];
      if ( !v53 )
        goto LABEL_69;
      v33 = v53 - 1;
    }
    v34 = a3[1];
    v35 = 1;
    v36 = 0;
    v37 = ((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4);
    v38 = (((v37 | ((unsigned __int64)(((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4)) << 32)) - 1 - (v37 << 32)) >> 22)
        ^ ((v37 | ((unsigned __int64)(((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4)) << 32)) - 1 - (v37 << 32));
    v39 = ((9 * (((v38 - 1 - (v38 << 13)) >> 8) ^ (v38 - 1 - (v38 << 13)))) >> 15)
        ^ (9 * (((v38 - 1 - (v38 << 13)) >> 8) ^ (v38 - 1 - (v38 << 13))));
    for ( j = v33 & (((v39 - 1 - (v39 << 27)) >> 31) ^ (v39 - 1 - ((_DWORD)v39 << 27))); ; j = v33 & v42 )
    {
      v18 = &v32->m128i_i64[3 * j];
      v41 = *v18;
      if ( *v18 == *a3 && v18[1] == v34 )
        break;
      if ( v41 == -8 )
      {
        if ( v18[1] == -8 )
        {
LABEL_64:
          if ( v36 )
            v18 = v36;
          goto LABEL_60;
        }
      }
      else if ( v41 == -16 && v18[1] == -16 && !v36 )
      {
        v36 = &v32->m128i_i64[3 * j];
      }
      v42 = v35 + j;
      ++v35;
    }
    goto LABEL_60;
  }
  if ( v23 - a2->m128i_i32[3] - v25 <= v23 >> 3 )
  {
    sub_3992C50(a2, v23);
    if ( (a2->m128i_i8[8] & 1) != 0 )
    {
      v43 = a2 + 1;
      v44 = 3;
      goto LABEL_47;
    }
    v54 = a2[1].m128i_i32[2];
    v43 = (__m128i *)a2[1].m128i_i64[0];
    if ( v54 )
    {
      v44 = v54 - 1;
LABEL_47:
      v45 = a3[1];
      v46 = 1;
      v36 = 0;
      v47 = ((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4);
      v48 = (((v47 | ((unsigned __int64)(((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4)) << 32)) - 1 - (v47 << 32)) >> 22)
          ^ ((v47 | ((unsigned __int64)(((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4)) << 32)) - 1 - (v47 << 32));
      v49 = ((9 * (((v48 - 1 - (v48 << 13)) >> 8) ^ (v48 - 1 - (v48 << 13)))) >> 15)
          ^ (9 * (((v48 - 1 - (v48 << 13)) >> 8) ^ (v48 - 1 - (v48 << 13))));
      for ( k = v44 & (((v49 - 1 - (v49 << 27)) >> 31) ^ (v49 - 1 - ((_DWORD)v49 << 27))); ; k = v44 & v52 )
      {
        v18 = &v43->m128i_i64[3 * k];
        v51 = *v18;
        if ( *v18 == *a3 && v18[1] == v45 )
          break;
        if ( v51 == -8 )
        {
          if ( v18[1] == -8 )
            goto LABEL_64;
        }
        else if ( v51 == -16 && v18[1] == -16 && !v36 )
        {
          v36 = &v43->m128i_i64[3 * k];
        }
        v52 = v46 + k;
        ++v46;
      }
LABEL_60:
      v24 = a2->m128i_u32[2];
      goto LABEL_17;
    }
LABEL_69:
    a2->m128i_i32[2] = (2 * ((unsigned __int32)a2->m128i_i32[2] >> 1) + 2) | a2->m128i_i32[2] & 1;
    BUG();
  }
LABEL_17:
  a2->m128i_i32[2] = (2 * (v24 >> 1) + 2) | v24 & 1;
  if ( *v18 != -8 || v18[1] != -8 )
    --a2->m128i_i32[3];
  *v18 = *a3;
  v18[1] = a3[1];
  v18[2] = *a4;
  if ( (a2->m128i_i8[8] & 1) != 0 )
  {
    v27 = a2 + 1;
    v28 = 96;
  }
  else
  {
    v27 = (__m128i *)a2[1].m128i_i64[0];
    v28 = 24LL * a2[1].m128i_u32[2];
  }
  v29 = a2->m128i_i64[0];
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v18;
  *(_QWORD *)(a1 + 8) = v29;
  *(_QWORD *)(a1 + 24) = (char *)v27 + v28;
  *(_BYTE *)(a1 + 32) = 1;
  return a1;
}
