// Function: sub_375CDA0
// Address: 0x375cda0
//
__int64 __fastcall sub_375CDA0(__int64 a1, __m128i *a2, unsigned __int64 *a3, _DWORD *a4)
{
  __int64 result; // rax
  __int64 v8; // r11
  char v9; // cl
  __m128i *v10; // rdi
  int v11; // esi
  unsigned __int64 v12; // r8
  int v13; // r15d
  __int8 *v14; // r9
  unsigned int i; // edx
  __int8 *v16; // r10
  unsigned int v17; // edx
  unsigned int v18; // esi
  unsigned __int32 v19; // edx
  int v20; // edi
  unsigned int v21; // r8d
  __m128i *v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rdx
  int v26; // r14d
  __m128i *v27; // r8
  int v28; // ecx
  int v29; // r10d
  int v30; // r14d
  __int8 *v31; // r11
  unsigned int j; // edx
  unsigned int v33; // edx
  __m128i *v34; // r8
  int v35; // ecx
  int v36; // r10d
  int v37; // r14d
  unsigned int k; // edx
  unsigned int v39; // edx
  __int32 v40; // ecx
  __int32 v41; // ecx
  int v42; // edi
  int v43; // edi
  int v44; // [rsp+8h] [rbp-38h]
  __int64 v45; // [rsp+8h] [rbp-38h]
  __int64 v46; // [rsp+8h] [rbp-38h]

  result = a1;
  v8 = a2->m128i_i64[0];
  v9 = a2->m128i_i8[8] & 1;
  if ( v9 )
  {
    v10 = a2 + 1;
    v11 = 7;
  }
  else
  {
    v10 = (__m128i *)a2[1].m128i_i64[0];
    v18 = a2[1].m128i_u32[2];
    if ( !v18 )
    {
      v19 = a2->m128i_u32[2];
      v14 = 0;
      a2->m128i_i64[0] = v8 + 1;
      v20 = (v19 >> 1) + 1;
LABEL_10:
      v21 = 3 * v18;
      goto LABEL_11;
    }
    v11 = v18 - 1;
  }
  v12 = *a3;
  v13 = *((_DWORD *)a3 + 2);
  v44 = 1;
  v14 = 0;
  for ( i = v11 & (v13 + ((v12 >> 9) ^ (v12 >> 4))); ; i = v11 & v17 )
  {
    v16 = &v10->m128i_i8[24 * i];
    if ( v12 == *(_QWORD *)v16 && v13 == *((_DWORD *)v16 + 2) )
    {
      v25 = 192;
      if ( !v9 )
        v25 = 24LL * a2[1].m128i_u32[2];
      *(_QWORD *)result = a2;
      *(_QWORD *)(result + 8) = v8;
      *(_QWORD *)(result + 16) = v16;
      *(_QWORD *)(result + 24) = (char *)v10 + v25;
      *(_BYTE *)(result + 32) = 0;
      return result;
    }
    if ( !*(_QWORD *)v16 )
      break;
LABEL_6:
    v17 = v44 + i;
    ++v44;
  }
  v26 = *((_DWORD *)v16 + 2);
  if ( v26 != -1 )
  {
    if ( v26 == -2 && !v14 )
      v14 = &v10->m128i_i8[24 * i];
    goto LABEL_6;
  }
  v19 = a2->m128i_u32[2];
  if ( !v14 )
    v14 = v16;
  a2->m128i_i64[0] = v8 + 1;
  v20 = (v19 >> 1) + 1;
  if ( !v9 )
  {
    v18 = a2[1].m128i_u32[2];
    goto LABEL_10;
  }
  v21 = 24;
  v18 = 8;
LABEL_11:
  if ( v21 <= 4 * v20 )
  {
    v45 = result;
    sub_375C8C0(a2, 2 * v18);
    result = v45;
    if ( (a2->m128i_i8[8] & 1) != 0 )
    {
      v27 = a2 + 1;
      v28 = 7;
    }
    else
    {
      v40 = a2[1].m128i_i32[2];
      v27 = (__m128i *)a2[1].m128i_i64[0];
      if ( !v40 )
        goto LABEL_67;
      v28 = v40 - 1;
    }
    v29 = *((_DWORD *)a3 + 2);
    v30 = 1;
    v31 = 0;
    for ( j = v28 & (v29 + ((*a3 >> 9) ^ (*a3 >> 4))); ; j = v28 & v33 )
    {
      v14 = &v27->m128i_i8[24 * j];
      if ( *a3 == *(_QWORD *)v14 && v29 == *((_DWORD *)v14 + 2) )
        break;
      if ( !*(_QWORD *)v14 )
      {
        v42 = *((_DWORD *)v14 + 2);
        if ( v42 == -1 )
        {
LABEL_64:
          if ( v31 )
            v14 = v31;
          goto LABEL_51;
        }
        if ( v42 == -2 && !v31 )
          v31 = &v27->m128i_i8[24 * j];
      }
      v33 = v30 + j;
      ++v30;
    }
    goto LABEL_51;
  }
  if ( v18 - a2->m128i_i32[3] - v20 <= v18 >> 3 )
  {
    v46 = result;
    sub_375C8C0(a2, v18);
    result = v46;
    if ( (a2->m128i_i8[8] & 1) != 0 )
    {
      v34 = a2 + 1;
      v35 = 7;
      goto LABEL_38;
    }
    v41 = a2[1].m128i_i32[2];
    v34 = (__m128i *)a2[1].m128i_i64[0];
    if ( v41 )
    {
      v35 = v41 - 1;
LABEL_38:
      v36 = *((_DWORD *)a3 + 2);
      v37 = 1;
      v31 = 0;
      for ( k = v35 & (v36 + ((*a3 >> 9) ^ (*a3 >> 4))); ; k = v35 & v39 )
      {
        v14 = &v34->m128i_i8[24 * k];
        if ( *a3 == *(_QWORD *)v14 && v36 == *((_DWORD *)v14 + 2) )
          break;
        if ( !*(_QWORD *)v14 )
        {
          v43 = *((_DWORD *)v14 + 2);
          if ( v43 == -1 )
            goto LABEL_64;
          if ( v43 == -2 && !v31 )
            v31 = &v34->m128i_i8[24 * k];
        }
        v39 = v37 + k;
        ++v37;
      }
LABEL_51:
      v19 = a2->m128i_u32[2];
      goto LABEL_13;
    }
LABEL_67:
    a2->m128i_i32[2] = (2 * ((unsigned __int32)a2->m128i_i32[2] >> 1) + 2) | a2->m128i_i32[2] & 1;
    BUG();
  }
LABEL_13:
  a2->m128i_i32[2] = (2 * (v19 >> 1) + 2) | v19 & 1;
  if ( *(_QWORD *)v14 || *((_DWORD *)v14 + 2) != -1 )
    --a2->m128i_i32[3];
  *(_QWORD *)v14 = *a3;
  *((_DWORD *)v14 + 2) = *((_DWORD *)a3 + 2);
  *((_DWORD *)v14 + 4) = *a4;
  if ( (a2->m128i_i8[8] & 1) != 0 )
  {
    v22 = a2 + 1;
    v23 = 192;
  }
  else
  {
    v22 = (__m128i *)a2[1].m128i_i64[0];
    v23 = 24LL * a2[1].m128i_u32[2];
  }
  v24 = a2->m128i_i64[0];
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 16) = v14;
  *(_QWORD *)(result + 8) = v24;
  *(_QWORD *)(result + 24) = (char *)v22 + v23;
  *(_BYTE *)(result + 32) = 1;
  return result;
}
