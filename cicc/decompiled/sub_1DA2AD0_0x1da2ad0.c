// Function: sub_1DA2AD0
// Address: 0x1da2ad0
//
__int64 __fastcall sub_1DA2AD0(__m128i *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  char v8; // dl
  __m128i *v9; // rcx
  int v10; // esi
  int v11; // r10d
  unsigned int v12; // r8d
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned int v15; // eax
  __int64 *v16; // rdi
  __int64 result; // rax
  __int64 *v18; // r8
  __int64 v19; // r9
  unsigned int v20; // eax
  unsigned int v21; // esi
  unsigned __int32 v22; // eax
  int v23; // ecx
  unsigned int v24; // r8d
  __m128i *v25; // rcx
  int v26; // edx
  int v27; // r9d
  __int8 *v28; // r8
  unsigned int v29; // edi
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // rsi
  unsigned int i; // eax
  __int64 v33; // rsi
  unsigned int v34; // eax
  __m128i *v35; // rcx
  int v36; // edx
  int v37; // r9d
  unsigned int v38; // edi
  unsigned __int64 v39; // rsi
  unsigned __int64 v40; // rsi
  unsigned int j; // eax
  __int64 v42; // rsi
  unsigned int v43; // eax
  __int32 v44; // edx
  __int32 v45; // edx

  sub_1369D60(a1->m128i_i64, a2);
  v8 = a1[2].m128i_i8[8] & 1;
  if ( v8 )
  {
    v9 = a1 + 3;
    v10 = 7;
  }
  else
  {
    v21 = a1[3].m128i_u32[2];
    v9 = (__m128i *)a1[3].m128i_i64[0];
    if ( !v21 )
    {
      v22 = a1[2].m128i_u32[2];
      ++a1[2].m128i_i64[0];
      v16 = 0;
      v23 = (v22 >> 1) + 1;
LABEL_14:
      v24 = 3 * v21;
      goto LABEL_15;
    }
    v10 = v21 - 1;
  }
  v11 = 1;
  v12 = (unsigned int)a4 >> 9;
  v13 = (((v12 ^ ((unsigned int)a4 >> 4) | ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(v12 ^ ((unsigned int)a4 >> 4)) << 32)) >> 22)
      ^ ((v12 ^ ((unsigned int)a4 >> 4) | ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(v12 ^ ((unsigned int)a4 >> 4)) << 32));
  v14 = ((9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13)))) >> 15)
      ^ (9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13))));
  v15 = ((v14 - 1 - (v14 << 27)) >> 31) ^ (v14 - 1 - ((_DWORD)v14 << 27));
  v16 = 0;
  for ( result = v10 & v15; ; result = v10 & v20 )
  {
    v18 = &v9->m128i_i64[3 * (unsigned int)result];
    v19 = *v18;
    if ( *v18 == a3 && a4 == v18[1] )
      return result;
    if ( v19 == -8 )
      break;
    if ( v19 == -16 && v18[1] == -16 && !v16 )
      v16 = &v9->m128i_i64[3 * (unsigned int)result];
LABEL_10:
    v20 = v11 + result;
    ++v11;
  }
  if ( v18[1] != -8 )
    goto LABEL_10;
  v22 = a1[2].m128i_u32[2];
  if ( !v16 )
    v16 = v18;
  ++a1[2].m128i_i64[0];
  v23 = (v22 >> 1) + 1;
  if ( !v8 )
  {
    v21 = a1[3].m128i_u32[2];
    goto LABEL_14;
  }
  v24 = 24;
  v21 = 8;
LABEL_15:
  if ( 4 * v23 >= v24 )
  {
    sub_1DA2510(a1 + 2, 2 * v21);
    if ( (a1[2].m128i_i8[8] & 1) != 0 )
    {
      v25 = a1 + 3;
      v26 = 7;
    }
    else
    {
      v44 = a1[3].m128i_i32[2];
      v25 = (__m128i *)a1[3].m128i_i64[0];
      if ( !v44 )
        goto LABEL_63;
      v26 = v44 - 1;
    }
    v27 = 1;
    v28 = 0;
    v29 = (unsigned int)a4 >> 9;
    v30 = (((v29 ^ ((unsigned int)a4 >> 4)
           | ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32))
          - 1
          - ((unsigned __int64)(v29 ^ ((unsigned int)a4 >> 4)) << 32)) >> 22)
        ^ ((v29 ^ ((unsigned int)a4 >> 4) | ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32))
         - 1
         - ((unsigned __int64)(v29 ^ ((unsigned int)a4 >> 4)) << 32));
    v31 = ((9 * (((v30 - 1 - (v30 << 13)) >> 8) ^ (v30 - 1 - (v30 << 13)))) >> 15)
        ^ (9 * (((v30 - 1 - (v30 << 13)) >> 8) ^ (v30 - 1 - (v30 << 13))));
    for ( i = v26 & (((v31 - 1 - (v31 << 27)) >> 31) ^ (v31 - 1 - ((_DWORD)v31 << 27))); ; i = v26 & v34 )
    {
      v16 = &v25->m128i_i64[3 * i];
      v33 = *v16;
      if ( *v16 == a3 && a4 == v16[1] )
        break;
      if ( v33 == -8 )
      {
        if ( v16[1] == -8 )
        {
LABEL_58:
          if ( v28 )
            v16 = (__int64 *)v28;
          goto LABEL_54;
        }
      }
      else if ( v33 == -16 && v16[1] == -16 && !v28 )
      {
        v28 = &v25->m128i_i8[24 * i];
      }
      v34 = v27 + i;
      ++v27;
    }
    goto LABEL_54;
  }
  if ( v21 - a1[2].m128i_i32[3] - v23 > v21 >> 3 )
    goto LABEL_17;
  sub_1DA2510(a1 + 2, v21);
  if ( (a1[2].m128i_i8[8] & 1) == 0 )
  {
    v45 = a1[3].m128i_i32[2];
    v35 = (__m128i *)a1[3].m128i_i64[0];
    if ( v45 )
    {
      v36 = v45 - 1;
      goto LABEL_41;
    }
LABEL_63:
    a1[2].m128i_i32[2] = (2 * ((unsigned __int32)a1[2].m128i_i32[2] >> 1) + 2) | a1[2].m128i_i32[2] & 1;
    BUG();
  }
  v35 = a1 + 3;
  v36 = 7;
LABEL_41:
  v37 = 1;
  v28 = 0;
  v38 = (unsigned int)a4 >> 9;
  v39 = (((v38 ^ ((unsigned int)a4 >> 4) | ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(v38 ^ ((unsigned int)a4 >> 4)) << 32)) >> 22)
      ^ ((v38 ^ ((unsigned int)a4 >> 4) | ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(v38 ^ ((unsigned int)a4 >> 4)) << 32));
  v40 = ((9 * (((v39 - 1 - (v39 << 13)) >> 8) ^ (v39 - 1 - (v39 << 13)))) >> 15)
      ^ (9 * (((v39 - 1 - (v39 << 13)) >> 8) ^ (v39 - 1 - (v39 << 13))));
  for ( j = v36 & (((v40 - 1 - (v40 << 27)) >> 31) ^ (v40 - 1 - ((_DWORD)v40 << 27))); ; j = v36 & v43 )
  {
    v16 = &v35->m128i_i64[3 * j];
    v42 = *v16;
    if ( *v16 == a3 && a4 == v16[1] )
      break;
    if ( v42 == -8 )
    {
      if ( v16[1] == -8 )
        goto LABEL_58;
    }
    else if ( v42 == -16 && v16[1] == -16 && !v28 )
    {
      v28 = &v35->m128i_i8[24 * j];
    }
    v43 = v37 + j;
    ++v37;
  }
LABEL_54:
  v22 = a1[2].m128i_u32[2];
LABEL_17:
  result = (2 * (v22 >> 1) + 2) | v22 & 1;
  a1[2].m128i_i32[2] = result;
  if ( *v16 != -8 || v16[1] != -8 )
    --a1[2].m128i_i32[3];
  *v16 = a3;
  v16[1] = a4;
  *((_DWORD *)v16 + 4) = a2;
  return result;
}
