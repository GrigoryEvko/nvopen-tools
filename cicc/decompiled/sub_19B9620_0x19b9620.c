// Function: sub_19B9620
// Address: 0x19b9620
//
__int64 __fastcall sub_19B9620(
        const __m128i *a1,
        const __m128i *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        const __m128i *a6)
{
  char v8; // dl
  const __m128i *v9; // r8
  int v10; // esi
  __int64 v11; // rdi
  int v12; // r11d
  __int64 v13; // r9
  const __m128i **v14; // r10
  unsigned __int64 v15; // r9
  unsigned __int64 v16; // r9
  __int64 result; // rax
  __int64 v18; // r13
  int v19; // eax
  unsigned int v20; // esi
  unsigned __int32 v21; // eax
  int v22; // ecx
  unsigned int v23; // edi
  __int64 v24; // rax
  const __m128i *v25; // rdi
  int v26; // edx
  __int64 v27; // rsi
  int v28; // r11d
  __int64 v29; // r8
  unsigned __int64 v30; // r8
  unsigned __int64 v31; // r8
  int i; // eax
  int v33; // eax
  const __m128i *v34; // rdi
  int v35; // edx
  __int64 v36; // rsi
  int v37; // r11d
  __int64 v38; // r8
  unsigned __int64 v39; // r8
  unsigned __int64 v40; // r8
  int j; // eax
  int v42; // eax
  __int32 v43; // edx
  __int32 v44; // edx

  v8 = a1->m128i_i8[8] & 1;
  if ( v8 )
  {
    v9 = a1 + 1;
    v10 = 3;
  }
  else
  {
    v20 = a1[1].m128i_u32[2];
    v9 = (const __m128i *)a1[1].m128i_i64[0];
    if ( !v20 )
    {
      v21 = a1->m128i_u32[2];
      ++a1->m128i_i64[0];
      v14 = 0;
      v22 = (v21 >> 1) + 1;
LABEL_14:
      v23 = 3 * v20;
      goto LABEL_15;
    }
    v10 = v20 - 1;
  }
  v11 = a2->m128i_i64[1];
  v12 = 1;
  v13 = ((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4);
  v14 = 0;
  v15 = (((v13
         | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))
        - 1
        - (v13 << 32)) >> 22)
      ^ ((v13 | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))
       - 1
       - (v13 << 32));
  v16 = ((9 * (((v15 - 1 - (v15 << 13)) >> 8) ^ (v15 - 1 - (v15 << 13)))) >> 15)
      ^ (9 * (((v15 - 1 - (v15 << 13)) >> 8) ^ (v15 - 1 - (v15 << 13))));
  for ( result = v10 & ((unsigned int)((v16 - 1 - (v16 << 27)) >> 31) ^ ((_DWORD)v16 - 1 - ((_DWORD)v16 << 27)));
        ;
        result = v10 & (unsigned int)v19 )
  {
    a6 = &v9[(unsigned int)result];
    v18 = a6->m128i_i64[0];
    if ( a6->m128i_i64[0] == a2->m128i_i64[0] && a6->m128i_i64[1] == v11 )
      return result;
    if ( v18 == -8 )
      break;
    if ( v18 == -16 && a6->m128i_i64[1] == -16 && !v14 )
      v14 = (const __m128i **)&v9[(unsigned int)result];
LABEL_10:
    v19 = v12 + result;
    ++v12;
  }
  if ( a6->m128i_i64[1] != -8 )
    goto LABEL_10;
  v21 = a1->m128i_u32[2];
  if ( !v14 )
    v14 = (const __m128i **)a6;
  ++a1->m128i_i64[0];
  v22 = (v21 >> 1) + 1;
  if ( !v8 )
  {
    v20 = a1[1].m128i_u32[2];
    goto LABEL_14;
  }
  v23 = 12;
  v20 = 4;
LABEL_15:
  if ( 4 * v22 >= v23 )
  {
    sub_19B90A0(a1, 2 * v20);
    if ( (a1->m128i_i8[8] & 1) != 0 )
    {
      v25 = a1 + 1;
      v26 = 3;
    }
    else
    {
      v43 = a1[1].m128i_i32[2];
      v25 = (const __m128i *)a1[1].m128i_i64[0];
      if ( !v43 )
        goto LABEL_65;
      v26 = v43 - 1;
    }
    v27 = a2->m128i_i64[1];
    v28 = 1;
    v29 = ((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4);
    a6 = 0;
    v30 = (((v29
           | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))
          - 1
          - (v29 << 32)) >> 22)
        ^ ((v29
          | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))
         - 1
         - (v29 << 32));
    v31 = ((9 * (((v30 - 1 - (v30 << 13)) >> 8) ^ (v30 - 1 - (v30 << 13)))) >> 15)
        ^ (9 * (((v30 - 1 - (v30 << 13)) >> 8) ^ (v30 - 1 - (v30 << 13))));
    for ( i = v26 & (((v31 - 1 - (v31 << 27)) >> 31) ^ (v31 - 1 - ((_DWORD)v31 << 27))); ; i = v26 & v33 )
    {
      v14 = (const __m128i **)&v25[i];
      v9 = *v14;
      if ( *v14 == (const __m128i *)a2->m128i_i64[0] && v14[1] == (const __m128i *)v27 )
        break;
      if ( v9 == (const __m128i *)-8LL )
      {
        if ( v14[1] == (const __m128i *)-8LL )
        {
LABEL_60:
          if ( a6 )
            v14 = (const __m128i **)a6;
          goto LABEL_56;
        }
      }
      else if ( v9 == (const __m128i *)-16LL && v14[1] == (const __m128i *)-16LL && !a6 )
      {
        a6 = &v25[i];
      }
      v33 = v28 + i;
      ++v28;
    }
    goto LABEL_56;
  }
  if ( v20 - a1->m128i_i32[3] - v22 > v20 >> 3 )
    goto LABEL_17;
  sub_19B90A0(a1, v20);
  if ( (a1->m128i_i8[8] & 1) == 0 )
  {
    v44 = a1[1].m128i_i32[2];
    v34 = (const __m128i *)a1[1].m128i_i64[0];
    if ( v44 )
    {
      v35 = v44 - 1;
      goto LABEL_43;
    }
LABEL_65:
    a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
    BUG();
  }
  v34 = a1 + 1;
  v35 = 3;
LABEL_43:
  v36 = a2->m128i_i64[1];
  v37 = 1;
  v38 = ((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4);
  a6 = 0;
  v39 = (((v38
         | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))
        - 1
        - (v38 << 32)) >> 22)
      ^ ((v38 | ((unsigned __int64)(((unsigned int)a2->m128i_i64[0] >> 9) ^ ((unsigned int)a2->m128i_i64[0] >> 4)) << 32))
       - 1
       - (v38 << 32));
  v40 = ((9 * (((v39 - 1 - (v39 << 13)) >> 8) ^ (v39 - 1 - (v39 << 13)))) >> 15)
      ^ (9 * (((v39 - 1 - (v39 << 13)) >> 8) ^ (v39 - 1 - (v39 << 13))));
  for ( j = v35 & (((v40 - 1 - (v40 << 27)) >> 31) ^ (v40 - 1 - ((_DWORD)v40 << 27))); ; j = v35 & v42 )
  {
    v14 = (const __m128i **)&v34[j];
    v9 = *v14;
    if ( *v14 == (const __m128i *)a2->m128i_i64[0] && v14[1] == (const __m128i *)v36 )
      break;
    if ( v9 == (const __m128i *)-8LL )
    {
      if ( v14[1] == (const __m128i *)-8LL )
        goto LABEL_60;
    }
    else if ( v9 == (const __m128i *)-16LL && v14[1] == (const __m128i *)-16LL && !a6 )
    {
      a6 = &v34[j];
    }
    v42 = v37 + j;
    ++v37;
  }
LABEL_56:
  v21 = a1->m128i_u32[2];
LABEL_17:
  a1->m128i_i32[2] = (2 * (v21 >> 1) + 2) | v21 & 1;
  if ( *v14 != (const __m128i *)-8LL || v14[1] != (const __m128i *)-8LL )
    --a1->m128i_i32[3];
  *v14 = (const __m128i *)a2->m128i_i64[0];
  v14[1] = (const __m128i *)a2->m128i_i64[1];
  v24 = a1[5].m128i_u32[2];
  if ( (unsigned int)v24 >= a1[5].m128i_i32[3] )
  {
    sub_16CD150((__int64)a1[5].m128i_i64, &a1[6], 0, 16, (int)v9, (int)a6);
    v24 = a1[5].m128i_u32[2];
  }
  result = a1[5].m128i_i64[0] + 16 * v24;
  *(__m128i *)result = _mm_loadu_si128(a2);
  ++a1[5].m128i_i32[2];
  return result;
}
