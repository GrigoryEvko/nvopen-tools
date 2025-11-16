// Function: sub_38D5D60
// Address: 0x38d5d60
//
void __fastcall sub_38D5D60(__int64 a1, __int8 *a2, size_t a3)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  const void *v9; // rdi
  __m128i *v10; // rax
  __m128i *v11; // rsi
  __m128i *v12; // rdi
  const void *v13; // rdi
  const void *v14; // rdi
  __m128i *v15; // rdi
  __int64 v16; // [rsp+8h] [rbp-78h]
  __m128i **v17; // [rsp+10h] [rbp-70h]
  __int64 v18; // [rsp+18h] [rbp-68h]
  __int64 v19; // [rsp+28h] [rbp-58h] BYREF
  __m128i v20; // [rsp+30h] [rbp-50h] BYREF
  __m128i v21[4]; // [rsp+40h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a1 + 264);
  v16 = a3;
  v5 = *(_QWORD *)(v4 + 152);
  v17 = (__m128i **)v4;
  v18 = *(_QWORD *)(v4 + 160);
  v6 = (v18 - v5) >> 5;
  if ( (v18 - v5) >> 7 <= 0 )
  {
LABEL_34:
    if ( v6 != 2 )
    {
      if ( v6 != 3 )
      {
        if ( v6 != 1 )
          goto LABEL_16;
LABEL_37:
        if ( a3 != *(_QWORD *)(v5 + 8) || a3 && memcmp(*(const void **)v5, a2, a3) )
          goto LABEL_16;
        goto LABEL_10;
      }
      if ( a3 == *(_QWORD *)(v5 + 8) && (!a3 || !memcmp(*(const void **)v5, a2, a3)) )
      {
LABEL_10:
        if ( v18 == v5 )
          goto LABEL_16;
        return;
      }
      v5 += 32;
    }
    if ( a3 != *(_QWORD *)(v5 + 8) || a3 && memcmp(*(const void **)v5, a2, a3) )
    {
      v5 += 32;
      goto LABEL_37;
    }
    goto LABEL_10;
  }
  v7 = v5 + ((v18 - v5) >> 7 << 7);
  while ( 1 )
  {
    while ( a3 != *(_QWORD *)(v5 + 8) )
    {
      if ( a3 == *(_QWORD *)(v5 + 40) )
      {
        v8 = v5 + 32;
        if ( !a3 )
          goto LABEL_15;
        v9 = *(const void **)(v5 + 32);
        goto LABEL_14;
      }
      if ( a3 == *(_QWORD *)(v5 + 72) )
      {
        v8 = v5 + 64;
        if ( !a3 )
          goto LABEL_15;
        goto LABEL_42;
      }
LABEL_5:
      if ( a3 != *(_QWORD *)(v5 + 104) )
        goto LABEL_6;
      v8 = v5 + 96;
      if ( !a3 )
        goto LABEL_15;
      v13 = *(const void **)(v5 + 96);
LABEL_31:
      if ( !memcmp(v13, a2, a3) )
        goto LABEL_15;
      v5 += 128;
      if ( v5 == v7 )
      {
LABEL_33:
        v6 = (v18 - v5) >> 5;
        goto LABEL_34;
      }
    }
    if ( !a3 || !memcmp(*(const void **)v5, a2, a3) )
      goto LABEL_10;
    if ( a3 != *(_QWORD *)(v5 + 40) )
    {
      if ( a3 == *(_QWORD *)(v5 + 72) )
      {
        v8 = v5 + 64;
LABEL_42:
        v14 = *(const void **)(v5 + 64);
        goto LABEL_43;
      }
      goto LABEL_5;
    }
    v9 = *(const void **)(v5 + 32);
    v8 = v5 + 32;
LABEL_14:
    if ( !memcmp(v9, a2, a3) )
      break;
    if ( a3 != *(_QWORD *)(v5 + 72) )
      goto LABEL_5;
    v14 = *(const void **)(v5 + 64);
    v8 = v5 + 64;
LABEL_43:
    if ( !memcmp(v14, a2, a3) )
      break;
    if ( a3 == *(_QWORD *)(v5 + 104) )
    {
      v13 = *(const void **)(v5 + 96);
      v8 = v5 + 96;
      goto LABEL_31;
    }
LABEL_6:
    v5 += 128;
    if ( v5 == v7 )
      goto LABEL_33;
  }
LABEL_15:
  if ( v18 != v8 )
    return;
LABEL_16:
  if ( a2 )
  {
    v19 = a3;
    v20.m128i_i64[0] = (__int64)v21;
    if ( a3 > 0xF )
    {
      v20.m128i_i64[0] = sub_22409D0((__int64)&v20, (unsigned __int64 *)&v19, 0);
      v15 = (__m128i *)v20.m128i_i64[0];
      v21[0].m128i_i64[0] = v19;
    }
    else
    {
      if ( a3 == 1 )
      {
        v21[0].m128i_i8[0] = *a2;
        v10 = v21;
        goto LABEL_20;
      }
      if ( !a3 )
      {
        v10 = v21;
        goto LABEL_20;
      }
      v15 = v21;
    }
    memcpy(v15, a2, a3);
    v16 = v19;
    v10 = (__m128i *)v20.m128i_i64[0];
LABEL_20:
    v20.m128i_i64[1] = v16;
    v10->m128i_i8[v16] = 0;
  }
  else
  {
    v20.m128i_i64[1] = 0;
    v20.m128i_i64[0] = (__int64)v21;
    v21[0].m128i_i8[0] = 0;
  }
  v11 = v17[20];
  if ( v11 == v17[21] )
  {
    sub_8F99A0(v17 + 19, v11, &v20);
    v12 = (__m128i *)v20.m128i_i64[0];
  }
  else
  {
    v12 = (__m128i *)v20.m128i_i64[0];
    if ( v11 )
    {
      v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
      if ( (__m128i *)v20.m128i_i64[0] == v21 )
      {
        v11[1] = _mm_load_si128(v21);
      }
      else
      {
        v11->m128i_i64[0] = v20.m128i_i64[0];
        v11[1].m128i_i64[0] = v21[0].m128i_i64[0];
      }
      v20.m128i_i64[0] = (__int64)v21;
      v12 = v21;
      v11->m128i_i64[1] = v20.m128i_i64[1];
      v20.m128i_i64[1] = 0;
      v21[0].m128i_i8[0] = 0;
      v11 = v17[20];
    }
    v17[20] = v11 + 2;
  }
  if ( v12 != v21 )
    j_j___libc_free_0((unsigned __int64)v12);
}
