// Function: sub_E8EE00
// Address: 0xe8ee00
//
__int64 __fastcall sub_E8EE00(__int64 a1, __int64 a2, __int8 *a3, size_t a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v10; // r14
  __m128i *v11; // rdx
  __int64 v12; // rdx
  int v13; // eax
  __m128i *v14; // rdi
  __m128i *v15; // rdx
  __int64 v16; // rax
  __int64 result; // rax
  __m128i *v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // r12
  __int64 v21; // rcx
  __m128i *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  const __m128i *v25; // rdx
  __int64 *v26; // r14
  const __m128i *v27; // rax
  __m128i *v28; // rdx
  __int64 v29; // rcx
  const __m128i *v30; // rcx
  __int64 *v31; // r13
  __int64 v32; // [rsp+8h] [rbp-68h]
  __int64 v33; // [rsp+18h] [rbp-58h] BYREF
  __m128i *v34; // [rsp+20h] [rbp-50h] BYREF
  __int64 v35; // [rsp+28h] [rbp-48h]
  __m128i v36[4]; // [rsp+30h] [rbp-40h] BYREF

  v6 = a4;
  v10 = *(unsigned int *)(a2 + 64);
  v34 = v36;
  if ( &a3[a4] && !a3 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v33 = a4;
  if ( a4 > 0xF )
  {
    v34 = (__m128i *)sub_22409D0(&v34, &v33, 0);
    v18 = v34;
    v36[0].m128i_i64[0] = v33;
  }
  else
  {
    if ( a4 == 1 )
    {
      v36[0].m128i_i8[0] = *a3;
      v11 = v36;
      goto LABEL_6;
    }
    if ( !a4 )
    {
      v11 = v36;
      goto LABEL_6;
    }
    v18 = v36;
  }
  memcpy(v18, a3, a4);
  v6 = v33;
  v11 = v34;
LABEL_6:
  v35 = v6;
  v11->m128i_i8[v6] = 0;
  v12 = *(unsigned int *)(a1 + 16);
  v13 = v12;
  if ( *(_DWORD *)(a1 + 20) <= (unsigned int)v12 )
  {
    v19 = a1 + 24;
    v20 = sub_C8D7D0(a1 + 8, a1 + 24, 0, 0x28u, (unsigned __int64 *)&v33, a6);
    v21 = 40LL * *(unsigned int *)(a1 + 16);
    v22 = (__m128i *)(v21 + v20);
    if ( v21 + v20 )
    {
      v22->m128i_i64[0] = (__int64)v22[1].m128i_i64;
      if ( v34 == v36 )
      {
        v22[1] = _mm_load_si128(v36);
      }
      else
      {
        v22->m128i_i64[0] = (__int64)v34;
        v22[1].m128i_i64[0] = v36[0].m128i_i64[0];
      }
      v23 = v35;
      v22[2].m128i_i64[0] = v10;
      v34 = v36;
      v22->m128i_i64[1] = v23;
      v24 = *(unsigned int *)(a1 + 16);
      v35 = 0;
      v36[0].m128i_i8[0] = 0;
      v21 = 40 * v24;
    }
    v25 = *(const __m128i **)(a1 + 8);
    v26 = &v25->m128i_i64[(unsigned __int64)v21 / 8];
    if ( v25 != (const __m128i *)&v25->m128i_i8[v21] )
    {
      v27 = v25 + 1;
      v19 = v20 + 8 * ((unsigned __int64)(v21 - 40) >> 3) + 40;
      v28 = (__m128i *)v20;
      do
      {
        if ( v28 )
        {
          v28->m128i_i64[0] = (__int64)v28[1].m128i_i64;
          v30 = (const __m128i *)v27[-1].m128i_i64[0];
          if ( v27 == v30 )
          {
            v28[1] = _mm_loadu_si128(v27);
          }
          else
          {
            v28->m128i_i64[0] = (__int64)v30;
            v28[1].m128i_i64[0] = v27->m128i_i64[0];
          }
          v28->m128i_i64[1] = v27[-1].m128i_i64[1];
          v29 = v27[1].m128i_i64[0];
          v27[-1].m128i_i64[0] = (__int64)v27;
          v27[-1].m128i_i64[1] = 0;
          v27->m128i_i8[0] = 0;
          v28[2].m128i_i64[0] = v29;
        }
        v28 = (__m128i *)((char *)v28 + 40);
        v27 = (const __m128i *)((char *)v27 + 40);
      }
      while ( (__m128i *)v19 != v28 );
      v31 = *(__int64 **)(a1 + 8);
      v26 = &v31[5 * *(unsigned int *)(a1 + 16)];
      if ( v31 != v26 )
      {
        do
        {
          v26 -= 5;
          if ( (__int64 *)*v26 != v26 + 2 )
          {
            v19 = v26[2] + 1;
            j_j___libc_free_0(*v26, v19);
          }
        }
        while ( v31 != v26 );
        v26 = *(__int64 **)(a1 + 8);
      }
    }
    result = v33;
    if ( v26 != (__int64 *)(a1 + 24) )
    {
      v32 = v33;
      _libc_free(v26, v19);
      result = v32;
    }
    ++*(_DWORD *)(a1 + 16);
    v14 = v34;
    *(_QWORD *)(a1 + 8) = v20;
    *(_DWORD *)(a1 + 20) = result;
  }
  else
  {
    v14 = v34;
    v15 = (__m128i *)(*(_QWORD *)(a1 + 8) + 40 * v12);
    if ( v15 )
    {
      v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
      if ( v34 == v36 )
      {
        v15[1] = _mm_load_si128(v36);
      }
      else
      {
        v15->m128i_i64[0] = (__int64)v34;
        v15[1].m128i_i64[0] = v36[0].m128i_i64[0];
      }
      v16 = v35;
      v34 = v36;
      v15[2].m128i_i64[0] = v10;
      v14 = v36;
      v15->m128i_i64[1] = v16;
      v13 = *(_DWORD *)(a1 + 16);
      v35 = 0;
      v36[0].m128i_i8[0] = 0;
    }
    result = (unsigned int)(v13 + 1);
    *(_DWORD *)(a1 + 16) = result;
  }
  if ( v14 != v36 )
    return j_j___libc_free_0(v14, v36[0].m128i_i64[0] + 1);
  return result;
}
