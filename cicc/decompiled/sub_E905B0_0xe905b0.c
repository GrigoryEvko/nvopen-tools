// Function: sub_E905B0
// Address: 0xe905b0
//
__int64 __fastcall sub_E905B0(_QWORD *a1, const __m128i *a2, _DWORD *a3)
{
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 *v9; // rax
  _QWORD *v10; // r12
  __int64 *v11; // r15
  __int64 v12; // rdi
  __int64 result; // rax
  __int64 v14; // r12
  __int64 v15; // rbx
  __int32 v16; // edx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 *v19; // rax
  _QWORD *v20; // r8
  __int64 *v21; // r15
  __int64 v22; // rdi
  _QWORD *v23; // r8
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 *v26; // rax
  _QWORD *v27; // r13
  __int64 *v28; // rbx
  __int64 v29; // rdi
  __m128i *v30; // rsi
  _QWORD *v32; // [rsp+8h] [rbp-78h]
  _QWORD *v33; // [rsp+8h] [rbp-78h]
  _QWORD *v34; // [rsp+28h] [rbp-58h] BYREF
  __m128i v35; // [rsp+30h] [rbp-50h] BYREF
  __m128i v36[4]; // [rsp+40h] [rbp-40h] BYREF

  v5 = a3[2];
  v35.m128i_i32[0] = 0;
  v35.m128i_i64[1] = 0;
  if ( v5 )
    v6 = *(_QWORD *)(*(_QWORD *)a3 + 8LL);
  else
    v6 = a2[1].m128i_i64[0];
  v35.m128i_i64[1] = v6;
  v7 = sub_22077B0(96);
  if ( v7 )
  {
    *(_QWORD *)(v7 + 8) = 1;
    *(_QWORD *)v7 = v7 + 48;
    v8 = v35.m128i_i64[1];
    *(_QWORD *)(v7 + 16) = 0;
    *(_QWORD *)(v7 + 24) = 0;
    *(_DWORD *)(v7 + 32) = 1065353216;
    *(_QWORD *)(v7 + 40) = 0;
    *(_QWORD *)(v7 + 48) = 0;
    *(_QWORD *)(v7 + 56) = 0;
    *(_QWORD *)(v7 + 64) = 0;
    *(_QWORD *)(v7 + 72) = 0;
    *(_QWORD *)(v7 + 88) = 0;
    *(_QWORD *)(v7 + 80) = v8;
  }
  v36[0].m128i_i64[0] = v7;
  v9 = sub_E90130(a1, &v35, v36);
  v10 = (_QWORD *)v36[0].m128i_i64[0];
  v11 = v9;
  if ( v36[0].m128i_i64[0] )
  {
    v12 = *(_QWORD *)(v36[0].m128i_i64[0] + 56);
    if ( v12 )
      j_j___libc_free_0(v12, *(_QWORD *)(v36[0].m128i_i64[0] + 72) - v12);
    sub_E90070((__int64)v10);
    if ( (_QWORD *)*v10 != v10 + 6 )
      j_j___libc_free_0(*v10, 8LL * v10[1]);
    j_j___libc_free_0(v10, 96);
  }
  *(_QWORD *)(v11[3] + 88) = a1;
  result = (unsigned int)a3[2];
  v14 = v11[3];
  if ( (_DWORD)result )
  {
    v15 = *(_QWORD *)a3 + 16LL;
    v16 = **(_DWORD **)a3;
    if ( v15 != *(_QWORD *)a3 + 16 * result )
    {
      do
      {
        v36[0].m128i_i32[0] = v16;
        v36[0].m128i_i64[1] = *(_QWORD *)(v15 + 8);
        v17 = sub_22077B0(96);
        if ( v17 )
        {
          *(_QWORD *)(v17 + 8) = 1;
          *(_QWORD *)v17 = v17 + 48;
          v18 = v36[0].m128i_i64[1];
          *(_QWORD *)(v17 + 16) = 0;
          *(_QWORD *)(v17 + 24) = 0;
          *(_DWORD *)(v17 + 32) = 1065353216;
          *(_QWORD *)(v17 + 40) = 0;
          *(_QWORD *)(v17 + 48) = 0;
          *(_QWORD *)(v17 + 56) = 0;
          *(_QWORD *)(v17 + 64) = 0;
          *(_QWORD *)(v17 + 72) = 0;
          *(_QWORD *)(v17 + 88) = 0;
          *(_QWORD *)(v17 + 80) = v18;
        }
        v34 = (_QWORD *)v17;
        v19 = sub_E90130((_QWORD *)v14, v36, &v34);
        v20 = v34;
        v21 = v19;
        if ( v34 )
        {
          v22 = v34[7];
          if ( v22 )
          {
            v32 = v34;
            j_j___libc_free_0(v22, v34[9] - v22);
            v20 = v32;
          }
          v33 = v20;
          sub_E90070((__int64)v20);
          v23 = v33;
          if ( (_QWORD *)*v33 != v33 + 6 )
          {
            j_j___libc_free_0(*v33, 8LL * v33[1]);
            v23 = v33;
          }
          j_j___libc_free_0(v23, 96);
        }
        v15 += 16;
        *(_QWORD *)(v21[3] + 88) = v14;
        v16 = *(_DWORD *)(v15 - 16);
        v14 = v21[3];
      }
      while ( v15 != *(_QWORD *)a3 + 16LL * (unsigned int)a3[2] );
    }
    v36[0].m128i_i32[0] = v16;
    v36[0].m128i_i64[1] = a2[1].m128i_i64[0];
    v24 = sub_22077B0(96);
    if ( v24 )
    {
      *(_QWORD *)(v24 + 8) = 1;
      *(_QWORD *)v24 = v24 + 48;
      v25 = v36[0].m128i_i64[1];
      *(_QWORD *)(v24 + 16) = 0;
      *(_QWORD *)(v24 + 24) = 0;
      *(_DWORD *)(v24 + 32) = 1065353216;
      *(_QWORD *)(v24 + 40) = 0;
      *(_QWORD *)(v24 + 48) = 0;
      *(_QWORD *)(v24 + 56) = 0;
      *(_QWORD *)(v24 + 64) = 0;
      *(_QWORD *)(v24 + 72) = 0;
      *(_QWORD *)(v24 + 88) = 0;
      *(_QWORD *)(v24 + 80) = v25;
    }
    v34 = (_QWORD *)v24;
    v26 = sub_E90130((_QWORD *)v14, v36, &v34);
    v27 = v34;
    v28 = v26;
    if ( v34 )
    {
      v29 = v34[7];
      if ( v29 )
        j_j___libc_free_0(v29, v34[9] - v29);
      sub_E90070((__int64)v27);
      if ( (_QWORD *)*v27 != v27 + 6 )
        j_j___libc_free_0(*v27, 8LL * v27[1]);
      j_j___libc_free_0(v27, 96);
    }
    result = v28[3];
    *(_QWORD *)(result + 88) = v14;
    v14 = v28[3];
  }
  v30 = *(__m128i **)(v14 + 64);
  if ( v30 == *(__m128i **)(v14 + 72) )
    return sub_E90420((const __m128i **)(v14 + 56), v30, a2);
  if ( v30 )
  {
    result = (__int64)a2;
    *v30 = _mm_loadu_si128(a2);
    v30[1] = _mm_loadu_si128(a2 + 1);
    v30 = *(__m128i **)(v14 + 64);
  }
  *(_QWORD *)(v14 + 64) = v30 + 2;
  return result;
}
