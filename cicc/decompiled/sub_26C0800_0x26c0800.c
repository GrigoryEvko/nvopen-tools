// Function: sub_26C0800
// Address: 0x26c0800
//
void __fastcall sub_26C0800(__m128i *a1)
{
  unsigned __int64 v2; // rax
  __int32 v3; // edx
  __int64 *v4; // r12
  __int64 v5; // r15
  _QWORD *v6; // r14
  __int64 v7; // r13
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 v10; // rsi
  __m128i *v11; // rdx
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  __m128i v14; // xmm1
  _QWORD *v15; // r12
  unsigned __int64 v16; // rdi
  _QWORD *v17; // r12
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  __int32 v20; // xmm0_4
  __m128i v21; // xmm2
  unsigned __int64 *v22; // rdi
  __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rdx
  _QWORD *v27; // rbx
  unsigned __int64 v28; // rdi
  __int64 v29; // rdx
  unsigned __int64 v30; // [rsp+0h] [rbp-C0h]
  __int64 v31; // [rsp+8h] [rbp-B8h]
  __m128i v32; // [rsp+10h] [rbp-B0h] BYREF
  _QWORD *v33; // [rsp+20h] [rbp-A0h]
  __int64 v34; // [rsp+28h] [rbp-98h]
  int v35; // [rsp+30h] [rbp-90h]
  __int64 v36; // [rsp+38h] [rbp-88h]
  _QWORD v37[2]; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int128 s; // [rsp+50h] [rbp-70h] BYREF
  _QWORD *v39; // [rsp+60h] [rbp-60h] BYREF
  __int64 v40; // [rsp+68h] [rbp-58h]
  __m128i v41; // [rsp+70h] [rbp-50h] BYREF
  _QWORD v42[8]; // [rsp+80h] [rbp-40h] BYREF

  v2 = a1[7].m128i_u64[1];
  v3 = a1[8].m128i_i32[0];
  a1[7].m128i_i64[1] = 0;
  a1[8].m128i_i32[2] = 0;
  v30 = v2;
  LODWORD(v2) = a1[8].m128i_i32[1];
  a1[8].m128i_i64[0] = 0;
  if ( (_DWORD)v2 && v3 )
  {
    v4 = (__int64 *)v30;
    v31 = v30 + 8LL * (unsigned int)(v3 - 1) + 8;
    do
    {
      v5 = *v4;
      if ( *v4 != -8 && v5 )
      {
        v6 = *(_QWORD **)(v5 + 24);
        v7 = *(_QWORD *)v5 + 65LL;
        while ( v6 )
        {
          v8 = (unsigned __int64)v6;
          v6 = (_QWORD *)*v6;
          j_j___libc_free_0(v8);
        }
        memset(*(void **)(v5 + 8), 0, 8LL * *(_QWORD *)(v5 + 16));
        v9 = *(_QWORD *)(v5 + 8);
        *(_QWORD *)(v5 + 32) = 0;
        *(_QWORD *)(v5 + 24) = 0;
        if ( v9 != v5 + 56 )
          j_j___libc_free_0(v9);
        sub_C7D6A0(v5, v7, 8);
      }
      ++v4;
    }
    while ( (__int64 *)v31 != v4 );
  }
  _libc_free(v30);
  v10 = a1[18].m128i_i64[1];
  v11 = (__m128i *)a1[17].m128i_i64[0];
  v32.m128i_i64[0] = (__int64)v37;
  v12 = a1[17].m128i_u64[1];
  v13 = a1[18].m128i_i64[0];
  v32.m128i_i64[1] = 1;
  v14 = _mm_loadu_si128(a1 + 19);
  v40 = v10;
  v33 = 0;
  v34 = 0;
  v35 = 1065353216;
  v36 = 0;
  v37[0] = 0;
  s = __PAIR128__(v12, (unsigned __int64)v11);
  v39 = (_QWORD *)v13;
  v42[0] = 0;
  v41 = v14;
  if ( v11 == &a1[20] )
  {
    v29 = a1[20].m128i_i64[0];
    *(_QWORD *)&s = v42;
    v42[0] = v29;
  }
  if ( v13 )
    *(_QWORD *)(s + 8 * (*(_QWORD *)(v13 + 8) % v12)) = &v39;
  a1[17].m128i_i64[0] = (__int64)a1[20].m128i_i64;
  a1[19].m128i_i64[1] = 0;
  a1[17].m128i_i64[1] = 1;
  a1[20].m128i_i64[0] = 0;
  a1[18].m128i_i64[0] = 0;
  a1[18].m128i_i64[1] = 0;
  sub_26BB320(a1 + 17, &v32);
  sub_26BB320(&v32, (__m128i *)&s);
  v15 = v39;
  while ( v15 )
  {
    v16 = (unsigned __int64)v15;
    v15 = (_QWORD *)*v15;
    j_j___libc_free_0(v16);
  }
  memset((void *)s, 0, 8LL * *((_QWORD *)&s + 1));
  v40 = 0;
  v39 = 0;
  if ( (_QWORD *)s != v42 )
    j_j___libc_free_0(s);
  v17 = v33;
  while ( v17 )
  {
    v18 = (unsigned __int64)v17;
    v17 = (_QWORD *)*v17;
    j_j___libc_free_0(v18);
  }
  memset((void *)v32.m128i_i64[0], 0, 8 * v32.m128i_i64[1]);
  v34 = 0;
  v33 = 0;
  if ( (_QWORD *)v32.m128i_i64[0] != v37 )
    j_j___libc_free_0(v32.m128i_u64[0]);
  v19 = a1[15].m128i_i64[0];
  v41.m128i_i32[0] = 1065353216;
  v41.m128i_i64[1] = 0;
  v20 = a1[14].m128i_i32[2];
  v21 = _mm_loadu_si128(&v41);
  v22 = (unsigned __int64 *)a1[12].m128i_i64[1];
  v41.m128i_i64[1] = v19;
  *(_QWORD *)&s = v42;
  *(__m128i *)((char *)a1 + 232) = v21;
  v41.m128i_i32[0] = v20;
  if ( v22 == &a1[15].m128i_u64[1] )
  {
    v22 = v42;
  }
  else
  {
    *(_QWORD *)&s = v22;
    a1[12].m128i_i64[1] = (__int64)&a1[15].m128i_i64[1];
  }
  v23 = a1[14].m128i_i64[0];
  v24 = a1[13].m128i_u64[0];
  a1[14].m128i_i64[0] = 0;
  v25 = a1[13].m128i_i64[1];
  a1[13].m128i_i64[0] = 1;
  v40 = v23;
  v26 = a1[15].m128i_i64[1];
  *((_QWORD *)&s + 1) = v24;
  a1[13].m128i_i64[1] = 0;
  v39 = (_QWORD *)v25;
  a1[15].m128i_i64[1] = 0;
  v42[0] = v26;
  if ( v25 )
  {
    v22[*(_QWORD *)(v25 + 8) % v24] = (unsigned __int64)&v39;
    v27 = v39;
    while ( v27 )
    {
      v28 = (unsigned __int64)v27;
      v27 = (_QWORD *)*v27;
      j_j___libc_free_0(v28);
    }
    v24 = *((_QWORD *)&s + 1);
    v22 = (unsigned __int64 *)s;
  }
  memset(v22, 0, 8 * v24);
  v40 = 0;
  v39 = 0;
  if ( (_QWORD *)s != v42 )
    j_j___libc_free_0(s);
}
