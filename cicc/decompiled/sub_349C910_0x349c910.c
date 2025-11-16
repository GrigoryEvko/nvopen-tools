// Function: sub_349C910
// Address: 0x349c910
//
__int64 __fastcall sub_349C910(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  _QWORD *v9; // rbx
  __int64 v10; // r12
  __int64 v11; // r13
  unsigned int v12; // eax
  __int64 v13; // r8
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // r9
  unsigned __int64 v16; // r10
  int v17; // edx
  unsigned __int64 v18; // rdi
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // r13
  signed __int64 v22; // r13
  __int64 v23; // rbx
  __m128i *v24; // rax
  const __m128i *v25; // r12
  const __m128i *v26; // rcx
  const __m128i *v27; // rax
  __m128i v28; // xmm0
  __int64 v29; // rsi
  const __m128i *v30; // rax
  unsigned __int64 v32; // rdx
  const __m128i *v33; // r12
  __m128i *v34; // rdx
  __int64 v35; // r13
  char *v36; // r12
  const void *v37; // [rsp+0h] [rbp-70h]
  _QWORD *v38; // [rsp+10h] [rbp-60h]
  __m128i *v39; // [rsp+10h] [rbp-60h]
  const __m128i *v41; // [rsp+18h] [rbp-58h]
  _QWORD v42[2]; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v43; // [rsp+30h] [rbp-40h]

  *(_QWORD *)a1 = a1 + 16;
  v37 = (const void *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = 0x200000000LL;
  v7 = *(unsigned int *)(a3 + 24);
  if ( (unsigned int)v7 > 2 )
  {
    sub_C8D5F0(a1, v37, v7, 0x18u, a5, a6);
    v9 = *(_QWORD **)(a3 + 16);
    v35 = 4LL * *(unsigned int *)(a3 + 24);
    v38 = &v9[v35];
    if ( v9 == &v9[v35] )
      goto LABEL_12;
  }
  else
  {
    v8 = 4 * v7;
    v9 = *(_QWORD **)(a3 + 16);
    v38 = &v9[v8];
    if ( &v9[v8] == v9 )
    {
      v19 = a1 + 16;
      v21 = (__int64)v37 + 24 * *(unsigned int *)(a1 + 8);
      goto LABEL_25;
    }
  }
  do
  {
    v10 = *v9;
    v11 = v9[1];
    v12 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 2488LL))(a2, *v9, v11);
    if ( (!*(_BYTE *)(a3 + 10) || v12 <= 2) && (v12 != 2 || *(_DWORD *)(a3 + 4) == -1) )
    {
      v14 = *(unsigned int *)(a1 + 8);
      v15 = *(_QWORD *)a1;
      v16 = *(unsigned int *)(a1 + 12);
      v17 = *(_DWORD *)(a1 + 8);
      v18 = *(_QWORD *)a1 + 24 * v14;
      if ( v14 >= v16 )
      {
        v32 = v14 + 1;
        v42[0] = v10;
        v33 = (const __m128i *)v42;
        v42[1] = v11;
        v43 = v12;
        if ( v16 < v14 + 1 )
        {
          if ( v15 > (unsigned __int64)v42 || v18 <= (unsigned __int64)v42 )
          {
            sub_C8D5F0(a1, v37, v32, 0x18u, v13, v15);
            v15 = *(_QWORD *)a1;
            v14 = *(unsigned int *)(a1 + 8);
          }
          else
          {
            v36 = (char *)v42 - v15;
            sub_C8D5F0(a1, v37, v32, 0x18u, v13, v15);
            v15 = *(_QWORD *)a1;
            v14 = *(unsigned int *)(a1 + 8);
            v33 = (const __m128i *)&v36[*(_QWORD *)a1];
          }
        }
        v34 = (__m128i *)(v15 + 24 * v14);
        *v34 = _mm_loadu_si128(v33);
        v34[1].m128i_i64[0] = v33[1].m128i_i64[0];
        ++*(_DWORD *)(a1 + 8);
      }
      else
      {
        if ( v18 )
        {
          *(_QWORD *)v18 = v10;
          *(_QWORD *)(v18 + 8) = v11;
          *(_DWORD *)(v18 + 16) = v12;
          v17 = *(_DWORD *)(a1 + 8);
        }
        *(_DWORD *)(a1 + 8) = v17 + 1;
      }
    }
    v9 += 4;
  }
  while ( v38 != v9 );
LABEL_12:
  v19 = *(_QWORD *)a1;
  v20 = 24LL * *(unsigned int *)(a1 + 8);
  v21 = *(_QWORD *)a1 + v20;
  if ( !v20 )
  {
LABEL_25:
    v25 = 0;
    sub_3440EB0(v19, v21);
    goto LABEL_19;
  }
  v39 = *(__m128i **)a1;
  v41 = (const __m128i *)(*(_QWORD *)a1 + v20);
  v22 = 0xAAAAAAAAAAAAAAABLL * (v20 >> 3);
  while ( 1 )
  {
    v23 = 24 * v22;
    v24 = (__m128i *)sub_2207800(24 * v22);
    v25 = v24;
    if ( v24 )
      break;
    v22 >>= 1;
    if ( !v22 )
    {
      v21 = (__int64)v41;
      v19 = (__int64)v39;
      goto LABEL_25;
    }
  }
  v26 = (__m128i *)((char *)v24 + v23);
  *v24 = _mm_loadu_si128(v39);
  v24[1].m128i_i64[0] = v39[1].m128i_i64[0];
  v27 = (__m128i *)((char *)v24 + 24);
  if ( v26 == (const __m128i *)&v25[1].m128i_u64[1] )
  {
    v30 = v25;
  }
  else
  {
    do
    {
      v28 = _mm_loadu_si128((const __m128i *)((char *)v27 - 24));
      v29 = v27[-1].m128i_i64[1];
      v27 = (const __m128i *)((char *)v27 + 24);
      *(__m128i *)((char *)v27 - 24) = v28;
      v27[-1].m128i_i64[1] = v29;
    }
    while ( v26 != v27 );
    v30 = (const __m128i *)((char *)v25 + v23 - 24);
  }
  *v39 = _mm_loadu_si128(v30);
  v39[1].m128i_i32[0] = v30[1].m128i_i32[0];
  sub_349C820(v39, v41, v25, v22);
LABEL_19:
  j_j___libc_free_0((unsigned __int64)v25);
  return a1;
}
