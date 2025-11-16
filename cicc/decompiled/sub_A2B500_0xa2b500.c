// Function: sub_A2B500
// Address: 0xa2b500
//
__int64 __fastcall sub_A2B500(__int64 a1, __int64 *a2)
{
  __m128i v4; // rdi
  int v5; // eax
  unsigned int v6; // r13d
  __int64 v7; // rbx
  unsigned int v8; // r15d
  __int64 v9; // rdi
  bool v10; // zf
  __m128i *v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rbx
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 i; // rax
  const void *v18; // r15
  __int64 *v19; // rcx
  __int64 *v20; // rax
  __int64 result; // rax
  int v22; // esi
  int v23; // edx
  unsigned int v24; // esi
  __m128i v25; // xmm0
  __int64 v26; // r12
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // [rsp+8h] [rbp-58h]
  __m128i *v30; // [rsp+10h] [rbp-50h] BYREF
  __m128i *v31; // [rsp+18h] [rbp-48h] BYREF
  __m128i v32; // [rsp+20h] [rbp-40h] BYREF

  v4.m128i_i64[0] = (__int64)(a2 + 4);
  v4.m128i_i64[1] = *a2;
  v32 = v4;
  v5 = sub_A16070((char *)v4.m128i_i64[0], v4.m128i_i64[1]);
  if ( v5 )
  {
    v6 = **(_DWORD **)a1;
    if ( v5 == 1 )
      v6 = **(_DWORD **)(a1 + 16);
  }
  else
  {
    v6 = **(_DWORD **)(a1 + 8);
  }
  v7 = *(_QWORD *)(a1 + 24);
  v8 = *(_DWORD *)(v7 + 168);
  v9 = v7 + 152;
  v10 = (unsigned __int8)sub_A19D80(v7 + 152, &v32, &v30) == 0;
  v11 = v30;
  if ( !v10 )
    goto LABEL_4;
  v22 = *(_DWORD *)(v7 + 168);
  ++*(_QWORD *)(v7 + 152);
  v31 = v11;
  v23 = v22 + 1;
  v24 = *(_DWORD *)(v7 + 176);
  if ( 4 * v23 >= 3 * v24 )
  {
    v24 *= 2;
LABEL_41:
    sub_A2B260(v9, v24);
    sub_A19D80(v9, &v32, &v31);
    v23 = *(_DWORD *)(v7 + 168) + 1;
    v11 = v31;
    goto LABEL_22;
  }
  if ( v24 - *(_DWORD *)(v7 + 172) - v23 <= v24 >> 3 )
    goto LABEL_41;
LABEL_22:
  *(_DWORD *)(v7 + 168) = v23;
  if ( v11->m128i_i64[0] != -1 )
    --*(_DWORD *)(v7 + 172);
  v25 = _mm_loadu_si128(&v32);
  v11[1].m128i_i64[0] = 0;
  *v11 = v25;
LABEL_4:
  v11[1].m128i_i64[0] = v8;
  sub_9C8C60(*(_QWORD *)(a1 + 32), v8);
  v12 = *(_QWORD *)(a1 + 32);
  v13 = v32.m128i_i64[1];
  v14 = v32.m128i_i64[0];
  v15 = *(unsigned int *)(v12 + 8);
  if ( v32.m128i_i64[1] + v15 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
  {
    v29 = *(_QWORD *)(a1 + 32);
    sub_C8D5F0(v12, v12 + 16, v32.m128i_i64[1] + v15, 4);
    v12 = v29;
    v15 = *(unsigned int *)(v29 + 8);
  }
  v16 = *(_QWORD *)v12 + 4 * v15;
  if ( v13 > 0 )
  {
    for ( i = 0; i != v13; ++i )
      *(_DWORD *)(v16 + 4 * i) = *(char *)(v14 + i);
    LODWORD(v15) = *(_DWORD *)(v12 + 8);
  }
  v18 = a2 + 1;
  *(_DWORD *)(v12 + 8) = v15 + v13;
  sub_A214F0(**(_QWORD **)(a1 + 24), 1u, *(_QWORD *)(a1 + 32), v6);
  v19 = (__int64 *)((char *)a2 + 28);
  v20 = a2 + 1;
  while ( !*(_DWORD *)v20 )
  {
    if ( *((_DWORD *)v20 + 1) )
    {
      v20 = (__int64 *)((char *)v20 + 4);
      break;
    }
    if ( *((_DWORD *)v20 + 2) )
    {
      ++v20;
      break;
    }
    if ( *((_DWORD *)v20 + 3) )
    {
      v20 = (__int64 *)((char *)v20 + 12);
      break;
    }
    v20 += 2;
    if ( a2 + 3 == v20 )
    {
      v28 = ((char *)v19 - (char *)v20) >> 2;
      if ( v28 != 2 )
      {
        if ( v28 != 3 )
        {
          if ( v28 != 1 )
            goto LABEL_17;
LABEL_34:
          if ( !*(_DWORD *)v20 )
            goto LABEL_17;
          break;
        }
        if ( *(_DWORD *)v20 )
          break;
        v20 = (__int64 *)((char *)v20 + 4);
      }
      if ( *(_DWORD *)v20 )
        break;
      v20 = (__int64 *)((char *)v20 + 4);
      goto LABEL_34;
    }
  }
  if ( v19 != v20 )
  {
    v26 = *(_QWORD *)(a1 + 32);
    v27 = 0;
    *(_DWORD *)(v26 + 8) = 0;
    if ( *(unsigned int *)(v26 + 12) < 5uLL )
    {
      sub_C8D5F0(v26, v26 + 16, 5, 4);
      v27 = 4LL * *(unsigned int *)(v26 + 8);
    }
    memcpy((void *)(*(_QWORD *)v26 + v27), v18, 0x14u);
    *(_DWORD *)(v26 + 8) += 5;
    sub_A214F0(**(_QWORD **)(a1 + 24), 2u, *(_QWORD *)(a1 + 32), **(_DWORD **)(a1 + 40));
  }
LABEL_17:
  result = *(_QWORD *)(a1 + 32);
  *(_DWORD *)(result + 8) = 0;
  return result;
}
