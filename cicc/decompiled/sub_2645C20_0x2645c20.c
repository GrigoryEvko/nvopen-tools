// Function: sub_2645C20
// Address: 0x2645c20
//
__int64 __fastcall sub_2645C20(__int64 a1, const __m128i *a2)
{
  __m128i v4; // xmm0
  __int64 v5; // r9
  __int64 v6; // rax
  unsigned int v8; // esi
  int v9; // eax
  __int64 v10; // r13
  int v11; // eax
  __int64 v12; // rdx
  __m128i v13; // xmm1
  unsigned __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rcx
  const __m128i *v17; // rax
  __m128i *v18; // rdx
  unsigned __int64 v19; // r12
  __int64 v20; // rdi
  const void *v21; // rsi
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // [rsp+8h] [rbp-78h] BYREF
  __m128i v25; // [rsp+10h] [rbp-70h] BYREF
  int v26; // [rsp+20h] [rbp-60h]
  __m128i v27; // [rsp+30h] [rbp-50h] BYREF
  __int64 v28; // [rsp+40h] [rbp-40h]

  v4 = _mm_loadu_si128(a2);
  v26 = 0;
  v27 = v4;
  v25 = v4;
  if ( (unsigned __int8)sub_26404E0(a1, v25.m128i_i64, &v24) )
  {
    v6 = *(unsigned int *)(v24 + 16);
    return *(_QWORD *)(a1 + 32) + 24 * v6 + 16;
  }
  v8 = *(_DWORD *)(a1 + 24);
  v9 = *(_DWORD *)(a1 + 16);
  v10 = v24;
  ++*(_QWORD *)a1;
  v11 = v9 + 1;
  v27.m128i_i64[0] = v10;
  if ( 4 * v11 >= 3 * v8 )
  {
    sub_26459A0(a1, 2 * v8);
  }
  else
  {
    if ( v8 - *(_DWORD *)(a1 + 20) - v11 > v8 >> 3 )
      goto LABEL_6;
    sub_26459A0(a1, v8);
  }
  sub_26404E0(a1, v25.m128i_i64, &v27);
  v10 = v27.m128i_i64[0];
  v11 = *(_DWORD *)(a1 + 16) + 1;
LABEL_6:
  *(_DWORD *)(a1 + 16) = v11;
  if ( *(_QWORD *)v10 != -4096 || *(_DWORD *)(v10 + 8) != -1 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v10 = v25.m128i_i64[0];
  *(_DWORD *)(v10 + 8) = v25.m128i_i32[2];
  *(_DWORD *)(v10 + 16) = v26;
  v12 = *(unsigned int *)(a1 + 40);
  v13 = _mm_loadu_si128(a2);
  v14 = *(unsigned int *)(a1 + 44);
  v28 = 0;
  v15 = v12 + 1;
  v27 = v13;
  if ( v12 + 1 > v14 )
  {
    v19 = *(_QWORD *)(a1 + 32);
    v20 = a1 + 32;
    v21 = (const void *)(a1 + 48);
    if ( v19 > (unsigned __int64)&v27 )
    {
      sub_C8D5F0(v20, v21, v12 + 1, 0x18u, v15, v5);
      v16 = *(_QWORD *)(a1 + 32);
      v12 = *(unsigned int *)(a1 + 40);
      v17 = &v27;
    }
    else
    {
      v22 = 3 * v12;
      v23 = v12 + 1;
      if ( (unsigned __int64)&v27 < v19 + 8 * v22 )
      {
        sub_C8D5F0(v20, v21, v23, 0x18u, v15, v5);
        v16 = *(_QWORD *)(a1 + 32);
        v12 = *(unsigned int *)(a1 + 40);
        v17 = (__m128i *)((char *)&v27 + v16 - v19);
      }
      else
      {
        sub_C8D5F0(v20, v21, v23, 0x18u, v15, v5);
        v16 = *(_QWORD *)(a1 + 32);
        v12 = *(unsigned int *)(a1 + 40);
        v17 = &v27;
      }
    }
  }
  else
  {
    v16 = *(_QWORD *)(a1 + 32);
    v17 = &v27;
  }
  v18 = (__m128i *)(v16 + 24 * v12);
  *v18 = _mm_loadu_si128(v17);
  v18[1].m128i_i64[0] = v17[1].m128i_i64[0];
  v6 = *(unsigned int *)(a1 + 40);
  *(_DWORD *)(a1 + 40) = v6 + 1;
  *(_DWORD *)(v10 + 16) = v6;
  return *(_QWORD *)(a1 + 32) + 24 * v6 + 16;
}
