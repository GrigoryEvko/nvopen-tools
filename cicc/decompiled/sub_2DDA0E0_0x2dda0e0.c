// Function: sub_2DDA0E0
// Address: 0x2dda0e0
//
__int64 __fastcall sub_2DDA0E0(__int64 a1, const __m128i *a2)
{
  __m128i v4; // xmm0
  __int64 v5; // rax
  __int64 v6; // r9
  __int64 v7; // rax
  unsigned int v9; // esi
  int v10; // eax
  __int64 v11; // r14
  int v12; // eax
  __int64 v13; // r8
  __int64 v14; // rdx
  __int64 v15; // rax
  __m128i v16; // xmm2
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rsi
  __int64 v19; // rcx
  __m128i *v20; // rsi
  __int64 v21; // rdx
  __m128i *v22; // rdi
  __int64 v23; // rax
  __m128i v24; // xmm3
  _BYTE *v25; // rdi
  unsigned __int64 v26; // r13
  __int64 v27; // rdi
  _QWORD v28[2]; // [rsp+0h] [rbp-90h] BYREF
  __m128i v29; // [rsp+10h] [rbp-80h] BYREF
  __int64 v30; // [rsp+20h] [rbp-70h]
  int v31; // [rsp+28h] [rbp-68h]
  __m128i v32; // [rsp+30h] [rbp-60h] BYREF
  __int64 v33; // [rsp+40h] [rbp-50h]
  _BYTE *v34; // [rsp+48h] [rbp-48h]
  __int64 v35; // [rsp+50h] [rbp-40h]
  _BYTE v36[56]; // [rsp+58h] [rbp-38h] BYREF

  v4 = _mm_loadu_si128(a2);
  v5 = a2[1].m128i_i64[0];
  v31 = 0;
  v33 = v5;
  v30 = v5;
  v32 = v4;
  v29 = v4;
  if ( (unsigned __int8)sub_2DD7B60(a1, v29.m128i_i32, v28) )
  {
    v7 = *(unsigned int *)(v28[0] + 24LL);
    return *(_QWORD *)(a1 + 32) + 40 * v7 + 24;
  }
  v9 = *(_DWORD *)(a1 + 24);
  v10 = *(_DWORD *)(a1 + 16);
  v11 = v28[0];
  ++*(_QWORD *)a1;
  v12 = v10 + 1;
  v13 = 2 * v9;
  v32.m128i_i64[0] = v11;
  if ( 4 * v12 >= 3 * v9 )
  {
    sub_2DD9F10(a1, v13);
  }
  else
  {
    if ( v9 - *(_DWORD *)(a1 + 20) - v12 > v9 >> 3 )
      goto LABEL_6;
    sub_2DD9F10(a1, v9);
  }
  sub_2DD7B60(a1, v29.m128i_i32, &v32);
  v11 = v32.m128i_i64[0];
  v12 = *(_DWORD *)(a1 + 16) + 1;
LABEL_6:
  *(_DWORD *)(a1 + 16) = v12;
  if ( *(_DWORD *)v11 != -1 || *(_QWORD *)(v11 + 8) != -1 )
    --*(_DWORD *)(a1 + 20);
  *(_DWORD *)v11 = v29.m128i_i32[0];
  *(__m128i *)(v11 + 8) = _mm_loadu_si128((const __m128i *)&v29.m128i_u64[1]);
  *(_DWORD *)(v11 + 24) = v31;
  v14 = *(unsigned int *)(a1 + 40);
  v15 = a2[1].m128i_i64[0];
  v16 = _mm_loadu_si128(a2);
  v28[0] = &v29;
  v17 = *(unsigned int *)(a1 + 44);
  v18 = v14 + 1;
  v34 = v36;
  v33 = v15;
  v7 = v14;
  v28[1] = 0;
  v35 = 0;
  v32 = v16;
  if ( v14 + 1 > v17 )
  {
    v26 = *(_QWORD *)(a1 + 32);
    v27 = a1 + 32;
    if ( v26 > (unsigned __int64)&v32 || (unsigned __int64)&v32 >= v26 + 40 * v14 )
    {
      sub_2DD9530(v27, v18, v14, v17, v13, v6);
      v14 = *(unsigned int *)(a1 + 40);
      v19 = *(_QWORD *)(a1 + 32);
      v20 = &v32;
      v7 = v14;
    }
    else
    {
      sub_2DD9530(v27, v18, v14, v17, v13, v6);
      v19 = *(_QWORD *)(a1 + 32);
      v14 = *(unsigned int *)(a1 + 40);
      v20 = (__m128i *)((char *)&v32 + v19 - v26);
      v7 = v14;
    }
  }
  else
  {
    v19 = *(_QWORD *)(a1 + 32);
    v20 = &v32;
  }
  v21 = 5 * v14;
  v22 = (__m128i *)(v19 + 8 * v21);
  if ( v22 )
  {
    v23 = v20[1].m128i_i64[0];
    v24 = _mm_loadu_si128(v20);
    v22[2].m128i_i64[0] = 0;
    v22[1].m128i_i64[0] = v23;
    v22[1].m128i_i64[1] = (__int64)&v22[2].m128i_i64[1];
    *v22 = v24;
    if ( v20[2].m128i_i32[0] )
      sub_2DD33A0((__int64)&v22[1].m128i_i64[1], (char **)&v20[1].m128i_i64[1], v21, v19, v13, v6);
    v7 = *(unsigned int *)(a1 + 40);
  }
  v25 = v34;
  *(_DWORD *)(a1 + 40) = v7 + 1;
  if ( v25 != v36 )
  {
    _libc_free((unsigned __int64)v25);
    v7 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  }
  *(_DWORD *)(v11 + 24) = v7;
  return *(_QWORD *)(a1 + 32) + 40 * v7 + 24;
}
