// Function: sub_26BCD60
// Address: 0x26bcd60
//
char __fastcall sub_26BCD60(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        const __m128i a7,
        __int128 a8)
{
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r13
  unsigned __int64 v11; // rcx
  __m128i *v13; // rbx
  const __m128i *v14; // r12
  __int64 v15; // r8
  unsigned __int64 v16; // rsi
  bool v17; // cf
  bool v18; // zf
  __m128i v19; // xmm3
  unsigned __int64 v20; // r9
  unsigned __int64 v21; // rdx
  unsigned __int64 v23; // [rsp+8h] [rbp-108h]
  unsigned __int64 v24; // [rsp+8h] [rbp-108h]
  size_t v25; // [rsp+10h] [rbp-100h]
  size_t v26; // [rsp+10h] [rbp-100h]
  __int64 v27; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v28; // [rsp+18h] [rbp-F8h]
  int *v29; // [rsp+28h] [rbp-E8h]
  int *v30; // [rsp+28h] [rbp-E8h]
  _QWORD v31[2]; // [rsp+30h] [rbp-E0h] BYREF
  int v32[52]; // [rsp+40h] [rbp-D0h] BYREF

  LOBYTE(v8) = a2 - 1;
  v9 = a2;
  v10 = (a2 - 1) / 2;
  if ( a2 <= a3 )
  {
    v13 = (__m128i *)(a1 + 32 * a2);
    goto LABEL_12;
  }
  v11 = a8;
  while ( 1 )
  {
    v14 = (const __m128i *)(a1 + 32 * v10);
    if ( v14[1].m128i_i64[0] != v11 )
    {
      LOBYTE(v8) = v14[1].m128i_i64[0] < v11;
      goto LABEL_4;
    }
    v8 = v14->m128i_i64[1];
    v15 = a7.m128i_i64[1];
    if ( !v8 || !a7.m128i_i64[1] )
    {
      LOBYTE(v8) = v8 != 0;
LABEL_4:
      v13 = (__m128i *)(a1 + 32 * v9);
      if ( !(_BYTE)v8 )
        goto LABEL_12;
      goto LABEL_5;
    }
    v16 = *(_QWORD *)(a7.m128i_i64[1] + 112);
    v17 = *(_QWORD *)(v8 + 112) < v16;
    v18 = *(_QWORD *)(v8 + 112) == v16;
    if ( *(_QWORD *)(v8 + 112) == v16 )
    {
      v20 = *(_QWORD *)(v8 + 24);
      v29 = *(int **)(v8 + 16);
      if ( v29 )
      {
        v23 = v11;
        v27 = a7.m128i_i64[1];
        v25 = *(_QWORD *)(v8 + 24);
        sub_C7D030(v32);
        sub_C7D280(v32, v29, v25);
        sub_C7D290(v32, v31);
        v20 = v31[0];
        v11 = v23;
        v15 = v27;
      }
      v21 = *(_QWORD *)(v15 + 24);
      v30 = *(int **)(v15 + 16);
      if ( v30 )
      {
        v24 = v11;
        v28 = v20;
        v26 = *(_QWORD *)(v15 + 24);
        sub_C7D030(v32);
        sub_C7D280(v32, v30, v26);
        sub_C7D290(v32, v31);
        v21 = v31[0];
        v11 = v24;
        v20 = v28;
      }
      v17 = v21 < v20;
      v18 = v21 == v20;
    }
    LOBYTE(v8) = !v17 && !v18;
    v13 = (__m128i *)(a1 + 32 * v9);
    if ( v17 || v18 )
      goto LABEL_12;
LABEL_5:
    *v13 = _mm_loadu_si128(v14);
    v13[1] = _mm_loadu_si128(v14 + 1);
    v8 = (v10 - 1) / 2;
    v9 = v10;
    if ( a3 >= v10 )
      break;
    v10 = (v10 - 1) / 2;
  }
  v13 = (__m128i *)(a1 + 32 * v10);
LABEL_12:
  v19 = _mm_loadu_si128((const __m128i *)&a8);
  *v13 = _mm_loadu_si128(&a7);
  v13[1] = v19;
  return v8;
}
