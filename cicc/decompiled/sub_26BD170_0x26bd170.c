// Function: sub_26BD170
// Address: 0x26bd170
//
char __fastcall sub_26BD170(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128i a4,
        __int64 a5,
        _QWORD *a6,
        __int64 a7,
        __int128 a8)
{
  __int64 v8; // r10
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 i; // r12
  bool v12; // dl
  const __m128i *v13; // rbx
  __int64 v14; // r13
  __m128i *v15; // r12
  __int64 v16; // rax
  __int64 v17; // r14
  unsigned __int64 v18; // rsi
  _QWORD *v19; // rdx
  unsigned __int64 v20; // rsi
  bool v21; // cf
  bool v22; // zf
  unsigned __int64 v24; // r11
  unsigned __int64 v25; // rdx
  __int64 v26; // rdx
  __m128i *v27; // r13
  const __m128i *v28; // rax
  __int64 v29; // [rsp+0h] [rbp-120h]
  __int64 v30; // [rsp+0h] [rbp-120h]
  __int64 v31; // [rsp+8h] [rbp-118h]
  __int64 v32; // [rsp+8h] [rbp-118h]
  __int64 v33; // [rsp+10h] [rbp-110h]
  __int64 v34; // [rsp+10h] [rbp-110h]
  size_t v35; // [rsp+18h] [rbp-108h]
  size_t v36; // [rsp+18h] [rbp-108h]
  _QWORD *v37; // [rsp+20h] [rbp-100h]
  unsigned __int64 v38; // [rsp+20h] [rbp-100h]
  __int64 v39; // [rsp+28h] [rbp-F8h]
  int *v40; // [rsp+38h] [rbp-E8h]
  int *v41; // [rsp+38h] [rbp-E8h]
  _QWORD v42[2]; // [rsp+40h] [rbp-E0h] BYREF
  int v43[52]; // [rsp+50h] [rbp-D0h] BYREF

  v8 = a2;
  v9 = a3;
  v10 = (a3 - 1) / 2;
  if ( a2 >= v10 )
  {
    v14 = a2;
  }
  else
  {
    for ( i = a2; ; i = v14 )
    {
      v14 = 2 * (i + 1);
      v16 = v14 - 1;
      v13 = (const __m128i *)(a1 + ((i + 1) << 6));
      v17 = a1 + 32 * (v14 - 1);
      v18 = *(_QWORD *)(v17 + 16);
      if ( v13[1].m128i_i64[0] == v18 )
      {
        v19 = (_QWORD *)v13->m128i_i64[1];
        a6 = *(_QWORD **)(v17 + 8);
        if ( v19 && a6 )
        {
          v20 = a6[14];
          v21 = v19[14] < v20;
          v22 = v19[14] == v20;
          if ( v19[14] == v20 )
          {
            v24 = v19[3];
            v40 = (int *)v19[2];
            if ( v40 )
            {
              v29 = v10;
              v31 = v9;
              v33 = v8;
              v37 = *(_QWORD **)(v17 + 8);
              v35 = v19[3];
              sub_C7D030(v43);
              sub_C7D280(v43, v40, v35);
              sub_C7D290(v43, v42);
              v24 = v42[0];
              v10 = v29;
              v9 = v31;
              v8 = v33;
              a6 = v37;
              v16 = v14 - 1;
            }
            v25 = a6[3];
            v41 = (int *)a6[2];
            if ( v41 )
            {
              v30 = v10;
              v32 = v9;
              v34 = v8;
              v38 = v24;
              v39 = v16;
              v36 = a6[3];
              sub_C7D030(v43);
              sub_C7D280(v43, v41, v36);
              sub_C7D290(v43, v42);
              v25 = v42[0];
              v10 = v30;
              v9 = v32;
              v8 = v34;
              v24 = v38;
              v16 = v39;
            }
            v21 = v25 < v24;
            v22 = v25 == v24;
          }
          v12 = !v21 && !v22;
        }
        else
        {
          v12 = v19 != 0;
        }
      }
      else
      {
        v12 = v13[1].m128i_i64[0] < v18;
      }
      if ( v12 )
      {
        v13 = (const __m128i *)(a1 + 32 * (v14 - 1));
        v14 = v16;
      }
      v15 = (__m128i *)(a1 + 32 * i);
      a4 = _mm_loadu_si128(v13);
      *v15 = a4;
      v15[1] = _mm_loadu_si128(v13 + 1);
      if ( v14 >= v10 )
        break;
    }
  }
  if ( (v9 & 1) == 0 )
  {
    v9 = (v9 - 2) / 2;
    if ( v9 == v14 )
    {
      v26 = 2 * v14 + 1;
      v27 = (__m128i *)(a1 + 32 * v14);
      v28 = (const __m128i *)(a1 + 32 * v26);
      *v27 = _mm_loadu_si128(v28);
      v27[1] = _mm_loadu_si128(v28 + 1);
      v14 = v26;
    }
  }
  return sub_26BCD60(a1, v14, v8, v10, (__int64)a6, v9, a4, a8);
}
