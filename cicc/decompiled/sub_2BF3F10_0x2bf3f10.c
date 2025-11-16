// Function: sub_2BF3F10
// Address: 0x2bf3f10
//
__int64 __fastcall sub_2BF3F10(_QWORD *a1)
{
  _BYTE *v1; // rsi
  _BYTE *v2; // rdi
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  __int64 v7; // r9
  __int64 v8; // rcx
  __int64 v9; // r8
  unsigned __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdi
  __m128i *v13; // rdx
  const __m128i *v14; // rax
  __int64 v15; // r9
  const __m128i *v16; // rcx
  __int64 v17; // r8
  unsigned __int64 v18; // rbx
  __int64 v19; // rax
  __m128i *v20; // rdi
  __m128i *v21; // rdx
  const __m128i *v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rbx
  char v27; // al
  _BYTE v28[32]; // [rsp+0h] [rbp-220h] BYREF
  _BYTE v29[64]; // [rsp+20h] [rbp-200h] BYREF
  __int64 v30; // [rsp+60h] [rbp-1C0h]
  __int64 v31; // [rsp+68h] [rbp-1B8h]
  unsigned __int64 v32; // [rsp+70h] [rbp-1B0h]
  _QWORD v33[4]; // [rsp+80h] [rbp-1A0h] BYREF
  _BYTE v34[64]; // [rsp+A0h] [rbp-180h] BYREF
  __m128i *v35; // [rsp+E0h] [rbp-140h]
  unsigned __int64 v36; // [rsp+E8h] [rbp-138h]
  __int8 *v37; // [rsp+F0h] [rbp-130h]
  _QWORD v38[12]; // [rsp+100h] [rbp-120h] BYREF
  __int64 v39; // [rsp+160h] [rbp-C0h]
  __int64 v40; // [rsp+168h] [rbp-B8h]
  _BYTE v41[96]; // [rsp+178h] [rbp-A8h] BYREF
  __int64 v42; // [rsp+1D8h] [rbp-48h]
  const __m128i *v43; // [rsp+1E0h] [rbp-40h]

  v33[0] = *a1;
  sub_2BF3840(v38, v33);
  v1 = v29;
  v2 = v28;
  sub_C8CD80((__int64)v28, (__int64)v29, (__int64)v38, v3, v4, v5);
  v8 = v40;
  v9 = v39;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v10 = v40 - v39;
  if ( v40 == v39 )
  {
    v12 = 0;
  }
  else
  {
    if ( v10 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_39;
    v11 = sub_22077B0(v40 - v39);
    v8 = v40;
    v9 = v39;
    v12 = v11;
  }
  v30 = v12;
  v31 = v12;
  v32 = v12 + v10;
  if ( v9 != v8 )
  {
    v13 = (__m128i *)v12;
    v14 = (const __m128i *)v9;
    do
    {
      if ( v13 )
      {
        *v13 = _mm_loadu_si128(v14);
        v13[1].m128i_i64[0] = v14[1].m128i_i64[0];
      }
      v14 = (const __m128i *)((char *)v14 + 24);
      v13 = (__m128i *)((char *)v13 + 24);
    }
    while ( v14 != (const __m128i *)v8 );
    v12 += 8 * (((unsigned __int64)&v14[-2].m128i_u64[1] - v9) >> 3) + 24;
  }
  v31 = v12;
  v1 = v34;
  v2 = v33;
  sub_C8CD80((__int64)v33, (__int64)v34, (__int64)v41, v8, v9, v7);
  v16 = v43;
  v17 = v42;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v18 = (unsigned __int64)v43 - v42;
  if ( v43 == (const __m128i *)v42 )
  {
    v20 = 0;
    goto LABEL_13;
  }
  if ( v18 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_39:
    sub_4261EA(v2, v1, v6);
  v19 = sub_22077B0((unsigned __int64)v43 - v42);
  v16 = v43;
  v17 = v42;
  v20 = (__m128i *)v19;
LABEL_13:
  v35 = v20;
  v36 = (unsigned __int64)v20;
  v37 = &v20->m128i_i8[v18];
  if ( (const __m128i *)v17 == v16 )
  {
    v23 = (unsigned __int64)v20;
  }
  else
  {
    v21 = v20;
    v22 = (const __m128i *)v17;
    do
    {
      if ( v21 )
      {
        *v21 = _mm_loadu_si128(v22);
        v21[1].m128i_i64[0] = v22[1].m128i_i64[0];
      }
      v22 = (const __m128i *)((char *)v22 + 24);
      v21 = (__m128i *)((char *)v21 + 24);
    }
    while ( v22 != v16 );
    v23 = (unsigned __int64)&v20[1].m128i_u64[(((unsigned __int64)&v22[-2].m128i_u64[1] - v17) >> 3) + 1];
  }
  v36 = v23;
  while ( 1 )
  {
    v24 = v30;
    if ( v31 - v30 != v23 - (_QWORD)v20 )
      break;
    while ( 1 )
    {
      if ( v31 == v24 )
      {
        v25 = 0;
        sub_2AB1B10((__int64)v33);
        sub_2AB1B10((__int64)v28);
        sub_2AB1B10((__int64)v41);
        sub_2AB1B10((__int64)v38);
        return v25;
      }
      if ( *(_QWORD *)v24 != v20->m128i_i64[0] )
        break;
      v27 = *(_BYTE *)(v24 + 16);
      if ( v27 != v20[1].m128i_i8[0] || v27 && *(_QWORD *)(v24 + 8) != v20->m128i_i64[1] )
        goto LABEL_21;
      v24 += 24;
      v20 = (__m128i *)((char *)v20 + 24);
    }
    v25 = *(_QWORD *)(v31 - 24);
    if ( !*(_BYTE *)(v25 + 8) )
      goto LABEL_22;
LABEL_32:
    sub_2ADA290((__int64)v28, v31 - v30, v31, v24, v17, v15);
    v23 = v36;
    v20 = v35;
  }
LABEL_21:
  v25 = *(_QWORD *)(v31 - 24);
  if ( *(_BYTE *)(v25 + 8) )
    goto LABEL_32;
LABEL_22:
  if ( *(_BYTE *)(v25 + 128) )
    v25 = 0;
  sub_2AB1B10((__int64)v33);
  sub_2AB1B10((__int64)v28);
  sub_2AB1B10((__int64)v41);
  sub_2AB1B10((__int64)v38);
  return v25;
}
