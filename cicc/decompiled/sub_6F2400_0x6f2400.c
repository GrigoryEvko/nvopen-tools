// Function: sub_6F2400
// Address: 0x6f2400
//
const __m128i *__fastcall sub_6F2400(__int64 a1, __int64 a2, unsigned int a3, __int32 a4, _DWORD *a5, _DWORD *a6)
{
  __int64 v8; // r13
  char i; // al
  __int64 v12; // rbx
  unsigned int v13; // r8d
  const __m128i *v14; // rcx
  __int64 v15; // rax
  __m128i v16; // xmm1
  __int64 v17; // rax
  _QWORD *v18; // rcx
  __int64 v19; // rbx
  __int64 v20; // rcx
  int v21; // r8d
  const __m128i *result; // rax
  __m128i *v23; // rcx
  __int8 v24; // al
  char v25; // al
  __m128i *v26; // rax
  int v27; // r8d
  __int64 v28; // r13
  __m128i *v29; // rcx
  char v30; // al
  __int64 v31; // r13
  unsigned int v33; // [rsp+Ch] [rbp-64h]
  __int64 v34; // [rsp+18h] [rbp-58h] BYREF
  __m128i v35; // [rsp+20h] [rbp-50h] BYREF
  __int64 v36; // [rsp+30h] [rbp-40h]

  v8 = a1;
  for ( i = *(_BYTE *)(a1 + 24); i == 32; i = *(_BYTE *)(v8 + 24) )
  {
    v12 = *(_QWORD *)(a2 + 16);
    v13 = a3;
    v35 = 0;
    a3 = v12;
    if ( v12 == *(_QWORD *)(a2 + 8) )
    {
      v33 = v13;
      sub_6F2340((const __m128i **)a2);
      v13 = v33;
    }
    v14 = *(const __m128i **)a2;
    v15 = *(_QWORD *)a2 + 24 * v12;
    if ( v15 )
    {
      v35.m128i_i32[1] = a4;
      v35.m128i_i32[0] = 4 * v13 + 1;
      v16 = _mm_loadu_si128(&v35);
      v36 = v8;
      *(_QWORD *)(v15 + 16) = v8;
      *(__m128i *)v15 = v16;
      v14 = *(const __m128i **)a2;
    }
    *(_QWORD *)(a2 + 16) = v12 + 1;
    v17 = 24LL * (int)v12;
    v14->m128i_i64[(unsigned __int64)v17 / 8 + 1] = 0;
    if ( v13 == -1 )
    {
      *a6 = 0;
      v18 = *(_QWORD **)(v8 + 64);
      *(_QWORD *)(*(_QWORD *)a2 + v17 + 8) = v18;
      if ( (*(_BYTE *)(v8 + 27) & 4) != 0 )
        *(_QWORD *)(v8 + 64) = *v18;
    }
    else
    {
      *(_QWORD *)(*(_QWORD *)a2 + v17 + 8) = 0;
    }
    v8 = *(_QWORD *)(*(_QWORD *)(v8 + 56) + 192LL);
  }
  v19 = *(_QWORD *)(a2 + 16);
  v20 = *(_QWORD *)(a2 + 8);
  v21 = a3 & 0x3FFFFFFF;
  if ( i != 1 )
  {
LABEL_13:
    v35 = 0;
    if ( v20 == v19 )
    {
      sub_6F2340((const __m128i **)a2);
      v21 = a3 & 0x3FFFFFFF;
    }
    result = *(const __m128i **)a2;
    v23 = (__m128i *)(*(_QWORD *)a2 + 24 * v19);
    if ( v23 )
    {
      v24 = v35.m128i_i8[0];
      v35.m128i_i32[1] = a4;
      v23[1].m128i_i64[0] = v8;
      result = (const __m128i *)((4 * v21) | v24 & 3u);
      v35.m128i_i32[0] = (int)result;
      *v23 = _mm_loadu_si128(&v35);
    }
    *(_QWORD *)(a2 + 16) = v19 + 1;
    if ( a3 != -1 )
      return result;
LABEL_27:
    *a5 = 1;
    return (const __m128i *)a5;
  }
  v25 = *(_BYTE *)(v8 + 56);
  if ( v25 == 87 )
  {
    v34 = 0x100000001LL;
    v35 = 0;
    if ( v20 == v19 )
    {
      sub_6F2340((const __m128i **)a2);
      v21 = a3 & 0x3FFFFFFF;
    }
    v29 = (__m128i *)(*(_QWORD *)a2 + 24 * v19);
    if ( v29 )
    {
      v35.m128i_i32[1] = a4;
      v36 = v8;
      v30 = v35.m128i_i8[0] & 0xFC;
      v29[1].m128i_i64[0] = v8;
      v35.m128i_i32[0] = (4 * v21) | v30 & 1 | 2;
      *v29 = _mm_loadu_si128(&v35);
    }
    *(_QWORD *)(a2 + 16) = v19 + 1;
    v31 = *(_QWORD *)(v8 + 72);
    sub_6F2400(v31, a2, a3, (unsigned int)v19, a5, &v34);
    *(_DWORD *)(*(_QWORD *)a2 + 24LL * (int)v19) = (4 * *(_DWORD *)(a2 + 16))
                                                 | *(_DWORD *)(*(_QWORD *)a2 + 24LL * (int)v19) & 3;
    sub_6F2400(*(_QWORD *)(v31 + 16), a2, a3, (unsigned int)v19, a5, (char *)&v34 + 4);
    result = (const __m128i *)(HIDWORD(v34) | (unsigned int)v34);
    if ( !v34 )
      *a6 = 0;
  }
  else
  {
    if ( v25 != 88 )
      goto LABEL_13;
    v34 = 0;
    v35 = 0;
    if ( v20 == v19 )
    {
      sub_6F2340((const __m128i **)a2);
      v21 = a3 & 0x3FFFFFFF;
    }
    v26 = (__m128i *)(*(_QWORD *)a2 + 24 * v19);
    if ( v26 )
    {
      v35.m128i_i8[0] |= 3u;
      v35.m128i_i32[1] = a4;
      v36 = v8;
      v27 = (4 * v21) | v35.m128i_i8[0] & 3;
      v26[1].m128i_i64[0] = v8;
      v35.m128i_i32[0] = v27;
      *v26 = _mm_loadu_si128(&v35);
    }
    *(_QWORD *)(a2 + 16) = v19 + 1;
    v28 = *(_QWORD *)(v8 + 72);
    sub_6F2400(v28, a2, a3, (unsigned int)v19, &v34, a6);
    *(_DWORD *)(*(_QWORD *)a2 + 24LL * (int)v19) = (4 * *(_DWORD *)(a2 + 16))
                                                 | *(_DWORD *)(*(_QWORD *)a2 + 24LL * (int)v19) & 3;
    result = (const __m128i *)sub_6F2400(*(_QWORD *)(v28 + 16), a2, a3, (unsigned int)v19, (char *)&v34 + 4, a6);
    if ( (_DWORD)v34 )
    {
      result = (const __m128i *)HIDWORD(v34);
      if ( HIDWORD(v34) )
        goto LABEL_27;
    }
  }
  return result;
}
