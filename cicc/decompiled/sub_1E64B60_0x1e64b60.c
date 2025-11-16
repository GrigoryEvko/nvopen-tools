// Function: sub_1E64B60
// Address: 0x1e64b60
//
__int64 __fastcall sub_1E64B60(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // r15
  __int64 v5; // rax
  bool v7; // zf
  __int64 v8; // r9
  __int64 v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // rax
  bool v12; // cf
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 result; // rax
  __int64 v16; // r13
  __int64 v17; // r8
  __int64 v18; // rdx
  char v19; // si
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rax
  char v23; // si
  __int64 v24; // r12
  char v25; // dl
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-48h]
  __int64 v29; // [rsp+10h] [rbp-40h]
  __int64 v30; // [rsp+18h] [rbp-38h]
  __int64 v31; // [rsp+18h] [rbp-38h]

  v3 = a1[1];
  v4 = *a1;
  v5 = (v3 - *a1) >> 5;
  if ( v5 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = v5 == 0;
  v8 = a2;
  v9 = a2;
  v10 = (a1[1] - *a1) >> 5;
  v11 = 1;
  if ( !v7 )
    v11 = (a1[1] - *a1) >> 5;
  v12 = __CFADD__(v10, v11);
  v13 = v10 + v11;
  v14 = v8 - v4;
  if ( v12 )
  {
    v26 = 0x7FFFFFFFFFFFFFE0LL;
LABEL_26:
    v28 = a3;
    v29 = v8;
    v31 = v8 - v4;
    v27 = sub_22077B0(v26);
    v14 = v31;
    v8 = v29;
    v17 = v27;
    v16 = v27 + v26;
    a3 = v28;
    result = v27 + 32;
    goto LABEL_7;
  }
  if ( v13 )
  {
    if ( v13 > 0x3FFFFFFFFFFFFFFLL )
      v13 = 0x3FFFFFFFFFFFFFFLL;
    v26 = 32 * v13;
    goto LABEL_26;
  }
  result = 32;
  v16 = 0;
  v17 = 0;
LABEL_7:
  v18 = v17 + v14;
  if ( v18 )
  {
    *(_QWORD *)v18 = *(_QWORD *)a3;
    v19 = *(_BYTE *)(a3 + 24);
    *(_BYTE *)(v18 + 24) = v19;
    if ( v19 )
      *(__m128i *)(v18 + 8) = _mm_loadu_si128((const __m128i *)(a3 + 8));
  }
  if ( v8 != v4 )
  {
    v20 = v17;
    v21 = v4;
    v22 = v17 + v8 - v4;
    do
    {
      if ( v20 )
      {
        *(_QWORD *)v20 = *(_QWORD *)v21;
        v23 = *(_BYTE *)(v21 + 24);
        *(_BYTE *)(v20 + 24) = v23;
        if ( v23 )
          *(__m128i *)(v20 + 8) = _mm_loadu_si128((const __m128i *)(v21 + 8));
      }
      v20 += 32;
      v21 += 32;
    }
    while ( v22 != v20 );
    result = v22 + 32;
  }
  if ( v8 == v3 )
  {
    v24 = result;
  }
  else
  {
    v24 = result + v3 - v8;
    do
    {
      *(_QWORD *)result = *(_QWORD *)v9;
      v25 = *(_BYTE *)(v9 + 24);
      *(_BYTE *)(result + 24) = v25;
      if ( v25 )
        *(__m128i *)(result + 8) = _mm_loadu_si128((const __m128i *)(v9 + 8));
      result += 32;
      v9 += 32;
    }
    while ( result != v24 );
  }
  if ( v4 )
  {
    v30 = v17;
    result = j_j___libc_free_0(v4, a1[2] - v4);
    v17 = v30;
  }
  *a1 = v17;
  a1[1] = v24;
  a1[2] = v16;
  return result;
}
