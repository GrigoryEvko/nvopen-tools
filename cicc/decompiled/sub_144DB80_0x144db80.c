// Function: sub_144DB80
// Address: 0x144db80
//
unsigned __int64 __fastcall sub_144DB80(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // r15
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r9
  __int64 v12; // r8
  __int64 v13; // r14
  unsigned __int64 result; // rax
  char v15; // dl
  __int64 v16; // rdx
  __int64 v17; // rax
  char v18; // cl
  __int64 v19; // rdx
  __int64 v20; // rax
  char v21; // cl
  __m128i v22; // xmm1
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-48h]
  __int64 v27; // [rsp+10h] [rbp-40h]
  __int64 v28; // [rsp+18h] [rbp-38h]
  __int64 v29; // [rsp+18h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * ((v4 - *a1) >> 3);
  if ( v6 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xCCCCCCCCCCCCCCCDLL * ((a1[1] - *a1) >> 3);
  v8 = __CFADD__(v7, v6);
  v9 = v7 - 0x3333333333333333LL * ((a1[1] - *a1) >> 3);
  v10 = a2 - v5;
  if ( v8 )
  {
    v24 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_27:
    v26 = a3;
    v29 = v24;
    v25 = sub_22077B0(v24);
    v10 = a2 - v5;
    a3 = v26;
    v13 = v25;
    v11 = v25 + 40;
    v12 = v25 + v29;
    goto LABEL_7;
  }
  if ( v9 )
  {
    if ( v9 > 0x333333333333333LL )
      v9 = 0x333333333333333LL;
    v24 = 40 * v9;
    goto LABEL_27;
  }
  v11 = 40;
  v12 = 0;
  v13 = 0;
LABEL_7:
  result = v13 + v10;
  if ( v13 + v10 )
  {
    *(_QWORD *)result = *(_QWORD *)a3;
    v15 = *(_BYTE *)(a3 + 32);
    *(_BYTE *)(result + 32) = v15;
    if ( v15 )
    {
      v23 = *(_QWORD *)(a3 + 24);
      *(__m128i *)(result + 8) = _mm_loadu_si128((const __m128i *)(a3 + 8));
      *(_QWORD *)(result + 24) = v23;
    }
  }
  if ( a2 != v5 )
  {
    v16 = v13;
    v17 = v5;
    do
    {
      if ( v16 )
      {
        *(_QWORD *)v16 = *(_QWORD *)v17;
        v18 = *(_BYTE *)(v17 + 32);
        *(_BYTE *)(v16 + 32) = v18;
        if ( v18 )
        {
          *(__m128i *)(v16 + 8) = _mm_loadu_si128((const __m128i *)(v17 + 8));
          *(_QWORD *)(v16 + 24) = *(_QWORD *)(v17 + 24);
        }
      }
      v17 += 40;
      v16 += 40;
    }
    while ( a2 != v17 );
    result = (unsigned __int64)(a2 - 40 - v5) >> 3;
    v11 = v13 + 8 * result + 80;
  }
  if ( a2 != v4 )
  {
    v19 = v11;
    v20 = a2;
    do
    {
      *(_QWORD *)v19 = *(_QWORD *)v20;
      v21 = *(_BYTE *)(v20 + 32);
      *(_BYTE *)(v19 + 32) = v21;
      if ( v21 )
      {
        v22 = _mm_loadu_si128((const __m128i *)(v20 + 8));
        *(_QWORD *)(v19 + 24) = *(_QWORD *)(v20 + 24);
        *(__m128i *)(v19 + 8) = v22;
      }
      v20 += 40;
      v19 += 40;
    }
    while ( v4 != v20 );
    result = (unsigned __int64)(v4 - a2 - 40) >> 3;
    v11 += 8 * result + 40;
  }
  if ( v5 )
  {
    v27 = v12;
    v28 = v11;
    result = j_j___libc_free_0(v5, a1[2] - v5);
    v12 = v27;
    v11 = v28;
  }
  *a1 = v13;
  a1[1] = v11;
  a1[2] = v12;
  return result;
}
