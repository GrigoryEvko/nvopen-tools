// Function: sub_16FBDF0
// Address: 0x16fbdf0
//
__int64 __fastcall sub_16FBDF0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 *v6; // r13
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  signed __int64 v10; // rcx
  _QWORD *v11; // rcx
  unsigned __int64 *v12; // rbx
  unsigned __int64 *v13; // rax
  unsigned __int64 v14; // rcx
  unsigned __int64 *v15; // rdi
  __int64 v16; // rax
  __m128i v17; // xmm1
  __m128i *v18; // rdx
  __int64 v19; // rdx
  unsigned __int64 v20; // rdx
  __int128 v22; // [rsp+0h] [rbp-60h] BYREF
  __int64 v23; // [rsp+10h] [rbp-50h]
  __m128i *v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+20h] [rbp-40h]
  __m128i v26[3]; // [rsp+28h] [rbp-38h] BYREF

  v6 = (unsigned __int64 *)(*(_QWORD *)(a1 + 184) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (unsigned __int64 *)(a1 + 184) == v6 )
    goto LABEL_11;
  do
  {
    sub_16F7A50((__int64 *)a1, a2, a3, a4, a5);
    v7 = *(_QWORD **)(a1 + 232);
    v8 = *(_QWORD *)(a1 + 192);
    v9 = 3LL * *(unsigned int *)(a1 + 240);
    a2 = (unsigned __int64)&v7[v9];
    v10 = 0xAAAAAAAAAAAAAAABLL * ((v9 * 8) >> 3);
    if ( v10 >> 2 )
    {
      v11 = &v7[12 * (v10 >> 2)];
      while ( v8 != *v7 )
      {
        if ( v8 == v7[3] )
        {
          v7 += 3;
          goto LABEL_9;
        }
        if ( v8 == v7[6] )
        {
          v7 += 6;
          goto LABEL_9;
        }
        if ( v8 == v7[9] )
        {
          v7 += 9;
          goto LABEL_9;
        }
        v7 += 12;
        if ( v11 == v7 )
        {
          v10 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(a2 - (_QWORD)v7) >> 3);
          goto LABEL_22;
        }
      }
      goto LABEL_9;
    }
LABEL_22:
    if ( v10 == 2 )
      goto LABEL_26;
    if ( v10 == 3 )
    {
      if ( v8 == *v7 )
        goto LABEL_9;
      v7 += 3;
LABEL_26:
      if ( v8 == *v7 )
        goto LABEL_9;
      v7 += 3;
      goto LABEL_28;
    }
    if ( v10 != 1 )
      return v8 + 16;
LABEL_28:
    if ( v8 != *v7 )
      return v8 + 16;
LABEL_9:
    if ( (_QWORD *)a2 == v7 )
      return v8 + 16;
    v6 = (unsigned __int64 *)(a1 + 184);
LABEL_11:
    ;
  }
  while ( (unsigned __int8)sub_16FB8C0((_QWORD *)a1, a2) );
  v12 = *(unsigned __int64 **)(a1 + 192);
  while ( v12 != v6 )
  {
    while ( 1 )
    {
      v13 = v12;
      v12 = (unsigned __int64 *)v12[1];
      v14 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
      *v12 = v14 | *v12 & 7;
      *(_QWORD *)(v14 + 8) = v12;
      v15 = (unsigned __int64 *)v13[5];
      *v13 &= 7u;
      v13[1] = 0;
      if ( v15 == v13 + 7 )
        break;
      j_j___libc_free_0(v15, v13[7] + 1);
      if ( v12 == v6 )
        goto LABEL_16;
    }
  }
LABEL_16:
  v23 = 0;
  v24 = v26;
  v25 = 0;
  v26[0].m128i_i8[0] = 0;
  v22 = 0;
  v16 = sub_145CBF0((__int64 *)(a1 + 80), 72, 16);
  v17 = _mm_loadu_si128((const __m128i *)((char *)&v22 + 8));
  *(_QWORD *)v16 = 0;
  *(_DWORD *)(v16 + 16) = v22;
  *(_QWORD *)(v16 + 40) = v16 + 56;
  v18 = v24;
  *(_QWORD *)(v16 + 8) = 0;
  *(__m128i *)(v16 + 24) = v17;
  if ( v18 == v26 )
  {
    *(__m128i *)(v16 + 56) = _mm_loadu_si128(v26);
  }
  else
  {
    *(_QWORD *)(v16 + 40) = v18;
    *(_QWORD *)(v16 + 56) = v26[0].m128i_i64[0];
  }
  v19 = v25;
  v24 = v26;
  v25 = 0;
  *(_QWORD *)(v16 + 48) = v19;
  v26[0].m128i_i8[0] = 0;
  v20 = *v6;
  *(_QWORD *)(v16 + 8) = v6;
  v20 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v16 = v20;
  *(_QWORD *)(v20 + 8) = v16;
  *v6 = *v6 & 7 | v16;
  if ( v24 != v26 )
    j_j___libc_free_0(v24, v26[0].m128i_i64[0] + 1);
  return *(_QWORD *)(a1 + 192) + 16LL;
}
