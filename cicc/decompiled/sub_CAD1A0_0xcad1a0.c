// Function: sub_CAD1A0
// Address: 0xcad1a0
//
__int64 __fastcall sub_CAD1A0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 *v6; // rbx
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  signed __int64 v10; // rcx
  _QWORD *v11; // rcx
  unsigned __int64 *v12; // r13
  unsigned __int64 *v13; // rax
  unsigned __int64 v14; // rcx
  unsigned __int64 *v15; // rdi
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __m128i v18; // xmm1
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rdx
  __m128i *v22; // rdi
  __int128 v24; // [rsp+0h] [rbp-60h] BYREF
  __int64 v25; // [rsp+10h] [rbp-50h]
  __m128i *v26; // [rsp+18h] [rbp-48h]
  __int64 v27; // [rsp+20h] [rbp-40h]
  __m128i v28[3]; // [rsp+28h] [rbp-38h] BYREF

  v6 = (unsigned __int64 *)(a1 + 176);
  if ( a1 + 176 == (*(_QWORD *)(a1 + 176) & 0xFFFFFFFFFFFFFFF8LL) )
    goto LABEL_10;
  do
  {
    sub_CA81D0((__int64 *)a1, a2, a3, a4, a5);
    v7 = *(_QWORD **)(a1 + 224);
    v8 = *(_QWORD *)(a1 + 184);
    v9 = 3LL * *(unsigned int *)(a1 + 232);
    a2 = (unsigned __int64)&v7[v9];
    v10 = 0xAAAAAAAAAAAAAAABLL * ((v9 * 8) >> 3);
    if ( v10 >> 2 )
    {
      v11 = &v7[12 * (v10 >> 2)];
      while ( v8 != *v7 )
      {
        if ( v8 == v7[3] )
        {
          if ( (_QWORD *)a2 != v7 + 3 )
            goto LABEL_10;
          return v8 + 16;
        }
        if ( v8 == v7[6] )
        {
          if ( (_QWORD *)a2 != v7 + 6 )
            goto LABEL_10;
          return v8 + 16;
        }
        if ( v8 == v7[9] )
        {
          if ( (_QWORD *)a2 != v7 + 9 )
            goto LABEL_10;
          return v8 + 16;
        }
        v7 += 12;
        if ( v11 == v7 )
        {
          v10 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(a2 - (_QWORD)v7) >> 3);
          goto LABEL_24;
        }
      }
LABEL_9:
      if ( (_QWORD *)a2 == v7 )
        return v8 + 16;
      continue;
    }
LABEL_24:
    if ( v10 != 2 )
    {
      if ( v10 != 3 )
      {
        if ( v10 != 1 )
          return v8 + 16;
        goto LABEL_30;
      }
      if ( v8 == *v7 )
        goto LABEL_9;
      v7 += 3;
    }
    if ( v8 == *v7 )
      goto LABEL_9;
    v7 += 3;
LABEL_30:
    if ( v8 != *v7 || (_QWORD *)a2 == v7 )
      return v8 + 16;
LABEL_10:
    ;
  }
  while ( (unsigned __int8)sub_CACCB0(a1, a2) );
  v12 = *(unsigned __int64 **)(a1 + 184);
  while ( v6 != v12 )
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
      if ( v6 == v12 )
        goto LABEL_15;
    }
  }
LABEL_15:
  v16 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 160) += 72LL;
  *(_DWORD *)(a1 + 232) = 0;
  v25 = 0;
  v17 = (v16 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  v26 = v28;
  v27 = 0;
  v28[0].m128i_i8[0] = 0;
  v24 = 0;
  if ( *(_QWORD *)(a1 + 88) >= v17 + 72 && v16 )
  {
    *(_QWORD *)(a1 + 80) = v17 + 72;
    if ( !v17 )
    {
      MEMORY[8] = v6;
      BUG();
    }
  }
  else
  {
    v17 = sub_9D1E70(a1 + 80, 72, 72, 4);
  }
  *(_QWORD *)v17 = 0;
  *(_QWORD *)(v17 + 8) = 0;
  *(_DWORD *)(v17 + 16) = v24;
  v18 = _mm_loadu_si128((const __m128i *)((char *)&v24 + 8));
  *(_QWORD *)(v17 + 40) = v17 + 56;
  *(__m128i *)(v17 + 24) = v18;
  if ( v26 == v28 )
  {
    *(__m128i *)(v17 + 56) = _mm_loadu_si128(v28);
  }
  else
  {
    *(_QWORD *)(v17 + 40) = v26;
    *(_QWORD *)(v17 + 56) = v28[0].m128i_i64[0];
  }
  v19 = v27;
  v26 = v28;
  v27 = 0;
  *(_QWORD *)(v17 + 48) = v19;
  v20 = *(_QWORD *)(a1 + 176);
  v28[0].m128i_i8[0] = 0;
  v21 = *(_QWORD *)v17;
  v20 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(v17 + 8) = v6;
  *(_QWORD *)v17 = v20 | v21 & 7;
  *(_QWORD *)(v20 + 8) = v17;
  v22 = v26;
  *(_QWORD *)(a1 + 176) = *(_QWORD *)(a1 + 176) & 7LL | v17;
  if ( v22 != v28 )
    j_j___libc_free_0(v22, v28[0].m128i_i64[0] + 1);
  return *(_QWORD *)(a1 + 184) + 16LL;
}
