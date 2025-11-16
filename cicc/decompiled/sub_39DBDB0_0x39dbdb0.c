// Function: sub_39DBDB0
// Address: 0x39dbdb0
//
void __fastcall sub_39DBDB0(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rbx
  __int64 v5; // r13
  unsigned __int64 v6; // rdx
  __int64 v7; // rax
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int64 v10; // r13
  unsigned __int64 v11; // r14
  unsigned __int64 v12; // r13
  __int64 v13; // r15
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rbx
  __m128i v16; // xmm1
  __int64 v17; // rcx
  __m128i v18; // xmm2
  __int64 v19; // rcx
  __m128i v20; // xmm3
  __int64 v21; // rcx
  __m128i v22; // xmm4
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  __int64 v27; // rcx
  unsigned __int64 v28; // r12
  unsigned __int64 v29; // [rsp+0h] [rbp-60h]
  unsigned __int64 v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+10h] [rbp-50h]
  unsigned __int64 v33; // [rsp+20h] [rbp-40h]
  unsigned __int64 v34; // [rsp+28h] [rbp-38h]

  v33 = a2;
  if ( !a2 )
    return;
  v3 = a1[1];
  v4 = *a1;
  v34 = v3;
  v5 = v3 - *a1;
  v30 = v5 >> 8;
  if ( (__int64)(a1[2] - v3) >> 8 >= a2 )
  {
    v6 = v3;
    do
    {
      if ( v6 )
      {
        memset((void *)v6, 0, 0x100u);
        *(_BYTE *)(v6 + 104) = 1;
        *(_QWORD *)(v6 + 56) = v6 + 72;
        *(_QWORD *)(v6 + 112) = v6 + 128;
        *(_QWORD *)(v6 + 160) = v6 + 176;
        *(_QWORD *)(v6 + 208) = v6 + 224;
      }
      v6 += 256LL;
      --a2;
    }
    while ( a2 );
    a1[1] = (v33 << 8) + v3;
    return;
  }
  if ( 0x7FFFFFFFFFFFFFLL - v30 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v7 = a2;
  if ( v30 >= a2 )
    v7 = v5 >> 8;
  v8 = __CFADD__(v30, v7);
  v9 = v30 + v7;
  if ( v8 )
  {
    v28 = 0x7FFFFFFFFFFFFF00LL;
  }
  else
  {
    if ( !v9 )
    {
      v29 = 0;
      v31 = 0;
      goto LABEL_15;
    }
    if ( v9 > 0x7FFFFFFFFFFFFFLL )
      v9 = 0x7FFFFFFFFFFFFFLL;
    v28 = v9 << 8;
  }
  v31 = sub_22077B0(v28);
  v4 = *a1;
  v29 = v31 + v28;
  v34 = a1[1];
LABEL_15:
  v10 = v31 + v5;
  do
  {
    if ( v10 )
    {
      memset((void *)v10, 0, 0x100u);
      *(_BYTE *)(v10 + 104) = 1;
      *(_QWORD *)(v10 + 56) = v10 + 72;
      *(_QWORD *)(v10 + 112) = v10 + 128;
      *(_QWORD *)(v10 + 160) = v10 + 176;
      *(_QWORD *)(v10 + 208) = v10 + 224;
    }
    v10 += 256;
    --a2;
  }
  while ( a2 );
  if ( v4 != v34 )
  {
    v11 = v4 + 72;
    v12 = v4 + 128;
    v13 = v31;
    v14 = v4 + 176;
    v15 = v4 + 224;
    while ( 1 )
    {
      if ( v13 )
      {
        *(__m128i *)v13 = _mm_loadu_si128((const __m128i *)(v11 - 72));
        *(_QWORD *)(v13 + 16) = *(_QWORD *)(v11 - 56);
        *(_DWORD *)(v13 + 24) = *(_DWORD *)(v11 - 48);
        *(_QWORD *)(v13 + 32) = *(_QWORD *)(v11 - 40);
        *(_QWORD *)(v13 + 40) = *(_QWORD *)(v11 - 32);
        *(_DWORD *)(v13 + 48) = *(_DWORD *)(v11 - 24);
        *(_BYTE *)(v13 + 52) = *(_BYTE *)(v11 - 20);
        *(_BYTE *)(v13 + 53) = *(_BYTE *)(v11 - 19);
        *(_BYTE *)(v13 + 54) = *(_BYTE *)(v11 - 18);
        *(_QWORD *)(v13 + 56) = v13 + 72;
        v27 = *(_QWORD *)(v11 - 16);
        if ( v27 == v11 )
        {
          *(__m128i *)(v13 + 72) = _mm_loadu_si128((const __m128i *)v11);
        }
        else
        {
          *(_QWORD *)(v13 + 56) = v27;
          *(_QWORD *)(v13 + 72) = *(_QWORD *)v11;
        }
        *(_QWORD *)(v13 + 64) = *(_QWORD *)(v11 - 8);
        v16 = _mm_loadu_si128((const __m128i *)(v11 + 16));
        *(_QWORD *)(v11 - 16) = v11;
        *(_QWORD *)(v11 - 8) = 0;
        *(_BYTE *)v11 = 0;
        *(__m128i *)(v13 + 88) = v16;
        *(_BYTE *)(v13 + 104) = *(_BYTE *)(v11 + 32);
        *(_QWORD *)(v13 + 112) = v13 + 128;
        v17 = *(_QWORD *)(v11 + 40);
        if ( v17 == v12 )
        {
          *(__m128i *)(v13 + 128) = _mm_loadu_si128((const __m128i *)(v11 + 56));
        }
        else
        {
          *(_QWORD *)(v13 + 112) = v17;
          *(_QWORD *)(v13 + 128) = *(_QWORD *)(v11 + 56);
        }
        *(_QWORD *)(v13 + 120) = *(_QWORD *)(v11 + 48);
        v18 = _mm_loadu_si128((const __m128i *)(v11 + 72));
        *(_QWORD *)(v11 + 40) = v12;
        *(_QWORD *)(v11 + 48) = 0;
        *(_BYTE *)(v11 + 56) = 0;
        *(_QWORD *)(v13 + 160) = v13 + 176;
        *(__m128i *)(v13 + 144) = v18;
        v19 = *(_QWORD *)(v11 + 88);
        if ( v19 == v14 )
        {
          *(__m128i *)(v13 + 176) = _mm_loadu_si128((const __m128i *)(v11 + 104));
        }
        else
        {
          *(_QWORD *)(v13 + 160) = v19;
          *(_QWORD *)(v13 + 176) = *(_QWORD *)(v11 + 104);
        }
        *(_QWORD *)(v13 + 168) = *(_QWORD *)(v11 + 96);
        v20 = _mm_loadu_si128((const __m128i *)(v11 + 120));
        *(_QWORD *)(v11 + 88) = v14;
        *(_QWORD *)(v11 + 96) = 0;
        *(_BYTE *)(v11 + 104) = 0;
        *(_QWORD *)(v13 + 208) = v13 + 224;
        *(__m128i *)(v13 + 192) = v20;
        v21 = *(_QWORD *)(v11 + 136);
        if ( v21 == v15 )
        {
          *(__m128i *)(v13 + 224) = _mm_loadu_si128((const __m128i *)(v11 + 152));
        }
        else
        {
          *(_QWORD *)(v13 + 208) = v21;
          *(_QWORD *)(v13 + 224) = *(_QWORD *)(v11 + 152);
        }
        *(_QWORD *)(v13 + 216) = *(_QWORD *)(v11 + 144);
        v22 = _mm_loadu_si128((const __m128i *)(v11 + 168));
        *(_QWORD *)(v11 + 136) = v15;
        *(_QWORD *)(v11 + 144) = 0;
        *(_BYTE *)(v11 + 152) = 0;
        *(__m128i *)(v13 + 240) = v22;
      }
      v23 = *(_QWORD *)(v11 + 136);
      if ( v23 != v15 )
        j_j___libc_free_0(v23);
      v24 = *(_QWORD *)(v11 + 88);
      if ( v24 != v14 )
        j_j___libc_free_0(v24);
      v25 = *(_QWORD *)(v11 + 40);
      if ( v25 != v12 )
        j_j___libc_free_0(v25);
      v26 = *(_QWORD *)(v11 - 16);
      if ( v26 != v11 )
        j_j___libc_free_0(v26);
      v13 += 256;
      v12 += 256LL;
      v14 += 256LL;
      v15 += 256LL;
      if ( v34 == v11 + 184 )
        break;
      v11 += 256LL;
    }
    v34 = *a1;
  }
  if ( v34 )
    j_j___libc_free_0(v34);
  *a1 = v31;
  a1[1] = v31 + ((v33 + v30) << 8);
  a1[2] = v29;
}
