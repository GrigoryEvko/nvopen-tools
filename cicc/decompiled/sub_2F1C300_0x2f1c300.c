// Function: sub_2F1C300
// Address: 0x2f1c300
//
void __fastcall sub_2F1C300(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // r12
  unsigned __int64 v4; // rdx
  __int64 v5; // rax
  bool v6; // cf
  unsigned __int64 v7; // rax
  __int64 v8; // r12
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // r13
  __int64 v11; // r15
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rbx
  __m128i v14; // xmm1
  __int64 v15; // rcx
  __m128i v16; // xmm2
  __int64 v17; // rcx
  __m128i v18; // xmm3
  __int64 v19; // rcx
  __m128i v20; // xmm4
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  __int64 v25; // rcx
  unsigned __int64 v26; // r13
  unsigned __int64 v27; // [rsp+0h] [rbp-60h]
  unsigned __int64 v28; // [rsp+8h] [rbp-58h]
  __int64 v29; // [rsp+10h] [rbp-50h]
  unsigned __int64 v31; // [rsp+20h] [rbp-40h]
  unsigned __int64 v32; // [rsp+28h] [rbp-38h]

  v31 = a2;
  if ( !a2 )
    return;
  v2 = *a1;
  v32 = a1[1];
  v3 = v32 - *a1;
  v28 = 0xF83E0F83E0F83E1LL * (v3 >> 3);
  if ( a2 <= 0xF83E0F83E0F83E1LL * ((__int64)(a1[2] - v32) >> 3) )
  {
    v4 = a1[1];
    do
    {
      if ( v4 )
      {
        memset((void *)v4, 0, 0x108u);
        *(_BYTE *)(v4 + 112) = 1;
        *(_QWORD *)(v4 + 64) = v4 + 80;
        *(_QWORD *)(v4 + 120) = v4 + 136;
        *(_QWORD *)(v4 + 168) = v4 + 184;
        *(_QWORD *)(v4 + 216) = v4 + 232;
      }
      v4 += 264LL;
      --a2;
    }
    while ( a2 );
    a1[1] = v32 + 264 * v31;
    return;
  }
  if ( 0x7C1F07C1F07C1FLL - v28 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v5 = 0xF83E0F83E0F83E1LL * ((__int64)(v32 - *a1) >> 3);
  if ( a2 >= v28 )
    v5 = a2;
  v6 = __CFADD__(v28, v5);
  v7 = v28 + v5;
  if ( v6 )
  {
    v26 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v7 )
    {
      v27 = 0;
      v29 = 0;
      goto LABEL_15;
    }
    if ( v7 > 0x7C1F07C1F07C1FLL )
      v7 = 0x7C1F07C1F07C1FLL;
    v26 = 264 * v7;
  }
  v29 = sub_22077B0(v26);
  v2 = *a1;
  v27 = v29 + v26;
  v32 = a1[1];
LABEL_15:
  v8 = v29 + v3;
  do
  {
    if ( v8 )
    {
      memset((void *)v8, 0, 0x108u);
      *(_BYTE *)(v8 + 112) = 1;
      *(_QWORD *)(v8 + 64) = v8 + 80;
      *(_QWORD *)(v8 + 120) = v8 + 136;
      *(_QWORD *)(v8 + 168) = v8 + 184;
      *(_QWORD *)(v8 + 216) = v8 + 232;
    }
    v8 += 264;
    --a2;
  }
  while ( a2 );
  if ( v2 != v32 )
  {
    v9 = v2 + 80;
    v10 = v2 + 136;
    v11 = v29;
    v12 = v2 + 184;
    v13 = v2 + 232;
    while ( 1 )
    {
      if ( v11 )
      {
        *(__m128i *)v11 = _mm_loadu_si128((const __m128i *)(v9 - 80));
        *(_QWORD *)(v11 + 16) = *(_QWORD *)(v9 - 64);
        *(_DWORD *)(v11 + 24) = *(_DWORD *)(v9 - 56);
        *(_QWORD *)(v11 + 32) = *(_QWORD *)(v9 - 48);
        *(_QWORD *)(v11 + 40) = *(_QWORD *)(v9 - 40);
        *(_WORD *)(v11 + 48) = *(_WORD *)(v9 - 32);
        *(_DWORD *)(v11 + 52) = *(_DWORD *)(v9 - 28);
        *(_BYTE *)(v11 + 56) = *(_BYTE *)(v9 - 24);
        *(_BYTE *)(v11 + 57) = *(_BYTE *)(v9 - 23);
        *(_QWORD *)(v11 + 64) = v11 + 80;
        v25 = *(_QWORD *)(v9 - 16);
        if ( v25 == v9 )
        {
          *(__m128i *)(v11 + 80) = _mm_loadu_si128((const __m128i *)v9);
        }
        else
        {
          *(_QWORD *)(v11 + 64) = v25;
          *(_QWORD *)(v11 + 80) = *(_QWORD *)v9;
        }
        *(_QWORD *)(v11 + 72) = *(_QWORD *)(v9 - 8);
        v14 = _mm_loadu_si128((const __m128i *)(v9 + 16));
        *(_QWORD *)(v9 - 16) = v9;
        *(_QWORD *)(v9 - 8) = 0;
        *(_BYTE *)v9 = 0;
        *(__m128i *)(v11 + 96) = v14;
        *(_BYTE *)(v11 + 112) = *(_BYTE *)(v9 + 32);
        *(_QWORD *)(v11 + 120) = v11 + 136;
        v15 = *(_QWORD *)(v9 + 40);
        if ( v15 == v10 )
        {
          *(__m128i *)(v11 + 136) = _mm_loadu_si128((const __m128i *)(v9 + 56));
        }
        else
        {
          *(_QWORD *)(v11 + 120) = v15;
          *(_QWORD *)(v11 + 136) = *(_QWORD *)(v9 + 56);
        }
        *(_QWORD *)(v11 + 128) = *(_QWORD *)(v9 + 48);
        v16 = _mm_loadu_si128((const __m128i *)(v9 + 72));
        *(_QWORD *)(v9 + 40) = v10;
        *(_QWORD *)(v9 + 48) = 0;
        *(_BYTE *)(v9 + 56) = 0;
        *(_QWORD *)(v11 + 168) = v11 + 184;
        *(__m128i *)(v11 + 152) = v16;
        v17 = *(_QWORD *)(v9 + 88);
        if ( v17 == v12 )
        {
          *(__m128i *)(v11 + 184) = _mm_loadu_si128((const __m128i *)(v9 + 104));
        }
        else
        {
          *(_QWORD *)(v11 + 168) = v17;
          *(_QWORD *)(v11 + 184) = *(_QWORD *)(v9 + 104);
        }
        *(_QWORD *)(v11 + 176) = *(_QWORD *)(v9 + 96);
        v18 = _mm_loadu_si128((const __m128i *)(v9 + 120));
        *(_QWORD *)(v9 + 88) = v12;
        *(_QWORD *)(v9 + 96) = 0;
        *(_BYTE *)(v9 + 104) = 0;
        *(_QWORD *)(v11 + 216) = v11 + 232;
        *(__m128i *)(v11 + 200) = v18;
        v19 = *(_QWORD *)(v9 + 136);
        if ( v19 == v13 )
        {
          *(__m128i *)(v11 + 232) = _mm_loadu_si128((const __m128i *)(v9 + 152));
        }
        else
        {
          *(_QWORD *)(v11 + 216) = v19;
          *(_QWORD *)(v11 + 232) = *(_QWORD *)(v9 + 152);
        }
        *(_QWORD *)(v11 + 224) = *(_QWORD *)(v9 + 144);
        v20 = _mm_loadu_si128((const __m128i *)(v9 + 168));
        *(_QWORD *)(v9 + 136) = v13;
        *(_QWORD *)(v9 + 144) = 0;
        *(_BYTE *)(v9 + 152) = 0;
        *(__m128i *)(v11 + 248) = v20;
      }
      v21 = *(_QWORD *)(v9 + 136);
      if ( v21 != v13 )
        j_j___libc_free_0(v21);
      v22 = *(_QWORD *)(v9 + 88);
      if ( v22 != v12 )
        j_j___libc_free_0(v22);
      v23 = *(_QWORD *)(v9 + 40);
      if ( v23 != v10 )
        j_j___libc_free_0(v23);
      v24 = *(_QWORD *)(v9 - 16);
      if ( v24 != v9 )
        j_j___libc_free_0(v24);
      v11 += 264;
      v10 += 264LL;
      v12 += 264LL;
      v13 += 264LL;
      if ( v32 == v9 + 184 )
        break;
      v9 += 264LL;
    }
    v32 = *a1;
  }
  if ( v32 )
    j_j___libc_free_0(v32);
  *a1 = v29;
  a1[1] = v29 + 264 * (v28 + v31);
  a1[2] = v27;
}
