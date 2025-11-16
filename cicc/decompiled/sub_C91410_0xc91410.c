// Function: sub_C91410
// Address: 0xc91410
//
void __fastcall sub_C91410(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _BYTE *a4,
        __int64 a5,
        int a6,
        int a7,
        int a8,
        _BYTE *a9,
        __int64 a10,
        _BYTE *a11,
        __int64 a12,
        _QWORD *a13,
        __int64 a14,
        const __m128i *a15,
        __int64 a16)
{
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r15
  _QWORD *v22; // rax
  _QWORD *v23; // rdx
  __m128i *v24; // rdi
  const __m128i *v25; // r15
  __m128i *v26; // r13
  const __m128i *v27; // r14
  __int64 v28; // rdx
  __m128i v29; // xmm0
  _BYTE *m128i_i8; // rdi
  _BYTE *v31; // r8
  size_t v32; // r12
  __int64 v33; // rax
  __int64 v34; // rbx
  const __m128i *v35; // r12
  unsigned __int64 v36; // rax
  const __m128i *v37; // rbx
  const __m128i *v38; // rdi
  _BYTE *v39; // [rsp+28h] [rbp-48h]
  size_t v40[7]; // [rsp+38h] [rbp-38h] BYREF

  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  sub_C8D8C0((__int64 *)(a1 + 16), a4, (__int64)&a4[a5]);
  *(_DWORD *)(a1 + 48) = a6;
  *(_DWORD *)(a1 + 56) = a8;
  *(_DWORD *)(a1 + 52) = a7;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  sub_C8D8C0((__int64 *)(a1 + 64), a9, (__int64)&a9[a10]);
  *(_QWORD *)(a1 + 96) = a1 + 112;
  sub_C8D8C0((__int64 *)(a1 + 96), a11, (__int64)&a11[a12]);
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  v21 = a14;
  *(_QWORD *)(a1 + 144) = 0;
  if ( (unsigned __int64)(8 * a14) > 0x7FFFFFFFFFFFFFF8LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v22 = 0;
  if ( v21 * 8 )
  {
    v22 = (_QWORD *)sub_22077B0(8 * a14);
    v23 = &v22[v21];
    *(_QWORD *)(a1 + 128) = v22;
    *(_QWORD *)(a1 + 144) = &v22[v21];
    do
    {
      if ( v22 )
        *v22 = *a13;
      ++v22;
      ++a13;
    }
    while ( v23 != v22 );
  }
  *(_QWORD *)(a1 + 136) = v22;
  v24 = (__m128i *)(a1 + 168);
  v25 = a15;
  *(_QWORD *)(a1 + 160) = 0x400000000LL;
  v26 = (__m128i *)(a1 + 168);
  *(_QWORD *)(a1 + 152) = a1 + 168;
  v27 = &a15[3 * a16];
  LODWORD(v28) = 0;
  if ( (unsigned __int64)(48 * a16) > 0xC0 )
  {
    sub_C8F9C0(
      a1 + 152,
      0xAAAAAAAAAAAAAAABLL * ((48 * a16) >> 4),
      0,
      0xAAAAAAAAAAAAAAABLL * ((48 * a16) >> 4),
      v19,
      v20);
    v28 = *(unsigned int *)(a1 + 160);
    v24 = *(__m128i **)(a1 + 152);
    v26 = &v24[3 * v28];
  }
  if ( v27 != a15 )
  {
    while ( 1 )
    {
      if ( !v26 )
        goto LABEL_13;
      v29 = _mm_loadu_si128(v25);
      m128i_i8 = v26[2].m128i_i8;
      v26[1].m128i_i64[0] = (__int64)v26[2].m128i_i64;
      *v26 = v29;
      v31 = (_BYTE *)v25[1].m128i_i64[0];
      v32 = v25[1].m128i_u64[1];
      if ( &v31[v32] && !v31 )
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v40[0] = v25[1].m128i_u64[1];
      if ( v32 > 0xF )
        break;
      if ( v32 == 1 )
      {
        v26[2].m128i_i8[0] = *v31;
      }
      else if ( v32 )
      {
        goto LABEL_22;
      }
LABEL_12:
      v26[1].m128i_i64[1] = v32;
      m128i_i8[v32] = 0;
LABEL_13:
      v25 += 3;
      v26 += 3;
      if ( v27 == v25 )
      {
        v24 = *(__m128i **)(a1 + 152);
        LODWORD(v28) = *(_DWORD *)(a1 + 160);
        goto LABEL_24;
      }
    }
    v39 = v31;
    v33 = sub_22409D0(&v26[1], v40, 0);
    v31 = v39;
    v26[1].m128i_i64[0] = v33;
    m128i_i8 = (_BYTE *)v33;
    v26[2].m128i_i64[0] = v40[0];
LABEL_22:
    memcpy(m128i_i8, v31, v32);
    v32 = v40[0];
    m128i_i8 = (_BYTE *)v26[1].m128i_i64[0];
    goto LABEL_12;
  }
LABEL_24:
  *(_DWORD *)(a1 + 160) = v28 - 1431655765 * ((48 * a16) >> 4);
  v34 = 3LL * ((unsigned int)v28 - 1431655765 * (unsigned int)((48 * a16) >> 4));
  v35 = &v24[v34];
  if ( &v24[v34] != v24 )
  {
    _BitScanReverse64(&v36, 0xAAAAAAAAAAAAAAABLL * ((v34 * 16) >> 4));
    sub_C908F0(v24, &v24[v34], 2LL * (int)(63 - (v36 ^ 0x3F)));
    if ( (unsigned __int64)v34 <= 48 )
    {
      sub_C8DE50((__int64)v24, v35);
    }
    else
    {
      v37 = v24 + 48;
      sub_C8DE50((__int64)v24, v24 + 48);
      if ( v35 != &v24[48] )
      {
        do
        {
          v38 = v37;
          v37 += 3;
          sub_C8DB60(v38);
        }
        while ( v35 != v37 );
      }
    }
  }
}
