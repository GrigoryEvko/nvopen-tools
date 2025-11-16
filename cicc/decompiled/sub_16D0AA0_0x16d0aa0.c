// Function: sub_16D0AA0
// Address: 0x16d0aa0
//
void __fastcall sub_16D0AA0(
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
        __int64 a15,
        __int64 a16)
{
  int v19; // ecx
  int v20; // eax
  _BYTE *v21; // r10
  _BYTE *v22; // r9
  __int64 v23; // rbx
  __int64 v25; // rdx
  _QWORD *v26; // rax
  _QWORD *v27; // rdx
  __m128i *v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // r15
  __m128i *v31; // r14
  __m128i v32; // xmm0
  __int64 v33; // rbx
  const __m128i *v34; // r12
  unsigned __int64 v35; // rax
  const __m128i *v36; // rbx
  const __m128i *v37; // rdi
  _BYTE *v39; // [rsp+38h] [rbp-38h]

  v19 = a7;
  v20 = a8;
  *(_QWORD *)(a1 + 8) = a3;
  v21 = a9;
  v22 = a11;
  v23 = a15;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  if ( a4 )
  {
    sub_16CD2C0((__int64 *)(a1 + 16), a4, (__int64)&a4[a5]);
    v22 = a11;
    v21 = a9;
    v20 = a8;
    v19 = a7;
  }
  else
  {
    *(_QWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 32) = 0;
  }
  *(_DWORD *)(a1 + 56) = v20;
  *(_DWORD *)(a1 + 52) = v19;
  *(_DWORD *)(a1 + 48) = a6;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  if ( v21 )
  {
    v39 = v22;
    sub_16CD2C0((__int64 *)(a1 + 64), v21, (__int64)&v21[a10]);
    v22 = v39;
  }
  else
  {
    *(_QWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 80) = 0;
  }
  *(_QWORD *)(a1 + 96) = a1 + 112;
  if ( v22 )
  {
    sub_16CD2C0((__int64 *)(a1 + 96), v22, (__int64)&v22[a12]);
  }
  else
  {
    *(_QWORD *)(a1 + 104) = 0;
    *(_BYTE *)(a1 + 112) = 0;
  }
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  v25 = 8 * a14;
  if ( (unsigned __int64)(8 * a14) > 0x7FFFFFFFFFFFFFF8LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v26 = 0;
  if ( v25 )
  {
    v26 = (_QWORD *)sub_22077B0(v25);
    *(_QWORD *)(a1 + 128) = v26;
    v27 = &v26[a14];
    *(_QWORD *)(a1 + 144) = v27;
    do
    {
      if ( v26 )
        *v26 = *a13;
      ++v26;
      ++a13;
    }
    while ( v27 != v26 );
  }
  *(_QWORD *)(a1 + 136) = v26;
  v28 = (__m128i *)(a1 + 168);
  LODWORD(v29) = 0;
  *(_QWORD *)(a1 + 152) = a1 + 168;
  *(_QWORD *)(a1 + 160) = 0x400000000LL;
  v30 = a15 + 48 * a16;
  v31 = (__m128i *)(a1 + 168);
  if ( (unsigned __int64)(48 * a16) > 0xC0 )
  {
    sub_166DC00(a1 + 152, 0xAAAAAAAAAAAAAAABLL * ((48 * a16) >> 4));
    v29 = *(unsigned int *)(a1 + 160);
    v28 = *(__m128i **)(a1 + 152);
    v31 = &v28[3 * v29];
  }
  if ( v30 != a15 )
  {
    do
    {
      if ( v31 )
      {
        v32 = _mm_loadu_si128((const __m128i *)v23);
        v31[1].m128i_i64[0] = (__int64)v31[2].m128i_i64;
        *v31 = v32;
        sub_16CD370(v31[1].m128i_i64, *(_BYTE **)(v23 + 16), *(_QWORD *)(v23 + 16) + *(_QWORD *)(v23 + 24));
      }
      v23 += 48;
      v31 += 3;
    }
    while ( v30 != v23 );
    v28 = *(__m128i **)(a1 + 152);
    LODWORD(v29) = *(_DWORD *)(a1 + 160);
  }
  *(_DWORD *)(a1 + 160) = v29 - 1431655765 * ((48 * a16) >> 4);
  v33 = 3LL * ((unsigned int)v29 - 1431655765 * (unsigned int)((48 * a16) >> 4));
  v34 = &v28[v33];
  if ( &v28[v33] != v28 )
  {
    _BitScanReverse64(&v35, 0xAAAAAAAAAAAAAAABLL * ((v33 * 16) >> 4));
    sub_16CFF80(v28, &v28[v33], 2LL * (int)(63 - (v35 ^ 0x3F)));
    if ( (unsigned __int64)v33 <= 48 )
    {
      sub_16CD7F0((__int64)v28, v34);
    }
    else
    {
      v36 = v28 + 48;
      sub_16CD7F0((__int64)v28, v28 + 48);
      if ( v34 != &v28[48] )
      {
        do
        {
          v37 = v36;
          v36 += 3;
          sub_16CD500(v37);
        }
        while ( v34 != v36 );
      }
    }
  }
}
