// Function: sub_3422FA0
// Address: 0x3422fa0
//
void __fastcall sub_3422FA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // rsi
  __m128i v10; // xmm4
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __m128i v13; // xmm0
  __int64 *v14; // r15
  __m128i v15; // xmm0
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rdi
  unsigned __int64 v22; // rax
  int v23; // edx
  __int64 v24; // r9
  __m128i v25; // [rsp+0h] [rbp-2A0h] BYREF
  __m128i v26; // [rsp+10h] [rbp-290h] BYREF
  __m128i v27; // [rsp+20h] [rbp-280h] BYREF
  unsigned __int64 *v28; // [rsp+38h] [rbp-268h]
  __int64 v29; // [rsp+40h] [rbp-260h] BYREF
  int v30; // [rsp+48h] [rbp-258h]
  __int64 v31; // [rsp+50h] [rbp-250h] BYREF
  int v32; // [rsp+58h] [rbp-248h]
  unsigned __int64 *v33; // [rsp+60h] [rbp-240h] BYREF
  __int64 v34; // [rsp+68h] [rbp-238h]
  _BYTE v35[560]; // [rsp+70h] [rbp-230h] BYREF

  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_QWORD *)(a2 + 80);
  v28 = (unsigned __int64 *)v35;
  v33 = (unsigned __int64 *)v35;
  v34 = 0x2000000000LL;
  v29 = v9;
  if ( v9 )
  {
    sub_B96E90((__int64)&v29, v9, 1);
    v10 = _mm_loadu_si128((const __m128i *)(v8 + 40));
    v11 = (unsigned int)v34;
    v30 = *(_DWORD *)(a2 + 72);
    v12 = (unsigned int)v34 + 1LL;
    v13 = _mm_loadu_si128((const __m128i *)(v8 + 80));
    v26 = v10;
    v27 = _mm_loadu_si128((const __m128i *)v8);
    if ( v12 > HIDWORD(v34) )
    {
      v25 = v13;
      sub_C8D5F0((__int64)&v33, v28, v12, 0x10u, a5, a6);
      v11 = (unsigned int)v34;
      v13 = _mm_load_si128(&v25);
    }
  }
  else
  {
    v13 = _mm_loadu_si128((const __m128i *)(v8 + 80));
    v30 = *(_DWORD *)(a2 + 72);
    v11 = 0;
    v26 = _mm_loadu_si128((const __m128i *)(v8 + 40));
    v27 = _mm_loadu_si128((const __m128i *)v8);
  }
  v14 = (__int64 *)(v8 + 160);
  *(__m128i *)&v33[2 * v11] = v13;
  v15 = _mm_loadu_si128((const __m128i *)(v8 + 120));
  LODWORD(v34) = v34 + 1;
  v16 = (unsigned int)v34;
  if ( (unsigned __int64)(unsigned int)v34 + 1 > HIDWORD(v34) )
  {
    v25 = v15;
    sub_C8D5F0((__int64)&v33, v28, (unsigned int)v34 + 1LL, 0x10u, a5, a6);
    v16 = (unsigned int)v34;
    v15 = _mm_load_si128(&v25);
  }
  *(__m128i *)&v33[2 * v16] = v15;
  v17 = 5LL * *(unsigned int *)(a2 + 64);
  v18 = *(_QWORD *)(a2 + 40);
  v19 = (unsigned int)(v34 + 1);
  LODWORD(v34) = v34 + 1;
  if ( v14 != (__int64 *)(v18 + 8 * v17) )
  {
    do
    {
      v31 = v29;
      if ( v29 )
        sub_B96E90((__int64)&v31, v29, 1);
      v32 = v30;
      sub_3422DF0(a1, (__int64)&v33, *v14, v14[1], (__int64)&v31, a6, v15);
      if ( v31 )
        sub_B91220((__int64)&v31, v31);
      v14 += 5;
    }
    while ( v14 != (__int64 *)(*(_QWORD *)(a2 + 40) + 40LL * *(unsigned int *)(a2 + 64)) );
    v19 = (unsigned int)v34;
  }
  if ( v19 + 1 > (unsigned __int64)HIDWORD(v34) )
  {
    sub_C8D5F0((__int64)&v33, v28, v19 + 1, 0x10u, a5, a6);
    v19 = (unsigned int)v34;
  }
  *(__m128i *)&v33[2 * v19] = _mm_load_si128(&v27);
  LODWORD(v34) = v34 + 1;
  v20 = (unsigned int)v34;
  if ( (unsigned __int64)(unsigned int)v34 + 1 > HIDWORD(v34) )
  {
    sub_C8D5F0((__int64)&v33, v28, (unsigned int)v34 + 1LL, 0x10u, a5, a6);
    v20 = (unsigned int)v34;
  }
  *(__m128i *)&v33[2 * v20] = _mm_load_si128(&v26);
  v21 = *(__int64 **)(a1 + 64);
  LODWORD(v34) = v34 + 1;
  v22 = sub_33E5110(v21, 1, 0, 262, 0);
  sub_3415C10(*(const __m128i **)(a1 + 64), a2, 26, v22, v23, v24, v33, (unsigned int)v34);
  if ( v29 )
    sub_B91220((__int64)&v29, v29);
  if ( v33 != v28 )
    _libc_free((unsigned __int64)v33);
}
