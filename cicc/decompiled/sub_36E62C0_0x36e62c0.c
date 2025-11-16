// Function: sub_36E62C0
// Address: 0x36e62c0
//
void __fastcall sub_36E62C0(__int64 a1, __int64 a2)
{
  int v4; // eax
  const __m128i *v5; // rsi
  __int64 v6; // rax
  _QWORD *v7; // r15
  __int64 v8; // r9
  __int64 v9; // rcx
  __int64 v10; // r8
  int v11; // eax
  __m128i *v12; // rdx
  const __m128i *v13; // rax
  const __m128i *v14; // rcx
  __m128i v15; // xmm1
  __int64 v16; // rax
  __m128i v17; // xmm0
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rcx
  _QWORD *v20; // rdi
  __int64 v21; // r15
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __m128i v25; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v26; // [rsp+20h] [rbp-90h] BYREF
  int v27; // [rsp+28h] [rbp-88h]
  unsigned __int64 *v28; // [rsp+30h] [rbp-80h] BYREF
  __int64 v29; // [rsp+38h] [rbp-78h]
  _BYTE v30[112]; // [rsp+40h] [rbp-70h] BYREF

  v4 = *(_DWORD *)(a2 + 64);
  v5 = *(const __m128i **)(a2 + 40);
  v6 = *(_QWORD *)(v5->m128i_i64[5 * (unsigned int)(v4 - 1)] + 96);
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v8 = *(_QWORD *)(a2 + 80);
  v9 = 120;
  v10 = (v7 == (_QWORD *)1) + 2LL;
  v26 = v8;
  if ( v7 != (_QWORD *)1 )
    v9 = 80;
  if ( v8 )
  {
    v25.m128i_i64[0] = v9;
    sub_B96E90((__int64)&v26, v8, 1);
    v5 = *(const __m128i **)(a2 + 40);
    v10 = (v7 == (_QWORD *)1) + 2LL;
    v9 = v25.m128i_i64[0];
  }
  v11 = *(_DWORD *)(a2 + 72);
  v28 = (unsigned __int64 *)v30;
  v12 = (__m128i *)v30;
  v29 = 0x400000000LL;
  v27 = v11;
  v13 = v5 + 5;
  v14 = (const __m128i *)((char *)v5 + v9 + 80);
  do
  {
    v15 = _mm_loadu_si128(v13);
    v13 = (const __m128i *)((char *)v13 + 40);
    ++v12;
    v12[-1] = v15;
  }
  while ( v14 != v13 );
  LODWORD(v29) = v10 + v29;
  v16 = (unsigned int)v29;
  v17 = _mm_loadu_si128(v5);
  v18 = (unsigned int)v29 + 1LL;
  if ( v18 > HIDWORD(v29) )
  {
    v25 = v17;
    sub_C8D5F0((__int64)&v28, v30, v18, 0x10u, v10, v8);
    v16 = (unsigned int)v29;
    v17 = _mm_load_si128(&v25);
  }
  *(__m128i *)&v28[2 * v16] = v17;
  v19 = *(_QWORD *)(a2 + 48);
  v20 = *(_QWORD **)(a1 + 64);
  LODWORD(v29) = v29 + 1;
  v21 = sub_33E66D0(
          v20,
          (unsigned int)(v7 == (_QWORD *)1) + 427,
          (__int64)&v26,
          v19,
          *(unsigned int *)(a2 + 68),
          (unsigned int)v29,
          v28,
          (unsigned int)v29);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v21, v22, v23, v24);
  sub_3421DB0(v21);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v28 != (unsigned __int64 *)v30 )
    _libc_free((unsigned __int64)v28);
  if ( v26 )
    sub_B91220((__int64)&v26, v26);
}
