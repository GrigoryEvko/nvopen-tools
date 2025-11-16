// Function: sub_36E5D40
// Address: 0x36e5d40
//
void __fastcall sub_36E5D40(__int64 a1, __int64 a2)
{
  int v3; // eax
  const __m128i *v4; // rsi
  __int64 v5; // rax
  _QWORD *v6; // r15
  __int64 v7; // r9
  __int64 v8; // rcx
  __int64 v9; // r8
  int v10; // eax
  const __m128i *v11; // rax
  const __m128i *v12; // rcx
  __m128i *v13; // rdx
  __m128i v14; // xmm1
  __int64 v15; // rax
  __m128i v16; // xmm0
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  _BOOL4 v20; // esi
  __int64 v21; // r9
  int v22; // esi
  __int64 v23; // r15
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __m128i v27; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v28; // [rsp+20h] [rbp-D0h] BYREF
  int v29; // [rsp+28h] [rbp-C8h]
  unsigned __int64 *v30; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v31; // [rsp+38h] [rbp-B8h]
  _BYTE v32[176]; // [rsp+40h] [rbp-B0h] BYREF

  v3 = *(_DWORD *)(a2 + 64);
  v4 = *(const __m128i **)(a2 + 40);
  v5 = *(_QWORD *)(v4->m128i_i64[5 * (unsigned int)(v3 - 1)] + 96);
  v6 = *(_QWORD **)(v5 + 24);
  if ( *(_DWORD *)(v5 + 32) > 0x40u )
    v6 = (_QWORD *)*v6;
  v7 = *(_QWORD *)(a2 + 80);
  v8 = 160;
  v9 = (v6 == (_QWORD *)1) + 3LL;
  v28 = v7;
  if ( v6 != (_QWORD *)1 )
    v8 = 120;
  if ( v7 )
  {
    v27.m128i_i64[0] = v8;
    sub_B96E90((__int64)&v28, v7, 1);
    v4 = *(const __m128i **)(a2 + 40);
    v9 = (v6 == (_QWORD *)1) + 3LL;
    v8 = v27.m128i_i64[0];
  }
  v10 = *(_DWORD *)(a2 + 72);
  v30 = (unsigned __int64 *)v32;
  v29 = v10;
  v11 = v4 + 5;
  v31 = 0x800000000LL;
  v12 = (const __m128i *)((char *)v4 + v8 + 80);
  v13 = (__m128i *)v32;
  do
  {
    v14 = _mm_loadu_si128(v11);
    v11 = (const __m128i *)((char *)v11 + 40);
    ++v13;
    v13[-1] = v14;
  }
  while ( v12 != v11 );
  LODWORD(v31) = v9 + v31;
  v15 = (unsigned int)v31;
  v16 = _mm_loadu_si128(v4);
  v17 = (unsigned int)v31 + 1LL;
  if ( v17 > HIDWORD(v31) )
  {
    v27 = v16;
    sub_C8D5F0((__int64)&v30, v32, v17, 0x10u, v9, v7);
    v15 = (unsigned int)v31;
    v16 = _mm_load_si128(&v27);
  }
  *(__m128i *)&v30[2 * v15] = v16;
  v18 = *(_QWORD *)(a1 + 64);
  LODWORD(v31) = v31 + 1;
  v19 = sub_2E79000(*(__int64 **)(v18 + 40));
  v20 = sub_AE2980(v19, 3u)[1] == 32;
  if ( v6 == (_QWORD *)1 )
    v22 = 2 * v20 + 430;
  else
    v22 = 2 * v20 + 429;
  v23 = sub_33E66D0(
          *(_QWORD **)(a1 + 64),
          v22,
          (__int64)&v28,
          *(_QWORD *)(a2 + 48),
          *(unsigned int *)(a2 + 68),
          v21,
          v30,
          (unsigned int)v31);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v23, v24, v25, v26);
  sub_3421DB0(v23);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v30 != (unsigned __int64 *)v32 )
    _libc_free((unsigned __int64)v30);
  if ( v28 )
    sub_B91220((__int64)&v28, v28);
}
