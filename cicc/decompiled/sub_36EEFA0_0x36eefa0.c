// Function: sub_36EEFA0
// Address: 0x36eefa0
//
void __fastcall sub_36EEFA0(__int64 a1, __int64 a2, char a3)
{
  const __m128i *v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // rsi
  _BOOL8 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rsi
  const __m128i *v13; // rbx
  int v14; // ecx
  __int64 v15; // rsi
  __int64 v16; // r8
  unsigned __int64 v17; // rax
  __m128i *v18; // rdx
  const __m128i *v19; // rdx
  __int64 v20; // rax
  __m128i v21; // xmm0
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  _DWORD *v25; // rax
  int v26; // eax
  __int64 v27; // r9
  __int64 v28; // r12
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __m128i v32; // [rsp+10h] [rbp-100h] BYREF
  unsigned __int64 v33; // [rsp+20h] [rbp-F0h]
  unsigned __int64 *v34; // [rsp+28h] [rbp-E8h]
  char v35; // [rsp+37h] [rbp-D9h]
  __int64 *v36; // [rsp+38h] [rbp-D8h]
  __int64 v37; // [rsp+40h] [rbp-D0h] BYREF
  int v38; // [rsp+48h] [rbp-C8h]
  unsigned __int64 *v39; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v40; // [rsp+58h] [rbp-B8h]
  _BYTE v41[176]; // [rsp+60h] [rbp-B0h] BYREF

  v5 = *(const __m128i **)(a2 + 40);
  v6 = *(unsigned int *)(a2 + 64) - 6LL;
  v7 = *(_QWORD *)(v5->m128i_i64[5 * (unsigned int)(*(_DWORD *)(a2 + 64) - 1)] + 96);
  v8 = *(_QWORD **)(v7 + 24);
  if ( *(_DWORD *)(v7 + 32) > 0x40u )
    v8 = (_QWORD *)*v8;
  v9 = *(_QWORD *)(a2 + 80);
  v36 = &v37;
  v10 = v8 == (_QWORD *)1;
  v37 = v9;
  v35 = v10;
  v11 = v10 + v6 + 2;
  if ( v9 )
  {
    v34 = (unsigned __int64 *)(v10 + v6 + 2);
    sub_B96E90((__int64)&v37, v9, 1);
    v5 = *(const __m128i **)(a2 + 40);
    v11 = (__int64)v34;
  }
  v12 = 5 * v11;
  v13 = v5 + 5;
  v14 = 0;
  v15 = v12;
  v38 = *(_DWORD *)(a2 + 72);
  v16 = (__int64)&v13->m128i_i64[v15];
  v40 = 0x800000000LL;
  v17 = 0xCCCCCCCCCCCCCCCDLL * ((v15 * 8) >> 3);
  v39 = (unsigned __int64 *)v41;
  v18 = (__m128i *)v41;
  if ( (unsigned __int64)v15 > 40 )
  {
    v33 = 0xCCCCCCCCCCCCCCCDLL * ((v15 * 8) >> 3);
    v32.m128i_i64[0] = (__int64)v41;
    v34 = (unsigned __int64 *)&v39;
    sub_C8D5F0((__int64)&v39, v41, v33, 0x10u, v16, (__int64)v41);
    v14 = v40;
    v16 = (__int64)&v13->m128i_i64[v15];
    LODWORD(v17) = -858993459 * ((v15 * 8) >> 3);
    v18 = (__m128i *)&v39[2 * (unsigned int)v40];
  }
  if ( v13 != (const __m128i *)v16 )
  {
    do
    {
      if ( v18 )
        *v18 = _mm_loadu_si128(v13);
      v13 = (const __m128i *)((char *)v13 + 40);
      ++v18;
    }
    while ( (const __m128i *)v16 != v13 );
    v14 = v40;
  }
  v19 = *(const __m128i **)(a2 + 40);
  LODWORD(v40) = v14 + v17;
  v20 = (unsigned int)(v14 + v17);
  v21 = _mm_loadu_si128(v19);
  v22 = (unsigned int)v20 + 1LL;
  if ( v22 > HIDWORD(v40) )
  {
    v34 = (unsigned __int64 *)v41;
    v32 = v21;
    sub_C8D5F0((__int64)&v39, v41, v22, 0x10u, v16, (__int64)v41);
    v20 = (unsigned int)v40;
    v21 = _mm_load_si128(&v32);
  }
  v34 = (unsigned __int64 *)v41;
  *(__m128i *)&v39[2 * v20] = v21;
  v23 = *(_QWORD *)(a1 + 64);
  LODWORD(v40) = v40 + 1;
  v24 = sub_2E79000(*(__int64 **)(v23 + 40));
  v25 = sub_AE2980(v24, 3u);
  v26 = sub_36D70A0(v6, v25[1] == 32, v35, a3, 0);
  v28 = sub_33E66D0(
          *(_QWORD **)(a1 + 64),
          v26,
          (__int64)v36,
          *(_QWORD *)(a2 + 48),
          *(unsigned int *)(a2 + 68),
          v27,
          v39,
          (unsigned int)v40);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v28, v29, v30, v31);
  sub_3421DB0(v28);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v39 != v34 )
    _libc_free((unsigned __int64)v39);
  if ( v37 )
    sub_B91220((__int64)v36, v37);
}
