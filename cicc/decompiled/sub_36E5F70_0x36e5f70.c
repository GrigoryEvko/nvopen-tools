// Function: sub_36E5F70
// Address: 0x36e5f70
//
void __fastcall sub_36E5F70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // eax
  __int64 v8; // rsi
  __int64 v9; // rdx
  _QWORD *v10; // rbx
  __int64 v11; // rax
  _QWORD *v12; // r12
  __int64 v13; // r8
  int v14; // eax
  const __m128i *v15; // rdx
  __m128i *v16; // rax
  __m128i v17; // xmm1
  unsigned __int64 v18; // rcx
  __int64 v19; // rax
  unsigned __int64 v20; // r8
  __m128i v21; // xmm0
  __int64 v22; // rax
  __int64 v23; // rax
  _BOOL4 v24; // esi
  __int64 v25; // r9
  int v26; // esi
  __int64 v27; // r12
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __m128i v31; // xmm0
  const __m128i *v32; // rsi
  __m128i v33; // xmm0
  __m128i v34; // [rsp+0h] [rbp-F0h] BYREF
  unsigned __int64 *v35; // [rsp+18h] [rbp-D8h]
  __int64 v36; // [rsp+20h] [rbp-D0h] BYREF
  int v37; // [rsp+28h] [rbp-C8h]
  unsigned __int64 *v38; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v39; // [rsp+38h] [rbp-B8h]
  _BYTE v40[64]; // [rsp+40h] [rbp-B0h] BYREF
  char v41; // [rsp+80h] [rbp-70h] BYREF

  v7 = *(_DWORD *)(a2 + 64);
  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_QWORD *)(*(_QWORD *)(v8 + 40LL * (unsigned int)(v7 - 1)) + 96LL);
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  v11 = *(_QWORD *)(*(_QWORD *)(v8 + 40LL * (unsigned int)(v7 - 2)) + 96LL);
  v12 = *(_QWORD **)(v11 + 24);
  if ( *(_DWORD *)(v11 + 32) > 0x40u )
    v12 = (_QWORD *)*v12;
  v13 = *(_QWORD *)(a2 + 80);
  v36 = v13;
  if ( v13 )
  {
    sub_B96E90((__int64)&v36, v13, 1);
    v8 = *(_QWORD *)(a2 + 40);
  }
  v14 = *(_DWORD *)(a2 + 72);
  v15 = (const __m128i *)(v8 + 80);
  v39 = 0x800000000LL;
  v37 = v14;
  v16 = (__m128i *)v40;
  v35 = (unsigned __int64 *)v40;
  v38 = (unsigned __int64 *)v40;
  do
  {
    v17 = _mm_loadu_si128(v15);
    ++v16;
    v15 = (const __m128i *)((char *)v15 + 40);
    v16[-1] = v17;
  }
  while ( v16 != (__m128i *)&v41 );
  v18 = HIDWORD(v39);
  LODWORD(v39) = v39 + 4;
  v19 = (unsigned int)v39;
  v20 = (unsigned int)v39 + 1LL;
  if ( v12 == (_QWORD *)1 )
  {
    v33 = _mm_loadu_si128((const __m128i *)(v8 + 240));
    if ( HIDWORD(v39) < v20 )
    {
      v34 = v33;
      sub_C8D5F0((__int64)&v38, v35, (unsigned int)v39 + 1LL, 0x10u, v20, a6);
      v19 = (unsigned int)v39;
      v33 = _mm_load_si128(&v34);
    }
    *(__m128i *)&v38[2 * v19] = v33;
    v8 = *(_QWORD *)(a2 + 40);
    v18 = HIDWORD(v39);
    LODWORD(v39) = v39 + 1;
    v19 = (unsigned int)v39;
    v20 = (unsigned int)v39 + 1LL;
  }
  if ( v10 != (_QWORD *)1 )
  {
    v21 = _mm_loadu_si128((const __m128i *)v8);
    if ( v18 >= v20 )
      goto LABEL_12;
    goto LABEL_25;
  }
  v31 = _mm_loadu_si128((const __m128i *)(v8 + 280));
  if ( v18 < v20 )
  {
    v34 = v31;
    sub_C8D5F0((__int64)&v38, v35, v20, 0x10u, v20, a6);
    v19 = (unsigned int)v39;
    v31 = _mm_load_si128(&v34);
  }
  *(__m128i *)&v38[2 * v19] = v31;
  v32 = *(const __m128i **)(a2 + 40);
  v19 = (unsigned int)(v39 + 1);
  v20 = v19 + 1;
  LODWORD(v39) = v39 + 1;
  v21 = _mm_loadu_si128(v32);
  if ( HIDWORD(v39) < (unsigned __int64)(v19 + 1) )
  {
LABEL_25:
    v34 = v21;
    sub_C8D5F0((__int64)&v38, v35, v20, 0x10u, v20, a6);
    v19 = (unsigned int)v39;
    v21 = _mm_load_si128(&v34);
  }
LABEL_12:
  *(__m128i *)&v38[2 * v19] = v21;
  v22 = *(_QWORD *)(a1 + 64);
  LODWORD(v39) = v39 + 1;
  v23 = sub_2E79000(*(__int64 **)(v22 + 40));
  v24 = sub_AE2980(v23, 3u)[1] == 32;
  if ( v12 == (_QWORD *)1 )
  {
    if ( v10 == (_QWORD *)1 )
      v26 = 4 * v24 + 414;
    else
      v26 = 4 * v24 + 413;
  }
  else if ( v10 == (_QWORD *)1 )
  {
    v26 = 4 * v24 + 412;
  }
  else
  {
    v26 = 4 * v24 + 411;
  }
  v27 = sub_33E66D0(
          *(_QWORD **)(a1 + 64),
          v26,
          (__int64)&v36,
          *(_QWORD *)(a2 + 48),
          *(unsigned int *)(a2 + 68),
          v25,
          v38,
          (unsigned int)v39);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v27, v28, v29, v30);
  sub_3421DB0(v27);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v38 != v35 )
    _libc_free((unsigned __int64)v38);
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
}
