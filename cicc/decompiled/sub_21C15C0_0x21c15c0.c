// Function: sub_21C15C0
// Address: 0x21c15c0
//
__int64 __fastcall sub_21C15C0(__int64 a1, __int64 a2)
{
  unsigned __int16 v2; // ax
  int v4; // ecx
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 v8; // rbx
  const __m128i *v9; // r15
  __int64 v10; // rdx
  const __m128i *v11; // rbx
  unsigned __int64 v12; // r8
  __m128i *v13; // rax
  const __m128i *v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rsi
  _QWORD *v17; // r15
  __int64 *v18; // r10
  unsigned int v19; // eax
  __int64 v20; // rcx
  int v21; // r8d
  __int64 v22; // r11
  __int64 v23; // r15
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // [rsp-100h] [rbp-100h]
  unsigned __int64 v29; // [rsp-F8h] [rbp-F8h]
  unsigned __int64 v30; // [rsp-F8h] [rbp-F8h]
  __int64 v31; // [rsp-F0h] [rbp-F0h]
  int v32; // [rsp-E8h] [rbp-E8h]
  unsigned int v33; // [rsp-E8h] [rbp-E8h]
  unsigned int v34; // [rsp-E0h] [rbp-E0h]
  unsigned int v35; // [rsp-E0h] [rbp-E0h]
  __int64 v36; // [rsp-D8h] [rbp-D8h] BYREF
  int v37; // [rsp-D0h] [rbp-D0h]
  __int64 *v38; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v39; // [rsp-C0h] [rbp-C0h]
  _BYTE v40[184]; // [rsp-B8h] [rbp-B8h] BYREF

  v2 = *(_WORD *)(a2 + 24) - 684;
  if ( v2 > 0xADu )
    return 0;
  v4 = 0;
  v6 = word_433D800[v2];
  v7 = *(unsigned int *)(a2 + 56);
  v38 = (__int64 *)v40;
  v8 = *(_QWORD *)(a2 + 32);
  v39 = 0x800000000LL;
  v9 = (const __m128i *)(v8 + 40 * v7);
  v10 = 40 * v7 - 40;
  v11 = (const __m128i *)(v8 + 40);
  v12 = 0xCCCCCCCCCCCCCCCDLL * (v10 >> 3);
  v13 = (__m128i *)v40;
  if ( (unsigned __int64)v10 > 0x140 )
  {
    v33 = v6;
    v30 = 0xCCCCCCCCCCCCCCCDLL * (v10 >> 3);
    sub_16CD150((__int64)&v38, v40, v30, 16, v12, v6);
    v4 = v39;
    v6 = v33;
    LODWORD(v12) = v30;
    v13 = (__m128i *)&v38[2 * (unsigned int)v39];
  }
  if ( v11 != v9 )
  {
    do
    {
      if ( v13 )
        *v13 = _mm_loadu_si128(v11);
      v11 = (const __m128i *)((char *)v11 + 40);
      ++v13;
    }
    while ( v9 != v11 );
    v4 = v39;
  }
  v14 = *(const __m128i **)(a2 + 32);
  LODWORD(v39) = v12 + v4;
  v15 = (unsigned int)(v12 + v4);
  if ( HIDWORD(v39) <= (unsigned int)(v12 + v4) )
  {
    v35 = v6;
    sub_16CD150((__int64)&v38, v40, 0, 16, v12, v6);
    v15 = (unsigned int)v39;
    v6 = v35;
  }
  *(__m128i *)&v38[2 * v15] = _mm_loadu_si128(v14);
  v16 = *(_QWORD *)(a2 + 72);
  v17 = *(_QWORD **)(a1 + 272);
  v18 = v38;
  v19 = v39 + 1;
  v20 = *(_QWORD *)(a2 + 40);
  v21 = *(_DWORD *)(a2 + 60);
  v36 = v16;
  LODWORD(v39) = v19;
  v22 = v19;
  if ( v16 )
  {
    v28 = v20;
    v32 = v21;
    v29 = (unsigned __int64)v38;
    v31 = v19;
    v34 = v6;
    sub_1623A60((__int64)&v36, v16, 2);
    v20 = v28;
    v21 = v32;
    v18 = (__int64 *)v29;
    v22 = v31;
    v6 = v34;
  }
  v37 = *(_DWORD *)(a2 + 64);
  v23 = sub_1D23DE0(v17, v6, (__int64)&v36, v20, v21, v6, v18, v22);
  sub_1D444E0(*(_QWORD *)(a1 + 272), a2, v23);
  sub_1D49010(v23);
  sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v24, v25, v26, v27);
  if ( v36 )
    sub_161E7C0((__int64)&v36, v36);
  if ( v38 != (__int64 *)v40 )
    _libc_free((unsigned __int64)v38);
  return 1;
}
