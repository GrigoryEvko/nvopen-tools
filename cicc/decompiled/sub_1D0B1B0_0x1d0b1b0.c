// Function: sub_1D0B1B0
// Address: 0x1d0b1b0
//
void __fastcall sub_1D0B1B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdx
  const __m128i *v10; // rbx
  int v11; // ecx
  unsigned __int64 v12; // rsi
  const __m128i *v13; // r12
  unsigned __int64 v14; // r14
  __m128i *v15; // rdx
  unsigned int v16; // ecx
  __int64 v17; // rdx
  _QWORD *v18; // rdx
  int v19; // eax
  int v20; // edx
  int v21; // r8d
  int v22; // edx
  __int64 v23; // r14
  __int64 v24; // r12
  __int64 v25; // [rsp+0h] [rbp-F0h]
  __int64 v26; // [rsp+8h] [rbp-E8h]
  __int64 v27; // [rsp+8h] [rbp-E8h]
  __int64 v28; // [rsp+10h] [rbp-E0h]
  __int64 v29; // [rsp+10h] [rbp-E0h]
  __int64 v30; // [rsp+18h] [rbp-D8h]
  __int64 v31; // [rsp+18h] [rbp-D8h]
  __int64 v32; // [rsp+20h] [rbp-D0h]
  __int64 v33; // [rsp+20h] [rbp-D0h]
  int v34; // [rsp+28h] [rbp-C8h]
  __int64 v35; // [rsp+28h] [rbp-C8h]
  _BYTE *v36; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v37; // [rsp+38h] [rbp-B8h]
  _BYTE v38[176]; // [rsp+40h] [rbp-B0h] BYREF

  v9 = *(unsigned int *)(a1 + 56);
  v10 = *(const __m128i **)(a1 + 32);
  v37 = 0x800000000LL;
  v36 = v38;
  v11 = 0;
  v12 = 40 * v9;
  v13 = (const __m128i *)((char *)v10 + 40 * v9);
  v14 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v9) >> 3);
  v15 = (__m128i *)v38;
  if ( v12 > 0x140 )
  {
    v25 = a4;
    v26 = a3;
    v28 = a6;
    v30 = a5;
    v32 = a2;
    sub_16CD150((__int64)&v36, v38, v14, 16, a5, a6);
    v11 = v37;
    a4 = v25;
    a3 = v26;
    a6 = v28;
    a5 = v30;
    a2 = v32;
    v15 = (__m128i *)&v36[16 * (unsigned int)v37];
  }
  if ( v10 != v13 )
  {
    do
    {
      if ( v15 )
        *v15 = _mm_loadu_si128(v10);
      v10 = (const __m128i *)((char *)v10 + 40);
      ++v15;
    }
    while ( v13 != v10 );
    v11 = v37;
  }
  v16 = v14 + v11;
  LODWORD(v37) = v16;
  v17 = v16;
  if ( a5 )
  {
    if ( v16 >= HIDWORD(v37) )
    {
      v27 = a4;
      v29 = a3;
      v31 = a6;
      v33 = a5;
      v35 = a2;
      sub_16CD150((__int64)&v36, v38, 0, 16, a5, a6);
      v17 = (unsigned int)v37;
      a4 = v27;
      a3 = v29;
      a6 = v31;
      a5 = v33;
      a2 = v35;
    }
    v18 = &v36[16 * v17];
    *v18 = a5;
    v18[1] = a6;
    LODWORD(v37) = v37 + 1;
  }
  v34 = a2;
  v19 = sub_1D25C30(a2, a3, a4);
  v21 = v20;
  v22 = *(unsigned __int16 *)(a1 + 24);
  if ( (v22 & 0x8000u) == 0 )
  {
    sub_1D2E3C0(v34, a1, (__int16)v22, v19, v21, v19, (__int64)v36, (unsigned int)v37);
  }
  else
  {
    v23 = *(_QWORD *)(a1 + 88);
    v24 = *(_QWORD *)(a1 + 96);
    sub_1D2E3C0(v34, a1, v22, v19, v21, v19, (__int64)v36, (unsigned int)v37);
    *(_QWORD *)(a1 + 88) = v23;
    *(_QWORD *)(a1 + 96) = v24;
  }
  if ( v36 != v38 )
    _libc_free((unsigned __int64)v36);
}
