// Function: sub_3813200
// Address: 0x3813200
//
__int64 __fastcall sub_3813200(__int64 a1, unsigned __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rbx
  __int64 v7; // r8
  unsigned __int64 v9; // rax
  const __m128i *v10; // r13
  int v11; // ecx
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rdx
  __m128i *v14; // rax
  const __m128i *v15; // r14
  __int64 v16; // rsi
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // r9
  _BYTE *v20; // r13
  int v21; // edx
  __int64 v22; // rsi
  _QWORD *v23; // r11
  unsigned __int64 v24; // r13
  __int64 v25; // r14
  unsigned int *v26; // rcx
  __int64 v27; // r8
  __int64 v28; // rsi
  unsigned __int8 *v29; // r13
  unsigned int i; // r14d
  __int64 v31; // rdx
  __int128 v33; // [rsp-10h] [rbp-D0h]
  unsigned int *v34; // [rsp+8h] [rbp-B8h]
  __int64 v35; // [rsp+10h] [rbp-B0h]
  _QWORD *v36; // [rsp+18h] [rbp-A8h]
  int v37; // [rsp+18h] [rbp-A8h]
  int v38; // [rsp+20h] [rbp-A0h]
  __int64 v39; // [rsp+40h] [rbp-80h] BYREF
  int v40; // [rsp+48h] [rbp-78h]
  _BYTE *v41; // [rsp+50h] [rbp-70h] BYREF
  __int64 v42; // [rsp+58h] [rbp-68h]
  _BYTE v43[96]; // [rsp+60h] [rbp-60h] BYREF

  v7 = a3;
  v9 = *(unsigned int *)(a2 + 64);
  v10 = *(const __m128i **)(a2 + 40);
  v42 = 0x300000000LL;
  v11 = 0;
  v12 = 40 * v9;
  v13 = v9;
  v14 = (__m128i *)v43;
  v15 = (const __m128i *)((char *)v10 + v12);
  v41 = v43;
  if ( v12 > 0x78 )
  {
    v37 = v7;
    v38 = v13;
    sub_C8D5F0((__int64)&v41, v43, v13, 0x10u, v7, a6);
    v11 = v42;
    LODWORD(v7) = v37;
    LODWORD(v13) = v38;
    v14 = (__m128i *)&v41[16 * (unsigned int)v42];
  }
  if ( v10 != v15 )
  {
    do
    {
      if ( v14 )
        *v14 = _mm_loadu_si128(v10);
      v10 = (const __m128i *)((char *)v10 + 40);
      ++v14;
    }
    while ( v15 != v10 );
    v11 = v42;
  }
  v16 = *(_QWORD *)(a2 + 40);
  LODWORD(v42) = v11 + v13;
  v17 = 16LL * (unsigned int)v7;
  v18 = sub_380F170(a1, *(_QWORD *)(v16 + 40LL * (unsigned int)v7), *(_QWORD *)(v16 + 40LL * (unsigned int)v7 + 8));
  v20 = &v41[v17];
  *(_QWORD *)v20 = v18;
  *((_DWORD *)v20 + 2) = v21;
  v22 = *(_QWORD *)(a2 + 80);
  v23 = *(_QWORD **)(a1 + 8);
  v24 = (unsigned __int64)v41;
  v25 = (unsigned int)v42;
  v26 = *(unsigned int **)(a2 + 48);
  v39 = v22;
  v27 = *(unsigned int *)(a2 + 68);
  if ( v22 )
  {
    v34 = v26;
    v35 = *(unsigned int *)(a2 + 68);
    v36 = v23;
    sub_B96E90((__int64)&v39, v22, 1);
    v26 = v34;
    v27 = v35;
    v23 = v36;
  }
  *((_QWORD *)&v33 + 1) = v25;
  *(_QWORD *)&v33 = v24;
  v28 = *(unsigned int *)(a2 + 24);
  v40 = *(_DWORD *)(a2 + 72);
  v29 = sub_3411630(v23, v28, (__int64)&v39, v26, v27, v19, v33);
  if ( v39 )
    sub_B91220((__int64)&v39, v39);
  for ( i = 0; i < *(_DWORD *)(a2 + 68); ++i )
  {
    v31 = i;
    v6 = v31 | v6 & 0xFFFFFFFF00000000LL;
    sub_3760E70(a1, a2, v31, (unsigned __int64)v29, v6);
  }
  if ( v41 != v43 )
    _libc_free((unsigned __int64)v41);
  return 0;
}
