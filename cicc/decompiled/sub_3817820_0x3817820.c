// Function: sub_3817820
// Address: 0x3817820
//
__int64 *__fastcall sub_3817820(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        __m128i a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  int v8; // edx
  __int64 v11; // rax
  const __m128i *v12; // rbx
  unsigned __int64 v13; // r8
  unsigned __int64 v14; // rcx
  __m128 *v15; // rax
  const __m128i *v16; // r14
  __int64 v17; // r11
  __int64 v18; // rbx
  unsigned int *v19; // roff
  __int64 (__fastcall *v20)(__int64, __int64, unsigned int, __int64); // r10
  __int64 v21; // rax
  unsigned __int16 v22; // di
  __int64 v23; // r8
  __int64 v24; // rax
  int v25; // r9d
  __int64 v26; // r8
  __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // r15
  unsigned __int8 *v30; // rax
  _BYTE *v31; // rbx
  int v32; // edx
  __int64 *v33; // r12
  __int64 v35; // rdx
  __int64 v36; // [rsp+8h] [rbp-C8h]
  int v37; // [rsp+10h] [rbp-C0h]
  __int64 v38; // [rsp+20h] [rbp-B0h]
  __int64 v39; // [rsp+40h] [rbp-90h] BYREF
  int v40; // [rsp+48h] [rbp-88h]
  __int64 v41; // [rsp+50h] [rbp-80h]
  _BYTE *v42; // [rsp+60h] [rbp-70h] BYREF
  __int64 v43; // [rsp+68h] [rbp-68h]
  _BYTE v44[96]; // [rsp+70h] [rbp-60h] BYREF

  v8 = 0;
  v11 = *(unsigned int *)(a2 + 64);
  v12 = *(const __m128i **)(a2 + 40);
  v43 = 0x300000000LL;
  v11 *= 5;
  v13 = 0xCCCCCCCCCCCCCCCDLL * v11;
  v14 = 8 * v11;
  v15 = (__m128 *)v44;
  v16 = (const __m128i *)((char *)v12 + v14);
  v42 = v44;
  if ( v14 > 0x78 )
  {
    v37 = v13;
    sub_C8D5F0((__int64)&v42, v44, v13, 0x10u, v13, a7);
    v8 = v43;
    LODWORD(v13) = v37;
    v15 = (__m128 *)&v42[16 * (unsigned int)v43];
  }
  if ( v12 != v16 )
  {
    do
    {
      if ( v15 )
      {
        a4 = _mm_loadu_si128(v12);
        *v15 = (__m128)a4;
      }
      v12 = (const __m128i *)((char *)v12 + 40);
      ++v15;
    }
    while ( v16 != v12 );
    v8 = v43;
  }
  v17 = *a1;
  v18 = a3;
  LODWORD(v43) = v13 + v8;
  v19 = (unsigned int *)(*(_QWORD *)(a2 + 40) + 40LL * a3);
  v20 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v17 + 592LL);
  v21 = *(_QWORD *)(*(_QWORD *)v19 + 48LL) + 16LL * v19[2];
  v22 = *(_WORD *)v21;
  v23 = *(_QWORD *)(v21 + 8);
  v24 = a1[1];
  if ( v20 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v39, v17, *(_QWORD *)(v24 + 64), v22, v23);
    v26 = v41;
    v27 = (unsigned __int16)v40;
  }
  else
  {
    v27 = v20(v17, *(_QWORD *)(v24 + 64), v22, v23);
    v26 = v35;
  }
  v28 = *(_QWORD *)(a2 + 80);
  v29 = a1[1];
  v39 = v28;
  if ( v28 )
  {
    v36 = v26;
    v38 = v27;
    sub_B96E90((__int64)&v39, v28, 1);
    v26 = v36;
    v27 = v38;
  }
  v40 = *(_DWORD *)(a2 + 72);
  v30 = sub_33FAF80(v29, 215, (__int64)&v39, v27, v26, v25, a4);
  v31 = &v42[16 * v18];
  *(_QWORD *)v31 = v30;
  *((_DWORD *)v31 + 2) = v32;
  if ( v39 )
    sub_B91220((__int64)&v39, v39);
  v33 = sub_33EC210((_QWORD *)a1[1], (__int64 *)a2, (__int64)v42, (unsigned int)v43);
  if ( v42 != v44 )
    _libc_free((unsigned __int64)v42);
  return v33;
}
