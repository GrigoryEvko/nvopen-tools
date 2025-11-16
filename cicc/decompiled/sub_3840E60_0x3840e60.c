// Function: sub_3840E60
// Address: 0x3840e60
//
__m128i *__fastcall sub_3840E60(__int64 a1, __int64 a2, unsigned int a3, __m128i a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r9
  __int64 v7; // r10
  int v8; // ecx
  __int64 v9; // r15
  unsigned __int64 v10; // rax
  char v11; // r14
  const __m128i *v12; // r12
  unsigned __int64 v13; // rsi
  bool v14; // r14
  unsigned __int64 v15; // rdx
  const __m128i *v16; // r13
  __m128 *v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rsi
  __int64 v20; // rdx
  unsigned __int8 *v21; // rax
  int v22; // edx
  int v23; // esi
  unsigned __int64 v24; // rdx
  __int64 v25; // r10
  __int64 v26; // r9
  unsigned __int16 v27; // ax
  __int64 v28; // rsi
  unsigned __int64 *v29; // r12
  _QWORD *v30; // r15
  __int64 v31; // r13
  __int64 v32; // rdi
  __m128i *v33; // rax
  __int32 v34; // edx
  __m128i *v35; // r12
  __int64 v37; // rax
  _BYTE *v38; // r15
  int v39; // edx
  unsigned __int8 *v40; // rax
  int v41; // edx
  int v42; // esi
  char v43; // [rsp+Ch] [rbp-104h]
  __int64 v44; // [rsp+10h] [rbp-100h]
  unsigned __int64 v45; // [rsp+10h] [rbp-100h]
  __int64 v46; // [rsp+18h] [rbp-F8h]
  unsigned __int16 v47; // [rsp+18h] [rbp-F8h]
  __int64 v48; // [rsp+18h] [rbp-F8h]
  __int64 v49; // [rsp+20h] [rbp-F0h]
  __int16 v50; // [rsp+20h] [rbp-F0h]
  __int64 v51; // [rsp+20h] [rbp-F0h]
  __int64 v52; // [rsp+20h] [rbp-F0h]
  __int64 v53; // [rsp+28h] [rbp-E8h]
  const __m128i *v54; // [rsp+28h] [rbp-E8h]
  __int64 v55; // [rsp+28h] [rbp-E8h]
  int v56; // [rsp+28h] [rbp-E8h]
  __int64 v57; // [rsp+28h] [rbp-E8h]
  int v58; // [rsp+38h] [rbp-D8h]
  __int64 v59; // [rsp+70h] [rbp-A0h] BYREF
  int v60; // [rsp+78h] [rbp-98h]
  _BYTE *v61; // [rsp+80h] [rbp-90h] BYREF
  __int64 v62; // [rsp+88h] [rbp-88h]
  _BYTE v63[128]; // [rsp+90h] [rbp-80h] BYREF

  v6 = a2;
  v7 = a1;
  v8 = 0;
  v9 = a3;
  v10 = *(unsigned int *)(a2 + 64);
  v11 = *(_BYTE *)(a2 + 33);
  v61 = v63;
  v12 = *(const __m128i **)(a2 + 40);
  v13 = 40 * v10;
  v14 = (v11 & 4) != 0;
  v15 = v10;
  v16 = (const __m128i *)((char *)v12 + 40 * v10);
  v62 = 0x500000000LL;
  v17 = (__m128 *)v63;
  if ( v13 > 0xC8 )
  {
    v48 = v6;
    v56 = v15;
    sub_C8D5F0((__int64)&v61, v63, v15, 0x10u, a6, v6);
    v8 = v62;
    v6 = v48;
    v7 = a1;
    LODWORD(v15) = v56;
    v17 = (__m128 *)&v61[16 * (unsigned int)v62];
  }
  if ( v12 != v16 )
  {
    do
    {
      if ( v17 )
      {
        a4 = _mm_loadu_si128(v12);
        *v17 = (__m128)a4;
      }
      v12 = (const __m128i *)((char *)v12 + 40);
      ++v17;
    }
    while ( v16 != v12 );
    v8 = v62;
  }
  v18 = *(_QWORD *)(v6 + 40);
  LODWORD(v62) = v8 + v15;
  if ( (_DWORD)v9 == 2 )
  {
    v52 = v6;
    v57 = v7;
    v40 = sub_375B580(
            v7,
            *(_QWORD *)(v18 + 80),
            a4,
            *(_QWORD *)(v18 + 88),
            *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v18 + 40) + 48LL) + 16LL * *(unsigned int *)(v18 + 48)),
            *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v18 + 40) + 48LL) + 16LL * *(unsigned int *)(v18 + 48) + 8));
    v25 = v57;
    v26 = v52;
    v42 = v41;
    v24 = (unsigned __int64)v61;
    *((_QWORD *)v61 + 4) = v40;
    *(_DWORD *)(v24 + 40) = v42;
    v43 = v14;
  }
  else if ( (_DWORD)v9 == 4 )
  {
    v49 = v6;
    v43 = v14;
    v19 = *(_QWORD *)(v18 + 160);
    v20 = *(_QWORD *)(v18 + 168);
    v53 = v7;
    if ( (*(_WORD *)(v6 + 32) & 0x380) != 0 )
      v21 = sub_37AF270(v7, v19, v20, a4);
    else
      v21 = sub_383B380(v7, v19, v20);
    v23 = v22;
    v24 = (unsigned __int64)v61;
    *((_QWORD *)v61 + 8) = v21;
    *(_DWORD *)(v24 + 72) = v23;
    v25 = v53;
    v26 = v49;
  }
  else
  {
    v51 = v6;
    v55 = v7;
    v37 = sub_37AE0F0(v7, *(_QWORD *)(v18 + 40 * v9), *(_QWORD *)(v18 + 40 * v9 + 8));
    v38 = &v61[16 * v9];
    v43 = 1;
    v58 = v39;
    v26 = v51;
    v25 = v55;
    *(_QWORD *)v38 = v37;
    v24 = (unsigned __int64)v61;
    *((_DWORD *)v38 + 2) = v58;
  }
  v27 = *(_WORD *)(v26 + 32);
  v28 = *(_QWORD *)(v26 + 80);
  v29 = (unsigned __int64 *)v24;
  v46 = v25;
  v30 = *(_QWORD **)(v25 + 8);
  v31 = (unsigned int)v62;
  v59 = v28;
  v50 = (v27 >> 7) & 7;
  v54 = *(const __m128i **)(v26 + 112);
  if ( v28 )
  {
    v44 = v26;
    sub_B96E90((__int64)&v59, v28, 1);
    v26 = v44;
    v32 = *(_QWORD *)(v46 + 8);
  }
  else
  {
    v32 = (__int64)v30;
  }
  v45 = *(_QWORD *)(v26 + 104);
  v47 = *(_WORD *)(v26 + 96);
  v60 = *(_DWORD *)(v26 + 72);
  v33 = sub_33ED250(v32, 1, 0);
  v35 = sub_33E7ED0(v30, (unsigned __int64)v33, v34, v47, v45, (__int64)&v59, v29, v31, v54, v50, v43);
  if ( v59 )
    sub_B91220((__int64)&v59, v59);
  if ( v61 != v63 )
    _libc_free((unsigned __int64)v61);
  return v35;
}
