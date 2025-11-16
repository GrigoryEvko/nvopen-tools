// Function: sub_383D760
// Address: 0x383d760
//
__int64 *__fastcall sub_383D760(__int64 a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  int v7; // ecx
  __int64 v9; // rax
  const __m128i *v10; // rbx
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rsi
  const __m128i *v13; // r14
  __m128 *v14; // rax
  bool v15; // zf
  __int64 v16; // rax
  unsigned __int64 v17; // r10
  __int64 v18; // rbx
  __int64 v19; // rsi
  unsigned __int16 *v20; // rax
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  unsigned int v24; // edx
  int v25; // edx
  unsigned __int8 *v26; // rbx
  int v27; // ecx
  unsigned __int64 v28; // rax
  unsigned __int8 *v29; // rax
  int v30; // edx
  int v31; // esi
  __int64 v32; // rdx
  __int64 *v33; // r12
  unsigned __int8 *v35; // rax
  int v36; // edx
  int v37; // ebx
  unsigned __int8 *v38; // rdx
  unsigned __int64 v39; // rax
  unsigned __int8 *v40; // rax
  int v41; // edx
  int v42; // esi
  __int64 v43; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v44; // [rsp+10h] [rbp-B0h]
  __int64 v45; // [rsp+10h] [rbp-B0h]
  __int64 v46; // [rsp+18h] [rbp-A8h]
  __int64 v47; // [rsp+18h] [rbp-A8h]
  int v48; // [rsp+18h] [rbp-A8h]
  int v49; // [rsp+18h] [rbp-A8h]
  __int64 v50; // [rsp+60h] [rbp-60h] BYREF
  int v51; // [rsp+68h] [rbp-58h]
  _BYTE *v52; // [rsp+70h] [rbp-50h] BYREF
  __int64 v53; // [rsp+78h] [rbp-48h]
  _BYTE v54[64]; // [rsp+80h] [rbp-40h] BYREF

  v7 = 0;
  v9 = *(unsigned int *)(a2 + 64);
  v10 = *(const __m128i **)(a2 + 40);
  v52 = v54;
  v9 *= 5;
  v11 = 0xCCCCCCCCCCCCCCCDLL * v9;
  v12 = 8 * v9;
  v53 = 0x100000000LL;
  v13 = (const __m128i *)((char *)v10 + 8 * v9);
  v14 = (__m128 *)v54;
  if ( v12 > 0x28 )
  {
    v49 = v11;
    sub_C8D5F0((__int64)&v52, v54, v11, 0x10u, a6, a7);
    v7 = v53;
    LODWORD(v11) = v49;
    v14 = (__m128 *)&v52[16 * (unsigned int)v53];
  }
  if ( v10 != v13 )
  {
    do
    {
      if ( v14 )
      {
        a3 = _mm_loadu_si128(v10);
        *v14 = (__m128)a3;
      }
      v10 = (const __m128i *)((char *)v10 + 40);
      ++v14;
    }
    while ( v13 != v10 );
    v7 = v53;
  }
  v15 = *(_DWORD *)(a2 + 24) == 391;
  LODWORD(v53) = v11 + v7;
  v16 = *(_QWORD *)(a2 + 40);
  if ( v15 )
  {
    v35 = sub_383B380(a1, *(_QWORD *)(v16 + 40), *(_QWORD *)(v16 + 48));
    v37 = v36;
    v38 = v35;
    v39 = (unsigned __int64)v52;
    *((_QWORD *)v52 + 2) = v38;
    *(_DWORD *)(v39 + 24) = v37;
    v40 = sub_383B380(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
    v42 = v41;
    v32 = (__int64)v52;
    *((_QWORD *)v52 + 4) = v40;
    *(_DWORD *)(v32 + 40) = v42;
  }
  else
  {
    v17 = *(_QWORD *)(v16 + 40);
    v18 = *(_QWORD *)(v16 + 48);
    v19 = *(_QWORD *)(v17 + 80);
    v20 = (unsigned __int16 *)(*(_QWORD *)(v17 + 48) + 16LL * *(unsigned int *)(v16 + 48));
    v21 = *v20;
    v22 = *((_QWORD *)v20 + 1);
    v50 = v19;
    if ( v19 )
    {
      v43 = v21;
      v44 = v17;
      v46 = v22;
      sub_B96E90((__int64)&v50, v19, 1);
      v21 = v43;
      v17 = v44;
      v22 = v46;
    }
    v45 = v21;
    v47 = v22;
    v51 = *(_DWORD *)(v17 + 72);
    v23 = sub_37AE0F0(a1, v17, v18);
    v26 = sub_34070B0(*(_QWORD **)(a1 + 8), v23, v18 & 0xFFFFFFFF00000000LL | v24, (__int64)&v50, v45, v47, a3);
    v27 = v25;
    if ( v50 )
    {
      v48 = v25;
      sub_B91220((__int64)&v50, v50);
      v27 = v48;
    }
    v28 = (unsigned __int64)v52;
    *((_QWORD *)v52 + 2) = v26;
    *(_DWORD *)(v28 + 24) = v27;
    v29 = sub_37AF270(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL), a3);
    v31 = v30;
    v32 = (__int64)v52;
    *((_QWORD *)v52 + 4) = v29;
    *(_DWORD *)(v32 + 40) = v31;
  }
  v33 = sub_33EC210(*(_QWORD **)(a1 + 8), (__int64 *)a2, v32, (unsigned int)v53);
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
  return v33;
}
