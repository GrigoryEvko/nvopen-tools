// Function: sub_383D470
// Address: 0x383d470
//
__int64 *__fastcall sub_383D470(__int64 a1, __int64 a2, unsigned int a3, __m128i a4)
{
  __int64 v4; // r8
  int v5; // edx
  __int64 v8; // rax
  const __m128i *v9; // rbx
  unsigned __int64 v10; // r9
  unsigned __int64 v11; // rcx
  const __m128i *v12; // r15
  __m128 *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rbx
  unsigned int *v16; // rax
  unsigned __int64 v17; // r10
  __int64 v18; // rcx
  __int64 v19; // rsi
  unsigned __int16 *v20; // rax
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  unsigned int v24; // edx
  unsigned __int8 *v25; // rax
  int v26; // edx
  int v27; // ecx
  unsigned __int8 *v28; // rdi
  _BYTE *v29; // rbx
  __int64 v30; // rdx
  __int64 *v31; // r12
  unsigned __int8 *v33; // rax
  int v34; // edx
  int v35; // esi
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // [rsp+0h] [rbp-F0h]
  __int64 v39; // [rsp+8h] [rbp-E8h]
  __int64 v40; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v41; // [rsp+10h] [rbp-E0h]
  __int64 v42; // [rsp+10h] [rbp-E0h]
  int v43; // [rsp+10h] [rbp-E0h]
  int v44; // [rsp+10h] [rbp-E0h]
  __int64 v45; // [rsp+18h] [rbp-D8h]
  __int64 v46; // [rsp+18h] [rbp-D8h]
  unsigned __int8 *v47; // [rsp+18h] [rbp-D8h]
  int v48; // [rsp+18h] [rbp-D8h]
  __int64 v49; // [rsp+40h] [rbp-B0h] BYREF
  int v50; // [rsp+48h] [rbp-A8h]
  _BYTE *v51; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v52; // [rsp+58h] [rbp-98h]
  _BYTE v53[144]; // [rsp+60h] [rbp-90h] BYREF

  v4 = a3;
  v5 = 0;
  v8 = *(unsigned int *)(a2 + 64);
  v9 = *(const __m128i **)(a2 + 40);
  v51 = v53;
  v8 *= 5;
  v10 = 0xCCCCCCCCCCCCCCCDLL * v8;
  v11 = 8 * v8;
  v52 = 0x600000000LL;
  v12 = (const __m128i *)((char *)v9 + 8 * v8);
  v13 = (__m128 *)v53;
  if ( v11 > 0xF0 )
  {
    v44 = v4;
    v48 = v10;
    sub_C8D5F0((__int64)&v51, v53, v10, 0x10u, v4, v10);
    v5 = v52;
    LODWORD(v4) = v44;
    LODWORD(v10) = v48;
    v13 = (__m128 *)&v51[16 * (unsigned int)v52];
  }
  if ( v9 != v12 )
  {
    do
    {
      if ( v13 )
      {
        a4 = _mm_loadu_si128(v9);
        *v13 = (__m128)a4;
      }
      v9 = (const __m128i *)((char *)v9 + 40);
      ++v13;
    }
    while ( v12 != v9 );
    v5 = v52;
  }
  LODWORD(v52) = v10 + v5;
  v14 = *(_QWORD *)(a2 + 40);
  if ( (_DWORD)v4 == 2 )
  {
    v33 = sub_383B380(a1, *(_QWORD *)(v14 + 80), *(_QWORD *)(v14 + 88));
    v35 = v34;
    v36 = (__int64)v51;
    *((_QWORD *)v51 + 4) = v33;
    v37 = (unsigned int)v52;
    *(_DWORD *)(v36 + 40) = v35;
    v31 = sub_33EC210(*(_QWORD **)(a1 + 8), (__int64 *)a2, v36, v37);
  }
  else
  {
    v15 = (unsigned int)v4;
    v16 = (unsigned int *)(v14 + 40LL * (unsigned int)v4);
    v17 = *(_QWORD *)v16;
    v18 = *((_QWORD *)v16 + 1);
    v19 = *(_QWORD *)(*(_QWORD *)v16 + 80LL);
    v20 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v16 + 48LL) + 16LL * v16[2]);
    v21 = *v20;
    v22 = *((_QWORD *)v20 + 1);
    v49 = v19;
    if ( v19 )
    {
      v38 = v21;
      v39 = v18;
      v41 = v17;
      v45 = v22;
      sub_B96E90((__int64)&v49, v19, 1);
      v21 = v38;
      v18 = v39;
      v17 = v41;
      v22 = v45;
    }
    v40 = v21;
    v42 = v22;
    v46 = v18;
    v50 = *(_DWORD *)(v17 + 72);
    v23 = sub_37AE0F0(a1, v17, v18);
    v25 = sub_34070B0(*(_QWORD **)(a1 + 8), v23, v46 & 0xFFFFFFFF00000000LL | v24, (__int64)&v49, v40, v42, a4);
    v27 = v26;
    v28 = v25;
    if ( v49 )
    {
      v43 = v26;
      v47 = v25;
      sub_B91220((__int64)&v49, v49);
      v27 = v43;
      v28 = v47;
    }
    v29 = &v51[16 * v15];
    *(_QWORD *)v29 = v28;
    v30 = (__int64)v51;
    *((_DWORD *)v29 + 2) = v27;
    v31 = sub_33EC210(*(_QWORD **)(a1 + 8), (__int64 *)a2, v30, (unsigned int)v52);
  }
  if ( v51 != v53 )
    _libc_free((unsigned __int64)v51);
  return v31;
}
