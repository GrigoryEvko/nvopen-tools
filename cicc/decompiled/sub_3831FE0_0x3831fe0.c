// Function: sub_3831FE0
// Address: 0x3831fe0
//
__int64 *__fastcall sub_3831FE0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // edx
  __int64 v9; // rax
  const __m128i *v10; // rbx
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // rcx
  __m128i *v13; // rax
  const __m128i *v14; // r14
  const __m128i *v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rsi
  unsigned __int16 *v18; // rax
  unsigned __int64 v19; // rcx
  unsigned int v20; // r15d
  __int64 v21; // rax
  unsigned int v22; // edx
  unsigned int v23; // ebx
  unsigned __int64 v24; // rdx
  __int64 v25; // r10
  unsigned __int16 *v26; // rax
  __int64 v27; // rdx
  unsigned __int8 *v28; // rbx
  int v29; // edx
  int v30; // r15d
  _BYTE *v31; // rdx
  __int64 *v32; // r12
  __int128 v34; // [rsp-10h] [rbp-120h]
  __int64 v35; // [rsp+0h] [rbp-110h]
  unsigned __int64 v36; // [rsp+8h] [rbp-108h]
  _QWORD *v37; // [rsp+8h] [rbp-108h]
  __int64 v38; // [rsp+10h] [rbp-100h]
  __m128i v39; // [rsp+20h] [rbp-F0h]
  int v40; // [rsp+20h] [rbp-F0h]
  __int64 v41; // [rsp+40h] [rbp-D0h] BYREF
  int v42; // [rsp+48h] [rbp-C8h]
  _BYTE *v43; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v44; // [rsp+58h] [rbp-B8h]
  _BYTE v45[176]; // [rsp+60h] [rbp-B0h] BYREF

  v7 = 0;
  v9 = *(unsigned int *)(a2 + 64);
  v10 = *(const __m128i **)(a2 + 40);
  v44 = 0x800000000LL;
  v9 *= 5;
  v11 = 0xCCCCCCCCCCCCCCCDLL * v9;
  v12 = 8 * v9;
  v13 = (__m128i *)v45;
  v14 = (const __m128i *)((char *)v10 + v12);
  v43 = v45;
  if ( v12 > 0x140 )
  {
    v40 = v11;
    sub_C8D5F0((__int64)&v43, v45, v11, 0x10u, v11, a6);
    v7 = v44;
    LODWORD(v11) = v40;
    v13 = (__m128i *)&v43[16 * (unsigned int)v44];
  }
  if ( v10 != v14 )
  {
    do
    {
      if ( v13 )
        *v13 = _mm_loadu_si128(v10);
      v10 = (const __m128i *)((char *)v10 + 40);
      ++v13;
    }
    while ( v14 != v10 );
    v7 = v44;
  }
  LODWORD(v44) = v11 + v7;
  v38 = a3;
  v15 = (const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL * a3);
  v16 = v15->m128i_i64[0];
  v17 = *(_QWORD *)(v15->m128i_i64[0] + 80);
  v39 = _mm_loadu_si128(v15);
  v18 = (unsigned __int16 *)(*(_QWORD *)(v15->m128i_i64[0] + 48) + 16LL * v15->m128i_u32[2]);
  v19 = *((_QWORD *)v18 + 1);
  v20 = *v18;
  v41 = v17;
  v36 = v19;
  if ( v17 )
    sub_B96E90((__int64)&v41, v17, 1);
  v42 = *(_DWORD *)(v16 + 72);
  v21 = sub_37AE0F0(a1, v39.m128i_u64[0], v39.m128i_i64[1]);
  v23 = v22;
  v24 = v36;
  v35 = v21;
  v37 = *(_QWORD **)(a1 + 8);
  v25 = sub_33F7D60(v37, v20, v24);
  v26 = (unsigned __int16 *)(*(_QWORD *)(v35 + 48) + 16LL * v23);
  *((_QWORD *)&v34 + 1) = v27;
  *(_QWORD *)&v34 = v25;
  v28 = sub_3406EB0(
          v37,
          0xDEu,
          (__int64)&v41,
          *v26,
          *((_QWORD *)v26 + 1),
          v35,
          __PAIR128__(v23 | v39.m128i_i64[1] & 0xFFFFFFFF00000000LL, v35),
          v34);
  v30 = v29;
  if ( v41 )
    sub_B91220((__int64)&v41, v41);
  v31 = &v43[16 * v38];
  *(_QWORD *)v31 = v28;
  *((_DWORD *)v31 + 2) = v30;
  v32 = sub_33EC210(*(_QWORD **)(a1 + 8), (__int64 *)a2, (__int64)v43, (unsigned int)v44);
  if ( v43 != v45 )
    _libc_free((unsigned __int64)v43);
  return v32;
}
