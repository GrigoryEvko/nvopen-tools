// Function: sub_332D1E0
// Address: 0x332d1e0
//
void __fastcall sub_332D1E0(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v9; // eax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  __m128i v13; // xmm0
  __int64 v14; // rdx
  __int64 v15; // r9
  __int64 v16; // r8
  __int32 v17; // esi
  __int64 v18; // rax
  const __m128i *v19; // r10
  __int64 v20; // r11
  const __m128i *v21; // rax
  unsigned __int64 v22; // rdx
  __m128i *v23; // rcx
  __int64 v24; // rax
  __int64 v25; // r11
  unsigned __int32 v26; // esi
  __int64 *v27; // rcx
  int v28; // eax
  __int64 v29; // r12
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rax
  __m128i v33; // xmm0
  unsigned __int64 v34; // rcx
  __m128i v35; // xmm0
  __int64 v36; // rax
  _BYTE *v37; // rdi
  int v38; // [rsp-118h] [rbp-118h]
  int v39; // [rsp-118h] [rbp-118h]
  __int64 *v40; // [rsp-110h] [rbp-110h]
  const __m128i *v41; // [rsp-110h] [rbp-110h]
  int v42; // [rsp-108h] [rbp-108h]
  const __m128i *v43; // [rsp-108h] [rbp-108h]
  int v44; // [rsp-100h] [rbp-100h]
  int v45; // [rsp-100h] [rbp-100h]
  __m128i v46; // [rsp-F8h] [rbp-F8h] BYREF
  __int64 v47; // [rsp-E8h] [rbp-E8h] BYREF
  int v48; // [rsp-E0h] [rbp-E0h]
  __m128i v49; // [rsp-D8h] [rbp-D8h] BYREF
  __m128i v50; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v51; // [rsp-B8h] [rbp-B8h]
  __int64 v52; // [rsp-B0h] [rbp-B0h]
  __int64 v53; // [rsp-A8h] [rbp-A8h]
  __int64 v54; // [rsp-A0h] [rbp-A0h]
  char v55; // [rsp-98h] [rbp-98h]
  __m128i v56; // [rsp-88h] [rbp-88h] BYREF
  _BYTE v57[120]; // [rsp-78h] [rbp-78h] BYREF

  if ( a3 == 729 )
    BUG();
  v9 = *(_DWORD *)(a2 + 24);
  if ( v9 > 239 )
  {
    if ( (unsigned int)(v9 - 242) > 1 )
      goto LABEL_5;
  }
  else if ( v9 <= 237 && (unsigned int)(v9 - 101) > 0x2F )
  {
LABEL_5:
    sub_332CFC0(&v56, a1, a3, a2, (__int64 (__fastcall *)(__int64, __int64, unsigned int))(v9 == 259), a6);
    v12 = *(unsigned int *)(a4 + 8);
    v13 = _mm_load_si128(&v56);
    if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
    {
      v46 = v13;
      sub_C8D5F0(a4, (const void *)(a4 + 16), v12 + 1, 0x10u, v10, v11);
      v12 = *(unsigned int *)(a4 + 8);
      v13 = _mm_load_si128(&v46);
    }
    *(__m128i *)(*(_QWORD *)a4 + 16 * v12) = v13;
    ++*(_DWORD *)(a4 + 8);
    return;
  }
  v14 = *(unsigned int *)(a2 + 64);
  v15 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v16 = **(unsigned __int16 **)(a2 + 48);
  v56.m128i_i64[1] = 0x400000000LL;
  v17 = 0;
  v18 = *(_QWORD *)(a2 + 40);
  v56.m128i_i64[0] = (__int64)v57;
  v19 = (const __m128i *)(v18 + 40 * v14);
  v20 = 40 * v14 - 40;
  v21 = (const __m128i *)(v18 + 40);
  v22 = 0xCCCCCCCCCCCCCCCDLL * (v20 >> 3);
  v23 = (__m128i *)v57;
  if ( (unsigned __int64)v20 > 0xA0 )
  {
    v39 = v16;
    v41 = v19;
    v43 = v21;
    v45 = v15;
    v46.m128i_i64[0] = 0xCCCCCCCCCCCCCCCDLL * (v20 >> 3);
    sub_C8D5F0((__int64)&v56, v57, v22, 0x10u, v16, v15);
    v17 = v56.m128i_i32[2];
    v19 = v41;
    v21 = v43;
    LODWORD(v16) = v39;
    LODWORD(v15) = v45;
    LODWORD(v22) = v46.m128i_i32[0];
    v23 = (__m128i *)(v56.m128i_i64[0] + 16LL * v56.m128i_u32[2]);
    if ( v41 != v43 )
      goto LABEL_12;
  }
  else if ( v19 != v21 )
  {
    do
    {
LABEL_12:
      if ( v23 )
        *v23 = _mm_loadu_si128(v21);
      v21 = (const __m128i *)((char *)v21 + 40);
      ++v23;
    }
    while ( v19 != v21 );
    v17 = v56.m128i_i32[2];
  }
  v24 = *(_QWORD *)(a2 + 80);
  v51 = 0;
  v56.m128i_i32[2] = v17 + v22;
  v25 = *(_QWORD *)(a1 + 8);
  v26 = v17 + v22;
  v52 = 0;
  v27 = *(__int64 **)(a2 + 40);
  v53 = 0;
  v54 = 0;
  v55 = 12;
  v47 = v24;
  if ( v24 )
  {
    v38 = v16;
    v40 = v27;
    v42 = v15;
    v44 = v25;
    v46.m128i_i64[0] = (__int64)&v47;
    sub_B96E90((__int64)&v47, v24, 1);
    v26 = v56.m128i_u32[2];
    LODWORD(v16) = v38;
    v27 = v40;
    LODWORD(v15) = v42;
    LODWORD(v25) = v44;
  }
  v28 = *(_DWORD *)(a2 + 72);
  v29 = *(_QWORD *)(a1 + 16);
  v46.m128i_i64[0] = (__int64)&v47;
  v48 = v28;
  sub_3494590(
    (unsigned int)&v49,
    v25,
    v29,
    a3,
    v16,
    v15,
    v56.m128i_i64[0],
    v26,
    v51,
    v52,
    v53,
    v54,
    v55,
    (__int64)&v47,
    *v27,
    v27[1]);
  if ( v47 )
    sub_B91220(v46.m128i_i64[0], v47);
  v32 = *(unsigned int *)(a4 + 8);
  v33 = _mm_load_si128(&v49);
  if ( v32 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
  {
    v46 = v33;
    sub_C8D5F0(a4, (const void *)(a4 + 16), v32 + 1, 0x10u, v30, v31);
    v32 = *(unsigned int *)(a4 + 8);
    v33 = _mm_load_si128(&v46);
  }
  *(__m128i *)(*(_QWORD *)a4 + 16 * v32) = v33;
  v34 = *(unsigned int *)(a4 + 12);
  v35 = _mm_load_si128(&v50);
  v36 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
  *(_DWORD *)(a4 + 8) = v36;
  if ( v36 + 1 > v34 )
  {
    v46 = v35;
    sub_C8D5F0(a4, (const void *)(a4 + 16), v36 + 1, 0x10u, v30, v31);
    v36 = *(unsigned int *)(a4 + 8);
    v35 = _mm_load_si128(&v46);
  }
  *(__m128i *)(*(_QWORD *)a4 + 16 * v36) = v35;
  v37 = (_BYTE *)v56.m128i_i64[0];
  ++*(_DWORD *)(a4 + 8);
  if ( v37 != v57 )
    _libc_free((unsigned __int64)v37);
}
