// Function: sub_21413B0
// Address: 0x21413b0
//
unsigned __int64 __fastcall sub_21413B0(
        _QWORD *a1,
        unsigned __int64 a2,
        unsigned int a3,
        __m128 a4,
        double a5,
        __m128i a6,
        __int64 a7,
        __int64 a8,
        int a9)
{
  unsigned __int64 v10; // r12
  __int64 v11; // rax
  const __m128i *v12; // r14
  int v13; // esi
  unsigned __int64 v14; // rcx
  const __m128i *v15; // r15
  unsigned __int64 v16; // rdx
  __m128 *v17; // rax
  __int64 *v18; // rax
  __int64 v19; // r14
  __int64 v20; // rax
  _BYTE *v21; // r8
  int v22; // edx
  __int64 v23; // rdx
  __int64 *v24; // rax
  const __m128i *v25; // r9
  __int64 v26; // r14
  unsigned __int64 v27; // rsi
  const __m128i *v28; // r9
  unsigned __int64 v30; // r8
  unsigned __int64 v31; // r10
  __int64 v32; // r11
  __int64 v33; // rsi
  unsigned __int8 *v34; // rax
  unsigned __int64 v35; // rcx
  unsigned int v36; // r15d
  __int64 v37; // rax
  unsigned int v38; // edx
  unsigned __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int128 v43; // rax
  int v44; // edx
  __int64 *v45; // r15
  int v46; // ecx
  __int64 v47; // rax
  int v48; // edx
  int v49; // esi
  __int64 v50; // [rsp+8h] [rbp-F8h]
  unsigned __int64 v51; // [rsp+10h] [rbp-F0h]
  __int64 v52; // [rsp+10h] [rbp-F0h]
  __int64 v53; // [rsp+18h] [rbp-E8h]
  unsigned int v54; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v55; // [rsp+20h] [rbp-E0h]
  unsigned int v56; // [rsp+20h] [rbp-E0h]
  int v57; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v58; // [rsp+28h] [rbp-D8h]
  __int64 *v59; // [rsp+28h] [rbp-D8h]
  int v60; // [rsp+28h] [rbp-D8h]
  int v61; // [rsp+38h] [rbp-C8h]
  __int64 v62; // [rsp+60h] [rbp-A0h] BYREF
  int v63; // [rsp+68h] [rbp-98h]
  _BYTE *v64; // [rsp+70h] [rbp-90h] BYREF
  __int64 v65; // [rsp+78h] [rbp-88h]
  _BYTE v66[128]; // [rsp+80h] [rbp-80h] BYREF

  v10 = a2;
  v11 = *(unsigned int *)(a2 + 56);
  v12 = *(const __m128i **)(a2 + 32);
  v13 = 0;
  v64 = v66;
  v14 = 40 * v11;
  v65 = 0x500000000LL;
  v15 = (const __m128i *)((char *)v12 + 40 * v11);
  v16 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v11) >> 3);
  v17 = (__m128 *)v66;
  if ( v14 > 0xC8 )
  {
    v54 = a3;
    v57 = v16;
    sub_16CD150((__int64)&v64, v66, v16, 16, a3, a9);
    v13 = v65;
    a3 = v54;
    LODWORD(v16) = v57;
    v17 = (__m128 *)&v64[16 * (unsigned int)v65];
  }
  if ( v12 != v15 )
  {
    do
    {
      if ( v17 )
      {
        a4 = (__m128)_mm_loadu_si128(v12);
        *v17 = a4;
      }
      v12 = (const __m128i *)((char *)v12 + 40);
      ++v17;
    }
    while ( v15 != v12 );
    v13 = v65;
  }
  v18 = *(__int64 **)(v10 + 32);
  LODWORD(v65) = v13 + v16;
  if ( a3 == 2 )
  {
    v47 = sub_200E230(
            a1,
            v18[10],
            v18[11],
            **(unsigned __int8 **)(v10 + 40),
            *(_QWORD *)(*(_QWORD *)(v10 + 40) + 8LL),
            *(double *)a4.m128_u64,
            a5,
            *(double *)a6.m128i_i64);
    v49 = v48;
    v23 = (__int64)v64;
    *((_QWORD *)v64 + 4) = v47;
    *(_DWORD *)(v23 + 40) = v49;
  }
  else if ( a3 == 4 )
  {
    v30 = v18[20];
    v31 = v30;
    v32 = v18[21];
    v33 = *(_QWORD *)(v30 + 72);
    v34 = (unsigned __int8 *)(*(_QWORD *)(v30 + 40) + 16LL * *((unsigned int *)v18 + 42));
    v35 = *((_QWORD *)v34 + 1);
    v36 = *v34;
    v62 = v33;
    v58 = v35;
    if ( v33 )
    {
      v51 = v30;
      v53 = v32;
      v55 = v30;
      sub_1623A60((__int64)&v62, v33, 2);
      v31 = v51;
      v32 = v53;
      v30 = v55;
    }
    v50 = v32;
    v63 = *(_DWORD *)(v30 + 64);
    v37 = sub_2138AD0((__int64)a1, v31, v32);
    v56 = v38;
    v39 = v58;
    v52 = v37;
    v59 = (__int64 *)a1[1];
    *(_QWORD *)&v43 = sub_1D2EF30(v59, v36, v39, v40, v41, v42);
    v45 = sub_1D332F0(
            v59,
            148,
            (__int64)&v62,
            *(unsigned __int8 *)(*(_QWORD *)(v52 + 40) + 16LL * v56),
            *(const void ***)(*(_QWORD *)(v52 + 40) + 16LL * v56 + 8),
            0,
            *(double *)a4.m128_u64,
            a5,
            a6,
            v52,
            v56 | v50 & 0xFFFFFFFF00000000LL,
            v43);
    v46 = v44;
    if ( v62 )
    {
      v60 = v44;
      sub_161E7C0((__int64)&v62, v62);
      v46 = v60;
    }
    v23 = (__int64)v64;
    *((_QWORD *)v64 + 8) = v45;
    *(_DWORD *)(v23 + 72) = v46;
  }
  else
  {
    v19 = a3;
    v20 = sub_2138AD0((__int64)a1, v18[5 * a3], v18[5 * a3 + 1]);
    v21 = &v64[16 * v19];
    v61 = v22;
    *(_QWORD *)v21 = v20;
    v23 = (__int64)v64;
    *((_DWORD *)v21 + 2) = v61;
  }
  v24 = sub_1D2E160((_QWORD *)a1[1], (__int64 *)v10, v23, (unsigned int)v65);
  v26 = (__int64)v24;
  if ( v24 != (__int64 *)v10 )
  {
    sub_2013400((__int64)a1, v10, 0, (__int64)v24, 0, v25);
    v27 = v10;
    v10 = 0;
    sub_2013400((__int64)a1, v27, 1, v26, (__m128i *)1, v28);
  }
  if ( v64 != v66 )
    _libc_free((unsigned __int64)v64);
  return v10;
}
