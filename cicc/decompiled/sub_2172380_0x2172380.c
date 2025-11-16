// Function: sub_2172380
// Address: 0x2172380
//
void __fastcall sub_2172380(__int64 a1, __int64 *a2, __int64 a3, __m128i a4, __m128 a5, __m128i a6)
{
  __int64 v7; // r14
  __int64 v8; // rsi
  __int64 v9; // rsi
  bool v10; // zf
  __int64 v11; // rax
  int v12; // r9d
  int v13; // r8d
  __int64 *v14; // r12
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // r13d
  unsigned __int8 *v18; // rax
  __int64 *v19; // r12
  __int64 v20; // r13
  int v21; // esi
  unsigned __int8 *v22; // rax
  const __m128i *v23; // rax
  __m128i v24; // xmm2
  int v25; // r8d
  int v26; // r9d
  __int64 v27; // r12
  __int64 v28; // rdx
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 *v31; // rax
  int v32; // ecx
  unsigned int v33; // eax
  __int64 v34; // r10
  unsigned int v35; // r14d
  __int64 v36; // rax
  unsigned int *v37; // r12
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // r13
  __int64 v41; // r12
  __int128 v42; // rax
  __m128i v43; // rax
  __int128 v44; // rax
  __int64 *v45; // rax
  __int64 *v46; // r12
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // r13
  __int64 v50; // rax
  __int64 *v51; // rax
  __int64 v52; // r9
  __int64 v53; // r13
  unsigned __int8 v54; // r12
  __int64 v55; // rax
  __int64 *v56; // rdi
  int v57; // r13d
  __int64 v58; // r15
  int v59; // edx
  int v60; // r8d
  int v61; // r9d
  __int64 v62; // r12
  __int64 v63; // rax
  __int64 *v64; // rax
  __int64 *v65; // rax
  __int64 *v66; // rdi
  __int64 v67; // [rsp-8h] [rbp-188h]
  __int64 v69; // [rsp+28h] [rbp-158h]
  __m128i v70; // [rsp+30h] [rbp-150h] BYREF
  __int64 *v71; // [rsp+40h] [rbp-140h]
  __int64 v72; // [rsp+48h] [rbp-138h]
  __int64 v73; // [rsp+50h] [rbp-130h] BYREF
  int v74; // [rsp+58h] [rbp-128h]
  unsigned __int8 *v75; // [rsp+60h] [rbp-120h] BYREF
  __int64 v76; // [rsp+68h] [rbp-118h]
  _BYTE v77[80]; // [rsp+70h] [rbp-110h] BYREF
  __int64 *v78; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v79; // [rsp+C8h] [rbp-B8h]
  __m128i v80; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 *v81; // [rsp+E0h] [rbp-A0h]
  __int64 v82; // [rsp+E8h] [rbp-98h]

  v7 = a1;
  v8 = *(_QWORD *)(a1 + 72);
  v73 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v73, v8, 2);
  v9 = 3673;
  v10 = *(_WORD *)(a1 + 24) == 221;
  v74 = *(_DWORD *)(a1 + 64);
  if ( !v10 )
    v9 = 3703;
  v11 = sub_1D38BB0((__int64)a2, v9, (__int64)&v73, 5, 0, 0, a4, *(double *)a5.m128_u64, a6, 0);
  v13 = 2;
  v14 = (__int64 *)v11;
  v75 = v77;
  v76 = 0x500000000LL;
  v15 = 0;
  v72 = v16;
  v17 = 2;
  v71 = v14;
  while ( 1 )
  {
    v18 = &v75[16 * v15];
    *(_QWORD *)v18 = 6;
    *((_QWORD *)v18 + 1) = 0;
    v15 = (unsigned int)(v76 + 1);
    LODWORD(v76) = v76 + 1;
    if ( v17 == 1 )
      break;
    v17 = 1;
    if ( HIDWORD(v76) <= (unsigned int)v15 )
    {
      v70.m128i_i64[0] = (__int64)&v75;
      sub_16CD150((__int64)&v75, v77, 0, 16, v13, v12);
      v15 = (unsigned int)v76;
    }
  }
  v19 = v71;
  v20 = v72;
  if ( (unsigned int)v15 >= HIDWORD(v76) )
  {
    sub_16CD150((__int64)&v75, v77, 0, 16, v13, v12);
    v15 = (unsigned int)v76;
  }
  v21 = 14;
  v22 = &v75[16 * v15];
  *(_QWORD *)v22 = 1;
  *((_QWORD *)v22 + 1) = 0;
  v78 = (__int64 *)&v80;
  v79 = 0x800000000LL;
  v23 = *(const __m128i **)(a1 + 32);
  LODWORD(v76) = v76 + 1;
  v10 = *(_WORD *)(a1 + 24) == 221;
  v24 = _mm_loadu_si128(v23);
  v81 = v19;
  if ( !v10 )
    v21 = 0;
  v82 = v20;
  v80 = v24;
  LODWORD(v79) = 2;
  v27 = sub_1D38BB0((__int64)a2, (v21 << 16) | 5u, (__int64)&v73, 5, 0, 0, a4, *(double *)a5.m128_u64, v24, 0);
  v29 = v28;
  v30 = (unsigned int)v79;
  if ( (unsigned int)v79 >= HIDWORD(v79) )
  {
    sub_16CD150((__int64)&v78, &v80, 0, 16, v25, v26);
    v30 = (unsigned int)v79;
  }
  v31 = &v78[2 * v30];
  *v31 = v27;
  v31[1] = v29;
  v32 = *(_DWORD *)(a1 + 56);
  v33 = v79 + 1;
  LODWORD(v79) = v79 + 1;
  LODWORD(v71) = v32;
  if ( v32 != 1 )
  {
    v34 = a1;
    v35 = 1;
    do
    {
      while ( 1 )
      {
        v37 = (unsigned int *)(*(_QWORD *)(v34 + 32) + 40LL * v35);
        if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v37 + 40LL) + 16LL * v37[2]) != 7 )
          break;
        v69 = v34;
        v38 = sub_1D309E0(
                a2,
                158,
                (__int64)&v73,
                50,
                0,
                0,
                *(double *)a4.m128i_i64,
                *(double *)a5.m128_u64,
                *(double *)v24.m128i_i64,
                *(_OWORD *)v37);
        v40 = v39;
        v41 = v38;
        *(_QWORD *)&v42 = sub_1D38E70((__int64)a2, 0, (__int64)&v73, 0, a4, *(double *)a5.m128_u64, v24);
        v43.m128i_i64[0] = (__int64)sub_1D332F0(
                                      a2,
                                      106,
                                      (__int64)&v73,
                                      6,
                                      0,
                                      0,
                                      *(double *)a4.m128i_i64,
                                      *(double *)a5.m128_u64,
                                      v24,
                                      v41,
                                      v40,
                                      v42);
        v70 = v43;
        *(_QWORD *)&v44 = sub_1D38E70((__int64)a2, 1, (__int64)&v73, 0, a4, *(double *)a5.m128_u64, v24);
        v45 = sub_1D332F0(
                a2,
                106,
                (__int64)&v73,
                6,
                0,
                0,
                *(double *)a4.m128i_i64,
                *(double *)a5.m128_u64,
                v24,
                v41,
                v40,
                v44);
        v34 = v69;
        v46 = v45;
        v47 = (unsigned int)v79;
        v49 = v48;
        if ( (unsigned int)v79 >= HIDWORD(v79) )
        {
          sub_16CD150((__int64)&v78, &v80, 0, 16, v25, v26);
          v47 = (unsigned int)v79;
          v34 = v69;
        }
        a5 = (__m128)_mm_load_si128(&v70);
        *(__m128 *)&v78[2 * v47] = a5;
        v50 = (unsigned int)(v79 + 1);
        LODWORD(v79) = v50;
        if ( HIDWORD(v79) <= (unsigned int)v50 )
        {
          v70.m128i_i64[0] = v34;
          sub_16CD150((__int64)&v78, &v80, 0, 16, v25, v26);
          v50 = (unsigned int)v79;
          v34 = v70.m128i_i64[0];
        }
        v51 = &v78[2 * v50];
        ++v35;
        *v51 = (__int64)v46;
        v51[1] = v49;
        LODWORD(v79) = v79 + 1;
        if ( v35 == (_DWORD)v71 )
          goto LABEL_26;
      }
      v36 = (unsigned int)v79;
      if ( (unsigned int)v79 >= HIDWORD(v79) )
      {
        v70.m128i_i64[0] = v34;
        sub_16CD150((__int64)&v78, &v80, 0, 16, v25, v26);
        v36 = (unsigned int)v79;
        v34 = v70.m128i_i64[0];
      }
      a4 = _mm_loadu_si128((const __m128i *)v37);
      ++v35;
      *(__m128i *)&v78[2 * v36] = a4;
      LODWORD(v79) = v79 + 1;
    }
    while ( v35 != (_DWORD)v71 );
LABEL_26:
    v33 = v79;
    v7 = v34;
  }
  v52 = *(_QWORD *)(v7 + 104);
  v53 = *(_QWORD *)(v7 + 96);
  v72 = v33;
  v54 = *(_BYTE *)(v7 + 88);
  v70.m128i_i64[0] = v52;
  v71 = v78;
  v55 = sub_1D25C30((__int64)a2, v75, (unsigned int)v76);
  v67 = v53;
  v56 = a2;
  v57 = 0;
  v58 = 0;
  v62 = sub_1D24DC0(v56, 0x2Cu, (__int64)&v73, v55, v59, v70.m128i_i64[0], v71, v72, v54, v67);
  v63 = *(unsigned int *)(a3 + 8);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v63 )
    goto LABEL_39;
  while ( 1 )
  {
    v64 = (__int64 *)(*(_QWORD *)a3 + 16 * v63);
    *v64 = v62;
    v64[1] = v58;
    v63 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
    *(_DWORD *)(a3 + 8) = v63;
    if ( v57 == 1 )
      break;
    v57 = 1;
    v58 = 1;
    if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v63 )
    {
LABEL_39:
      sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v60, v61);
      v63 = *(unsigned int *)(a3 + 8);
    }
  }
  if ( (unsigned int)v63 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v60, v61);
    v63 = *(unsigned int *)(a3 + 8);
  }
  v65 = (__int64 *)(*(_QWORD *)a3 + 16 * v63);
  *v65 = v62;
  v66 = v78;
  v65[1] = 2;
  ++*(_DWORD *)(a3 + 8);
  if ( v66 != (__int64 *)&v80 )
    _libc_free((unsigned __int64)v66);
  if ( v75 != v77 )
    _libc_free((unsigned __int64)v75);
  if ( v73 )
    sub_161E7C0((__int64)&v73, v73);
}
