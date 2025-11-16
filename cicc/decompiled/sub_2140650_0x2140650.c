// Function: sub_2140650
// Address: 0x2140650
//
__int64 *__fastcall sub_2140650(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned int *v6; // rax
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r10
  __int64 v9; // r11
  __int64 v10; // rsi
  unsigned __int8 *v11; // rax
  unsigned __int64 v12; // rcx
  unsigned int v13; // r13d
  __int64 v14; // rax
  unsigned int v15; // edx
  unsigned int v16; // r12d
  unsigned __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int128 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // r12
  __m128i v25; // xmm0
  __int64 v26; // rsi
  unsigned __int8 *v27; // rax
  unsigned __int64 v28; // rcx
  unsigned int v29; // r13d
  __int64 v30; // rax
  unsigned int v31; // edx
  unsigned int v32; // r12d
  unsigned __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // r10
  const void ***v38; // rax
  __int64 v39; // rdx
  __int64 *v40; // r12
  __int64 v41; // rdx
  __int64 v42; // r13
  __int64 v43; // rsi
  __int64 *v44; // r14
  const void **v45; // r8
  __int64 v46; // rcx
  __int64 v47; // rsi
  __int64 *v48; // r12
  __int128 v50; // [rsp-10h] [rbp-A0h]
  __int128 v51; // [rsp-10h] [rbp-A0h]
  __int64 v52; // [rsp+8h] [rbp-88h]
  unsigned __int64 v53; // [rsp+10h] [rbp-80h]
  __int64 *v54; // [rsp+10h] [rbp-80h]
  unsigned __int64 v55; // [rsp+18h] [rbp-78h]
  __int64 *v56; // [rsp+20h] [rbp-70h]
  __int64 v57; // [rsp+20h] [rbp-70h]
  __int64 v58; // [rsp+28h] [rbp-68h]
  unsigned __int64 v59; // [rsp+30h] [rbp-60h]
  __int64 v60; // [rsp+30h] [rbp-60h]
  __int64 v61; // [rsp+30h] [rbp-60h]
  __int64 v62; // [rsp+38h] [rbp-58h]
  unsigned __int64 v63; // [rsp+40h] [rbp-50h]
  __int64 *v64; // [rsp+40h] [rbp-50h]
  const void **v65; // [rsp+40h] [rbp-50h]
  __int64 v66; // [rsp+50h] [rbp-40h] BYREF
  int v67; // [rsp+58h] [rbp-38h]

  v6 = *(unsigned int **)(a2 + 32);
  v7 = *(_QWORD *)v6;
  v8 = *(_QWORD *)v6;
  v9 = *((_QWORD *)v6 + 1);
  v10 = *(_QWORD *)(*(_QWORD *)v6 + 72LL);
  v11 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v6 + 40LL) + 16LL * v6[2]);
  v12 = *((_QWORD *)v11 + 1);
  v13 = *v11;
  v66 = v10;
  v63 = v12;
  if ( v10 )
  {
    v59 = v8;
    v62 = v9;
    sub_1623A60((__int64)&v66, v10, 2);
    v8 = v59;
    v9 = v62;
  }
  v58 = v9;
  v67 = *(_DWORD *)(v7 + 64);
  v14 = sub_2138AD0(a1, v8, v9);
  v16 = v15;
  v17 = v63;
  v60 = v14;
  v64 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v21 = sub_1D2EF30(v64, v13, v17, v18, v19, v20);
  v56 = sub_1D332F0(
          v64,
          148,
          (__int64)&v66,
          *(unsigned __int8 *)(*(_QWORD *)(v60 + 40) + 16LL * v16),
          *(const void ***)(*(_QWORD *)(v60 + 40) + 16LL * v16 + 8),
          0,
          a3,
          a4,
          a5,
          v60,
          v16 | v58 & 0xFFFFFFFF00000000LL,
          v21);
  v55 = v22;
  if ( v66 )
    sub_161E7C0((__int64)&v66, v66);
  v61 = (__int64)v56;
  v23 = *(_QWORD *)(a2 + 32);
  v24 = *(_QWORD *)(v23 + 40);
  v25 = _mm_loadu_si128((const __m128i *)(v23 + 40));
  v26 = *(_QWORD *)(v24 + 72);
  v27 = (unsigned __int8 *)(*(_QWORD *)(v24 + 40) + 16LL * *(unsigned int *)(v23 + 48));
  v28 = *((_QWORD *)v27 + 1);
  v29 = *v27;
  v66 = v26;
  v53 = v28;
  if ( v26 )
    sub_1623A60((__int64)&v66, v26, 2);
  v67 = *(_DWORD *)(v24 + 64);
  v30 = sub_2138AD0(a1, v25.m128i_u64[0], v25.m128i_i64[1]);
  v32 = v31;
  v33 = v53;
  v52 = v30;
  v54 = *(__int64 **)(a1 + 8);
  v37 = sub_1D2EF30(v54, v29, v33, v34, v35, v36);
  v38 = (const void ***)(*(_QWORD *)(v52 + 40) + 16LL * v32);
  *((_QWORD *)&v50 + 1) = v39;
  *(_QWORD *)&v50 = v37;
  v40 = sub_1D332F0(
          v54,
          148,
          (__int64)&v66,
          *(unsigned __int8 *)v38,
          v38[1],
          0,
          *(double *)v25.m128i_i64,
          a4,
          a5,
          v52,
          v32 | v25.m128i_i64[1] & 0xFFFFFFFF00000000LL,
          v50);
  v42 = v41;
  if ( v66 )
    sub_161E7C0((__int64)&v66, v66);
  v43 = *(_QWORD *)(a2 + 72);
  v44 = *(__int64 **)(a1 + 8);
  v45 = *(const void ***)(v56[5] + 16LL * (unsigned int)v55 + 8);
  v46 = *(unsigned __int8 *)(v56[5] + 16LL * (unsigned int)v55);
  v66 = v43;
  if ( v43 )
  {
    v57 = v46;
    v65 = v45;
    sub_1623A60((__int64)&v66, v43, 2);
    v46 = v57;
    v45 = v65;
  }
  *((_QWORD *)&v51 + 1) = v42;
  *(_QWORD *)&v51 = v40;
  v47 = *(unsigned __int16 *)(a2 + 24);
  v67 = *(_DWORD *)(a2 + 64);
  v48 = sub_1D332F0(v44, v47, (__int64)&v66, v46, v45, 0, *(double *)v25.m128i_i64, a4, a5, v61, v55, v51);
  if ( v66 )
    sub_161E7C0((__int64)&v66, v66);
  return v48;
}
