// Function: sub_383C920
// Address: 0x383c920
//
unsigned __int8 *__fastcall sub_383C920(__int64 a1, unsigned __int64 a2, int a3)
{
  unsigned __int64 v3; // r10
  const __m128i *v4; // rax
  __int64 v5; // rsi
  __m128i v6; // xmm0
  __int64 v7; // r14
  unsigned __int32 v8; // ebx
  unsigned __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // rax
  bool v12; // zf
  __int16 v13; // dx
  unsigned __int64 v14; // rax
  unsigned __int8 *v15; // rax
  unsigned int v16; // edx
  __int64 v17; // rbx
  unsigned __int8 *v18; // r14
  unsigned int v19; // edx
  unsigned __int8 *v20; // r12
  unsigned __int64 v21; // r13
  unsigned int *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // r9
  unsigned __int8 *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r14
  __int64 v29; // rbx
  unsigned __int8 *v30; // r13
  _QWORD *v31; // r12
  __int128 v32; // rax
  __int64 v33; // r9
  __int128 v34; // rax
  _QWORD *v35; // r12
  __int64 v36; // rbx
  __int128 v37; // rax
  __int64 v38; // rax
  unsigned int v39; // edx
  __int64 v40; // r9
  unsigned __int64 v41; // r10
  unsigned __int8 *v42; // rax
  unsigned __int8 *v43; // rax
  unsigned int v44; // edx
  __int64 v46; // rbx
  unsigned int v47; // eax
  _QWORD *v48; // r12
  __int128 v49; // rax
  __int64 v50; // r9
  unsigned __int8 *v51; // rax
  _QWORD *v52; // r12
  __int64 v53; // rdx
  unsigned __int16 *v54; // rax
  __int64 v55; // rax
  __int64 v56; // rdx
  __int128 v57; // rax
  __int64 v58; // r9
  unsigned __int8 *v59; // rax
  unsigned int v60; // edx
  __int128 v61; // [rsp-40h] [rbp-120h]
  __int128 v62; // [rsp-20h] [rbp-100h]
  __int128 v63; // [rsp-20h] [rbp-100h]
  __int128 v64; // [rsp-20h] [rbp-100h]
  __int128 v65; // [rsp-10h] [rbp-F0h]
  __int128 v66; // [rsp-10h] [rbp-F0h]
  unsigned __int8 *v67; // [rsp+0h] [rbp-E0h]
  __int128 v68; // [rsp+10h] [rbp-D0h]
  __int128 v69; // [rsp+10h] [rbp-D0h]
  __int64 v70; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v71; // [rsp+28h] [rbp-B8h]
  __int64 v72; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v73; // [rsp+30h] [rbp-B0h]
  unsigned int v74; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v75; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v76; // [rsp+40h] [rbp-A0h]
  unsigned __int8 *v77; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v78; // [rsp+40h] [rbp-A0h]
  __int128 v79; // [rsp+50h] [rbp-90h]
  __int64 v80; // [rsp+90h] [rbp-50h] BYREF
  int v81; // [rsp+98h] [rbp-48h]
  unsigned int v82; // [rsp+A0h] [rbp-40h] BYREF
  unsigned __int64 v83; // [rsp+A8h] [rbp-38h]

  v3 = a2;
  if ( a3 == 1 )
    return sub_38159C0((__int64 *)a1, a2);
  v4 = *(const __m128i **)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = _mm_loadu_si128(v4);
  v7 = v4->m128i_i64[0];
  v80 = v5;
  v8 = v4->m128i_u32[2];
  v9 = v4[2].m128i_u64[1];
  v10 = v4[3].m128i_i64[0];
  if ( v5 )
  {
    v75 = v3;
    sub_B96E90((__int64)&v80, v5, 1);
    v3 = v75;
  }
  v76 = v3;
  v81 = *(_DWORD *)(v3 + 72);
  v11 = *(_QWORD *)(v7 + 48) + 16LL * v8;
  v12 = *(_DWORD *)(v3 + 24) == 80;
  v13 = *(_WORD *)v11;
  v14 = *(_QWORD *)(v11 + 8);
  LOWORD(v82) = v13;
  v83 = v14;
  if ( v12 )
  {
    v59 = sub_383B380(a1, v6.m128i_u64[0], v6.m128i_i64[1]);
    v17 = v60;
    v18 = v59;
    v20 = sub_383B380(a1, v9, v10);
  }
  else
  {
    v15 = sub_37AF270(a1, v6.m128i_u64[0], v6.m128i_i64[1], v6);
    v17 = v16;
    v18 = v15;
    v20 = sub_37AF270(a1, v9, v10, v6);
  }
  v21 = v19 | v10 & 0xFFFFFFFF00000000LL;
  v22 = (unsigned int *)sub_33E5110(
                          *(__int64 **)(a1 + 8),
                          *(unsigned __int16 *)(*((_QWORD *)v18 + 6) + 16 * v17),
                          *(_QWORD *)(*((_QWORD *)v18 + 6) + 16 * v17 + 8),
                          *(unsigned __int16 *)(*(_QWORD *)(v76 + 48) + 16LL),
                          *(_QWORD *)(*(_QWORD *)(v76 + 48) + 24LL));
  v23 = *(unsigned int *)(v76 + 24);
  *((_QWORD *)&v65 + 1) = v21;
  *(_QWORD *)&v65 = v20;
  v73 = v76;
  v26 = sub_3411F20(
          *(_QWORD **)(a1 + 8),
          v23,
          (__int64)&v80,
          v22,
          v24,
          v25,
          __PAIR128__(v17 | v6.m128i_i64[1] & 0xFFFFFFFF00000000LL, (unsigned __int64)v18),
          v65);
  v28 = v27;
  v29 = (unsigned int)v27;
  v77 = v26;
  v30 = v26;
  if ( *(_DWORD *)(v73 + 24) == 81 )
  {
    v46 = 16LL * (unsigned int)v27;
    v47 = sub_32844A0((unsigned __int16 *)&v82, v23);
    v48 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)&v49 = sub_3400E40(
                        (__int64)v48,
                        v47,
                        *(unsigned __int16 *)(v46 + *((_QWORD *)v77 + 6)),
                        *(_QWORD *)(v46 + *((_QWORD *)v77 + 6) + 8),
                        (__int64)&v80,
                        v6);
    *((_QWORD *)&v64 + 1) = v28;
    *(_QWORD *)&v64 = v30;
    v51 = sub_3406EB0(
            v48,
            0xC0u,
            (__int64)&v80,
            *(unsigned __int16 *)(*((_QWORD *)v77 + 6) + v46),
            *(_QWORD *)(*((_QWORD *)v77 + 6) + v46 + 8),
            v50,
            v64,
            v49);
    v52 = *(_QWORD **)(a1 + 8);
    v72 = v53;
    v67 = v51;
    v54 = (unsigned __int16 *)(*((_QWORD *)v51 + 6) + 16LL * (unsigned int)v53);
    *(_QWORD *)&v69 = sub_3400BD0((__int64)v52, 0, (__int64)&v80, *v54, *((_QWORD *)v54 + 1), 0, v6, 0);
    v55 = *(_QWORD *)(v73 + 48);
    LOWORD(v46) = *(_WORD *)(v55 + 16);
    *((_QWORD *)&v69 + 1) = v56;
    v70 = *(_QWORD *)(v55 + 24);
    *(_QWORD *)&v57 = sub_33ED040(v52, 0x16u);
    *((_QWORD *)&v61 + 1) = v72;
    *(_QWORD *)&v61 = v67;
    v38 = sub_340F900(v52, 0xD0u, (__int64)&v80, (unsigned __int16)v46, v70, v58, v61, v69, v57);
    v41 = v73;
  }
  else
  {
    v31 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)&v32 = sub_33F7D60(v31, v82, v83);
    *((_QWORD *)&v62 + 1) = v28;
    *(_QWORD *)&v62 = v30;
    *(_QWORD *)&v34 = sub_3406EB0(
                        v31,
                        0xDEu,
                        (__int64)&v80,
                        *(unsigned __int16 *)(*((_QWORD *)v77 + 6) + 16 * v29),
                        *(_QWORD *)(*((_QWORD *)v77 + 6) + 16 * v29 + 8),
                        v33,
                        v62,
                        v32);
    v35 = *(_QWORD **)(a1 + 8);
    v68 = v34;
    *(_QWORD *)&v34 = *(_QWORD *)(v73 + 48);
    v71 = v73;
    v36 = *(_QWORD *)(v34 + 24);
    v74 = *(unsigned __int16 *)(v34 + 16);
    *(_QWORD *)&v37 = sub_33ED040(v35, 0x16u);
    *((_QWORD *)&v63 + 1) = v28;
    *(_QWORD *)&v63 = v30;
    v38 = sub_340F900(v35, 0xD0u, (__int64)&v80, v74, v36, *((__int64 *)&v68 + 1), v68, v63, v37);
    v41 = v71;
  }
  *(_QWORD *)&v79 = v38;
  *((_QWORD *)&v79 + 1) = v39;
  v42 = v77;
  v78 = v41;
  *((_QWORD *)&v66 + 1) = 1;
  *(_QWORD *)&v66 = v42;
  v43 = sub_3406EB0(
          *(_QWORD **)(a1 + 8),
          0xBBu,
          (__int64)&v80,
          *(unsigned __int16 *)(*(_QWORD *)(v41 + 48) + 16LL),
          *(_QWORD *)(*(_QWORD *)(v41 + 48) + 24LL),
          v40,
          v79,
          v66);
  sub_3760E70(a1, v78, 1, (unsigned __int64)v43, v44 | *((_QWORD *)&v79 + 1) & 0xFFFFFFFF00000000LL);
  if ( v80 )
    sub_B91220((__int64)&v80, v80);
  return v30;
}
