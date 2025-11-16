// Function: sub_2140C50
// Address: 0x2140c50
//
__int64 *__fastcall sub_2140C50(__int64 a1, unsigned __int64 a2, __m128 a3, double a4, __m128i a5)
{
  unsigned __int64 v5; // r11
  unsigned int *v6; // rax
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rsi
  unsigned __int8 *v11; // rax
  unsigned __int64 v12; // rcx
  unsigned int v13; // r12d
  unsigned int v14; // edx
  unsigned int v15; // ebx
  unsigned __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int128 v20; // rax
  __int64 *v21; // rax
  unsigned __int64 v22; // r11
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  unsigned __int64 v25; // rbx
  __int64 v26; // r14
  __int64 v27; // rsi
  unsigned __int8 *v28; // rax
  unsigned __int64 v29; // rcx
  unsigned int v30; // r12d
  __int64 v31; // rax
  unsigned int v32; // edx
  unsigned int v33; // ebx
  unsigned __int64 v34; // rdx
  __int64 v35; // r8
  __int64 v36; // r9
  __int128 v37; // rax
  __int64 *v38; // rax
  unsigned __int64 v39; // r11
  __int64 *v40; // r12
  __int64 v41; // rdx
  __int64 v42; // r13
  __int64 v43; // rsi
  unsigned int v44; // r14d
  const void **v45; // r8
  unsigned int v46; // ebx
  bool v47; // si
  __int64 *v48; // r12
  unsigned __int64 v49; // rdx
  unsigned __int64 v50; // r13
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int128 v54; // rax
  __int64 *v55; // r9
  __int64 v56; // rdx
  __int64 v57; // rbx
  __int64 v58; // rax
  const void **v59; // r14
  __int64 v60; // r8
  __int64 v61; // rax
  __int64 v62; // rdx
  unsigned int v63; // edx
  const __m128i *v64; // r9
  __int128 v66; // [rsp-20h] [rbp-B0h]
  __int128 v67; // [rsp-10h] [rbp-A0h]
  unsigned __int64 v68; // [rsp+0h] [rbp-90h]
  unsigned __int64 v69; // [rsp+8h] [rbp-88h]
  __int64 v70; // [rsp+8h] [rbp-88h]
  unsigned __int64 v71; // [rsp+10h] [rbp-80h]
  unsigned __int64 v72; // [rsp+10h] [rbp-80h]
  __int64 *v73; // [rsp+10h] [rbp-80h]
  unsigned __int64 v74; // [rsp+10h] [rbp-80h]
  unsigned __int64 v75; // [rsp+10h] [rbp-80h]
  unsigned __int64 v76; // [rsp+18h] [rbp-78h]
  unsigned __int64 v77; // [rsp+18h] [rbp-78h]
  unsigned __int64 v78; // [rsp+18h] [rbp-78h]
  unsigned __int64 v79; // [rsp+18h] [rbp-78h]
  unsigned __int64 v80; // [rsp+20h] [rbp-70h]
  __int64 *v81; // [rsp+20h] [rbp-70h]
  __int64 v82; // [rsp+20h] [rbp-70h]
  __int64 *v83; // [rsp+20h] [rbp-70h]
  unsigned __int64 v84; // [rsp+20h] [rbp-70h]
  unsigned __int64 v85; // [rsp+28h] [rbp-68h]
  __int16 *v86; // [rsp+28h] [rbp-68h]
  unsigned __int64 v87; // [rsp+30h] [rbp-60h]
  __int64 v88; // [rsp+30h] [rbp-60h]
  __int64 *v89; // [rsp+30h] [rbp-60h]
  const void **v90; // [rsp+30h] [rbp-60h]
  const void **v91; // [rsp+30h] [rbp-60h]
  __int64 *v92; // [rsp+30h] [rbp-60h]
  __int64 *v93; // [rsp+40h] [rbp-50h]
  __int64 v94; // [rsp+50h] [rbp-40h] BYREF
  int v95; // [rsp+58h] [rbp-38h]

  v5 = a2;
  v6 = *(unsigned int **)(a2 + 32);
  v7 = *(_QWORD *)v6;
  v8 = *(_QWORD *)v6;
  v9 = *((_QWORD *)v6 + 1);
  v10 = *(_QWORD *)(*(_QWORD *)v6 + 72LL);
  v11 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v6 + 40LL) + 16LL * v6[2]);
  v12 = *((_QWORD *)v11 + 1);
  v13 = *v11;
  v94 = v10;
  v80 = v12;
  if ( v10 )
  {
    v87 = v5;
    sub_1623A60((__int64)&v94, v10, 2);
    v5 = v87;
  }
  v71 = v5;
  v95 = *(_DWORD *)(v7 + 64);
  v88 = sub_2138AD0(a1, v8, v9);
  v15 = v14;
  v16 = v80;
  v81 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v20 = sub_1D2EF30(v81, v13, v16, v17, v18, v19);
  v21 = sub_1D332F0(
          v81,
          148,
          (__int64)&v94,
          *(unsigned __int8 *)(*(_QWORD *)(v88 + 40) + 16LL * v15),
          *(const void ***)(*(_QWORD *)(v88 + 40) + 16LL * v15 + 8),
          0,
          *(double *)a3.m128_u64,
          a4,
          a5,
          v88,
          v15 | v9 & 0xFFFFFFFF00000000LL,
          v20);
  v22 = v71;
  v89 = v21;
  v76 = v23;
  if ( v94 )
  {
    sub_161E7C0((__int64)&v94, v94);
    v22 = v71;
  }
  v82 = (__int64)v89;
  v85 = v76;
  v24 = *(_QWORD *)(v22 + 32);
  v25 = *(_QWORD *)(v24 + 40);
  v26 = *(_QWORD *)(v24 + 48);
  v27 = *(_QWORD *)(v25 + 72);
  v28 = (unsigned __int8 *)(*(_QWORD *)(v25 + 40) + 16LL * *(unsigned int *)(v24 + 48));
  v29 = *((_QWORD *)v28 + 1);
  v30 = *v28;
  v94 = v27;
  v72 = v29;
  if ( v27 )
  {
    v69 = v22;
    sub_1623A60((__int64)&v94, v27, 2);
    v22 = v69;
  }
  v68 = v22;
  v95 = *(_DWORD *)(v25 + 64);
  v31 = sub_2138AD0(a1, v25, v26);
  v33 = v32;
  v34 = v72;
  v70 = v31;
  v73 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v37 = sub_1D2EF30(v73, v30, v34, a1, v35, v36);
  v38 = sub_1D332F0(
          v73,
          148,
          (__int64)&v94,
          *(unsigned __int8 *)(*(_QWORD *)(v70 + 40) + 16LL * v33),
          *(const void ***)(*(_QWORD *)(v70 + 40) + 16LL * v33 + 8),
          0,
          *(double *)a3.m128_u64,
          a4,
          a5,
          v70,
          v33 | v26 & 0xFFFFFFFF00000000LL,
          v37);
  v39 = v68;
  v40 = v38;
  v42 = v41;
  if ( v94 )
  {
    sub_161E7C0((__int64)&v94, v94);
    v39 = v68;
  }
  v43 = *(_QWORD *)(v39 + 72);
  v44 = *(unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(v39 + 32) + 40LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(v39 + 32) + 8LL));
  v74 = *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v39 + 32) + 40LL)
                  + 16LL * *(unsigned int *)(*(_QWORD *)(v39 + 32) + 8LL)
                  + 8);
  v45 = *(const void ***)(v89[5] + 16LL * (unsigned int)v76 + 8);
  v46 = *(unsigned __int8 *)(v89[5] + 16LL * (unsigned int)v76);
  v94 = v43;
  if ( v43 )
  {
    v77 = v39;
    v90 = v45;
    sub_1623A60((__int64)&v94, v43, 2);
    v39 = v77;
    v45 = v90;
  }
  v47 = *(_WORD *)(v39 + 24) != 70;
  v78 = v39;
  v95 = *(_DWORD *)(v39 + 64);
  v91 = v45;
  *((_QWORD *)&v67 + 1) = v42;
  *(_QWORD *)&v67 = v40;
  v48 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          (unsigned int)v47 + 52,
          (__int64)&v94,
          v46,
          v45,
          0,
          *(double *)a3.m128_u64,
          a4,
          a5,
          v82,
          v85,
          v67);
  v50 = v49;
  v83 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v54 = sub_1D2EF30(v83, v44, v74, v51, v52, v53);
  v55 = sub_1D332F0(v83, 148, (__int64)&v94, v46, v91, 0, *(double *)a3.m128_u64, a4, a5, (__int64)v48, v50, v54);
  v57 = v56;
  v75 = v78;
  v58 = *(_QWORD *)(v78 + 40);
  v84 = (unsigned __int64)v55;
  v86 = (__int16 *)v56;
  v59 = *(const void ***)(v58 + 24);
  v92 = *(__int64 **)(a1 + 8);
  v79 = *(unsigned __int8 *)(v58 + 16);
  v61 = sub_1D28D50(v92, 0x16u, v56, v79, v60, (__int64)v55);
  *((_QWORD *)&v66 + 1) = v50;
  *(_QWORD *)&v66 = v48;
  v93 = sub_1D3A900(v92, 0x89u, (__int64)&v94, v79, v59, 0, a3, a4, a5, v84, v86, v66, v61, v62);
  sub_2013400(a1, v75, 1, (__int64)v93, (__m128i *)(v57 & 0xFFFFFFFF00000000LL | v63), v64);
  if ( v94 )
    sub_161E7C0((__int64)&v94, v94);
  return v48;
}
