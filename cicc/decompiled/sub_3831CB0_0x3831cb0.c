// Function: sub_3831CB0
// Address: 0x3831cb0
//
unsigned __int8 *__fastcall sub_3831CB0(__int64 a1, unsigned __int64 a2, __m128i a3)
{
  unsigned __int64 v3; // r10
  unsigned int *v5; // rax
  unsigned __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rsi
  unsigned __int16 *v9; // rax
  __int64 v10; // r9
  unsigned int v11; // r14d
  __int64 v12; // rax
  unsigned int v13; // edx
  unsigned __int8 *v14; // rax
  unsigned __int64 v15; // r10
  __int64 v16; // rdx
  unsigned __int8 *v17; // rbx
  __int64 v18; // rax
  unsigned __int64 v19; // r9
  __int64 v20; // r15
  __int64 v21; // rsi
  unsigned __int16 *v22; // rax
  __int64 v23; // rcx
  unsigned int v24; // r14d
  __int64 v25; // rax
  unsigned int v26; // edx
  unsigned __int8 *v27; // rax
  unsigned __int64 v28; // r10
  unsigned __int8 *v29; // r14
  __int64 v30; // rdx
  __int64 v31; // r15
  __int64 v32; // rsi
  __int64 v33; // r9
  unsigned int v34; // r11d
  unsigned __int16 *v35; // rdx
  __int64 v36; // rbx
  __int64 v37; // rcx
  _QWORD *v38; // rdi
  unsigned int v39; // esi
  unsigned __int8 *v40; // r14
  __int64 v41; // rdx
  __int64 v42; // r15
  unsigned __int8 *v43; // r9
  __int64 v44; // rdx
  __int64 v45; // rbx
  __int64 v46; // rax
  __int128 v47; // rax
  __int64 v48; // r9
  unsigned int v49; // edx
  __int128 v51; // [rsp-20h] [rbp-A0h]
  __int128 v52; // [rsp-10h] [rbp-90h]
  unsigned __int64 v53; // [rsp+0h] [rbp-80h]
  __int64 v54; // [rsp+0h] [rbp-80h]
  unsigned __int64 v55; // [rsp+0h] [rbp-80h]
  unsigned __int64 v56; // [rsp+8h] [rbp-78h]
  unsigned __int64 v57; // [rsp+8h] [rbp-78h]
  unsigned int v58; // [rsp+8h] [rbp-78h]
  unsigned int v59; // [rsp+8h] [rbp-78h]
  __int64 v60; // [rsp+8h] [rbp-78h]
  unsigned __int64 v61; // [rsp+10h] [rbp-70h]
  __int64 v62; // [rsp+10h] [rbp-70h]
  unsigned __int64 v63; // [rsp+10h] [rbp-70h]
  __int64 v64; // [rsp+10h] [rbp-70h]
  unsigned __int64 v65; // [rsp+18h] [rbp-68h]
  __int64 v66; // [rsp+18h] [rbp-68h]
  __int64 v67; // [rsp+18h] [rbp-68h]
  unsigned __int64 v68; // [rsp+18h] [rbp-68h]
  _QWORD *v69; // [rsp+18h] [rbp-68h]
  __int64 v70; // [rsp+20h] [rbp-60h]
  __int64 v71; // [rsp+20h] [rbp-60h]
  __int128 v72; // [rsp+20h] [rbp-60h]
  __int128 v73; // [rsp+20h] [rbp-60h]
  unsigned __int64 v74; // [rsp+30h] [rbp-50h]
  __int64 v75; // [rsp+40h] [rbp-40h] BYREF
  int v76; // [rsp+48h] [rbp-38h]

  v3 = a2;
  v5 = *(unsigned int **)(a2 + 40);
  v6 = *(_QWORD *)v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = *(_QWORD *)(*(_QWORD *)v5 + 80LL);
  v9 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v5 + 48LL) + 16LL * v5[2]);
  v10 = *((_QWORD *)v9 + 1);
  v11 = *v9;
  v75 = v8;
  if ( v8 )
  {
    v65 = v3;
    v70 = v10;
    sub_B96E90((__int64)&v75, v8, 1);
    v3 = v65;
    v10 = v70;
  }
  v61 = v3;
  v71 = v10;
  v76 = *(_DWORD *)(v6 + 72);
  v12 = sub_37AE0F0(a1, v6, v7);
  v14 = sub_34070B0(*(_QWORD **)(a1 + 8), v12, v7 & 0xFFFFFFFF00000000LL | v13, (__int64)&v75, v11, v71, a3);
  v15 = v61;
  v66 = v16;
  v17 = v14;
  if ( v75 )
  {
    sub_B91220((__int64)&v75, v75);
    v15 = v61;
  }
  *(_QWORD *)&v72 = v17;
  *((_QWORD *)&v72 + 1) = v66;
  v18 = *(_QWORD *)(v15 + 40);
  v19 = *(_QWORD *)(v18 + 40);
  v20 = *(_QWORD *)(v18 + 48);
  v21 = *(_QWORD *)(v19 + 80);
  v22 = (unsigned __int16 *)(*(_QWORD *)(v19 + 48) + 16LL * *(unsigned int *)(v18 + 48));
  v23 = *((_QWORD *)v22 + 1);
  v24 = *v22;
  v75 = v21;
  v62 = v23;
  if ( v21 )
  {
    v53 = v19;
    v56 = v15;
    sub_B96E90((__int64)&v75, v21, 1);
    v19 = v53;
    v15 = v56;
  }
  v57 = v15;
  v76 = *(_DWORD *)(v19 + 72);
  v25 = sub_37AE0F0(a1, v19, v20);
  v27 = sub_34070B0(*(_QWORD **)(a1 + 8), v25, v20 & 0xFFFFFFFF00000000LL | v26, (__int64)&v75, v24, v62, a3);
  v28 = v57;
  v29 = v27;
  v31 = v30;
  if ( v75 )
  {
    sub_B91220((__int64)&v75, v75);
    v28 = v57;
  }
  v32 = *(_QWORD *)(v28 + 80);
  v33 = *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v28 + 40) + 48LL)
                  + 16LL * *(unsigned int *)(*(_QWORD *)(v28 + 40) + 8LL)
                  + 8);
  v34 = *(unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(v28 + 40) + 48LL)
                            + 16LL * *(unsigned int *)(*(_QWORD *)(v28 + 40) + 8LL));
  v35 = (unsigned __int16 *)(*((_QWORD *)v17 + 6) + 16LL * (unsigned int)v66);
  v36 = *((_QWORD *)v35 + 1);
  v37 = *v35;
  v75 = v32;
  if ( v32 )
  {
    v54 = v37;
    v58 = v34;
    v63 = v28;
    v67 = v33;
    sub_B96E90((__int64)&v75, v32, 1);
    v37 = v54;
    v34 = v58;
    v28 = v63;
    v33 = v67;
  }
  *((_QWORD *)&v52 + 1) = v31;
  *(_QWORD *)&v52 = v29;
  v38 = *(_QWORD **)(a1 + 8);
  v39 = (*(_DWORD *)(v28 + 24) != 77) + 56;
  v68 = v28;
  v59 = v34;
  v64 = v33;
  v76 = *(_DWORD *)(v28 + 72);
  v40 = sub_3406EB0(v38, v39, (__int64)&v75, v37, v36, v33, v72, v52);
  v42 = v41;
  v43 = sub_34070B0(*(_QWORD **)(a1 + 8), (__int64)v40, v41, (__int64)&v75, v59, v64, a3);
  v45 = v44;
  *((_QWORD *)&v73 + 1) = v44;
  v46 = *(_QWORD *)(v68 + 48);
  v55 = v68;
  *(_QWORD *)&v73 = v43;
  v69 = *(_QWORD **)(a1 + 8);
  v60 = *(_QWORD *)(v46 + 24);
  LODWORD(v64) = *(unsigned __int16 *)(v46 + 16);
  *(_QWORD *)&v47 = sub_33ED040(v69, 0x16u);
  *((_QWORD *)&v51 + 1) = v42;
  *(_QWORD *)&v51 = v40;
  v74 = sub_340F900(v69, 0xD0u, (__int64)&v75, v64, v60, v48, v73, v51, v47);
  sub_3760E70(a1, v55, 1, v74, v45 & 0xFFFFFFFF00000000LL | v49);
  if ( v75 )
    sub_B91220((__int64)&v75, v75);
  return v40;
}
