// Function: sub_383C140
// Address: 0x383c140
//
unsigned __int8 *__fastcall sub_383C140(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // r10
  __int64 v7; // r11
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  unsigned int v11; // r13d
  __int64 v12; // r14
  unsigned int v13; // edx
  unsigned int v14; // r12d
  unsigned __int64 v15; // rdx
  __int128 v16; // rax
  __int64 v17; // r9
  __int64 v18; // r9
  unsigned __int8 *v19; // r12
  __int64 v20; // rdx
  __int64 v21; // r13
  unsigned __int16 *v22; // rax
  unsigned int v23; // r11d
  __int64 v24; // rsi
  __int64 v25; // r8
  unsigned int v26; // r14d
  _BOOL4 v27; // esi
  unsigned __int8 *v28; // r12
  __int64 v29; // rdx
  __int64 v30; // r13
  __int128 v31; // rax
  __int64 v32; // r9
  __int128 v33; // rax
  __int64 v34; // r14
  __int128 v35; // rax
  __int64 v36; // r9
  unsigned int v37; // edx
  __int128 v39; // [rsp-20h] [rbp-B0h]
  __int128 v40; // [rsp-20h] [rbp-B0h]
  __int128 v41; // [rsp-20h] [rbp-B0h]
  __int128 v42; // [rsp-10h] [rbp-A0h]
  __int64 v43; // [rsp+0h] [rbp-90h]
  __int64 v44; // [rsp+0h] [rbp-90h]
  __int64 v45; // [rsp+8h] [rbp-88h]
  __int64 v46; // [rsp+8h] [rbp-88h]
  unsigned __int64 v47; // [rsp+10h] [rbp-80h]
  _QWORD *v48; // [rsp+10h] [rbp-80h]
  unsigned __int64 v49; // [rsp+10h] [rbp-80h]
  __int128 v50; // [rsp+10h] [rbp-80h]
  __int128 v51; // [rsp+20h] [rbp-70h]
  _QWORD *v52; // [rsp+20h] [rbp-70h]
  _QWORD *v53; // [rsp+20h] [rbp-70h]
  unsigned int v54; // [rsp+30h] [rbp-60h]
  unsigned int v55; // [rsp+30h] [rbp-60h]
  __int64 v56; // [rsp+30h] [rbp-60h]
  unsigned __int64 v57; // [rsp+40h] [rbp-50h]
  __int64 v58; // [rsp+50h] [rbp-40h] BYREF
  int v59; // [rsp+58h] [rbp-38h]

  HIWORD(v11) = 0;
  *(_QWORD *)&v51 = sub_383B380(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v3 = *(_QWORD *)(a2 + 40);
  *((_QWORD *)&v51 + 1) = v4;
  v5 = *(_QWORD *)(v3 + 40);
  v6 = v5;
  v7 = *(_QWORD *)(v3 + 48);
  v8 = *(_QWORD *)(v5 + 80);
  v9 = *(_QWORD *)(v5 + 48) + 16LL * *(unsigned int *)(v3 + 48);
  v10 = *(_QWORD *)(v9 + 8);
  LOWORD(v11) = *(_WORD *)v9;
  v58 = v8;
  v47 = v10;
  if ( v8 )
  {
    v45 = v7;
    sub_B96E90((__int64)&v58, v8, 1);
    v6 = v5;
    v7 = v45;
  }
  v46 = v7;
  v59 = *(_DWORD *)(v5 + 72);
  v12 = sub_37AE0F0(a1, v6, v7);
  v14 = v13;
  v15 = v47;
  v48 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)&v16 = sub_33F7D60(v48, v11, v15);
  *((_QWORD *)&v39 + 1) = v14 | v46 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v39 = v12;
  v19 = sub_3406EB0(
          v48,
          0xDEu,
          (__int64)&v58,
          *(unsigned __int16 *)(*(_QWORD *)(v12 + 48) + 16LL * v14),
          *(_QWORD *)(*(_QWORD *)(v12 + 48) + 16LL * v14 + 8),
          v17,
          v39,
          v16);
  v21 = v20;
  if ( v58 )
    sub_B91220((__int64)&v58, v58);
  v22 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL));
  v23 = *v22;
  v24 = *(_QWORD *)(a2 + 80);
  v25 = *(_QWORD *)(*(_QWORD *)(v51 + 48) + 16LL * DWORD2(v51) + 8);
  v26 = *(unsigned __int16 *)(*(_QWORD *)(v51 + 48) + 16LL * DWORD2(v51));
  v49 = *((_QWORD *)v22 + 1);
  v58 = v24;
  if ( v24 )
  {
    v54 = v23;
    v43 = v25;
    sub_B96E90((__int64)&v58, v24, 1);
    v23 = v54;
    v25 = v43;
  }
  v27 = *(_DWORD *)(a2 + 24) != 76;
  v44 = v25;
  v59 = *(_DWORD *)(a2 + 72);
  v55 = v23;
  *((_QWORD *)&v42 + 1) = v21;
  *(_QWORD *)&v42 = v19;
  v28 = sub_3406EB0(*(_QWORD **)(a1 + 8), v27 + 56, (__int64)&v58, v26, v25, v18, v51, v42);
  v30 = v29;
  v52 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)&v31 = sub_33F7D60(v52, v55, v49);
  *((_QWORD *)&v40 + 1) = v30;
  *(_QWORD *)&v40 = v28;
  *(_QWORD *)&v33 = sub_3406EB0(v52, 0xDEu, (__int64)&v58, v26, v44, v32, v40, v31);
  v34 = *((_QWORD *)&v33 + 1);
  v50 = v33;
  v53 = *(_QWORD **)(a1 + 8);
  v56 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 24LL);
  LODWORD(v44) = *(unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL);
  *(_QWORD *)&v35 = sub_33ED040(v53, 0x16u);
  *((_QWORD *)&v41 + 1) = v30;
  *(_QWORD *)&v41 = v28;
  v57 = sub_340F900(v53, 0xD0u, (__int64)&v58, v44, v56, v36, v50, v41, v35);
  sub_3760E70(a1, a2, 1, v57, v34 & 0xFFFFFFFF00000000LL | v37);
  if ( v58 )
    sub_B91220((__int64)&v58, v58);
  return v28;
}
