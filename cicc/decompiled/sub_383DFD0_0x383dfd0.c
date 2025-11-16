// Function: sub_383DFD0
// Address: 0x383dfd0
//
unsigned __int8 *__fastcall sub_383DFD0(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned __int64 *v8; // rax
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // r15
  __int64 v11; // r12
  __int64 v12; // r13
  __int64 v13; // rdx
  __int64 v14; // r9
  __int64 v15; // rdx
  unsigned __int8 *v16; // r10
  __int64 v17; // r11
  __int64 v18; // rsi
  __int64 v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rsi
  unsigned __int8 *v22; // rax
  __int64 v23; // rsi
  unsigned __int8 *v24; // r12
  unsigned __int8 *v26; // r12
  __int64 v27; // rdx
  __int64 v28; // r13
  unsigned __int8 *v29; // rax
  __int64 v30; // rsi
  unsigned __int8 *v31; // r14
  __int64 v32; // rdx
  __int64 v33; // r15
  _QWORD *v34; // r10
  __int64 v35; // r8
  __int64 v36; // rcx
  unsigned int v37; // esi
  unsigned __int8 *v38; // rax
  __int128 v39; // [rsp-30h] [rbp-B0h]
  __int128 v40; // [rsp-20h] [rbp-A0h]
  __int128 v41; // [rsp-20h] [rbp-A0h]
  __int128 v42; // [rsp-20h] [rbp-A0h]
  __int128 v43; // [rsp-10h] [rbp-90h]
  __int128 v44; // [rsp-10h] [rbp-90h]
  __int128 v45; // [rsp-10h] [rbp-90h]
  unsigned __int8 *v46; // [rsp+0h] [rbp-80h]
  __int64 v47; // [rsp+8h] [rbp-78h]
  __int64 v48; // [rsp+10h] [rbp-70h]
  __int64 v49; // [rsp+18h] [rbp-68h]
  __int64 v50; // [rsp+18h] [rbp-68h]
  __int64 v51; // [rsp+20h] [rbp-60h]
  _QWORD *v52; // [rsp+28h] [rbp-58h]
  _QWORD *v53; // [rsp+28h] [rbp-58h]
  __int128 v54; // [rsp+30h] [rbp-50h]
  __int64 v55; // [rsp+40h] [rbp-40h] BYREF
  int v56; // [rsp+48h] [rbp-38h]

  v8 = *(unsigned __int64 **)(a2 + 40);
  if ( *(_DWORD *)(a2 + 64) == 2 )
  {
    v26 = sub_383B380(a1, *v8, v8[1]);
    v28 = v27;
    v29 = sub_383B380(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
    v30 = *(_QWORD *)(a2 + 80);
    v31 = v29;
    v33 = v32;
    v34 = *(_QWORD **)(a1 + 8);
    v35 = *(_QWORD *)(*((_QWORD *)v26 + 6) + 16LL * (unsigned int)v28 + 8);
    v36 = *(unsigned __int16 *)(*((_QWORD *)v26 + 6) + 16LL * (unsigned int)v28);
    v55 = v30;
    if ( v30 )
    {
      v50 = v36;
      v51 = v35;
      v53 = v34;
      sub_B96E90((__int64)&v55, v30, 1);
      v36 = v50;
      v35 = v51;
      v34 = v53;
    }
    *((_QWORD *)&v45 + 1) = v33;
    *(_QWORD *)&v45 = v31;
    v37 = *(_DWORD *)(a2 + 24);
    *((_QWORD *)&v42 + 1) = v28;
    *(_QWORD *)&v42 = v26;
    v56 = *(_DWORD *)(a2 + 72);
    v38 = sub_3406EB0(v34, v37, (__int64)&v55, v36, v35, (__int64)&v55, v42, v45);
    v23 = v55;
    v24 = v38;
    if ( v55 )
      goto LABEL_5;
  }
  else
  {
    v9 = v8[15];
    v10 = v8[16];
    v11 = v8[10];
    v12 = v8[11];
    *((_QWORD *)&v43 + 1) = v10;
    *(_QWORD *)&v43 = v9;
    *(_QWORD *)&v54 = sub_382E5B0(a1, *v8, v8[1], v11, v12, a3, a7, v43);
    *((_QWORD *)&v54 + 1) = v13;
    *((_QWORD *)&v40 + 1) = v10;
    *(_QWORD *)&v40 = v9;
    v16 = sub_382E5B0(
            a1,
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
            v11,
            v12,
            a3,
            v14,
            v40);
    v17 = v15;
    v52 = *(_QWORD **)(a1 + 8);
    v18 = *(_QWORD *)(a2 + 80);
    v19 = *(_QWORD *)(*(_QWORD *)(v54 + 48) + 16LL * DWORD2(v54) + 8);
    v20 = *(unsigned __int16 *)(*(_QWORD *)(v54 + 48) + 16LL * DWORD2(v54));
    v55 = v18;
    if ( v18 )
    {
      v48 = v20;
      v46 = v16;
      v47 = v15;
      v49 = v19;
      sub_B96E90((__int64)&v55, v18, 1);
      v20 = v48;
      v16 = v46;
      v17 = v47;
      v19 = v49;
    }
    v21 = *(unsigned int *)(a2 + 24);
    *((_QWORD *)&v44 + 1) = v10;
    *(_QWORD *)&v44 = v9;
    *((_QWORD *)&v41 + 1) = v12;
    *(_QWORD *)&v41 = v11;
    *((_QWORD *)&v39 + 1) = v17;
    *(_QWORD *)&v39 = v16;
    v56 = *(_DWORD *)(a2 + 72);
    v22 = sub_33FC130(v52, v21, (__int64)&v55, v20, v19, (__int64)&v55, v54, v39, v41, v44);
    v23 = v55;
    v24 = v22;
    if ( v55 )
LABEL_5:
      sub_B91220((__int64)&v55, v23);
  }
  return v24;
}
