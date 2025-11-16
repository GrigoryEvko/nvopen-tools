// Function: sub_383C440
// Address: 0x383c440
//
unsigned __int8 *__fastcall sub_383C440(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // r10
  __int64 v7; // r11
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  unsigned int v11; // r14d
  __int64 v12; // rax
  unsigned int v13; // edx
  unsigned int v14; // ebx
  unsigned __int64 v15; // rdx
  __int128 v16; // rax
  __int64 v17; // rdx
  _QWORD *v18; // rbx
  __int64 v19; // rax
  __int16 v20; // dx
  __int64 v21; // rax
  __int16 v22; // dx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r9
  __int64 v26; // rsi
  unsigned int *v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // r13
  __int64 v30; // rsi
  unsigned __int8 *v31; // r13
  __int128 v33; // [rsp-20h] [rbp-C0h]
  unsigned int *v34; // [rsp+0h] [rbp-A0h]
  __int64 v35; // [rsp+8h] [rbp-98h]
  __int64 v36; // [rsp+10h] [rbp-90h]
  __int64 v37; // [rsp+10h] [rbp-90h]
  __int64 v38; // [rsp+18h] [rbp-88h]
  unsigned __int64 v39; // [rsp+20h] [rbp-80h]
  _QWORD *v40; // [rsp+20h] [rbp-80h]
  __int128 v41; // [rsp+20h] [rbp-80h]
  __int128 v42; // [rsp+30h] [rbp-70h]
  __int64 v43; // [rsp+40h] [rbp-60h] BYREF
  int v44; // [rsp+48h] [rbp-58h]
  __int64 v45; // [rsp+50h] [rbp-50h] BYREF
  __int64 v46; // [rsp+58h] [rbp-48h]
  __int16 v47; // [rsp+60h] [rbp-40h]
  __int64 v48; // [rsp+68h] [rbp-38h]

  HIWORD(v11) = 0;
  *(_QWORD *)&v42 = sub_383B380(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v3 = *(_QWORD *)(a2 + 40);
  *((_QWORD *)&v42 + 1) = v4;
  v5 = *(_QWORD *)(v3 + 40);
  v6 = v5;
  v7 = *(_QWORD *)(v3 + 48);
  v8 = *(_QWORD *)(v5 + 80);
  v9 = *(_QWORD *)(v5 + 48) + 16LL * *(unsigned int *)(v3 + 48);
  v10 = *(_QWORD *)(v9 + 8);
  LOWORD(v11) = *(_WORD *)v9;
  v45 = v8;
  v39 = v10;
  if ( v8 )
  {
    v38 = v7;
    sub_B96E90((__int64)&v45, v8, 1);
    v6 = v5;
    v7 = v38;
  }
  v35 = v7;
  LODWORD(v46) = *(_DWORD *)(v5 + 72);
  v12 = sub_37AE0F0(a1, v6, v7);
  v14 = v13;
  v15 = v39;
  v36 = v12;
  v40 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)&v16 = sub_33F7D60(v40, v11, v15);
  *((_QWORD *)&v33 + 1) = v14 | v35 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v33 = v36;
  *(_QWORD *)&v41 = sub_3406EB0(
                      v40,
                      0xDEu,
                      (__int64)&v45,
                      *(unsigned __int16 *)(*(_QWORD *)(v36 + 48) + 16LL * v14),
                      *(_QWORD *)(*(_QWORD *)(v36 + 48) + 16LL * v14 + 8),
                      v36,
                      v33,
                      v16);
  *((_QWORD *)&v41 + 1) = v17;
  if ( v45 )
    sub_B91220((__int64)&v45, v45);
  v18 = *(_QWORD **)(a1 + 8);
  v19 = *(_QWORD *)(v42 + 48) + 16LL * DWORD2(v42);
  v20 = *(_WORD *)v19;
  v46 = *(_QWORD *)(v19 + 8);
  v21 = *(_QWORD *)(a2 + 48);
  LOWORD(v45) = v20;
  v22 = *(_WORD *)(v21 + 16);
  v48 = *(_QWORD *)(v21 + 24);
  v23 = *(_QWORD *)(a2 + 40);
  v47 = v22;
  v37 = v23;
  v24 = sub_33E5830(v18, (unsigned __int16 *)&v45, 2);
  v26 = *(_QWORD *)(a2 + 80);
  v27 = (unsigned int *)v24;
  v29 = v28;
  v43 = v26;
  if ( v26 )
  {
    v34 = (unsigned int *)v24;
    sub_B96E90((__int64)&v43, v26, 1);
    v27 = v34;
  }
  v30 = *(unsigned int *)(a2 + 24);
  v44 = *(_DWORD *)(a2 + 72);
  v31 = sub_3412970(v18, v30, (__int64)&v43, v27, v29, v25, v42, v41, *(_OWORD *)(v37 + 80));
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
  sub_3760E70(a1, a2, 1, (unsigned __int64)v31, 1);
  return v31;
}
