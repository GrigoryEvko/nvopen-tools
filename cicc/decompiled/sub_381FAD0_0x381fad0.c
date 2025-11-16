// Function: sub_381FAD0
// Address: 0x381fad0
//
void __fastcall sub_381FAD0(__int64 *a1, __int64 a2, unsigned int *a3, __int64 a4, __m128i a5)
{
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // r8
  __int128 v9; // rax
  __int64 v10; // r15
  __int64 (__fastcall *v11)(__int64, __int64, __int64, __int64, __int64); // r14
  __int64 v12; // rax
  unsigned __int16 v13; // ax
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // r15
  __int128 v17; // rax
  __int64 v18; // r9
  unsigned int v19; // edx
  int v20; // r9d
  __int64 v21; // rdx
  int v22; // r9d
  unsigned __int8 *v23; // r14
  __int64 v24; // rdx
  __int64 v25; // r15
  __int64 v26; // rdx
  __int64 v27; // rdx
  char v28; // al
  __int64 v29; // rax
  __int128 v30; // rax
  __int64 v31; // r9
  __int64 v32; // rdx
  unsigned __int8 *v33; // r10
  __int64 v34; // r11
  unsigned int v35; // ecx
  __int64 v36; // r8
  __int64 v37; // r14
  __int64 v38; // r13
  __int64 v39; // r15
  __int16 v40; // ax
  unsigned int v41; // esi
  __int64 v42; // rax
  __int64 v43; // r8
  unsigned int v44; // edx
  unsigned __int8 *v45; // rax
  __int64 v46; // rsi
  int v47; // edx
  bool v48; // al
  __int64 v49; // rax
  __int128 v50; // [rsp-40h] [rbp-130h]
  __int128 v51; // [rsp-30h] [rbp-120h]
  __int128 v52; // [rsp-10h] [rbp-100h]
  _QWORD *v53; // [rsp+0h] [rbp-F0h]
  __int64 v54; // [rsp+0h] [rbp-F0h]
  __int64 v55; // [rsp+8h] [rbp-E8h]
  __int64 v56; // [rsp+8h] [rbp-E8h]
  unsigned int v57; // [rsp+8h] [rbp-E8h]
  unsigned int v58; // [rsp+8h] [rbp-E8h]
  __int64 v59; // [rsp+10h] [rbp-E0h]
  __int64 v60; // [rsp+10h] [rbp-E0h]
  unsigned __int8 *v61; // [rsp+10h] [rbp-E0h]
  __int64 v62; // [rsp+18h] [rbp-D8h]
  __int64 v63; // [rsp+20h] [rbp-D0h]
  __int128 v64; // [rsp+20h] [rbp-D0h]
  __int128 v65; // [rsp+30h] [rbp-C0h]
  _QWORD *v66; // [rsp+30h] [rbp-C0h]
  __int64 v69; // [rsp+70h] [rbp-80h] BYREF
  int v70; // [rsp+78h] [rbp-78h]
  __int64 v71; // [rsp+80h] [rbp-70h] BYREF
  __int64 v72; // [rsp+88h] [rbp-68h]
  __int64 v73; // [rsp+90h] [rbp-60h] BYREF
  char v74; // [rsp+98h] [rbp-58h]
  __int16 v75; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v76; // [rsp+A8h] [rbp-48h]
  __int64 v77; // [rsp+B0h] [rbp-40h]
  __int64 v78; // [rsp+B8h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 80);
  v69 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v69, v6, 1);
  v70 = *(_DWORD *)(a2 + 72);
  sub_375E510((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), (__int64)a3, a4);
  v7 = *(_QWORD *)(*(_QWORD *)a3 + 48LL) + 16LL * a3[2];
  v8 = *(_QWORD *)(v7 + 8);
  v53 = (_QWORD *)a1[1];
  LOWORD(v71) = *(_WORD *)v7;
  v72 = v8;
  *(_QWORD *)&v9 = sub_3400BD0((__int64)v53, 0, (__int64)&v69, (unsigned int)v71, v8, 0, a5, 0);
  v10 = *a1;
  v65 = v9;
  *(_QWORD *)&v9 = a1[1];
  v59 = v72;
  v11 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 528LL);
  v63 = v71;
  v55 = *(_QWORD *)(v9 + 64);
  v12 = sub_2E79000(*(__int64 **)(v9 + 40));
  v13 = v11(v10, v12, v55, v63, v59);
  v56 = v14;
  LODWORD(v59) = v13;
  v15 = *(_QWORD *)a3;
  v16 = *((_QWORD *)a3 + 1);
  *(_QWORD *)&v17 = sub_33ED040(v53, 0x16u);
  *((_QWORD *)&v50 + 1) = v16;
  *(_QWORD *)&v50 = v15;
  v60 = sub_340F900(v53, 0xD0u, (__int64)&v69, v59, v56, v18, v50, v65, v17);
  v57 = v19;
  *(_QWORD *)&v64 = sub_33FAF80(a1[1], 203, (__int64)&v69, (unsigned int)v71, v72, v20, a5);
  *((_QWORD *)&v64 + 1) = v21;
  v23 = sub_33FAF80(a1[1], *(unsigned int *)(a2 + 24), (__int64)&v69, (unsigned int)v71, v72, v22, a5);
  v25 = v24;
  v66 = (_QWORD *)a1[1];
  if ( (_WORD)v71 )
  {
    if ( (_WORD)v71 == 1 || (unsigned __int16)(v71 - 504) <= 7u )
      BUG();
    v49 = 16LL * ((unsigned __int16)v71 - 1);
    v27 = *(_QWORD *)&byte_444C4A0[v49];
    v28 = byte_444C4A0[v49 + 8];
  }
  else
  {
    v77 = sub_3007260((__int64)&v71);
    v78 = v26;
    v27 = v77;
    v28 = v78;
  }
  v73 = v27;
  v74 = v28;
  v29 = sub_CA1930(&v73);
  *(_QWORD *)&v30 = sub_3400BD0((__int64)v66, v29, (__int64)&v69, (unsigned int)v71, v72, 0, a5, 0);
  *((_QWORD *)&v51 + 1) = v25;
  *(_QWORD *)&v51 = v23;
  v33 = sub_3406EB0(v66, 0x38u, (__int64)&v69, (unsigned int)v71, v72, v31, v51, v30);
  v34 = v32;
  v35 = v71;
  v36 = v72;
  v37 = v60;
  v38 = *(_QWORD *)(v60 + 48) + 16LL * v57;
  v39 = v57;
  v40 = *(_WORD *)v38;
  v76 = *(_QWORD *)(v38 + 8);
  v75 = v40;
  if ( v40 )
  {
    v41 = ((unsigned __int16)(v40 - 17) < 0xD4u) + 205;
  }
  else
  {
    v54 = v72;
    v58 = v71;
    v61 = v33;
    v62 = v32;
    v48 = sub_30070B0((__int64)&v75);
    v36 = v54;
    v35 = v58;
    v33 = v61;
    v34 = v62;
    v41 = 205 - (!v48 - 1);
  }
  *((_QWORD *)&v52 + 1) = v34;
  *(_QWORD *)&v52 = v33;
  v42 = sub_340EC60(v66, v41, (__int64)&v69, v35, v36, 0, v37, v39, v64, v52);
  v43 = v72;
  *(_QWORD *)a3 = v42;
  a3[2] = v44;
  v45 = sub_3400BD0(a1[1], 0, (__int64)&v69, (unsigned int)v71, v43, 0, a5, 0);
  v46 = v69;
  *(_QWORD *)a4 = v45;
  *(_DWORD *)(a4 + 8) = v47;
  if ( v46 )
    sub_B91220((__int64)&v69, v46);
}
