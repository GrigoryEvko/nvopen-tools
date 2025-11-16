// Function: sub_381F530
// Address: 0x381f530
//
void __fastcall sub_381F530(__int64 *a1, __int64 a2, unsigned int *a3, unsigned __int8 **a4, __m128i a5)
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
  unsigned __int8 *v15; // r14
  unsigned __int8 *v16; // r15
  __int128 v17; // rax
  __int64 v18; // r9
  unsigned int v19; // edx
  int v20; // r9d
  unsigned __int8 *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r15
  unsigned __int8 *v24; // r14
  int v25; // r9d
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rdx
  char v29; // al
  __int64 v30; // rax
  __int128 v31; // rax
  __int64 v32; // r9
  __int64 v33; // rdx
  unsigned __int8 *v34; // r10
  __int64 v35; // r11
  unsigned int v36; // ecx
  __int64 v37; // r8
  __int64 v38; // r14
  __int64 v39; // r13
  __int64 v40; // r15
  __int16 v41; // ax
  unsigned int v42; // esi
  __int64 v43; // rax
  __int64 v44; // r8
  unsigned int v45; // edx
  unsigned __int8 *v46; // rax
  __int64 v47; // rsi
  int v48; // edx
  bool v49; // al
  __int64 v50; // rax
  __int128 v51; // [rsp-40h] [rbp-130h]
  __int128 v52; // [rsp-30h] [rbp-120h]
  __int128 v53; // [rsp-10h] [rbp-100h]
  _QWORD *v54; // [rsp+0h] [rbp-F0h]
  __int64 v55; // [rsp+0h] [rbp-F0h]
  __int64 v56; // [rsp+8h] [rbp-E8h]
  __int64 v57; // [rsp+8h] [rbp-E8h]
  unsigned int v58; // [rsp+8h] [rbp-E8h]
  unsigned int v59; // [rsp+8h] [rbp-E8h]
  __int64 v60; // [rsp+10h] [rbp-E0h]
  __int64 v61; // [rsp+10h] [rbp-E0h]
  unsigned __int8 *v62; // [rsp+10h] [rbp-E0h]
  __int64 v63; // [rsp+18h] [rbp-D8h]
  __int64 v64; // [rsp+20h] [rbp-D0h]
  __int128 v65; // [rsp+20h] [rbp-D0h]
  __int128 v66; // [rsp+30h] [rbp-C0h]
  _QWORD *v67; // [rsp+30h] [rbp-C0h]
  __int64 v70; // [rsp+70h] [rbp-80h] BYREF
  int v71; // [rsp+78h] [rbp-78h]
  __int64 v72; // [rsp+80h] [rbp-70h] BYREF
  __int64 v73; // [rsp+88h] [rbp-68h]
  __int64 v74; // [rsp+90h] [rbp-60h] BYREF
  char v75; // [rsp+98h] [rbp-58h]
  __int16 v76; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v77; // [rsp+A8h] [rbp-48h]
  __int64 v78; // [rsp+B0h] [rbp-40h]
  __int64 v79; // [rsp+B8h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 80);
  v70 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v70, v6, 1);
  v71 = *(_DWORD *)(a2 + 72);
  sub_375E510((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), (__int64)a3, (__int64)a4);
  v7 = *(_QWORD *)(*(_QWORD *)a3 + 48LL) + 16LL * a3[2];
  v8 = *(_QWORD *)(v7 + 8);
  v54 = (_QWORD *)a1[1];
  LOWORD(v72) = *(_WORD *)v7;
  v73 = v8;
  *(_QWORD *)&v9 = sub_3400BD0((__int64)v54, 0, (__int64)&v70, (unsigned int)v72, v8, 0, a5, 0);
  v10 = *a1;
  v66 = v9;
  *(_QWORD *)&v9 = a1[1];
  v60 = v73;
  v11 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 528LL);
  v64 = v72;
  v56 = *(_QWORD *)(v9 + 64);
  v12 = sub_2E79000(*(__int64 **)(v9 + 40));
  v13 = v11(v10, v12, v56, v64, v60);
  v57 = v14;
  LODWORD(v60) = v13;
  v15 = *a4;
  v16 = a4[1];
  *(_QWORD *)&v17 = sub_33ED040(v54, 0x16u);
  *((_QWORD *)&v51 + 1) = v16;
  *(_QWORD *)&v51 = v15;
  v61 = sub_340F900(v54, 0xD0u, (__int64)&v70, v60, v57, v18, v51, v66, v17);
  v58 = v19;
  v21 = sub_33FAF80(a1[1], *(unsigned int *)(a2 + 24), (__int64)&v70, (unsigned int)v72, v73, v20, a5);
  v23 = v22;
  v24 = v21;
  *(_QWORD *)&v65 = sub_33FAF80(a1[1], 204, (__int64)&v70, (unsigned int)v72, v73, v25, a5);
  *((_QWORD *)&v65 + 1) = v26;
  v67 = (_QWORD *)a1[1];
  if ( (_WORD)v72 )
  {
    if ( (_WORD)v72 == 1 || (unsigned __int16)(v72 - 504) <= 7u )
      BUG();
    v50 = 16LL * ((unsigned __int16)v72 - 1);
    v28 = *(_QWORD *)&byte_444C4A0[v50];
    v29 = byte_444C4A0[v50 + 8];
  }
  else
  {
    v78 = sub_3007260((__int64)&v72);
    v79 = v27;
    v28 = v78;
    v29 = v79;
  }
  v74 = v28;
  v75 = v29;
  v30 = sub_CA1930(&v74);
  *(_QWORD *)&v31 = sub_3400BD0((__int64)v67, v30, (__int64)&v70, (unsigned int)v72, v73, 0, a5, 0);
  *((_QWORD *)&v52 + 1) = v23;
  *(_QWORD *)&v52 = v24;
  v34 = sub_3406EB0(v67, 0x38u, (__int64)&v70, (unsigned int)v72, v73, v32, v52, v31);
  v35 = v33;
  v36 = v72;
  v37 = v73;
  v38 = v61;
  v39 = *(_QWORD *)(v61 + 48) + 16LL * v58;
  v40 = v58;
  v41 = *(_WORD *)v39;
  v77 = *(_QWORD *)(v39 + 8);
  v76 = v41;
  if ( v41 )
  {
    v42 = ((unsigned __int16)(v41 - 17) < 0xD4u) + 205;
  }
  else
  {
    v55 = v73;
    v59 = v72;
    v62 = v34;
    v63 = v33;
    v49 = sub_30070B0((__int64)&v76);
    v37 = v55;
    v36 = v59;
    v34 = v62;
    v35 = v63;
    v42 = 205 - (!v49 - 1);
  }
  *((_QWORD *)&v53 + 1) = v35;
  *(_QWORD *)&v53 = v34;
  v43 = sub_340EC60(v67, v42, (__int64)&v70, v36, v37, 0, v38, v40, v65, v53);
  v44 = v73;
  *(_QWORD *)a3 = v43;
  a3[2] = v45;
  v46 = sub_3400BD0(a1[1], 0, (__int64)&v70, (unsigned int)v72, v44, 0, a5, 0);
  v47 = v70;
  *a4 = v46;
  *((_DWORD *)a4 + 2) = v48;
  if ( v47 )
    sub_B91220((__int64)&v70, v47);
}
