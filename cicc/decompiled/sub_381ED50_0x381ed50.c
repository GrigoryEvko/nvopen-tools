// Function: sub_381ED50
// Address: 0x381ed50
//
void __fastcall sub_381ED50(__int64 *a1, __int64 a2, unsigned int *a3, __int64 a4)
{
  __int64 v6; // rsi
  __m128i v7; // xmm0
  __int64 v8; // rdi
  __int64 v9; // rax
  __int16 v10; // dx
  __int64 v11; // rax
  unsigned int v12; // eax
  int v13; // r9d
  unsigned __int16 v14; // r13
  unsigned __int64 v15; // r12
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rdx
  unsigned int v20; // r12d
  unsigned int v21; // r9d
  __int64 v22; // r14
  __int64 v23; // r13
  __int64 v24; // rbx
  __int64 v25; // r8
  __int64 *v26; // rbx
  char v27; // al
  _QWORD *v28; // r14
  unsigned __int16 *v29; // rax
  __int64 v30; // r15
  __int128 v31; // rax
  __int64 v32; // r9
  unsigned __int8 *v33; // rax
  __int64 v34; // rdx
  _QWORD *v35; // r13
  unsigned __int8 *v36; // r14
  __int64 v37; // rdx
  __int64 v38; // r15
  __int64 v39; // rax
  __int64 v40; // rax
  unsigned __int16 v41; // ax
  __int64 v42; // rdx
  __int128 v43; // rax
  __int64 v44; // r9
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r15
  __int64 v48; // r14
  __int64 v49; // rax
  int v50; // ecx
  unsigned int v51; // edx
  int v52; // edx
  __int64 v53; // rax
  __int64 v54; // rsi
  int v55; // edx
  __int64 (__fastcall *v56)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // rdx
  unsigned __int8 *v61; // rax
  __int64 v62; // rcx
  __int64 v63; // r8
  unsigned int v64; // edx
  int v65; // edx
  __int128 v66; // rax
  __int64 v67; // rax
  __int128 v68; // rax
  __int64 v69; // r9
  unsigned __int8 *v70; // rax
  __int64 *v71; // r14
  __int64 v72; // rdx
  __int64 v73; // r13
  unsigned __int8 *v74; // r12
  unsigned int v75; // eax
  __int64 v76; // rdx
  unsigned int *v77; // r14
  __int64 v78; // rdx
  __int64 v79; // r9
  unsigned __int8 *v80; // rax
  __int64 v81; // r8
  unsigned int v82; // edx
  __int64 v83; // r9
  int v84; // edx
  __int64 v85; // r9
  unsigned int v86; // edx
  __int64 v87; // r9
  int v88; // edx
  __int64 v89; // rdx
  __int128 v90; // [rsp-20h] [rbp-190h]
  __int128 v91; // [rsp-10h] [rbp-180h]
  __int128 v92; // [rsp+0h] [rbp-170h]
  __int128 v93; // [rsp+0h] [rbp-170h]
  __int128 v94; // [rsp+0h] [rbp-170h]
  __int128 v95; // [rsp+0h] [rbp-170h]
  __int64 v96; // [rsp+18h] [rbp-158h]
  __int64 (__fastcall *v97)(__int64, __int64, __int64, __int64, __int64); // [rsp+20h] [rbp-150h]
  __int64 *v98; // [rsp+28h] [rbp-148h]
  __int64 v99; // [rsp+28h] [rbp-148h]
  __int64 v100; // [rsp+28h] [rbp-148h]
  __int64 v101; // [rsp+30h] [rbp-140h]
  unsigned int v102; // [rsp+40h] [rbp-130h]
  unsigned int v103; // [rsp+40h] [rbp-130h]
  __int64 v104; // [rsp+40h] [rbp-130h]
  __int128 v105; // [rsp+40h] [rbp-130h]
  __int64 v106; // [rsp+40h] [rbp-130h]
  unsigned int v109; // [rsp+78h] [rbp-F8h]
  unsigned __int8 *v110; // [rsp+90h] [rbp-E0h]
  __int64 v111; // [rsp+E0h] [rbp-90h] BYREF
  int v112; // [rsp+E8h] [rbp-88h]
  __int64 v113; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v114; // [rsp+F8h] [rbp-78h]
  __int128 v115; // [rsp+100h] [rbp-70h] BYREF
  unsigned __int64 v116; // [rsp+110h] [rbp-60h]
  __int64 v117; // [rsp+118h] [rbp-58h]
  __int128 v118; // [rsp+120h] [rbp-50h] BYREF
  __int64 v119; // [rsp+130h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 80);
  v111 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v111, v6, 1);
  v112 = *(_DWORD *)(a2 + 72);
  v7 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  sub_375E510((__int64)a1, v7.m128i_u64[0], v7.m128i_i64[1], (__int64)a3, a4);
  v8 = a1[1];
  v9 = *(_QWORD *)(*(_QWORD *)a3 + 48LL) + 16LL * a3[2];
  v10 = *(_WORD *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  LOWORD(v113) = v10;
  v114 = v11;
  v12 = sub_33D4D80(v8, v7.m128i_i64[0], v7.m128i_i64[1], 0);
  v14 = v113;
  v15 = v12;
  if ( (_WORD)v113 )
  {
    if ( (unsigned __int16)(v113 - 17) <= 0xD3u )
    {
      v16 = 0;
      v14 = word_4456580[(unsigned __int16)v113 - 1];
LABEL_6:
      LOWORD(v118) = v14;
      *((_QWORD *)&v118 + 1) = v16;
      if ( !v14 )
        goto LABEL_7;
LABEL_19:
      if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
LABEL_28:
        BUG();
      v18 = a1[1];
      if ( v15 <= *(_QWORD *)&byte_444C4A0[16 * v14 - 16] )
        goto LABEL_8;
LABEL_22:
      v61 = sub_33FAF80(v18, 189, (__int64)&v111, (unsigned int)v113, v114, v13, v7);
      v62 = (unsigned int)v113;
      v63 = v114;
      *(_QWORD *)a3 = v61;
      a3[2] = v64;
      *(_QWORD *)a4 = sub_3400BD0(a1[1], 0, (__int64)&v111, v62, v63, 0, v7, 0);
      *(_DWORD *)(a4 + 8) = v65;
      goto LABEL_24;
    }
LABEL_5:
    v16 = v114;
    goto LABEL_6;
  }
  if ( !sub_30070B0((__int64)&v113) )
    goto LABEL_5;
  v14 = sub_3009970((__int64)&v113, v7.m128i_i64[0], v57, v58, v59);
  *((_QWORD *)&v118 + 1) = v60;
  LOWORD(v118) = v14;
  if ( v14 )
    goto LABEL_19;
LABEL_7:
  v17 = sub_3007260((__int64)&v118);
  v18 = a1[1];
  v116 = v17;
  v117 = v19;
  if ( v15 > v17 )
    goto LABEL_22;
LABEL_8:
  v98 = a1;
  HIWORD(v20) = WORD1(v113);
  v21 = (unsigned __int16)v113;
  v22 = *a1;
  v23 = v114;
  v24 = *(_QWORD *)(v18 + 64);
  while ( 1 )
  {
    LOWORD(v20) = v21;
    v102 = v21;
    sub_2FE6CC0((__int64)&v118, v22, v24, v20, v23);
    if ( !(_BYTE)v118 )
      break;
    if ( (_BYTE)v118 != 2 )
      goto LABEL_28;
    v56 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v22 + 592LL);
    if ( v56 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v118, v22, v24, v20, v23);
      v21 = WORD4(v118);
      v23 = v119;
    }
    else
    {
      v20 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, __int64, __int64, _QWORD))v56)(
              v22,
              v24,
              v20,
              v23,
              v25,
              v102);
      v23 = v89;
      v21 = v20;
    }
  }
  v26 = v98;
  v27 = sub_3813820(v22, 0x49u, v102, 0, v25);
  v28 = (_QWORD *)v98[1];
  if ( !v27 )
  {
    v29 = *(unsigned __int16 **)(a2 + 48);
    v30 = *((_QWORD *)v29 + 1);
    v103 = *v29;
    *(_QWORD *)&v31 = sub_3400BD0(v98[1], 0, (__int64)&v111, *v29, v30, 0, v7, 0);
    v33 = sub_3406EB0(v28, 0x39u, (__int64)&v111, v103, v30, v32, v31, *(_OWORD *)&v7);
    *(_QWORD *)&v115 = 0;
    DWORD2(v115) = 0;
    *(_QWORD *)&v118 = 0;
    DWORD2(v118) = 0;
    sub_375BC20(v98, (__int64)v33, v34, (__int64)&v115, (__int64)&v118, v7);
    v35 = (_QWORD *)v98[1];
    v36 = sub_3400BD0((__int64)v35, 0, (__int64)&v111, (unsigned int)v113, v114, 0, v7, 0);
    v38 = v37;
    v39 = v98[1];
    v104 = v113;
    v101 = v114;
    v96 = *v98;
    v97 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*v98 + 528LL);
    v99 = *(_QWORD *)(v39 + 64);
    v40 = sub_2E79000(*(__int64 **)(v39 + 40));
    v41 = v97(v96, v40, v99, v104, v101);
    v100 = v42;
    LODWORD(v101) = v41;
    v105 = *(_OWORD *)a4;
    *(_QWORD *)&v43 = sub_33ED040(v35, 0x14u);
    *((_QWORD *)&v90 + 1) = v38;
    *(_QWORD *)&v90 = v36;
    v45 = sub_340F900(v35, 0xD0u, (__int64)&v111, v101, v100, v44, v105, v90, v43);
    v47 = v46;
    v48 = v45;
    v49 = sub_3288B20(v26[1], (int)&v111, v113, v114, v45, v46, v115, *(_OWORD *)a3, 0);
    v50 = v114;
    v109 = v51;
    v52 = v113;
    *(_QWORD *)a3 = v49;
    a3[2] = v109;
    v53 = sub_3288B20(v26[1], (int)&v111, v52, v50, v48, v47, v118, *(_OWORD *)a4, 0);
    v54 = v111;
    *(_QWORD *)a4 = v53;
    *(_DWORD *)(a4 + 8) = v55;
    if ( v54 )
      sub_B91220((__int64)&v111, v54);
    return;
  }
  *(_QWORD *)&v66 = sub_2D5B750((unsigned __int16 *)&v113);
  v118 = v66;
  v67 = sub_CA1930(&v118);
  *(_QWORD *)&v68 = sub_3400E40((__int64)v28, v67 - 1, v113, v114, (__int64)&v111, v7);
  v70 = sub_3406EB0(v28, 0xBFu, (__int64)&v111, (unsigned int)v113, v114, v69, *(_OWORD *)a4, v68);
  v71 = (__int64 *)v98[1];
  v73 = v72;
  v74 = v70;
  v75 = sub_38137B0(*v98, (__int64)v71, v113, v114);
  *((_QWORD *)&v92 + 1) = v73;
  *(_QWORD *)&v92 = v74;
  v77 = (unsigned int *)sub_33E5110(v71, (unsigned int)v113, v114, v75, v76);
  v106 = v78;
  v80 = sub_3406EB0((_QWORD *)v98[1], 0xBCu, (__int64)&v111, (unsigned int)v113, v114, v79, *(_OWORD *)a3, v92);
  v81 = v114;
  *(_QWORD *)a3 = v80;
  a3[2] = v82;
  *((_QWORD *)&v93 + 1) = v73;
  *(_QWORD *)&v93 = v74;
  *(_QWORD *)a4 = sub_3406EB0((_QWORD *)v98[1], 0xBCu, (__int64)&v111, (unsigned int)v113, v81, v83, *(_OWORD *)a4, v93);
  *(_DWORD *)(a4 + 8) = v84;
  *((_QWORD *)&v94 + 1) = v73;
  *(_QWORD *)&v94 = v74;
  v110 = sub_3411F20((_QWORD *)v98[1], 79, (__int64)&v111, v77, v106, v85, *(_OWORD *)a3, v94);
  *(_QWORD *)a3 = v110;
  a3[2] = v86;
  *((_QWORD *)&v95 + 1) = 1;
  *(_QWORD *)&v95 = v110;
  *((_QWORD *)&v91 + 1) = v73;
  *(_QWORD *)&v91 = v74;
  *(_QWORD *)a4 = sub_3412970((_QWORD *)v98[1], 73, (__int64)&v111, v77, v106, v87, *(_OWORD *)a4, v91, v95);
  *(_DWORD *)(a4 + 8) = v88;
LABEL_24:
  if ( v111 )
    sub_B91220((__int64)&v111, v111);
}
