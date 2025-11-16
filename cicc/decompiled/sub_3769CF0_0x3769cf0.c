// Function: sub_3769CF0
// Address: 0x3769cf0
//
__int64 __fastcall sub_3769CF0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v5; // rsi
  const __m128i *v6; // rax
  __int128 v7; // xmm0
  unsigned int v8; // r14d
  unsigned __int16 *v9; // rdx
  int v10; // eax
  __int64 v11; // rdx
  char v12; // cl
  unsigned __int64 v13; // rdi
  int v14; // esi
  unsigned __int16 *v15; // rax
  unsigned int v16; // r14d
  __int64 v17; // r15
  __int64 *v18; // rdx
  unsigned __int16 v19; // ax
  __int64 v20; // r9
  __int64 v21; // r8
  unsigned __int16 v22; // r10
  _BYTE *v23; // r14
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 v27; // rax
  char v28; // dl
  unsigned __int16 v29; // ax
  __int64 v30; // rdx
  char v31; // al
  __int64 v32; // r15
  __int64 v33; // rax
  __int16 v34; // ax
  __int64 v35; // rdx
  __int64 v36; // r9
  unsigned __int8 *v37; // rax
  int v38; // r9d
  _QWORD *v39; // r11
  unsigned __int8 *v40; // r14
  __int64 v41; // rdx
  __int64 v42; // r15
  __int64 v43; // rax
  unsigned int v44; // edx
  __int64 v45; // r12
  __int128 v46; // rax
  __int64 v47; // r9
  __int128 v48; // rax
  __int64 v49; // r9
  unsigned __int8 *v50; // rax
  _QWORD *v51; // r10
  unsigned __int16 *v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // rbx
  __int64 v55; // rsi
  __int64 v56; // r8
  unsigned int v57; // edx
  __int64 v58; // rcx
  __int64 v59; // r13
  __int16 v60; // ax
  __int64 v61; // rsi
  __int64 v62; // r15
  unsigned int v63; // esi
  unsigned __int8 *v64; // rcx
  bool v65; // al
  unsigned __int8 *v66; // rax
  __int128 v67; // [rsp-30h] [rbp-120h]
  __int128 v68; // [rsp-20h] [rbp-110h]
  __int128 v69; // [rsp-10h] [rbp-100h]
  unsigned __int16 v70; // [rsp+Ch] [rbp-E4h]
  int v71; // [rsp+10h] [rbp-E0h]
  __int128 v72; // [rsp+10h] [rbp-E0h]
  __int64 v73; // [rsp+30h] [rbp-C0h]
  __int64 v74; // [rsp+38h] [rbp-B8h]
  unsigned __int32 v75; // [rsp+40h] [rbp-B0h]
  __int64 v76; // [rsp+40h] [rbp-B0h]
  unsigned __int32 v77; // [rsp+50h] [rbp-A0h]
  unsigned int v78; // [rsp+50h] [rbp-A0h]
  __int64 *v79; // [rsp+58h] [rbp-98h]
  __int64 v80; // [rsp+58h] [rbp-98h]
  _QWORD *v81; // [rsp+58h] [rbp-98h]
  __int64 v82; // [rsp+58h] [rbp-98h]
  unsigned int v83; // [rsp+58h] [rbp-98h]
  bool v84; // [rsp+60h] [rbp-90h]
  __int64 (__fastcall *v85)(_BYTE *, __int64, __int64, _QWORD, __int64); // [rsp+60h] [rbp-90h]
  __int64 v86; // [rsp+68h] [rbp-88h]
  unsigned int v87; // [rsp+68h] [rbp-88h]
  _QWORD *v88; // [rsp+68h] [rbp-88h]
  unsigned __int8 *v89; // [rsp+68h] [rbp-88h]
  unsigned __int64 v90; // [rsp+78h] [rbp-78h]
  __int64 v91; // [rsp+80h] [rbp-70h] BYREF
  int v92; // [rsp+88h] [rbp-68h]
  __int64 v93; // [rsp+90h] [rbp-60h] BYREF
  __int64 v94; // [rsp+98h] [rbp-58h]
  __int64 v95; // [rsp+A0h] [rbp-50h]
  __int64 v96; // [rsp+A8h] [rbp-48h]
  __int64 v97; // [rsp+B0h] [rbp-40h] BYREF
  __int64 v98; // [rsp+B8h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v91 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v91, v5, 1);
  v92 = *(_DWORD *)(a2 + 72);
  v6 = *(const __m128i **)(a2 + 40);
  v7 = (__int128)_mm_loadu_si128(v6);
  v8 = v6[8].m128i_u32[0];
  v73 = v6[2].m128i_i64[1];
  v77 = v6[3].m128i_u32[0];
  v74 = v6[5].m128i_i64[0];
  v75 = v6[5].m128i_u32[2];
  v86 = v6[7].m128i_i64[1];
  v9 = (unsigned __int16 *)(*(_QWORD *)(v6->m128i_i64[0] + 48) + 16LL * v6->m128i_u32[2]);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  LOWORD(v93) = v10;
  v94 = v11;
  if ( (_WORD)v10 )
  {
    v84 = (unsigned __int16)(v10 - 17) <= 0x9Eu;
    v12 = (unsigned __int16)(v10 - 176) <= 0x34u;
    LOBYTE(v13) = v12;
    v14 = word_4456340[v10 - 1];
  }
  else
  {
    v84 = sub_30070D0((__int64)&v93);
    v90 = sub_3007240((__int64)&v93);
    v14 = v90;
    v13 = HIDWORD(v90);
    v12 = BYTE4(v90);
  }
  v71 = v8;
  v15 = (unsigned __int16 *)(*(_QWORD *)(v86 + 48) + 16LL * v8);
  v16 = *v15;
  v17 = *((_QWORD *)v15 + 1);
  v18 = (__int64 *)(*a1)[8];
  BYTE4(v97) = v13;
  LODWORD(v97) = v14;
  v79 = v18;
  if ( v12 )
  {
    v19 = sub_2D43AD0(v16, v14);
    v21 = 0;
    v22 = v19;
    if ( v19 )
      goto LABEL_7;
  }
  else
  {
    v29 = sub_2D43050(v16, v14);
    v21 = 0;
    v22 = v29;
    if ( v29 )
      goto LABEL_7;
  }
  v2 = sub_3009450(v79, v16, v17, v97, 0, v20);
  v22 = v2;
  v21 = v30;
LABEL_7:
  v23 = a1[1];
  LOWORD(v2) = v22;
  if ( v84 )
  {
    v24 = 1;
    if ( v22 != 1 )
    {
      if ( !v22 )
        goto LABEL_10;
      v24 = v22;
      if ( !*(_QWORD *)&v23[8 * v22 + 112] )
        goto LABEL_10;
    }
    if ( (v23[500 * v24 + 6570] & 0xFB) != 0 )
      goto LABEL_10;
  }
  else
  {
    if ( v22 == 1 )
    {
      v31 = v23[7084];
      if ( v31 && v31 != 4 )
        goto LABEL_10;
      v27 = 1;
    }
    else
    {
      if ( !v22 )
        goto LABEL_10;
      v27 = v22;
      if ( !*(_QWORD *)&v23[8 * v22 + 112] )
        goto LABEL_10;
      v28 = v23[500 * v22 + 6584];
      if ( v28 )
      {
        if ( v28 != 4 || !*(_QWORD *)&v23[8 * v22 + 112] )
          goto LABEL_10;
      }
    }
    if ( (v23[500 * v27 + 6582] & 0xFB) != 0 )
    {
LABEL_10:
      v25 = 0;
      goto LABEL_11;
    }
  }
  v80 = v21;
  v70 = v22;
  v32 = (*a1)[8];
  v85 = *(__int64 (__fastcall **)(_BYTE *, __int64, __int64, _QWORD, __int64))(*(_QWORD *)v23 + 528LL);
  v33 = sub_2E79000((__int64 *)(*a1)[5]);
  v34 = v85(v23, v33, v32, (unsigned int)v2, v80);
  if ( (_WORD)v93 != v34 || !v34 && v94 != v35 )
    goto LABEL_10;
  v37 = sub_3402A00(*a1, (unsigned __int64 *)&v91, (unsigned int)v2, v80, (__m128i)v7, v80, v36);
  v95 = v2;
  v39 = *a1;
  v40 = v37;
  v42 = v41;
  v96 = v80;
  if ( (unsigned __int16)(v70 - 176) <= 0x34u )
  {
    if ( *(_DWORD *)(v86 + 24) == 51 )
    {
      v97 = 0;
      LODWORD(v98) = 0;
      v66 = (unsigned __int8 *)sub_33F17F0(v39, 51, (__int64)&v97, v95, v96);
      v64 = v66;
      if ( v97 )
      {
        v83 = v44;
        v89 = v66;
        sub_B91220((__int64)&v97, v97);
        v64 = v89;
        v44 = v83;
      }
    }
    else
    {
      v64 = sub_33FAF80((__int64)v39, 168, (__int64)&v91, v95, v96, v38, (__m128i)v7);
    }
    v43 = (__int64)v64;
  }
  else
  {
    v43 = sub_32886A0((__int64)v39, (unsigned int)v95, v96, (int)&v91, v86, v71);
  }
  v45 = v94;
  v87 = v93;
  *(_QWORD *)&v72 = v43;
  v81 = *a1;
  *((_QWORD *)&v72 + 1) = v44;
  *(_QWORD *)&v46 = sub_33ED040(*a1, 0xCu);
  *((_QWORD *)&v67 + 1) = v42;
  *(_QWORD *)&v67 = v40;
  *(_QWORD *)&v48 = sub_340F900(v81, 0xD0u, (__int64)&v91, v87, v45, v47, v67, v72, v46);
  v50 = sub_3406EB0(*a1, 0xBAu, (__int64)&v91, (unsigned int)v93, v94, v49, v7, v48);
  v51 = *a1;
  v52 = *(unsigned __int16 **)(a2 + 48);
  v54 = v53;
  v55 = *((_QWORD *)v50 + 6) + 16LL * (unsigned int)v53;
  v56 = *((_QWORD *)v52 + 1);
  v57 = *v52;
  v58 = (__int64)v50;
  v59 = v77;
  v60 = *(_WORD *)v55;
  v61 = *(_QWORD *)(v55 + 8);
  LOWORD(v97) = v60;
  v98 = v61;
  v62 = v75;
  if ( v60 )
  {
    v63 = ((unsigned __int16)(v60 - 17) < 0xD4u) + 205;
  }
  else
  {
    v78 = v57;
    v76 = v58;
    v82 = v56;
    v88 = v51;
    v65 = sub_30070B0((__int64)&v97);
    v57 = v78;
    v58 = v76;
    v56 = v82;
    v51 = v88;
    v63 = 205 - (!v65 - 1);
  }
  *((_QWORD *)&v69 + 1) = v62;
  *(_QWORD *)&v69 = v74;
  *((_QWORD *)&v68 + 1) = v59;
  *(_QWORD *)&v68 = v73;
  v25 = sub_340EC60(v51, v63, (__int64)&v91, v57, v56, 0, v58, v54, v68, v69);
LABEL_11:
  if ( v91 )
    sub_B91220((__int64)&v91, v91);
  return v25;
}
