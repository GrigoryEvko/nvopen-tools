// Function: sub_34680A0
// Address: 0x34680a0
//
__int64 __fastcall sub_34680A0(_DWORD *a1, __int64 a2, _QWORD *a3, __m128i a4)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rbx
  __int32 v10; // ecx
  __int64 v11; // rax
  unsigned __int16 *v12; // rax
  __int64 v13; // rcx
  unsigned int v14; // ebx
  __int16 *v15; // rax
  __int16 v16; // di
  __int64 v17; // rax
  __int64 *v18; // rdi
  __int64 (__fastcall *v19)(_DWORD *, __int64, __int64 *, _QWORD, __int64); // r13
  __int64 v20; // rax
  __int32 v21; // eax
  __int64 v22; // rsi
  __int64 v23; // rdx
  unsigned int v24; // esi
  __int128 v25; // rax
  __int64 v26; // r9
  __m128i v27; // rax
  __int128 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // r9
  __int64 v31; // r9
  __int64 v32; // r12
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // r13
  __int64 (*v35)(); // rax
  unsigned __int16 v36; // bx
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  char v40; // bl
  int v41; // eax
  __m128i v42; // xmm1
  __int64 v43; // rbx
  unsigned __int8 *v44; // rax
  unsigned int v45; // edx
  __int64 v46; // r12
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int128 v50; // rax
  unsigned __int8 *v51; // rax
  __int64 v52; // rcx
  __int64 v53; // r10
  unsigned __int8 *v54; // r8
  __int64 v55; // rdx
  __int64 v56; // r9
  __int64 v57; // rax
  __int16 v58; // dx
  __int64 v59; // rax
  unsigned int v60; // esi
  unsigned int v61; // edx
  unsigned int v62; // ebx
  unsigned __int8 *v63; // rax
  __int64 v64; // rdx
  __int64 v65; // r11
  unsigned __int8 *v66; // r10
  unsigned int v67; // ecx
  __int64 v68; // r8
  __int64 v69; // rsi
  __int64 v70; // r9
  __int64 v71; // rax
  __int16 v72; // dx
  __int64 v73; // rax
  bool v74; // al
  bool v76; // al
  unsigned __int16 v77; // cx
  __int128 v78; // [rsp-20h] [rbp-150h]
  __int128 v79; // [rsp-20h] [rbp-150h]
  __int128 v80; // [rsp-10h] [rbp-140h]
  __int128 v81; // [rsp-10h] [rbp-140h]
  __int128 v82; // [rsp-10h] [rbp-140h]
  __int128 v83; // [rsp+0h] [rbp-130h]
  unsigned int v84; // [rsp+14h] [rbp-11Ch]
  __int64 v85; // [rsp+18h] [rbp-118h]
  __m128i v86; // [rsp+30h] [rbp-100h] BYREF
  __int64 v87; // [rsp+40h] [rbp-F0h]
  __int64 v88; // [rsp+48h] [rbp-E8h]
  unsigned __int8 *v89; // [rsp+50h] [rbp-E0h]
  __int64 v90; // [rsp+58h] [rbp-D8h]
  __int64 v91; // [rsp+60h] [rbp-D0h]
  __int64 v92; // [rsp+68h] [rbp-C8h]
  unsigned __int8 *v93; // [rsp+70h] [rbp-C0h]
  __int64 v94; // [rsp+78h] [rbp-B8h]
  __int128 v95; // [rsp+80h] [rbp-B0h]
  __int64 v96; // [rsp+90h] [rbp-A0h]
  __int64 *v97; // [rsp+98h] [rbp-98h]
  __int64 v98; // [rsp+A0h] [rbp-90h]
  unsigned __int64 v99; // [rsp+A8h] [rbp-88h]
  __m128i v100; // [rsp+B0h] [rbp-80h]
  __m128i v101; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v102; // [rsp+D0h] [rbp-60h] BYREF
  int v103; // [rsp+D8h] [rbp-58h]
  __int64 v104; // [rsp+E0h] [rbp-50h]
  __int64 v105; // [rsp+E8h] [rbp-48h]
  __m128i v106; // [rsp+F0h] [rbp-40h] BYREF

  v7 = a3[8];
  LODWORD(v95) = *(_DWORD *)(a2 + 24);
  v8 = *(_QWORD *)(a2 + 40);
  v97 = (__int64 *)v7;
  v9 = *(_QWORD *)v8;
  v89 = *(unsigned __int8 **)(v8 + 40);
  v10 = *(_DWORD *)(v8 + 48);
  v11 = *(unsigned int *)(v8 + 8);
  v91 = v9;
  v86.m128i_i32[0] = v10;
  v88 = v11;
  v12 = (unsigned __int16 *)(*(_QWORD *)(v9 + 48) + 16 * v11);
  v13 = *((_QWORD *)v12 + 1);
  v14 = *v12;
  v15 = *(__int16 **)(a2 + 48);
  v87 = v13;
  v16 = *v15;
  v17 = *((_QWORD *)v15 + 1);
  LOWORD(v93) = v16;
  v18 = (__int64 *)a3[5];
  v96 = v17;
  v19 = *(__int64 (__fastcall **)(_DWORD *, __int64, __int64 *, _QWORD, __int64))(*(_QWORD *)a1 + 528LL);
  v20 = sub_2E79000(v18);
  v21 = v19(a1, v20, v97, v14, v87);
  v22 = *(_QWORD *)(a2 + 80);
  v101.m128i_i32[0] = v21;
  v101.m128i_i64[1] = v23;
  v102 = v22;
  v97 = &v102;
  if ( v22 )
    sub_B96E90((__int64)&v102, v22, 1);
  v103 = *(_DWORD *)(a2 + 72);
  v24 = 8 * ((_DWORD)v95 != 185) + 12;
  v84 = 8 * ((_DWORD)v95 != 185) + 10;
  v85 = v86.m128i_u32[0];
  *(_QWORD *)&v95 = v89;
  *((_QWORD *)&v95 + 1) = v86.m128i_u32[0];
  *(_QWORD *)&v25 = sub_33ED040(a3, v24);
  *((_QWORD *)&v78 + 1) = v88;
  *(_QWORD *)&v78 = v91;
  v27.m128i_i64[0] = sub_340F900(a3, 0xD0u, (__int64)v97, v101.m128i_u32[0], v101.m128i_i64[1], v26, v78, v95, v25);
  v86 = v27;
  *(_QWORD *)&v95 = v89;
  *((_QWORD *)&v95 + 1) = v85;
  *(_QWORD *)&v28 = sub_33ED040(a3, v84);
  v29 = 208;
  *((_QWORD *)&v79 + 1) = v88;
  *(_QWORD *)&v79 = v91;
  v32 = sub_340F900(a3, 0xD0u, (__int64)v97, v101.m128i_u32[0], v101.m128i_i64[1], v30, v79, v95, v28);
  v34 = v33;
  v35 = *(__int64 (**)())(*(_QWORD *)a1 + 1760LL);
  if ( v35 != sub_2FE3680 )
  {
    v29 = v14;
    if ( ((unsigned __int8 (__fastcall *)(_DWORD *, _QWORD, __int64))v35)(a1, v14, v87) )
      goto LABEL_21;
  }
  v36 = v101.m128i_i16[0];
  if ( v101.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v101.m128i_i16[0] - 17) <= 0xD3u )
    {
      v106.m128i_i64[1] = 0;
      v36 = word_4456580[v101.m128i_u16[0] - 1];
      v106.m128i_i16[0] = v36;
      if ( !v36 )
        goto LABEL_8;
      goto LABEL_36;
    }
    goto LABEL_6;
  }
  *(_QWORD *)&v95 = &v101;
  if ( !sub_30070B0((__int64)&v101) )
  {
LABEL_6:
    v37 = v101.m128i_i64[1];
    goto LABEL_7;
  }
  v36 = sub_3009970((__int64)&v101, v29, v47, v48, v49);
LABEL_7:
  v106.m128i_i16[0] = v36;
  v106.m128i_i64[1] = v37;
  if ( !v36 )
  {
LABEL_8:
    v38 = sub_3007260((__int64)&v106);
    v104 = v38;
    v105 = v39;
    goto LABEL_9;
  }
LABEL_36:
  if ( v36 == 1 || (unsigned __int16)(v36 - 504) <= 7u )
    BUG();
  v38 = *(_QWORD *)&byte_444C4A0[16 * v36 - 16];
LABEL_9:
  if ( v38 == 1 )
  {
LABEL_21:
    *(_QWORD *)&v50 = sub_3400BD0((__int64)a3, 0, (__int64)v97, (unsigned __int16)v93, v96, 0, a4, 0);
    v95 = v50;
    v51 = sub_3400BD0((__int64)a3, 1, (__int64)v97, (unsigned __int16)v93, v96, 0, a4, 0);
    v52 = (unsigned int)v34;
    v53 = v32;
    v54 = v51;
    v56 = v55;
    v57 = *(_QWORD *)(v32 + 48) + 16LL * (unsigned int)v34;
    v58 = *(_WORD *)v57;
    v59 = *(_QWORD *)(v57 + 8);
    v106.m128i_i16[0] = v58;
    v106.m128i_i64[1] = v59;
    if ( v58 )
    {
      v60 = ((unsigned __int16)(v58 - 17) < 0xD4u) + 205;
    }
    else
    {
      v88 = (unsigned int)v34;
      v89 = v54;
      v90 = v56;
      v91 = v32;
      v76 = sub_30070B0((__int64)&v106);
      v52 = (unsigned int)v34;
      v54 = v89;
      v56 = v90;
      v53 = v32;
      v60 = 205 - (!v76 - 1);
    }
    *((_QWORD *)&v81 + 1) = v56;
    *(_QWORD *)&v81 = v54;
    *(_QWORD *)&v95 = sub_340EC60(
                        a3,
                        v60,
                        (__int64)v97,
                        (unsigned __int16)v93,
                        v96,
                        0,
                        v53,
                        v52 | v34 & 0xFFFFFFFF00000000LL,
                        v81,
                        v95);
    v62 = v61;
    v63 = sub_34015B0((__int64)a3, (__int64)v97, (unsigned __int16)v93, v96, 0, 0, a4);
    v65 = v64;
    v66 = v63;
    v67 = (unsigned __int16)v93;
    v68 = v95;
    v69 = v86.m128i_i64[0];
    v70 = v62;
    v71 = *(_QWORD *)(v86.m128i_i64[0] + 48) + 16LL * v86.m128i_u32[2];
    v72 = *(_WORD *)v71;
    v73 = *(_QWORD *)(v71 + 8);
    v106.m128i_i16[0] = v72;
    v106.m128i_i64[1] = v73;
    if ( v72 )
    {
      v74 = (unsigned __int16)(v72 - 17) <= 0xD3u;
    }
    else
    {
      v89 = (unsigned __int8 *)(unsigned __int16)v93;
      v91 = v95;
      v92 = v62;
      v93 = v66;
      v94 = v65;
      *(_QWORD *)&v95 = v86.m128i_i64[0];
      v74 = sub_30070B0((__int64)&v106);
      v67 = (unsigned int)v89;
      v68 = v91;
      v70 = v62;
      v66 = v93;
      v65 = v94;
      v69 = v86.m128i_i64[0];
    }
    *((_QWORD *)&v83 + 1) = v70;
    *(_QWORD *)&v83 = v68;
    *((_QWORD *)&v82 + 1) = v65;
    *(_QWORD *)&v82 = v66;
    v46 = sub_340EC60(a3, 205 - ((unsigned int)!v74 - 1), (__int64)v97, v67, v96, 0, v69, v86.m128i_i64[1], v82, v83);
    goto LABEL_26;
  }
  a4 = _mm_loadu_si128(&v101);
  v106 = a4;
  if ( !v101.m128i_i16[0] )
  {
    *(_QWORD *)&v95 = &v106;
    v40 = sub_3007030((__int64)&v106);
    if ( !sub_30070B0((__int64)&v106) )
    {
      if ( !v40 )
      {
LABEL_13:
        v41 = a1[15];
        goto LABEL_14;
      }
      goto LABEL_40;
    }
    goto LABEL_34;
  }
  v77 = v101.m128i_i16[0] - 17;
  if ( (unsigned __int16)(v101.m128i_i16[0] - 10) > 6u && (unsigned __int16)(v101.m128i_i16[0] - 126) > 0x31u )
  {
    if ( v77 > 0xD3u )
      goto LABEL_13;
    goto LABEL_34;
  }
  if ( v77 <= 0xD3u )
  {
LABEL_34:
    v41 = a1[17];
    goto LABEL_14;
  }
LABEL_40:
  v41 = a1[16];
LABEL_14:
  if ( !v41 )
    goto LABEL_21;
  if ( v41 == 2 )
  {
    v42 = _mm_load_si128(&v86);
    v98 = v32;
    v100 = v42;
    v32 = v86.m128i_i64[0];
    v86.m128i_i64[0] = v98;
    v99 = v34;
    v34 = v42.m128i_u32[2] | v34 & 0xFFFFFFFF00000000LL;
    v86.m128i_i64[1] = (unsigned int)v99 | v86.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  }
  v43 = (__int64)v97;
  *((_QWORD *)&v80 + 1) = v34;
  *(_QWORD *)&v80 = v32;
  v44 = sub_3406EB0(a3, 0x39u, (__int64)v97, v101.m128i_u32[0], v101.m128i_i64[1], v31, v80, *(_OWORD *)&v86);
  v46 = (__int64)sub_33FB160((__int64)a3, (__int64)v44, v45, v43, (unsigned __int16)v93, v96, a4);
LABEL_26:
  if ( v102 )
    sub_B91220((__int64)v97, v102);
  return v46;
}
