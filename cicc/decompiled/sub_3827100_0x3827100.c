// Function: sub_3827100
// Address: 0x3827100
//
void __fastcall sub_3827100(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rax
  __int16 v9; // dx
  __int64 v10; // rax
  __int64 v11; // r13
  unsigned int v12; // r15d
  const __m128i *v13; // rax
  __m128i v14; // xmm0
  __int64 v15; // rax
  unsigned __int16 v16; // r14
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // rsi
  unsigned __int16 v20; // ax
  unsigned __int16 v21; // r13
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rdx
  _QWORD *v26; // r13
  __int128 v27; // rax
  __int64 v28; // r9
  __int128 v29; // rax
  _QWORD *v30; // r13
  __int64 v31; // rdx
  __int128 v32; // rax
  __int64 v33; // r12
  __int64 v34; // rdx
  __int64 v35; // r13
  __int64 v36; // rax
  unsigned int v37; // eax
  __int64 v38; // rdx
  __int128 v39; // rax
  __int64 v40; // r9
  __int128 v41; // rax
  __int64 v42; // r9
  __int128 v43; // rax
  __int64 v44; // r9
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r13
  __int64 v48; // r12
  __int64 v49; // r9
  __int64 v50; // rax
  __int64 v51; // r8
  int v52; // edx
  __int64 v53; // r9
  __int64 v54; // rax
  __int64 v55; // rsi
  int v56; // edx
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // rdx
  __int128 v61; // [rsp-30h] [rbp-160h]
  __int128 v62; // [rsp-30h] [rbp-160h]
  __int128 v63; // [rsp-30h] [rbp-160h]
  __int128 v64; // [rsp-30h] [rbp-160h]
  __int128 v65; // [rsp+0h] [rbp-130h]
  __int64 v66; // [rsp+18h] [rbp-118h]
  __int64 (__fastcall *v69)(__int64, __int64, __int64, _QWORD, __int64); // [rsp+30h] [rbp-100h]
  __int128 v70; // [rsp+30h] [rbp-100h]
  __int128 v71; // [rsp+30h] [rbp-100h]
  __int64 v72; // [rsp+40h] [rbp-F0h]
  unsigned __int16 v73; // [rsp+40h] [rbp-F0h]
  __int64 v74; // [rsp+40h] [rbp-F0h]
  __int128 v75; // [rsp+40h] [rbp-F0h]
  __int128 v76; // [rsp+50h] [rbp-E0h]
  __int128 v77; // [rsp+80h] [rbp-B0h] BYREF
  __int128 v78; // [rsp+90h] [rbp-A0h] BYREF
  __int128 v79; // [rsp+A0h] [rbp-90h] BYREF
  __int128 v80; // [rsp+B0h] [rbp-80h] BYREF
  unsigned int v81; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v82; // [rsp+C8h] [rbp-68h]
  __int64 v83; // [rsp+D0h] [rbp-60h] BYREF
  int v84; // [rsp+D8h] [rbp-58h]
  unsigned __int16 v85; // [rsp+E0h] [rbp-50h] BYREF
  __int64 v86; // [rsp+E8h] [rbp-48h]
  __int64 v87; // [rsp+F0h] [rbp-40h]
  __int64 v88; // [rsp+F8h] [rbp-38h]

  v5 = *(unsigned __int64 **)(a2 + 40);
  DWORD2(v77) = 0;
  DWORD2(v78) = 0;
  DWORD2(v79) = 0;
  DWORD2(v80) = 0;
  v6 = v5[1];
  *(_QWORD *)&v77 = 0;
  *(_QWORD *)&v78 = 0;
  *(_QWORD *)&v79 = 0;
  *(_QWORD *)&v80 = 0;
  sub_375E510((__int64)a1, *v5, v6, (__int64)&v79, (__int64)&v80);
  sub_375E510(
    (__int64)a1,
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
    (__int64)&v77,
    (__int64)&v78);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *(_QWORD *)(v77 + 48) + 16LL * DWORD2(v77);
  v9 = *(_WORD *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v83 = v7;
  LOWORD(v81) = v9;
  v82 = v10;
  if ( v7 )
    sub_B96E90((__int64)&v83, v7, 1);
  v11 = *a1;
  v12 = *(_DWORD *)(a2 + 24);
  v84 = *(_DWORD *)(a2 + 72);
  v13 = *(const __m128i **)(a2 + 40);
  v14 = _mm_loadu_si128(v13 + 5);
  v15 = *(_QWORD *)(v13[5].m128i_i64[0] + 48) + 16LL * v13[5].m128i_u32[2];
  v16 = *(_WORD *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  v18 = a1[1];
  v69 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)v11 + 528LL);
  v72 = *(_QWORD *)(v18 + 64);
  v19 = sub_2E79000(*(__int64 **)(v18 + 40));
  v20 = v69(v11, v19, v72, v16, v17);
  v21 = v81;
  v66 = v22;
  v73 = v20;
  if ( (_WORD)v81 )
  {
    if ( (unsigned __int16)(v81 - 17) <= 0xD3u )
    {
      v21 = word_4456580[(unsigned __int16)v81 - 1];
      v86 = 0;
      v85 = v21;
      if ( !v21 )
        goto LABEL_7;
      goto LABEL_12;
    }
    goto LABEL_5;
  }
  if ( !sub_30070B0((__int64)&v81) )
  {
LABEL_5:
    v23 = v82;
    goto LABEL_6;
  }
  v21 = sub_3009970((__int64)&v81, v19, v57, v58, v59);
  v23 = v60;
LABEL_6:
  v85 = v21;
  v86 = v23;
  if ( !v21 )
  {
LABEL_7:
    v87 = sub_3007260((__int64)&v85);
    LODWORD(v24) = v87;
    v88 = v25;
    goto LABEL_8;
  }
LABEL_12:
  if ( v21 == 1 || (unsigned __int16)(v21 - 504) <= 7u )
    BUG();
  v24 = *(_QWORD *)&byte_444C4A0[16 * v21 - 16];
LABEL_8:
  v26 = (_QWORD *)a1[1];
  *(_QWORD *)&v27 = sub_3400BD0((__int64)v26, (unsigned int)v24, (__int64)&v83, v16, v17, 0, v14, 0);
  *(_QWORD *)&v29 = sub_3406EB0(v26, 0xBAu, (__int64)&v83, v16, v17, v28, *(_OWORD *)&v14, v27);
  v30 = (_QWORD *)a1[1];
  v70 = v29;
  *(_QWORD *)&v65 = sub_3400BD0((__int64)v30, 0, (__int64)&v83, v16, v17, 0, v14, 0);
  *((_QWORD *)&v65 + 1) = v31;
  *(_QWORD *)&v32 = sub_33ED040(v30, 5 * (unsigned int)(v12 == 195) + 17);
  v33 = sub_340F900(v30, 0xD0u, (__int64)&v83, v73, v66, *((__int64 *)&v65 + 1), v70, v65, v32);
  v35 = v34;
  v74 = *a1;
  v36 = sub_2E79000(*(__int64 **)(a1[1] + 40));
  v37 = sub_2FE6750(v74, v81, v82, v36);
  *(_QWORD *)&v39 = sub_33FAFB0(a1[1], v14.m128i_i64[0], v14.m128i_u32[2], (__int64)&v83, v37, v38, v14);
  *((_QWORD *)&v61 + 1) = v35;
  *(_QWORD *)&v61 = v33;
  v76 = v39;
  *(_QWORD *)&v41 = sub_340F900((_QWORD *)a1[1], 0xCDu, (__int64)&v83, v81, v82, v40, v61, v77, v78);
  *((_QWORD *)&v62 + 1) = v35;
  *(_QWORD *)&v62 = v33;
  v71 = v41;
  *(_QWORD *)&v43 = sub_340F900((_QWORD *)a1[1], 0xCDu, (__int64)&v83, v81, v82, v42, v62, v78, v79);
  *((_QWORD *)&v63 + 1) = v35;
  *(_QWORD *)&v63 = v33;
  v75 = v43;
  v45 = sub_340F900((_QWORD *)a1[1], 0xCDu, (__int64)&v83, v81, v82, v44, v63, v79, v80);
  v47 = v46;
  v48 = v45;
  v50 = sub_340F900((_QWORD *)a1[1], v12, (__int64)&v83, v81, v82, v49, v75, v71, v76);
  v51 = v82;
  *(_QWORD *)a3 = v50;
  *(_DWORD *)(a3 + 8) = v52;
  *((_QWORD *)&v64 + 1) = v47;
  *(_QWORD *)&v64 = v48;
  v54 = sub_340F900((_QWORD *)a1[1], v12, (__int64)&v83, v81, v51, v53, v64, v75, v76);
  v55 = v83;
  *(_QWORD *)a4 = v54;
  *(_DWORD *)(a4 + 8) = v56;
  if ( v55 )
    sub_B91220((__int64)&v83, v55);
}
