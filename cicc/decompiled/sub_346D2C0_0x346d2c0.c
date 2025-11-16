// Function: sub_346D2C0
// Address: 0x346d2c0
//
unsigned __int8 *__fastcall sub_346D2C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  const __m128i *v8; // rax
  unsigned __int16 *v9; // rdx
  __int64 v10; // rsi
  __m128i v11; // xmm0
  __int64 v12; // r14
  __int64 v13; // rbx
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rdx
  _QWORD *v17; // rax
  unsigned __int16 *v18; // rbx
  __int64 v19; // rcx
  __int64 v20; // r15
  __int64 v21; // r14
  __int64 (__fastcall *v22)(__int64, __int64, __int64, __int64, __int64); // rbx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rbx
  __int128 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // r9
  __int64 v29; // r15
  unsigned int v30; // edx
  int v31; // eax
  __int64 v32; // rdx
  unsigned int v33; // edx
  int v34; // r9d
  unsigned int v35; // edx
  __int128 v36; // rax
  __int64 v37; // r9
  __int128 v38; // rax
  __int128 v39; // rax
  __int128 v40; // rax
  __int64 v41; // r9
  __int128 v42; // rax
  __int64 v43; // r9
  unsigned int v44; // edx
  __int128 v45; // rax
  __int64 v46; // r9
  __int128 v47; // rax
  __int64 v48; // r9
  unsigned __int8 *v49; // rax
  unsigned int v50; // r8d
  __int64 v51; // r13
  unsigned __int64 v52; // rdx
  unsigned __int64 v53; // rcx
  unsigned __int8 *v54; // rdx
  __int64 v55; // r10
  __int64 v56; // rax
  __int16 v57; // si
  __int64 v58; // rax
  __int64 v59; // r11
  unsigned int v60; // esi
  unsigned int v61; // edx
  __int128 v62; // rax
  __int64 v63; // r9
  int v64; // r9d
  int v65; // r9d
  __int64 v66; // rcx
  __int64 v67; // r8
  int v68; // r9d
  unsigned __int8 *v69; // r13
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r8
  __int16 v74; // ax
  __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rdx
  bool v78; // al
  int v79; // esi
  __int128 v80; // [rsp-20h] [rbp-160h]
  __int128 v81; // [rsp-10h] [rbp-150h]
  __int64 v82; // [rsp+0h] [rbp-140h]
  unsigned int v83; // [rsp+8h] [rbp-138h]
  unsigned int v84; // [rsp+10h] [rbp-130h]
  __int128 v85; // [rsp+20h] [rbp-120h]
  unsigned __int8 *v86; // [rsp+20h] [rbp-120h]
  unsigned __int64 v87; // [rsp+28h] [rbp-118h]
  __int64 v88; // [rsp+30h] [rbp-110h]
  __int128 v89; // [rsp+30h] [rbp-110h]
  __int128 v90; // [rsp+40h] [rbp-100h]
  unsigned __int64 v91; // [rsp+48h] [rbp-F8h]
  unsigned __int8 *v92; // [rsp+90h] [rbp-B0h]
  unsigned int v93; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v94; // [rsp+C8h] [rbp-78h]
  __int64 v95; // [rsp+D0h] [rbp-70h] BYREF
  int v96; // [rsp+D8h] [rbp-68h]
  __int64 v97; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v98; // [rsp+E8h] [rbp-58h]
  __int64 v99; // [rsp+F0h] [rbp-50h] BYREF
  __int64 v100; // [rsp+F8h] [rbp-48h]
  __int16 v101; // [rsp+100h] [rbp-40h] BYREF
  __int64 v102; // [rsp+108h] [rbp-38h]

  v8 = *(const __m128i **)(a2 + 40);
  v9 = *(unsigned __int16 **)(a2 + 48);
  v10 = *(_QWORD *)(a2 + 80);
  v11 = _mm_loadu_si128(v8);
  v12 = v8->m128i_i64[0];
  v13 = v8->m128i_u32[2];
  v14 = *v9;
  v95 = v10;
  v15 = *((_QWORD *)v9 + 1);
  LOWORD(v93) = v14;
  v94 = v15;
  if ( v10 )
  {
    sub_B96E90((__int64)&v95, v10, 1);
    v14 = (unsigned __int16)v93;
  }
  v96 = *(_DWORD *)(a2 + 72);
  if ( !(_WORD)v14 )
  {
    if ( sub_30070B0((__int64)&v93) && (unsigned __int16)sub_3009970((__int64)&v93, v10, v70, v71, v72) == 10 )
      goto LABEL_6;
    goto LABEL_19;
  }
  if ( (unsigned __int16)(v14 - 17) <= 0xD3u )
  {
    if ( word_4456580[v14 - 1] == 10 )
      goto LABEL_6;
LABEL_19:
    v69 = 0;
    goto LABEL_20;
  }
  if ( (_WORD)v14 != 10 )
    goto LABEL_19;
LABEL_6:
  v16 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 96LL);
  v17 = *(_QWORD **)(v16 + 24);
  if ( *(_DWORD *)(v16 + 32) > 0x40u )
    v17 = (_QWORD *)*v17;
  if ( v17 != (_QWORD *)1 )
  {
    v18 = (unsigned __int16 *)(*(_QWORD *)(v12 + 48) + 16 * v13);
    v19 = *v18;
    v20 = *((_QWORD *)v18 + 1);
    v21 = *(_QWORD *)(a3 + 64);
    v22 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1 + 528LL);
    v88 = v19;
    v23 = sub_2E79000(*(__int64 **)(a3 + 40));
    LODWORD(v20) = v22(a1, v23, v21, v88, v20);
    v25 = v24;
    *(_QWORD *)&v26 = sub_33ED040((_QWORD *)a3, 8u);
    v27 = 208;
    v29 = sub_340F900((_QWORD *)a3, 0xD0u, (__int64)&v95, v20, v25, v28, *(_OWORD *)&v11, *(_OWORD *)&v11, v26);
    v84 = v30;
    if ( (_WORD)v93 )
    {
      if ( (unsigned __int16)(v93 - 17) > 0xD3u )
        goto LABEL_11;
      v27 = word_4456340[(unsigned __int16)v93 - 1];
      if ( (unsigned __int16)(v93 - 176) > 0x34u )
        v74 = sub_2D43050(12, v27);
      else
        v74 = sub_2D43AD0(12, v27);
      v75 = 0;
    }
    else
    {
      if ( !sub_30070B0((__int64)&v93) )
      {
LABEL_11:
        v97 = 12;
        v98 = 0;
        goto LABEL_12;
      }
      v27 = 12;
      v74 = sub_3009490((unsigned __int16 *)&v93, 0xCu, 0);
    }
    LOWORD(v97) = v74;
    v98 = v75;
LABEL_12:
    v31 = sub_327FF20((unsigned __int16 *)&v97, v27);
    v100 = v32;
    LODWORD(v99) = v31;
    sub_346C5C0(a1, (unsigned int)v97, v98, v11.m128i_i64[0], v11.m128i_i64[1], (__int64)&v95, v11, a3);
    v91 = v33 | v11.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v90 = sub_33FAF80(a3, 234, (__int64)&v95, (unsigned int)v99, v100, v34, v11);
    *((_QWORD *)&v90 + 1) = v35 | v91 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v36 = sub_3400BD0(a3, (__int64)&dword_400000, (__int64)&v95, (unsigned int)v99, v100, 0, v11, 0);
    *(_QWORD *)&v38 = sub_3406EB0((_QWORD *)a3, 0xBBu, (__int64)&v95, (unsigned int)v99, v100, v37, v90, v36);
    v89 = v38;
    *(_QWORD *)&v39 = sub_3400BD0(a3, 1, (__int64)&v95, (unsigned int)v99, v100, 0, v11, 0);
    v85 = v39;
    *(_QWORD *)&v40 = sub_3400E40(a3, 16, v99, v100, (__int64)&v95, v11);
    *(_QWORD *)&v42 = sub_3406EB0((_QWORD *)a3, 0xC0u, (__int64)&v95, (unsigned int)v99, v100, v41, v90, v40);
    v81 = v85;
    *((_QWORD *)&v85 + 1) = *((_QWORD *)&v42 + 1);
    v92 = sub_3406EB0((_QWORD *)a3, 0xBAu, (__int64)&v95, (unsigned int)v99, v100, v43, v42, v81);
    *((_QWORD *)&v85 + 1) = v44 | *((_QWORD *)&v85 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v45 = sub_3400BD0(a3, 0x7FFF, (__int64)&v95, (unsigned int)v99, v100, 0, v11, 0);
    *((_QWORD *)&v80 + 1) = *((_QWORD *)&v85 + 1);
    *(_QWORD *)&v80 = v92;
    *(_QWORD *)&v47 = sub_3406EB0((_QWORD *)a3, 0x38u, (__int64)&v95, (unsigned int)v99, v100, v46, v45, v80);
    v49 = sub_3406EB0((_QWORD *)a3, 0x38u, (__int64)&v95, (unsigned int)v99, v100, v48, v90, v47);
    v50 = v99;
    v51 = v100;
    v53 = v52;
    v54 = v49;
    v55 = v29;
    v56 = *(_QWORD *)(v29 + 48) + 16LL * v84;
    v57 = *(_WORD *)v56;
    v58 = *(_QWORD *)(v56 + 8);
    v59 = v84;
    v101 = v57;
    v102 = v58;
    if ( v57 )
    {
      v60 = ((unsigned __int16)(v57 - 17) < 0xD4u) + 205;
    }
    else
    {
      v83 = v99;
      v86 = v54;
      v87 = v53;
      v78 = sub_30070B0((__int64)&v101);
      v50 = v83;
      v55 = v29;
      v59 = v84;
      v54 = v86;
      v53 = v87;
      v60 = 205 - (!v78 - 1);
    }
    *(_QWORD *)&v90 = sub_340EC60(
                        (_QWORD *)a3,
                        v60,
                        (__int64)&v95,
                        v50,
                        v51,
                        0,
                        v55,
                        v59,
                        v89,
                        __PAIR128__(v53, (unsigned __int64)v54));
    *((_QWORD *)&v90 + 1) = v61 | *((_QWORD *)&v90 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v62 = sub_3400E40(a3, 16, v99, v100, (__int64)&v95, v11);
    sub_3406EB0((_QWORD *)a3, 0xC0u, (__int64)&v95, (unsigned int)v99, v100, v63, v90, v62);
    sub_33FAF80(a3, 234, (__int64)&v95, (unsigned int)v99, v100, v64, v11);
    if ( (_WORD)v99 )
    {
      if ( (unsigned __int16)(v99 - 17) > 0xD3u )
      {
LABEL_16:
        v66 = 6;
        v67 = 0;
LABEL_17:
        sub_33FAF80(a3, 216, (__int64)&v95, v66, v67, v65, v11);
        v69 = sub_33FAF80(a3, 234, (__int64)&v95, v93, v94, v68, v11);
        goto LABEL_20;
      }
      v79 = word_4456340[(unsigned __int16)v99 - 1];
      if ( (unsigned __int16)(v99 - 176) <= 0x34u )
        LOWORD(v76) = sub_2D43AD0(6, v79);
      else
        LOWORD(v76) = sub_2D43050(6, v79);
      v67 = 0;
    }
    else
    {
      if ( !sub_30070B0((__int64)&v99) )
        goto LABEL_16;
      v76 = sub_3009490((unsigned __int16 *)&v99, 6u, 0);
      v82 = v76;
      v67 = v77;
    }
    v66 = v82;
    LOWORD(v66) = v76;
    goto LABEL_17;
  }
  v69 = sub_33FAF80(a3, 241, (__int64)&v95, v93, v94, a6, v11);
LABEL_20:
  if ( v95 )
    sub_B91220((__int64)&v95, v95);
  return v69;
}
