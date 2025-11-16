// Function: sub_3457D60
// Address: 0x3457d60
//
__int64 __fastcall sub_3457D60(__int64 a1, __int64 a2, __int64 a3, __m128i a4)
{
  __int64 v6; // rsi
  __int64 *v7; // rdi
  __int16 *v8; // rax
  __int16 v9; // dx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned int v13; // eax
  int v14; // r9d
  unsigned __int16 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int8 *v18; // r12
  unsigned __int64 v19; // r13
  __int64 v20; // rax
  __int64 v21; // r11
  __int64 v22; // rdx
  unsigned __int16 v23; // r15
  __int64 v24; // rax
  __int64 v25; // r15
  __int64 (__fastcall *v26)(__int64, __int64, __int64, _QWORD, __int64); // rbx
  __int64 v27; // rax
  unsigned int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // rbx
  int v31; // r9d
  __int128 v32; // rax
  __int64 v33; // rdx
  __int128 v34; // rax
  __int64 v35; // r12
  int v36; // edx
  unsigned __int8 *v37; // rax
  __int64 v38; // r13
  __int64 v39; // r8
  unsigned __int8 *v40; // r10
  __int64 v41; // rdx
  __int64 v42; // r11
  unsigned int v43; // edx
  __int64 v44; // rax
  __int64 v45; // rbx
  __int64 v46; // r9
  unsigned __int16 v47; // ax
  bool v48; // al
  unsigned int v49; // esi
  __int64 v50; // r13
  int v51; // r11d
  __int128 v52; // rax
  __int64 v53; // r9
  __int128 v54; // rax
  __int64 v55; // r9
  unsigned int v56; // edx
  __int64 v57; // r8
  __int64 v58; // rsi
  __int64 v59; // rcx
  __int64 v61; // rax
  char v62; // al
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // rdx
  __int64 v68; // rax
  __int128 v69; // [rsp-30h] [rbp-110h]
  __int128 v70; // [rsp-20h] [rbp-100h]
  __int128 v71; // [rsp-10h] [rbp-F0h]
  __int128 v72; // [rsp-10h] [rbp-F0h]
  unsigned int v73; // [rsp+10h] [rbp-D0h]
  unsigned int v74; // [rsp+18h] [rbp-C8h]
  unsigned int v75; // [rsp+18h] [rbp-C8h]
  __int64 v76; // [rsp+20h] [rbp-C0h]
  __int128 v77; // [rsp+20h] [rbp-C0h]
  unsigned int v78; // [rsp+30h] [rbp-B0h]
  __int128 v79; // [rsp+30h] [rbp-B0h]
  __int64 v80; // [rsp+38h] [rbp-A8h]
  unsigned int v82; // [rsp+40h] [rbp-A0h]
  unsigned __int8 *v83; // [rsp+40h] [rbp-A0h]
  int v84; // [rsp+40h] [rbp-A0h]
  __int64 v85; // [rsp+48h] [rbp-98h]
  __int64 v86; // [rsp+70h] [rbp-70h] BYREF
  int v87; // [rsp+78h] [rbp-68h]
  __int64 v88; // [rsp+80h] [rbp-60h] BYREF
  __int64 v89; // [rsp+88h] [rbp-58h]
  __int64 v90; // [rsp+90h] [rbp-50h]
  __int64 v91; // [rsp+98h] [rbp-48h]
  unsigned __int16 v92; // [rsp+A0h] [rbp-40h] BYREF
  __int64 v93; // [rsp+A8h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 80);
  v86 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v86, v6, 1);
  v7 = *(__int64 **)(a3 + 40);
  v87 = *(_DWORD *)(a2 + 72);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v88) = v9;
  v89 = v10;
  v11 = sub_2E79000(v7);
  v12 = (unsigned int)v88;
  v13 = sub_2FE6750(a1, (unsigned int)v88, v89, v11);
  v15 = v88;
  v74 = v13;
  v16 = *(_QWORD *)(a2 + 40);
  v76 = v17;
  v18 = *(unsigned __int8 **)v16;
  v19 = *(_QWORD *)(v16 + 8);
  if ( (_WORD)v88 )
  {
    if ( (unsigned __int16)(v88 - 17) <= 0xD3u )
    {
      v15 = word_4456580[(unsigned __int16)v88 - 1];
      v93 = 0;
      v92 = v15;
      if ( !v15 )
        goto LABEL_7;
LABEL_26:
      if ( v15 == 1 || (unsigned __int16)(v15 - 504) <= 7u )
        BUG();
      v21 = *(_QWORD *)&byte_444C4A0[16 * v15 - 16];
      goto LABEL_8;
    }
    goto LABEL_5;
  }
  if ( !sub_30070B0((__int64)&v88) )
  {
LABEL_5:
    v20 = v89;
    goto LABEL_6;
  }
  v15 = sub_3009970((__int64)&v88, v12, v64, v65, v66);
  v20 = v67;
LABEL_6:
  v92 = v15;
  v93 = v20;
  if ( v15 )
    goto LABEL_26;
LABEL_7:
  v90 = sub_3007260((__int64)&v92);
  LODWORD(v21) = v90;
  v91 = v22;
LABEL_8:
  v78 = v21;
  v23 = v88;
  if ( *(_DWORD *)(a2 + 24) == 204 )
  {
    v68 = 1;
    if ( (_WORD)v88 == 1 )
      goto LABEL_47;
    if ( !(_WORD)v88 )
      goto LABEL_30;
    v68 = (unsigned __int16)v88;
    if ( *(_QWORD *)(a1 + 8LL * (unsigned __int16)v88 + 112) )
    {
LABEL_47:
      if ( (*(_BYTE *)(a1 + 500 * v68 + 6613) & 0xFB) == 0 )
      {
        v59 = (unsigned int)v88;
        v58 = 199;
        v57 = v89;
LABEL_21:
        v50 = (__int64)sub_33FAF80(a3, v58, (__int64)&v86, v59, v57, v14, a4);
        goto LABEL_22;
      }
    }
  }
  v24 = 1;
  if ( (_WORD)v88 != 1 )
  {
    if ( (_WORD)v88 )
    {
      v24 = (unsigned __int16)v88;
      if ( !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v88 + 112) )
        goto LABEL_16;
      goto LABEL_10;
    }
LABEL_30:
    if ( !sub_30070B0((__int64)&v88) )
      goto LABEL_17;
    goto LABEL_31;
  }
LABEL_10:
  v73 = v21;
  if ( (*(_BYTE *)(a1 + 500 * v24 + 6618) & 0xFB) == 0 )
  {
    v25 = *(_QWORD *)(a3 + 64);
    v26 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 528LL);
    v27 = sub_2E79000(*(__int64 **)(a3 + 40));
    v28 = v26(a1, v27, v25, (unsigned int)v88, v89);
    v30 = v29;
    v82 = v28;
    *(_QWORD *)&v32 = sub_33FAF80(a3, 204, (__int64)&v86, (unsigned int)v88, v89, v31, a4);
    v77 = v32;
    *(_QWORD *)&v79 = sub_3400BD0(a3, 0, (__int64)&v86, (unsigned int)v88, v89, 0, a4, 0);
    *((_QWORD *)&v79 + 1) = v33;
    *(_QWORD *)&v34 = sub_33ED040((_QWORD *)a3, 0x11u);
    *((_QWORD *)&v69 + 1) = v19;
    *(_QWORD *)&v69 = v18;
    v35 = sub_340F900((_QWORD *)a3, 0xD0u, (__int64)&v86, v82, v30, *((__int64 *)&v79 + 1), v69, v79, v34);
    LODWORD(v30) = v36;
    v37 = sub_3400BD0(a3, v73, (__int64)&v86, (unsigned int)v88, v89, 0, a4, 0);
    v38 = v89;
    v39 = v35;
    v40 = v37;
    v42 = v41;
    v43 = v88;
    v44 = (unsigned int)v30;
    v45 = *(_QWORD *)(v35 + 48) + 16LL * (unsigned int)v30;
    v46 = v44;
    v47 = *(_WORD *)v45;
    v93 = *(_QWORD *)(v45 + 8);
    v92 = v47;
    if ( v47 )
    {
      v49 = ((unsigned __int16)(v47 - 17) < 0xD4u) + 205;
    }
    else
    {
      v75 = v88;
      v80 = v46;
      v83 = v40;
      v85 = v42;
      v48 = sub_30070B0((__int64)&v92);
      v43 = v75;
      v39 = v35;
      v46 = v80;
      v40 = v83;
      v42 = v85;
      v49 = 205 - (!v48 - 1);
    }
    *((_QWORD *)&v71 + 1) = v42;
    *(_QWORD *)&v71 = v40;
    v50 = sub_340EC60((_QWORD *)a3, v49, (__int64)&v86, v43, v38, 0, v39, v46, v71, v77);
    goto LABEL_22;
  }
LABEL_16:
  if ( (unsigned __int16)(v88 - 17) > 0xD3u )
  {
LABEL_17:
    if ( v78 > 1 )
    {
      v51 = 0;
      do
      {
        v84 = v51;
        *(_QWORD *)&v52 = sub_3400BD0(a3, 1LL << v51, (__int64)&v86, v74, v76, 0, a4, 0);
        *((_QWORD *)&v70 + 1) = v19;
        *(_QWORD *)&v70 = v18;
        *(_QWORD *)&v54 = sub_3406EB0((_QWORD *)a3, 0xC0u, (__int64)&v86, (unsigned int)v88, v89, v53, v70, v52);
        *((_QWORD *)&v72 + 1) = v19;
        *(_QWORD *)&v72 = v18;
        v18 = sub_3406EB0((_QWORD *)a3, 0xBBu, (__int64)&v86, (unsigned int)v88, v89, v55, v72, v54);
        v51 = v84 + 1;
        v19 = v56 | v19 & 0xFFFFFFFF00000000LL;
      }
      while ( 1 << (v84 + 1) < v78 );
    }
    sub_34074A0((_QWORD *)a3, (__int64)&v86, (__int64)v18, v19, v88, v89, a4);
    v57 = v89;
    v58 = 200;
    v59 = (unsigned int)v88;
    goto LABEL_21;
  }
LABEL_31:
  if ( !v78 || ((v78 - 1) & v78) != 0 )
    goto LABEL_43;
  v61 = 1;
  if ( v23 != 1 && (!v23 || (v61 = v23, !*(_QWORD *)(a1 + 8LL * v23 + 112)))
    || (v62 = *(_BYTE *)(a1 + 500 * v61 + 6614)) != 0 && v62 != 4 )
  {
    if ( !sub_34447D0(a1, (unsigned int)v88, v89) )
      goto LABEL_43;
    v23 = v88;
  }
  v63 = 1;
  if ( v23 == 1 || v23 && (v63 = v23, *(_QWORD *)(a1 + 8LL * v23 + 112)) )
  {
    if ( (*(_BYTE *)(a1 + 500 * v63 + 6606) & 0xFB) == 0 && (unsigned __int8)sub_328C7F0(a1, 0xBBu, v88, v89, 0) )
      goto LABEL_17;
  }
LABEL_43:
  v50 = 0;
LABEL_22:
  if ( v86 )
    sub_B91220((__int64)&v86, v86);
  return v50;
}
