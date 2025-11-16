// Function: sub_380E540
// Address: 0x380e540
//
unsigned __int8 *__fastcall sub_380E540(__int64 a1, unsigned __int64 a2, __m128i a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r13d
  unsigned int v6; // r15d
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // r14
  int v12; // edx
  __int64 v13; // rsi
  unsigned __int8 *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // r14
  unsigned __int8 *v20; // r13
  unsigned __int16 *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // r8
  int v25; // eax
  __int64 v26; // r9
  __int16 *v27; // rax
  unsigned __int16 v28; // di
  __int64 v29; // r8
  __int64 v30; // r11
  __int64 (__fastcall *v31)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v32; // r13d
  int v33; // r9d
  __int16 v34; // cx
  __int64 v35; // r8
  __int64 v36; // rsi
  __int64 v37; // r12
  __int128 v38; // xmm1
  __int64 v39; // rdx
  __int64 v40; // r8
  unsigned __int16 v41; // ax
  __int16 v42; // cx
  __int64 v43; // rdx
  _QWORD *v44; // rcx
  __int64 v45; // rsi
  unsigned __int8 *v46; // r13
  __int64 v48; // rdx
  int v49; // eax
  __int64 v50; // rdx
  int *v51; // r14
  __int64 v52; // r9
  char v53; // si
  __int64 v54; // r8
  int v55; // ecx
  unsigned int v56; // edi
  unsigned int *v57; // rax
  unsigned __int8 *v58; // rcx
  __int64 v59; // rdx
  __int64 v60; // r8
  int *v61; // r13
  char v62; // di
  __int64 v63; // r8
  int v64; // ecx
  unsigned int v65; // esi
  __int64 v66; // rax
  int v67; // r9d
  __int64 v68; // r9
  __int64 v69; // rax
  __int16 v70; // dx
  __int64 v71; // rax
  unsigned int v72; // eax
  int v73; // eax
  __int64 v74; // rdx
  __int64 v75; // rax
  __int64 v76; // rcx
  _QWORD *v77; // r15
  __int64 v78; // rdx
  unsigned int v79; // edx
  unsigned __int16 *v80; // rax
  __int128 v81; // rax
  __int64 v82; // r9
  __int64 v83; // rax
  __int64 v84; // rax
  int v85; // eax
  int v86; // eax
  int v87; // r10d
  int v88; // r10d
  __int128 v89; // [rsp-20h] [rbp-120h]
  unsigned int v90; // [rsp+4h] [rbp-FCh]
  unsigned __int64 v91; // [rsp+8h] [rbp-F8h]
  __int64 v92; // [rsp+20h] [rbp-E0h]
  __int64 v93; // [rsp+20h] [rbp-E0h]
  __int16 v94; // [rsp+28h] [rbp-D8h]
  unsigned __int16 v95; // [rsp+30h] [rbp-D0h]
  __int128 v96; // [rsp+30h] [rbp-D0h]
  __int64 v97; // [rsp+70h] [rbp-90h] BYREF
  int v98; // [rsp+78h] [rbp-88h]
  unsigned __int16 v99; // [rsp+80h] [rbp-80h] BYREF
  __int64 v100; // [rsp+88h] [rbp-78h]
  __int128 v101; // [rsp+90h] [rbp-70h] BYREF
  __int128 v102; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v103; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v104; // [rsp+B8h] [rbp-48h]
  __int64 v105; // [rsp+C0h] [rbp-40h]

  v9 = *(_QWORD *)(a2 + 80);
  v97 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v97, v9, 1);
  v98 = *(_DWORD *)(a2 + 72);
  v10 = *(_QWORD *)(a2 + 40);
  v11 = *(_QWORD *)(v10 + 40);
  v12 = *(_DWORD *)(v11 + 24);
  if ( v12 != 11 && v12 != 35 )
    goto LABEL_5;
  a3 = _mm_loadu_si128((const __m128i *)v10);
  v38 = (__int128)_mm_loadu_si128((const __m128i *)(v10 + 40));
  v39 = *(_QWORD *)(*(_QWORD *)v10 + 48LL);
  v90 = *(_DWORD *)(v10 + 48);
  v40 = *(_QWORD *)(v39 + 8);
  v41 = *(_WORD *)v39;
  v99 = v41;
  v100 = v40;
  if ( v41 )
  {
    v93 = 0;
    v42 = word_4456580[v41 - 1];
  }
  else
  {
    v49 = sub_3009970((__int64)&v99, v9, v39, a5, v40);
    v40 = v100;
    HIWORD(v5) = HIWORD(v49);
    v42 = v49;
    v93 = v50;
    v41 = v99;
  }
  v43 = *(_QWORD *)(v11 + 96);
  LOWORD(v5) = v42;
  v44 = *(_QWORD **)(v43 + 24);
  if ( *(_DWORD *)(v43 + 32) > 0x40u )
    v44 = (_QWORD *)*v44;
  v91 = (unsigned __int64)v44;
  sub_2FE6CC0((__int64)&v103, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL), v41, v40);
  switch ( (_BYTE)v103 )
  {
    case 6:
      *(_QWORD *)&v101 = 0;
      DWORD2(v101) = 0;
      *(_QWORD *)&v102 = 0;
      DWORD2(v102) = 0;
      sub_375E8D0(a1, a3.m128i_u64[0], a3.m128i_i64[1], (__int64)&v101, (__int64)&v102);
      v69 = *(_QWORD *)(v101 + 48) + 16LL * DWORD2(v101);
      v70 = *(_WORD *)v69;
      v71 = *(_QWORD *)(v69 + 8);
      LOWORD(v103) = v70;
      v104 = v71;
      if ( v70 )
      {
        if ( (unsigned __int16)(v70 - 176) > 0x34u )
        {
LABEL_44:
          v72 = word_4456340[(unsigned __int16)v103 - 1];
LABEL_53:
          v77 = *(_QWORD **)(a1 + 8);
          v78 = v72;
          if ( v72 <= v91 )
          {
            v80 = (unsigned __int16 *)(*(_QWORD *)(v11 + 48) + 16LL * v90);
            *(_QWORD *)&v81 = sub_3400BD0(
                                *(_QWORD *)(a1 + 8),
                                v91 - v78,
                                (__int64)&v97,
                                *v80,
                                *((_QWORD *)v80 + 1),
                                0,
                                a3,
                                0);
            v58 = sub_3406EB0(v77, *(_DWORD *)(a2 + 24), (__int64)&v97, v5, v93, v82, v102, v81);
          }
          else
          {
            v58 = sub_3406EB0(*(_QWORD **)(a1 + 8), *(_DWORD *)(a2 + 24), (__int64)&v97, v5, v93, v68, v101, v38);
          }
          v60 = v79;
          goto LABEL_37;
        }
      }
      else if ( !sub_3007100((__int64)&v103) )
      {
        goto LABEL_52;
      }
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      if ( (_WORD)v103 )
      {
        if ( (unsigned __int16)(v103 - 176) <= 0x34u )
          sub_CA17B0(
            "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " MVT::getVectorElementCount() instead");
        goto LABEL_44;
      }
LABEL_52:
      v72 = sub_3007130((__int64)&v103, a3.m128i_i64[0]);
      goto LABEL_53;
    case 7:
      LODWORD(v103) = sub_375D5B0(a1, a3.m128i_u64[0], a3.m128i_i64[1]);
      v51 = sub_3805BC0(a1 + 1448, (int *)&v103);
      sub_37593F0(a1, v51);
      v53 = *(_BYTE *)(a1 + 512) & 1;
      if ( v53 )
      {
        v54 = a1 + 520;
        v55 = 7;
      }
      else
      {
        v76 = *(unsigned int *)(a1 + 528);
        v54 = *(_QWORD *)(a1 + 520);
        if ( !(_DWORD)v76 )
          goto LABEL_62;
        v55 = v76 - 1;
      }
      v56 = v55 & (37 * *v51);
      v57 = (unsigned int *)(v54 + 24LL * v56);
      v52 = *v57;
      if ( *v51 == (_DWORD)v52 )
      {
LABEL_36:
        *(_QWORD *)&v96 = *((_QWORD *)v57 + 1);
        *((_QWORD *)&v96 + 1) = v57[4] | a3.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        v58 = sub_3406EB0(*(_QWORD **)(a1 + 8), *(_DWORD *)(a2 + 24), (__int64)&v97, v5, v93, v52, v96, v38);
        v60 = v59;
LABEL_37:
        v46 = 0;
        sub_3760E70(a1, a2, 0, (unsigned __int64)v58, v60);
        goto LABEL_27;
      }
      v86 = 1;
      while ( (_DWORD)v52 != -1 )
      {
        v88 = v86 + 1;
        v56 = v55 & (v86 + v56);
        v57 = (unsigned int *)(v54 + 24LL * v56);
        v52 = *v57;
        if ( *v51 == (_DWORD)v52 )
          goto LABEL_36;
        v86 = v88;
      }
      if ( v53 )
      {
        v83 = 192;
        goto LABEL_63;
      }
      v76 = *(unsigned int *)(a1 + 528);
LABEL_62:
      v83 = 24 * v76;
LABEL_63:
      v57 = (unsigned int *)(v54 + v83);
      goto LABEL_36;
    case 5:
      LODWORD(v103) = sub_375D5B0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
      v61 = sub_3805BC0(a1 + 1256, (int *)&v103);
      sub_37593F0(a1, v61);
      v62 = *(_BYTE *)(a1 + 512) & 1;
      if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
      {
        v63 = a1 + 520;
        v64 = 7;
      }
      else
      {
        v75 = *(unsigned int *)(a1 + 528);
        v63 = *(_QWORD *)(a1 + 520);
        if ( !(_DWORD)v75 )
          goto LABEL_65;
        v64 = v75 - 1;
      }
      v65 = v64 & (37 * *v61);
      v66 = v63 + 24LL * v65;
      v67 = *(_DWORD *)v66;
      if ( *v61 == *(_DWORD *)v66 )
      {
LABEL_41:
        v58 = *(unsigned __int8 **)(v66 + 8);
        v60 = *(unsigned int *)(v66 + 16);
        goto LABEL_37;
      }
      v85 = 1;
      while ( v67 != -1 )
      {
        v87 = v85 + 1;
        v65 = v64 & (v85 + v65);
        v66 = v63 + 24LL * v65;
        v67 = *(_DWORD *)v66;
        if ( *v61 == *(_DWORD *)v66 )
          goto LABEL_41;
        v85 = v87;
      }
      if ( v62 )
      {
        v84 = 192;
        goto LABEL_66;
      }
      v75 = *(unsigned int *)(a1 + 528);
LABEL_65:
      v84 = 24 * v75;
LABEL_66:
      v66 = v63 + v84;
      goto LABEL_41;
  }
  v10 = *(_QWORD *)(a2 + 40);
LABEL_5:
  v13 = *(_QWORD *)v10;
  v14 = sub_375A8B0(a1, *(_QWORD *)v10, *(_QWORD *)(v10 + 8), a3);
  v19 = v18;
  v20 = v14;
  v21 = (unsigned __int16 *)(*((_QWORD *)v14 + 6) + 16LL * (unsigned int)v18);
  v22 = *v21;
  v23 = *((_QWORD *)v21 + 1);
  LOWORD(v103) = v22;
  v104 = v23;
  if ( (_WORD)v22 )
  {
    v24 = 0;
    LOWORD(v25) = word_4456580[(int)v22 - 1];
  }
  else
  {
    v25 = sub_3009970((__int64)&v103, v13, v22, v15, v16);
    HIWORD(v6) = HIWORD(v25);
    v24 = v48;
  }
  LOWORD(v6) = v25;
  *((_QWORD *)&v89 + 1) = v19;
  *(_QWORD *)&v89 = v20;
  sub_3406EB0(*(_QWORD **)(a1 + 8), 0x9Eu, (__int64)&v97, v6, v24, v17, v89, *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
  v26 = *(_QWORD *)a1;
  v27 = *(__int16 **)(a2 + 48);
  v28 = *v27;
  v29 = *((_QWORD *)v27 + 1);
  v30 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL);
  v95 = *v27;
  v31 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(**(_QWORD **)a1 + 592LL);
  if ( v31 == sub_2D56A50 )
  {
    HIWORD(v32) = 0;
    sub_2FE6CC0((__int64)&v103, v26, v30, v28, v29);
    v34 = v104;
    v35 = v105;
  }
  else
  {
    v73 = v31(v26, v30, v95, v29);
    HIWORD(v32) = HIWORD(v73);
    v35 = v74;
    v34 = v73;
  }
  v36 = *(_QWORD *)(a2 + 80);
  v37 = *(_QWORD *)(a1 + 8);
  v103 = v36;
  if ( v36 )
  {
    v92 = v35;
    v94 = v34;
    sub_B96E90((__int64)&v103, v36, 1);
    v35 = v92;
    v34 = v94;
  }
  LODWORD(v104) = *(_DWORD *)(a2 + 72);
  if ( v95 == 11 )
  {
    v45 = 236;
  }
  else if ( v34 == 11 )
  {
    v45 = 237;
  }
  else if ( v95 == 10 )
  {
    v45 = 240;
  }
  else
  {
    if ( v34 != 10 )
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    v45 = 241;
  }
  LOWORD(v32) = v34;
  v46 = sub_33FAF80(v37, v45, (__int64)&v103, v32, v35, v33, a3);
  if ( v103 )
    sub_B91220((__int64)&v103, v103);
LABEL_27:
  if ( v97 )
    sub_B91220((__int64)&v97, v97);
  return v46;
}
