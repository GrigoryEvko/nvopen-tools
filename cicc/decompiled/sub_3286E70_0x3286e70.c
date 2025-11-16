// Function: sub_3286E70
// Address: 0x3286e70
//
__int64 __fastcall sub_3286E70(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  int v7; // r11d
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // r13
  int v13; // eax
  bool v14; // zf
  __int64 v15; // rdi
  __int64 v16; // rax
  int v17; // r11d
  __int64 v18; // r12
  __int64 *v20; // rax
  char v21; // al
  int v22; // r9d
  __int128 *v23; // rax
  __int128 *v24; // rax
  bool v25; // al
  unsigned __int64 v26; // rdi
  __int128 v27; // rax
  int v28; // r9d
  __int64 v29; // rdi
  __int64 (*v30)(); // rax
  int v31; // eax
  __int64 v32; // rax
  int v33; // edx
  int v34; // eax
  _DWORD *v35; // rdi
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // r14
  __int64 v39; // rdx
  __int64 v40; // r15
  __int64 v41; // rdx
  int v42; // eax
  int v43; // edx
  int v44; // r9d
  int v45; // r9d
  __int64 v46; // rax
  __int64 v47; // rcx
  int v48; // r9d
  _DWORD *v49; // rdx
  unsigned __int16 v50; // cx
  int v51; // eax
  __int128 v52; // rax
  int v53; // r9d
  __int64 v54; // r14
  __int128 v55; // rax
  int v56; // r9d
  __int128 v57; // rax
  int v58; // r9d
  __int64 v59; // rax
  char v60; // al
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // rax
  int v65; // edx
  int v66; // r9d
  __int64 *v67; // r15
  __int128 v68; // rax
  int v69; // r9d
  __int64 v70; // rax
  char v71; // al
  int v72; // r9d
  char v73; // al
  int v74; // eax
  __int128 v75; // rax
  bool v76; // al
  __int128 v77; // rax
  int v78; // r9d
  __int64 v79; // r15
  __int128 v80; // rax
  int v81; // r9d
  __int128 v82; // rax
  int v83; // r9d
  __int128 v84; // [rsp-20h] [rbp-130h]
  __int128 v85; // [rsp-20h] [rbp-130h]
  __int128 v86; // [rsp-10h] [rbp-120h]
  _DWORD *v87; // [rsp+0h] [rbp-110h]
  int v88; // [rsp+Ch] [rbp-104h]
  int v89; // [rsp+10h] [rbp-100h]
  int v90; // [rsp+10h] [rbp-100h]
  __int64 v91; // [rsp+18h] [rbp-F8h]
  int v92; // [rsp+18h] [rbp-F8h]
  __int64 v93; // [rsp+18h] [rbp-F8h]
  bool v94; // [rsp+18h] [rbp-F8h]
  __int64 v95; // [rsp+18h] [rbp-F8h]
  char v96; // [rsp+18h] [rbp-F8h]
  unsigned __int16 v97; // [rsp+20h] [rbp-F0h]
  __int64 v98; // [rsp+20h] [rbp-F0h]
  __int128 v100; // [rsp+30h] [rbp-E0h]
  __int128 v101; // [rsp+30h] [rbp-E0h]
  __m128i v102; // [rsp+40h] [rbp-D0h]
  __m128i v103; // [rsp+50h] [rbp-C0h]
  __int128 v104; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v105; // [rsp+70h] [rbp-A0h] BYREF
  int v106; // [rsp+78h] [rbp-98h]
  __int128 v107; // [rsp+80h] [rbp-90h] BYREF
  __int128 v108; // [rsp+90h] [rbp-80h] BYREF
  int v109; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v110; // [rsp+A8h] [rbp-68h]
  unsigned __int64 v111; // [rsp+B0h] [rbp-60h] BYREF
  unsigned int v112; // [rsp+B8h] [rbp-58h]
  __int128 *v113; // [rsp+C0h] [rbp-50h]
  int v114; // [rsp+C8h] [rbp-48h]
  char v115; // [rsp+CCh] [rbp-44h]
  __int128 *v116; // [rsp+D0h] [rbp-40h]
  int v117; // [rsp+D8h] [rbp-38h]
  char v118; // [rsp+DCh] [rbp-34h]

  v7 = a3;
  *(_QWORD *)&v100 = a4;
  *(_QWORD *)&v104 = a2;
  *((_QWORD *)&v104 + 1) = a3;
  v10 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  *((_QWORD *)&v100 + 1) = a5;
  v11 = *(_QWORD *)(a6 + 80);
  v12 = *(_QWORD *)(v10 + 8);
  v97 = *(_WORD *)v10;
  v88 = a5;
  v105 = v11;
  if ( v11 )
  {
    v89 = a3;
    v91 = a6;
    sub_B96E90((__int64)&v105, v11, 1);
    v7 = v89;
    a6 = v91;
  }
  v13 = *(_DWORD *)(a6 + 72);
  v115 = 0;
  v14 = *(_DWORD *)(a4 + 24) == 190;
  *(_QWORD *)&v108 = 0;
  v106 = v13;
  v113 = &v107;
  *(_QWORD *)&v107 = 0;
  DWORD2(v107) = 0;
  DWORD2(v108) = 0;
  v109 = 190;
  LODWORD(v110) = 57;
  v112 = 64;
  v111 = 0;
  v116 = &v108;
  v118 = 0;
  if ( !v14 )
    goto LABEL_4;
  v20 = *(__int64 **)(a4 + 40);
  if ( *(_DWORD *)(*v20 + 24) != 57 )
    goto LABEL_4;
  v90 = v7;
  v93 = *v20;
  v21 = sub_32657E0((__int64)&v111, **(_QWORD **)(*v20 + 40));
  v7 = v90;
  if ( v21 )
  {
    v23 = v113;
    v103 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v93 + 40) + 40LL));
    *(_QWORD *)v113 = v103.m128i_i64[0];
    *((_DWORD *)v23 + 2) = v103.m128i_i32[2];
    if ( !v115 || v114 == (v114 & *(_DWORD *)(v93 + 28)) )
    {
      v24 = v116;
      v102 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a4 + 40) + 40LL));
      *(_QWORD *)v116 = v102.m128i_i64[0];
      *((_DWORD *)v24 + 2) = v102.m128i_i32[2];
      if ( !v118 )
      {
        if ( v112 > 0x40 && v111 )
        {
          j_j___libc_free_0_0(v111);
          v15 = *a1;
        }
        else
        {
          v15 = *a1;
        }
        goto LABEL_19;
      }
      v25 = (v117 & *(_DWORD *)(a4 + 28)) == v117;
      if ( v112 <= 0x40 )
        goto LABEL_18;
      v26 = v111;
      if ( !v111 )
        goto LABEL_18;
      goto LABEL_17;
    }
  }
  if ( v112 <= 0x40 || (v26 = v111, v25 = 0, !v111) )
  {
LABEL_4:
    v15 = *a1;
    goto LABEL_5;
  }
LABEL_17:
  v94 = v25;
  j_j___libc_free_0_0(v26);
  v7 = v90;
  v25 = v94;
LABEL_18:
  v15 = *a1;
  if ( v25 )
  {
LABEL_19:
    *(_QWORD *)&v27 = sub_3406EB0(v15, 190, (unsigned int)&v105, v97, v12, v22, v107, v108);
    v18 = sub_3406EB0(v15, 57, (unsigned int)&v105, v97, v12, v28, v104, v27);
    goto LABEL_7;
  }
LABEL_5:
  v92 = v7;
  v16 = sub_3271370(1, v104, *((__int64 *)&v104 + 1), a4, v15, (__int64)&v105);
  v17 = v92;
  if ( v16 )
  {
    v18 = v16;
    goto LABEL_7;
  }
  v29 = a1[1];
  v30 = *(__int64 (**)())(*(_QWORD *)v29 + 456LL);
  if ( v30 == sub_2FE3090 )
  {
LABEL_21:
    v31 = *(_DWORD *)(a2 + 24);
    goto LABEL_22;
  }
  v6 = v97;
  v60 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))v30)(v29, v97, v12);
  v17 = v92;
  v14 = v60 == 0;
  v31 = *(_DWORD *)(a2 + 24);
  if ( v14 && v31 == 56 )
  {
    v64 = *(_QWORD *)(a2 + 56);
    if ( !v64 )
      goto LABEL_31;
    v65 = 1;
    do
    {
      if ( v92 == *(_DWORD *)(v64 + 8) )
      {
        if ( !v65 )
          goto LABEL_31;
        v64 = *(_QWORD *)(v64 + 32);
        if ( !v64 )
          goto LABEL_87;
        if ( v92 == *(_DWORD *)(v64 + 8) )
          goto LABEL_31;
        v65 = 0;
      }
      v64 = *(_QWORD *)(v64 + 32);
    }
    while ( v64 );
    if ( v65 == 1 )
      goto LABEL_31;
LABEL_87:
    v73 = sub_33E0780(
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
            0,
            v61,
            v62,
            v63);
    v17 = v92;
    if ( v73 )
    {
      if ( *((int *)a1 + 6) > 2 || (v74 = *(_DWORD *)(a2 + 28), (v74 & 1) == 0) && (v74 & 2) == 0 )
      {
        v67 = a1;
        *(_QWORD *)&v75 = sub_34074A0(
                            *a1,
                            &v105,
                            **(_QWORD **)(a2 + 40),
                            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                            v97,
                            v12);
        v86 = v75;
        v85 = v100;
        goto LABEL_77;
      }
    }
    goto LABEL_21;
  }
LABEL_22:
  if ( v31 != 57 )
  {
LABEL_43:
    if ( v31 == 58 )
    {
      v46 = *(_QWORD *)(a2 + 40);
      if ( *(_QWORD *)v46 != a4 || *(_DWORD *)(v46 + 8) != v88 )
        goto LABEL_31;
      if ( (unsigned __int8)sub_326A930(*(_QWORD *)(v46 + 40), *(_QWORD *)(v46 + 48), 1u)
        && (unsigned __int8)sub_3286E00(&v104) )
      {
        LOWORD(v6) = v97;
        v79 = *a1;
        *(_QWORD *)&v80 = sub_3400BD0(*a1, 1, (unsigned int)&v105, v6, v12, 0, 0, v47);
        *(_QWORD *)&v82 = sub_3406EB0(
                            v79,
                            56,
                            (unsigned int)&v105,
                            v6,
                            v12,
                            v81,
                            *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                            v80);
        v59 = sub_3406EB0(*a1, 58, (unsigned int)&v105, v6, v12, v83, *(_OWORD *)*(_QWORD *)(a2 + 40), v82);
        goto LABEL_60;
      }
      v31 = *(_DWORD *)(a2 + 24);
    }
    if ( v31 != 213 || sub_3263630(**(_QWORD **)(a2 + 40), *(_DWORD *)(*(_QWORD *)(a2 + 40) + 8LL)) != 1 )
      goto LABEL_31;
    v110 = v12;
    LOWORD(v109) = v97;
    v49 = (_DWORD *)a1[1];
    if ( v97 )
    {
      v50 = v97 - 17;
      if ( (unsigned __int16)(v97 - 10) > 6u && (unsigned __int16)(v97 - 126) > 0x31u )
      {
        if ( v50 > 0xD3u )
        {
LABEL_55:
          v51 = v49[15];
          goto LABEL_56;
        }
        goto LABEL_96;
      }
      if ( v50 <= 0xD3u )
      {
LABEL_96:
        v51 = v49[17];
LABEL_56:
        if ( v51 == 1 )
        {
          LOWORD(v6) = v97;
          *(_QWORD *)&v52 = sub_33FAF80(*a1, 214, (unsigned int)&v105, v6, v12, v48, *(_OWORD *)*(_QWORD *)(a2 + 40));
          v18 = sub_3406EB0(*a1, 57, (unsigned int)&v105, v6, v12, v53, v100, v52);
          goto LABEL_7;
        }
        goto LABEL_31;
      }
    }
    else
    {
      v87 = (_DWORD *)a1[1];
      v96 = sub_3007030((__int64)&v109);
      v76 = sub_30070B0((__int64)&v109);
      v49 = v87;
      if ( v76 )
        goto LABEL_96;
      if ( !v96 )
        goto LABEL_55;
    }
    v51 = v49[16];
    goto LABEL_56;
  }
  v32 = *(_QWORD *)(a2 + 56);
  if ( v32 )
  {
    v33 = 1;
    do
    {
      if ( v17 == *(_DWORD *)(v32 + 8) )
      {
        if ( !v33 )
          goto LABEL_31;
        v32 = *(_QWORD *)(v32 + 32);
        if ( !v32 )
          goto LABEL_75;
        if ( v17 == *(_DWORD *)(v32 + 8) )
          goto LABEL_31;
        v33 = 0;
      }
      v32 = *(_QWORD *)(v32 + 32);
    }
    while ( v32 );
    if ( v33 == 1 )
      goto LABEL_31;
LABEL_75:
    if ( !(unsigned __int8)sub_326A930(
                             *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                             *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                             1u) )
    {
      if ( !(unsigned __int8)sub_326A930(**(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), 1u) )
      {
        v31 = *(_DWORD *)(a2 + 24);
        goto LABEL_43;
      }
      LOWORD(v6) = v97;
      *(_QWORD *)&v77 = sub_3406EB0(
                          *a1,
                          57,
                          (unsigned int)&v105,
                          v6,
                          v12,
                          v45,
                          v100,
                          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
      v70 = sub_3406EB0(*a1, 56, (unsigned int)&v105, v6, v12, v78, v77, *(_OWORD *)*(_QWORD *)(a2 + 40));
      goto LABEL_78;
    }
    v67 = a1;
    LOWORD(v6) = v97;
    *(_QWORD *)&v68 = sub_3406EB0(*a1, 56, (unsigned int)&v105, v6, v12, v66, *(_OWORD *)*(_QWORD *)(a2 + 40), v100);
    v86 = *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL);
    v85 = v68;
LABEL_77:
    v70 = sub_3406EB0(*v67, 57, (unsigned int)&v105, v6, v12, v69, v85, v86);
LABEL_78:
    v18 = v70;
    goto LABEL_7;
  }
LABEL_31:
  v34 = *(_DWORD *)(a4 + 24);
  if ( v34 == 222 )
  {
    if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 40) + 40LL) + 96LL) != 2 )
      goto LABEL_33;
    LOWORD(v6) = v97;
    v54 = *a1;
    *(_QWORD *)&v55 = sub_3400BD0(*a1, 1, (unsigned int)&v105, v6, v12, 0, 0);
    *(_QWORD *)&v57 = sub_3406EB0(v54, 186, (unsigned int)&v105, v6, v12, v56, *(_OWORD *)*(_QWORD *)(a4 + 40), v55);
    v59 = sub_3406EB0(*a1, 57, (unsigned int)&v105, v6, v12, v58, v104, v57);
LABEL_60:
    v18 = v59;
    goto LABEL_7;
  }
  if ( v34 == 72 )
  {
    v71 = sub_33CF170(*(_QWORD *)(*(_QWORD *)(a4 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a4 + 40) + 48LL));
    if ( !v88 )
    {
      if ( v71 )
      {
        v18 = sub_3412970(
                *a1,
                72,
                (unsigned int)&v105,
                *(_QWORD *)(a4 + 48),
                *(_DWORD *)(a4 + 68),
                v72,
                v104,
                *(_OWORD *)*(_QWORD *)(a4 + 40),
                *(_OWORD *)(*(_QWORD *)(a4 + 40) + 80LL));
        goto LABEL_7;
      }
    }
  }
LABEL_33:
  v35 = (_DWORD *)a1[1];
  v36 = 1;
  if ( (v97 == 1 || v97 && (v36 = v97, *(_QWORD *)&v35[2 * v97 + 28]))
    && (v35[125 * v36 + 1621] & 0xFB0000) == 0
    && (v37 = sub_32719C0(v35, v100, *((unsigned __int64 *)&v100 + 1), 0), v38 = v37, v40 = v39, v37) )
  {
    v95 = v37;
    LOWORD(v6) = v97;
    v98 = *a1;
    *(_QWORD *)&v101 = sub_3400BD0(*a1, 0, (unsigned int)&v105, v6, v12, 0, 0);
    *((_QWORD *)&v101 + 1) = v41;
    v42 = sub_33E5110(
            *a1,
            v6,
            v12,
            *(unsigned __int16 *)(*(_QWORD *)(v95 + 48) + 16LL * (unsigned int)v40),
            *(_QWORD *)(*(_QWORD *)(v95 + 48) + 16LL * (unsigned int)v40 + 8));
    *((_QWORD *)&v84 + 1) = v40;
    *(_QWORD *)&v84 = v38;
    v18 = sub_3412970(v98, 72, (unsigned int)&v105, v42, v43, v44, v104, v101, v84);
  }
  else
  {
    v18 = 0;
  }
LABEL_7:
  if ( v105 )
    sub_B91220((__int64)&v105, v105);
  return v18;
}
