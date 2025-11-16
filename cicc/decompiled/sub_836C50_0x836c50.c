// Function: sub_836C50
// Address: 0x836c50
//
__int64 __fastcall sub_836C50(
        const __m128i *a1,
        __int64 a2,
        const __m128i *a3,
        unsigned int a4,
        unsigned int a5,
        unsigned int a6,
        __int64 a7,
        int a8,
        int a9,
        __int64 a10,
        __m128i *a11,
        unsigned int *a12,
        __int64 **a13)
{
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 *v19; // r9
  __int64 v20; // r14
  __int64 v21; // r12
  __int64 v22; // rax
  const __m128i *v23; // r12
  __int64 v24; // rdx
  __int64 v25; // r8
  __int64 v26; // r9
  int v27; // r14d
  __int64 v28; // rax
  __int64 v29; // r8
  __int64 v30; // rcx
  const __m128i *v31; // r10
  __int16 v32; // ax
  bool v33; // r11
  _QWORD *v34; // rdi
  __int64 *v35; // r15
  __int64 v36; // rsi
  __int64 v37; // rcx
  __int64 v38; // rbx
  __int64 v39; // r8
  __int64 *v40; // r9
  __int64 v41; // rdx
  __int64 v42; // r13
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 *v46; // r9
  int v48; // eax
  bool v49; // al
  __int64 v50; // rax
  unsigned int v51; // eax
  __int64 *v52; // r13
  int v53; // eax
  const __m128i *v54; // rdi
  int v55; // eax
  int v56; // eax
  __int64 v57; // rdi
  __m128i *v58; // rax
  __m128i v59; // xmm2
  __m128i v60; // xmm3
  __int64 v61; // rax
  __int64 v62; // rax
  bool v63; // zf
  const __m128i *v64; // rax
  __int64 v65; // rdi
  const __m128i *v66; // rax
  int v67; // eax
  const __m128i *v68; // r10
  __int64 v69; // rdx
  int v70; // esi
  int v71; // eax
  unsigned __int8 v72; // [rsp+8h] [rbp-128h]
  const __m128i *v73; // [rsp+8h] [rbp-128h]
  const __m128i *v74; // [rsp+8h] [rbp-128h]
  const __m128i *v75; // [rsp+10h] [rbp-120h]
  bool v76; // [rsp+10h] [rbp-120h]
  const __m128i *v77; // [rsp+10h] [rbp-120h]
  bool v78; // [rsp+10h] [rbp-120h]
  const __m128i *v79; // [rsp+10h] [rbp-120h]
  bool v80; // [rsp+10h] [rbp-120h]
  const __m128i *v81; // [rsp+10h] [rbp-120h]
  bool v82; // [rsp+18h] [rbp-118h]
  const __m128i *v83; // [rsp+18h] [rbp-118h]
  bool v84; // [rsp+18h] [rbp-118h]
  _QWORD *v85; // [rsp+18h] [rbp-118h]
  bool v86; // [rsp+18h] [rbp-118h]
  bool v87; // [rsp+18h] [rbp-118h]
  __int64 v88; // [rsp+20h] [rbp-110h]
  __int64 v89; // [rsp+28h] [rbp-108h]
  unsigned int v91; // [rsp+34h] [rbp-FCh]
  char v92; // [rsp+38h] [rbp-F8h]
  int v93; // [rsp+38h] [rbp-F8h]
  const __m128i *v94; // [rsp+38h] [rbp-F8h]
  _BOOL4 v95; // [rsp+40h] [rbp-F0h]
  bool v96; // [rsp+44h] [rbp-ECh]
  int v97; // [rsp+44h] [rbp-ECh]
  bool v98; // [rsp+48h] [rbp-E8h]
  const __m128i *v99; // [rsp+48h] [rbp-E8h]
  __int64 v100; // [rsp+48h] [rbp-E8h]
  char v102; // [rsp+58h] [rbp-D8h]
  int v103; // [rsp+58h] [rbp-D8h]
  __int64 v105; // [rsp+60h] [rbp-D0h]
  unsigned int v106; // [rsp+60h] [rbp-D0h]
  __m128i *v107; // [rsp+60h] [rbp-D0h]
  const __m128i *v108; // [rsp+68h] [rbp-C8h]
  int v109; // [rsp+7Ch] [rbp-B4h] BYREF
  int v110; // [rsp+80h] [rbp-B0h] BYREF
  unsigned int v111; // [rsp+84h] [rbp-ACh] BYREF
  const __m128i *v112; // [rsp+88h] [rbp-A8h] BYREF
  char v113[8]; // [rsp+90h] [rbp-A0h] BYREF
  int v114; // [rsp+98h] [rbp-98h]
  __m128i v115; // [rsp+C0h] [rbp-70h] BYREF
  __m128i v116; // [rsp+D0h] [rbp-60h] BYREF
  __m128i v117[5]; // [rsp+E0h] [rbp-50h] BYREF

  v14 = a2;
  v108 = a1;
  v109 = 0;
  v110 = 0;
  v15 = ((__int64 (*)(void))sub_82BD70)();
  v20 = *(_QWORD *)(v15 + 1024);
  v21 = v15;
  if ( v20 == *(_QWORD *)(v15 + 1016) )
    sub_8332F0(v15, a2, v16, v17, v18, v19);
  v22 = *(_QWORD *)(v21 + 1008) + 40 * v20;
  if ( v22 )
  {
    *(_BYTE *)v22 &= 0xFCu;
    *(_QWORD *)(v22 + 8) = 0;
    *(_QWORD *)(v22 + 16) = 0;
    *(_QWORD *)(v22 + 24) = 0;
    *(_QWORD *)(v22 + 32) = 0;
  }
  *(_QWORD *)(v21 + 1024) = v20 + 1;
  v23 = a3;
  *a12 = 0;
  *(_OWORD *)a10 = 0;
  *(_OWORD *)(a10 + 16) = 0;
  *(_OWORD *)(a10 + 32) = 0;
  if ( a3[8].m128i_i8[12] == 12 )
  {
    do
      v23 = (const __m128i *)v23[10].m128i_i64[0];
    while ( v23[8].m128i_i8[12] == 12 );
  }
  else
  {
    v23 = a3;
  }
  if ( (unsigned int)sub_8D23B0(v23) && (unsigned int)sub_8D3A70(v23) )
  {
    a2 = 0;
    sub_8AD220(v23, 0);
  }
  v105 = *(_QWORD *)(v23->m128i_i64[0] + 96);
  if ( v14 )
  {
    if ( *(_BYTE *)(v14 + 8) )
    {
LABEL_10:
      v27 = 0;
      v28 = sub_6E1A20(v14);
      v92 = 0;
      LODWORD(v30) = 0;
      v31 = 0;
      v89 = v28;
      v112 = 0;
      v91 = a5;
      v108 = 0;
      v88 = 0;
      v95 = 0;
      v96 = 0;
      v102 = 0;
      goto LABEL_11;
    }
    v108 = (const __m128i *)(*(_QWORD *)(v14 + 24) + 8LL);
  }
  else if ( a1[1].m128i_i8[0] == 5 )
  {
    v14 = a1[9].m128i_i64[0];
    if ( !v14 )
      BUG();
    goto LABEL_10;
  }
  v30 = 0;
  v31 = (const __m128i *)v108->m128i_i64[0];
  if ( (*(_BYTE *)(v108->m128i_i64[0] + 140) & 0xFB) == 8 )
  {
    v100 = v108->m128i_i64[0];
    a2 = dword_4F077C4 != 2;
    v51 = sub_8D4C10(v108->m128i_i64[0], a2);
    v31 = (const __m128i *)v100;
    v30 = v51;
    if ( *(_BYTE *)(v100 + 140) == 12 )
    {
      do
        v31 = (const __m128i *)v31[10].m128i_i64[0];
      while ( v31[8].m128i_i8[12] == 12 );
    }
  }
  v89 = (__int64)v108[4].m128i_i64 + 4;
  if ( v31 == v23
    || (a2 = (__int64)v23,
        v103 = v30,
        v99 = v31,
        v48 = sub_8D97D0(v31, v23, 0, v30, v25),
        v31 = v99,
        LODWORD(v30) = v103,
        (v27 = v48) != 0) )
  {
    v96 = 0;
    v102 = 1;
    v88 = 0;
    v95 = (unsigned __int8)(v31[8].m128i_i8[12] - 9) <= 2u;
    v49 = 1;
  }
  else
  {
    if ( (unsigned __int8)(v99[8].m128i_i8[12] - 9) > 2u )
    {
      v14 = 0;
      v112 = 0;
      v95 = 0;
      v91 = a5;
      v88 = 0;
      v92 = 0;
      v96 = 0;
      v102 = 0;
      goto LABEL_11;
    }
    a2 = (__int64)v23;
    v93 = v103;
    v50 = sub_8D5CE0(v99, v23);
    v102 = 0;
    LODWORD(v30) = v93;
    v88 = v50;
    v31 = v99;
    v96 = v50 != 0;
    v49 = v50 != 0;
    v95 = 1;
  }
  v29 = a5;
  v112 = 0;
  LOBYTE(v24) = v49 && a5 != 0;
  v92 = v24;
  if ( (_BYTE)v24 )
  {
    v91 = 0;
    v14 = 0;
    v27 = 1;
  }
  else
  {
    v92 = v49;
    v14 = 0;
    v27 = 0;
    v91 = a5;
  }
LABEL_11:
  v32 = *(_WORD *)(v105 + 176) & 0x4008;
  v33 = v32 == 0x4000;
  v98 = a11 != 0;
  LOBYTE(a4) = v32 == 0x4000 && a4 != 0;
  if ( !(_BYTE)a4 || (v30 & 0xFFFFFFFE) != 0 )
  {
    LOBYTE(v24) = v14 != 0;
    LOBYTE(a4) = 0;
  }
  else
  {
    if ( v102 )
    {
      *(_BYTE *)(a10 + 16) |= 1u;
      v106 = 1;
      goto LABEL_16;
    }
    if ( v14 )
    {
      v79 = v31;
      v86 = v32 == 0x4000;
      v58 = sub_73C570(v23, 1);
      a2 = sub_72D600(v58);
      v34 = (_QWORD *)v14;
      sub_84A950(v14, a2, 0, 1, 0, v113);
      v33 = v86;
      v24 = a4;
      v31 = v79;
      if ( v114 != 7 )
      {
        *(_BYTE *)(a10 + 16) |= 1u;
        if ( a11 )
        {
          v59 = _mm_loadu_si128(&v116);
          v60 = _mm_loadu_si128(v117);
          *a11 = _mm_loadu_si128(&v115);
          a11[1] = v59;
          a11[2] = v60;
        }
        v98 = 0;
        v106 = 1;
        goto LABEL_18;
      }
    }
    else
    {
      v24 = 0;
    }
  }
  if ( dword_4F04C44 != -1
    || (v61 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v61 + 6) & 6) != 0)
    || *(_BYTE *)(v61 + 4) == 12 )
  {
    if ( (v23[11].m128i_i8[1] & 0x20) == 0 )
    {
      if ( !dword_4F077BC )
        goto LABEL_62;
      a2 = (unsigned int)qword_4F077B4;
      if ( (_DWORD)qword_4F077B4 )
        goto LABEL_62;
      if ( qword_4F077A8 > 0x18768u )
        goto LABEL_62;
      if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) & 0x10) != 0 )
        goto LABEL_62;
      if ( (a9 & 0x10) == 0 )
        goto LABEL_62;
      v72 = v24;
      v75 = v31;
      v82 = v33;
      v53 = sub_8D23B0(v23);
      v33 = v82;
      v31 = v75;
      v24 = v72;
      if ( !v53 )
      {
        if ( !v95 )
          goto LABEL_62;
        v54 = v75;
        v76 = v82;
        v83 = v31;
        v55 = sub_8D23B0(v54);
        v31 = v83;
        v33 = v76;
        v24 = v72;
        if ( !v55 )
          goto LABEL_62;
      }
      if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) != 0 )
      {
LABEL_62:
        if ( (_BYTE)v24 )
        {
          v77 = v31;
          v84 = v33;
          v56 = sub_82EC50(v14, a2, v24);
          v33 = v84;
          v31 = v77;
          if ( !v56 )
          {
LABEL_64:
            a9 |= 0x4000000u;
            v85 = *(_QWORD **)(v14 + 24);
            goto LABEL_65;
          }
        }
        else
        {
          v81 = v31;
          v87 = v33;
          if ( !sub_82ED00((__int64)v108, a2) )
          {
            v33 = v87;
            v31 = v81;
            goto LABEL_83;
          }
        }
      }
    }
    *(_BYTE *)(a10 + 17) |= 1u;
    v106 = 1;
    goto LABEL_16;
  }
  if ( (_BYTE)v24 )
    goto LABEL_64;
LABEL_83:
  v74 = v31;
  v80 = v33;
  v62 = sub_6E3060(v108);
  v31 = v74;
  v33 = v80;
  v85 = (_QWORD *)v62;
LABEL_65:
  v57 = *(_QWORD *)(v105 + 8);
  if ( v57 )
  {
    v73 = v31;
    v78 = v33;
    sub_8360D0(
      v57,
      0,
      0,
      (__int64)v85,
      (_QWORD *)v14,
      0,
      0,
      1,
      a6,
      v91 == 0,
      0,
      0,
      0,
      0,
      0,
      a9,
      10,
      (__int64 *)&v112,
      0,
      &v109,
      &v110);
    v31 = v73;
    v33 = v78;
  }
  if ( v95 && !v92 )
  {
    if ( !v91 && !a7 )
    {
      if ( (*(_BYTE *)(v105 + 176) & 8) == 0 && v33 )
      {
        v97 = qword_4D0495C == 0;
LABEL_105:
        v94 = v31;
        v67 = sub_8D23B0(v31);
        v68 = v94;
        if ( v67 )
        {
          v71 = sub_8D3A70(v94);
          v68 = v94;
          if ( v71 )
          {
            sub_8AD220(v94, 0);
            v68 = v94;
          }
        }
        if ( (*(_BYTE *)(v105 + 178) & 0x10) != 0
          || (v69 = *(_QWORD *)(v68->m128i_i64[0] + 96), (*(_BYTE *)(v69 + 178) & 8) != 0)
          || *(_QWORD *)(v69 + 48) )
        {
          v70 = (int)a3;
          if ( v97 )
          {
            v107 = sub_73C570(v23, 1);
            v70 = (int)v107;
            a6 = 0;
            a7 = sub_72D600(v107);
            v91 = 0;
            a8 = v97;
          }
          sub_83EB80((_DWORD)v108, v70, (_DWORD)a3, 0, 0, v91, a6, a7, a8, a9, (__int64)&v112);
        }
        goto LABEL_70;
      }
      if ( !qword_4D0495C )
      {
        if ( !dword_4D04460 )
          goto LABEL_70;
        v97 = a9 & 8;
        if ( (a9 & 8) != 0 )
          goto LABEL_70;
        goto LABEL_105;
      }
    }
    v97 = 0;
    goto LABEL_105;
  }
  if ( !v112 && ((unsigned __int8)a4 & v96) != 0 && v31 && !(unsigned int)sub_8D23B0(v31) )
  {
    *(_BYTE *)(a10 + 16) |= 1u;
    *(_BYTE *)(a10 + 36) |= 0x20u;
    *(_QWORD *)(a10 + 24) = v88;
    v106 = 1;
    goto LABEL_72;
  }
LABEL_70:
  sub_82D8D0((__int64 *)&v112, v89, &v111, a12, v29, v26);
  v106 = v111;
  if ( v111 )
  {
    v106 = 0;
    goto LABEL_72;
  }
  if ( v112 && !*a12 )
  {
    v63 = (unsigned int)sub_6E6010() == 0;
    v64 = v112;
    if ( v63 || (v65 = v112->m128i_i64[1]) == 0 )
    {
LABEL_97:
      *(__m128i *)a10 = _mm_loadu_si128(v64 + 4);
      *(__m128i *)(a10 + 16) = _mm_loadu_si128(v64 + 5);
      *(__m128i *)(a10 + 32) = _mm_loadu_si128(v64 + 6);
      if ( a11 )
      {
        v24 = v64[4].m128i_i64[0];
        v98 = 1;
        v106 = 1;
        if ( *(_BYTE *)(v24 + 174) == 1 )
        {
          v66 = (const __m128i *)v64[7].m128i_i64[1];
          v98 = 0;
          *a11 = _mm_loadu_si128(v66 + 3);
          a11[1] = _mm_loadu_si128(v66 + 4);
          a11[2] = _mm_loadu_si128(v66 + 5);
        }
      }
      else
      {
        v98 = 0;
        v106 = 1;
      }
      goto LABEL_72;
    }
    v106 = sub_884000(v65, 1);
    if ( v106 || !qword_4D03C50 || *(char *)(qword_4D03C50 + 18LL) >= 0 )
    {
      v64 = v112;
      goto LABEL_97;
    }
  }
LABEL_72:
  if ( !v14 )
    sub_6E1990(v85);
LABEL_16:
  v34 = (_QWORD *)dword_4D04460;
  if ( dword_4D04460 && v102 && a9 & 0x400 | a5 && v108 && v108[1].m128i_i8[1] == 2 )
  {
    *(_BYTE *)(a10 + 16) |= 2u;
    v106 = 1;
    goto LABEL_19;
  }
LABEL_18:
  v27 &= v106;
LABEL_19:
  v35 = (__int64 *)v112;
  v36 = *a12;
  if ( (_DWORD)v36 && (*(_BYTE *)(a10 + 16) |= 8u, *a12) && a13 )
  {
    *a13 = v35;
  }
  else
  {
    for ( ; v35; qword_4D03C68 = v52 )
    {
      v52 = v35;
      v35 = (__int64 *)*v35;
      sub_725130((__int64 *)v52[5]);
      v34 = (_QWORD *)v52[15];
      sub_82D8A0(v34);
      *v52 = (__int64)qword_4D03C68;
    }
  }
  if ( v27 )
    *(_BYTE *)(a10 + 16) |= 0x40u;
  if ( v98 )
  {
    *a11 = 0;
    a11[1] = 0;
    a11[2] = 0;
  }
  v38 = sub_82BD70(v34, v36, v24);
  v41 = *(_QWORD *)(v38 + 1008);
  v42 = *(_QWORD *)(v41 + 8 * (5LL * *(_QWORD *)(v38 + 1024) - 5) + 32);
  if ( v42 )
  {
    sub_823A00(*(_QWORD *)v42, 16LL * (unsigned int)(*(_DWORD *)(v42 + 8) + 1), v41, v37, v39, v40);
    sub_823A00(v42, 16, v43, v44, v45, v46);
  }
  --*(_QWORD *)(v38 + 1024);
  return v106;
}
