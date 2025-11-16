// Function: sub_6AD6A0
// Address: 0x6ad6a0
//
__int64 __fastcall sub_6AD6A0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdi
  int v5; // eax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // r14d
  __int64 v11; // rax
  __int64 v12; // rax
  int v14; // eax
  __int64 v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdi
  int v22; // eax
  _QWORD *v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rdx
  char v27; // al
  char v28; // al
  __m128i v29; // xmm1
  __m128i v30; // xmm2
  __m128i v31; // xmm3
  __m128i v32; // xmm4
  __m128i v33; // xmm5
  __m128i v34; // xmm6
  __m128i v35; // xmm7
  __m128i v36; // xmm0
  __int8 v37; // al
  __int64 v38; // rsi
  __m128i v39; // xmm2
  __m128i v40; // xmm3
  __m128i v41; // xmm4
  __m128i v42; // xmm5
  __m128i v43; // xmm6
  __m128i v44; // xmm7
  __m128i v45; // xmm1
  __m128i v46; // xmm2
  __m128i v47; // xmm4
  __m128i v48; // xmm5
  __m128i v49; // xmm6
  __m128i v50; // xmm7
  __m128i v51; // xmm0
  __m128i v52; // xmm1
  __m128i v53; // xmm3
  __m128i v54; // xmm2
  __m128i v55; // xmm4
  __m128i v56; // xmm5
  __m128i v57; // xmm6
  __m128i v58; // xmm7
  __m128i *v59; // rdi
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  char v64; // al
  __int64 v65; // rax
  __int64 i; // rdx
  int v67; // r14d
  int v68; // eax
  int v69; // eax
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r8
  __int64 v73; // r9
  int v74; // eax
  __int64 v75; // rdi
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r8
  __int64 v79; // r9
  _QWORD *v80; // r14
  _QWORD *v81; // r14
  char v82; // al
  int v83; // eax
  int v84; // eax
  int v85; // eax
  __int64 v86; // [rsp-8h] [rbp-1F0h]
  __int64 v87; // [rsp+0h] [rbp-1E8h]
  int v88; // [rsp+10h] [rbp-1D8h]
  int v89; // [rsp+14h] [rbp-1D4h]
  int v90; // [rsp+18h] [rbp-1D0h]
  __int64 v91; // [rsp+18h] [rbp-1D0h]
  int v92; // [rsp+28h] [rbp-1C0h]
  __int64 v93; // [rsp+28h] [rbp-1C0h]
  __int64 v94; // [rsp+30h] [rbp-1B8h]
  __int64 v95; // [rsp+38h] [rbp-1B0h] BYREF
  int v96; // [rsp+40h] [rbp-1A8h] BYREF
  __int16 v97; // [rsp+44h] [rbp-1A4h]
  int v98; // [rsp+48h] [rbp-1A0h] BYREF
  __int64 v99; // [rsp+50h] [rbp-198h] BYREF
  __m128i v100; // [rsp+58h] [rbp-190h] BYREF
  __m128i v101; // [rsp+68h] [rbp-180h] BYREF
  __m128i v102; // [rsp+78h] [rbp-170h] BYREF
  __m128i v103; // [rsp+88h] [rbp-160h] BYREF
  __m128i v104; // [rsp+98h] [rbp-150h] BYREF
  __m128i v105; // [rsp+A8h] [rbp-140h] BYREF
  __m128i v106; // [rsp+B8h] [rbp-130h] BYREF
  __m128i v107; // [rsp+C8h] [rbp-120h] BYREF
  __m128i v108; // [rsp+D8h] [rbp-110h] BYREF
  __m128i v109; // [rsp+E8h] [rbp-100h] BYREF
  __m128i v110; // [rsp+F8h] [rbp-F0h] BYREF
  __m128i v111; // [rsp+108h] [rbp-E0h] BYREF
  __m128i v112; // [rsp+118h] [rbp-D0h] BYREF
  __m128i v113; // [rsp+128h] [rbp-C0h] BYREF
  __m128i v114; // [rsp+138h] [rbp-B0h] BYREF
  __m128i v115; // [rsp+148h] [rbp-A0h] BYREF
  __m128i v116; // [rsp+158h] [rbp-90h] BYREF
  __m128i v117; // [rsp+168h] [rbp-80h] BYREF
  __m128i v118; // [rsp+178h] [rbp-70h] BYREF
  __m128i v119; // [rsp+188h] [rbp-60h] BYREF
  __m128i v120; // [rsp+198h] [rbp-50h] BYREF
  __m128i v121[4]; // [rsp+1A8h] [rbp-40h] BYREF

  v4 = 7;
  v5 = sub_6AD110(7, a1, &v96, &v95, &v98, &v99, &v100);
  v8 = v86;
  v9 = v87;
  v10 = v5;
  if ( unk_4F04C50 )
  {
    v11 = *(_QWORD *)(unk_4F04C50 + 32LL);
    if ( v11 )
    {
      if ( (*(_BYTE *)(v11 + 198) & 0x10) != 0 )
      {
        a1 = (__int64 *)&v96;
        v4 = 3645;
        sub_6851A0(0xE3Du, &v96, (__int64)"dynamic_cast");
        if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
          goto LABEL_5;
        sub_6E9250(&v96);
        goto LABEL_7;
      }
    }
  }
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
  {
LABEL_5:
    if ( (unsigned int)sub_6E5430(v4, a1, v8, v9, v6, v7) )
      sub_6851C0(0x39u, &v96);
    goto LABEL_7;
  }
  if ( (unsigned int)sub_6E9250(&v96) || !v10 )
  {
LABEL_7:
    sub_6E6450(&v100);
LABEL_8:
    sub_6E6260(a2);
    goto LABEL_9;
  }
  v14 = sub_8D32B0(v95);
  v15 = v95;
  v92 = v14;
  if ( !v14 )
  {
    if ( (unsigned int)sub_8D3D40(v95) )
    {
      v88 = 0;
      v89 = 0;
      v90 = 1;
      v94 = *(_QWORD *)&dword_4D03B80;
      goto LABEL_16;
    }
LABEL_67:
    v65 = v95;
    for ( i = *(unsigned __int8 *)(v95 + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v65 + 140) )
      v65 = *(_QWORD *)(v65 + 160);
    if ( (_BYTE)i && (unsigned int)sub_6E5430(v15, a1, i, v16, v17, v18) )
      sub_6851C0(0x2B7u, &v98);
    goto LABEL_7;
  }
  v94 = sub_8D46C0(v95);
  v92 = sub_8D32E0(v95);
  if ( v92 )
  {
    if ( (unsigned int)sub_8D3110(v95) )
    {
      v89 = 1;
      v92 = 1;
      if ( (unsigned int)sub_8D3A70(v94) )
        goto LABEL_39;
    }
    else
    {
      v89 = sub_8D3A70(v94);
      if ( v89 )
      {
        v89 = 0;
        v92 = 1;
        goto LABEL_39;
      }
      v92 = 1;
    }
LABEL_113:
    v15 = v94;
    if ( !(unsigned int)sub_8D3D40(v94) )
      goto LABEL_67;
    v88 = 0;
    v90 = 1;
    goto LABEL_16;
  }
  v89 = sub_8D3A70(v94);
  if ( !v89 )
  {
    v92 = sub_8D2600(v94);
    if ( v92 )
    {
      v90 = 0;
      v92 = 0;
      v88 = 1;
      goto LABEL_16;
    }
    goto LABEL_113;
  }
  v89 = 0;
LABEL_39:
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v94) && (unsigned int)sub_8D3A70(v94) )
  {
    a1 = 0;
    sub_8AD220(v94, 0);
  }
  v15 = v94;
  v90 = sub_8D23B0(v94);
  if ( v90 )
  {
    v88 = dword_4D04964;
    if ( !dword_4D04964 )
    {
      if ( dword_4F04C44 != -1 )
      {
        v90 = 1;
        v19 = v100.m128i_i64[0];
        goto LABEL_44;
      }
      v16 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( (*(_BYTE *)(v16 + 6) & 2) != 0 )
      {
        v90 = 1;
        v19 = v100.m128i_i64[0];
        goto LABEL_17;
      }
    }
    goto LABEL_67;
  }
  v88 = 0;
LABEL_16:
  v19 = v100.m128i_i64[0];
  if ( dword_4F04C44 == -1 )
  {
LABEL_17:
    v20 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v20 + 6) & 6) == 0 && *(_BYTE *)(v20 + 4) != 12 )
      goto LABEL_19;
  }
LABEL_44:
  if ( (unsigned int)sub_8DBE70(v19) )
    goto LABEL_45;
LABEL_19:
  if ( v90 )
    goto LABEL_45;
  if ( v92 )
  {
    v59 = (__m128i *)v19;
    if ( !(unsigned int)sub_8D3A70(v19) )
      goto LABEL_135;
    if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v19) && (unsigned int)sub_8D3A70(v19) )
    {
      a1 = 0;
      sub_8AD220(v19, 0);
    }
    v59 = (__m128i *)v19;
    if ( (unsigned int)sub_8D23B0(v19) )
      goto LABEL_135;
    if ( v89 )
    {
      v92 = v89;
      goto LABEL_50;
    }
    if ( v101.m128i_i8[1] != 1 || (v59 = &v100, v83 = sub_6ED0A0(&v100), v83) )
    {
LABEL_135:
      while ( 1 )
      {
        v64 = *(_BYTE *)(v19 + 140);
        if ( v64 != 12 )
          break;
        v19 = *(_QWORD *)(v19 + 160);
      }
      if ( v64 && (unsigned int)sub_6E5430(v59, a1, v60, v61, v62, v63) )
        sub_6851C0(v89 == 0 ? 697 : 1773, &v104.m128i_i32[1]);
      goto LABEL_7;
    }
  }
  else
  {
    sub_6F69D0(&v100, 0);
    v21 = v100.m128i_i64[0];
    v93 = v100.m128i_i64[0];
    v22 = sub_8D2EF0(v100.m128i_i64[0]);
    v26 = v93;
    if ( !v22 || (v67 = sub_8D2E30(v95), v21 = v93, v68 = sub_8D2E30(v93), v26 = v93, v67 != v68) )
    {
      v27 = *(_BYTE *)(v26 + 140);
      v19 = 0;
      if ( v27 == 12 )
        goto LABEL_23;
      goto LABEL_30;
    }
    v21 = sub_8D46C0(v93);
    v19 = v21;
    v69 = sub_8D3A70(v21);
    v26 = v93;
    if ( !v69 )
    {
LABEL_24:
      while ( 1 )
      {
        v28 = *(_BYTE *)(v26 + 140);
        if ( v28 != 12 )
          break;
LABEL_23:
        v26 = *(_QWORD *)(v26 + 160);
      }
      if ( !v28 )
        goto LABEL_7;
      if ( !v19 )
        goto LABEL_31;
      while ( 1 )
      {
        v27 = *(_BYTE *)(v19 + 140);
        if ( v27 != 12 )
          break;
        v19 = *(_QWORD *)(v19 + 160);
      }
LABEL_30:
      if ( !v27 )
        goto LABEL_7;
LABEL_31:
      if ( (unsigned int)sub_6E5430(v21, 0, v26, v23, v24, v25) )
        sub_6851C0(0x2B8u, &v104.m128i_i32[1]);
      goto LABEL_7;
    }
    if ( dword_4F077C4 == 2 )
    {
      v84 = sub_8D23B0(v21);
      v26 = v93;
      if ( v84 )
      {
        v85 = sub_8D3A70(v21);
        v26 = v93;
        if ( v85 )
        {
          sub_8AD220(v21, 0);
          v26 = v93;
        }
      }
    }
    v91 = v26;
    v92 = sub_8D23B0(v21);
    if ( v92 )
    {
      v26 = v91;
      if ( dword_4D04964 )
        goto LABEL_24;
      if ( dword_4F04C44 == -1 )
      {
        v23 = qword_4F04C68;
        if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
          goto LABEL_24;
      }
LABEL_45:
      sub_6F4200(&v100, v95, 7, 0);
      v29 = _mm_loadu_si128(&v101);
      v30 = _mm_loadu_si128(&v102);
      v31 = _mm_loadu_si128(&v103);
      v32 = _mm_loadu_si128(&v104);
      v33 = _mm_loadu_si128(&v105);
      *(__m128i *)a2 = _mm_loadu_si128(&v100);
      v34 = _mm_loadu_si128(&v106);
      v35 = _mm_loadu_si128(&v107);
      *(__m128i *)(a2 + 16) = v29;
      v36 = _mm_loadu_si128(&v108);
      v37 = v101.m128i_i8[0];
      *(__m128i *)(a2 + 32) = v30;
      *(__m128i *)(a2 + 48) = v31;
      *(__m128i *)(a2 + 64) = v32;
      *(__m128i *)(a2 + 80) = v33;
      *(__m128i *)(a2 + 96) = v34;
      *(__m128i *)(a2 + 112) = v35;
      *(__m128i *)(a2 + 128) = v36;
      if ( v37 != 2 )
      {
LABEL_46:
        if ( v37 == 5 || v37 == 1 )
          *(_QWORD *)(a2 + 144) = v109.m128i_i64[0];
        goto LABEL_9;
      }
LABEL_55:
      v47 = _mm_loadu_si128(&v110);
      v48 = _mm_loadu_si128(&v111);
      v49 = _mm_loadu_si128(&v112);
      v50 = _mm_loadu_si128(&v113);
      v51 = _mm_loadu_si128(&v114);
      *(__m128i *)(a2 + 144) = _mm_loadu_si128(&v109);
      v52 = _mm_loadu_si128(&v120);
      v53 = _mm_loadu_si128(&v115);
      *(__m128i *)(a2 + 160) = v47;
      v54 = _mm_loadu_si128(v121);
      v55 = _mm_loadu_si128(&v116);
      *(__m128i *)(a2 + 176) = v48;
      *(__m128i *)(a2 + 192) = v49;
      v56 = _mm_loadu_si128(&v117);
      v57 = _mm_loadu_si128(&v118);
      *(__m128i *)(a2 + 208) = v50;
      v58 = _mm_loadu_si128(&v119);
      *(__m128i *)(a2 + 224) = v51;
      *(__m128i *)(a2 + 240) = v53;
      *(__m128i *)(a2 + 256) = v55;
      *(__m128i *)(a2 + 272) = v56;
      *(__m128i *)(a2 + 288) = v57;
      *(__m128i *)(a2 + 304) = v58;
      *(__m128i *)(a2 + 320) = v52;
      *(__m128i *)(a2 + 336) = v54;
      goto LABEL_9;
    }
  }
LABEL_50:
  if ( (*(_BYTE *)(v19 + 140) & 0xFB) == 8
    && (unsigned int)sub_8D5780(v94, v19)
    && (unsigned int)sub_6E5430(v94, v19, v70, v71, v72, v73) )
  {
    sub_6851A0(0x2B6u, &v96, (__int64)"dynamic_cast");
  }
  v38 = v94;
  if ( (unsigned int)sub_8DEFB0(v19, v94, 1, 0)
    || (unsigned int)sub_8D3A70(v94) && (unsigned int)sub_8D3A70(v19) && (v38 = v94, sub_8D5CE0(v19, v94)) )
  {
    if ( v92 )
    {
      sub_6FAB30(&v100, v95, 1, 0, 0);
      goto LABEL_54;
    }
LABEL_83:
    v74 = sub_6E9880(&v100);
    sub_6FB850(v95, (unsigned int)&v100, 0, 1, 1, 0, 0, v74);
LABEL_54:
    v39 = _mm_loadu_si128(&v101);
    v40 = _mm_loadu_si128(&v102);
    v41 = _mm_loadu_si128(&v103);
    v42 = _mm_loadu_si128(&v104);
    v43 = _mm_loadu_si128(&v105);
    *(__m128i *)a2 = _mm_loadu_si128(&v100);
    v44 = _mm_loadu_si128(&v106);
    v45 = _mm_loadu_si128(&v107);
    *(__m128i *)(a2 + 16) = v39;
    v37 = v101.m128i_i8[0];
    v46 = _mm_loadu_si128(&v108);
    *(__m128i *)(a2 + 32) = v40;
    *(__m128i *)(a2 + 48) = v41;
    *(__m128i *)(a2 + 64) = v42;
    *(__m128i *)(a2 + 80) = v43;
    *(__m128i *)(a2 + 96) = v44;
    *(__m128i *)(a2 + 112) = v45;
    *(__m128i *)(a2 + 128) = v46;
    if ( v37 != 2 )
      goto LABEL_46;
    goto LABEL_55;
  }
  if ( v92 || !(unsigned int)sub_6E9880(&v100) )
  {
    v75 = v19;
    if ( (unsigned int)sub_8D3E60(v19) )
    {
      if ( !(unk_4D0484C | v88) )
      {
        if ( (unsigned int)sub_6E5430(v19, v38, v76, unk_4D0484C | (unsigned int)v88, v78, v79) )
          sub_6851C0(0x6D7u, &v96);
        goto LABEL_8;
      }
      goto LABEL_89;
    }
LABEL_99:
    while ( 1 )
    {
      v82 = *(_BYTE *)(v19 + 140);
      if ( v82 != 12 )
        break;
      v19 = *(_QWORD *)(v19 + 160);
    }
    if ( v82 && (unsigned int)sub_6E5430(v75, v38, v76, v77, v78, v79) )
      sub_6851C0(0x2BAu, &v104.m128i_i32[1]);
    goto LABEL_8;
  }
  if ( !v88 )
    goto LABEL_83;
  v75 = v19;
  if ( !(unsigned int)sub_8D3E60(v19) )
    goto LABEL_99;
LABEL_89:
  if ( v89 )
    sub_6FA350(&v100, v95);
  v80 = (_QWORD *)sub_6F6F40(&v100, 0);
  if ( !(unsigned int)sub_6E9880(&v100) )
    sub_8DCE90(*v80);
  if ( v92 )
  {
    v81 = (_QWORD *)sub_73DC30(19, v94, v80);
    sub_6E84C0(v81, v95);
    sub_6E7150(v81, a2);
    if ( v89 )
      sub_6ED1A0(a2);
  }
  else
  {
    v81 = (_QWORD *)sub_73DBF0(18, v95, v80);
    sub_6E70E0(v81, a2);
  }
  if ( !(unsigned int)sub_6E9880(&v100) )
    sub_8DCE90(*v81);
LABEL_9:
  *(_DWORD *)(a2 + 68) = v96;
  *(_WORD *)(a2 + 72) = v97;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a2 + 68);
  v12 = v99;
  *(_QWORD *)(a2 + 76) = v99;
  unk_4F061D8 = v12;
  sub_6E3280(a2, &v96);
  sub_6E41D0(a2, 0, 7, &v96, &v98, v95);
  return sub_6E26D0(2, a2);
}
