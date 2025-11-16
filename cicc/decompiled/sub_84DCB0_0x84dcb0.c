// Function: sub_84DCB0
// Address: 0x84dcb0
//
void __fastcall sub_84DCB0(unsigned __int8 a1, char *a2, _BOOL4 a3, __int64 *a4, _QWORD *a5, const __m128i *a6)
{
  const __m128i *v6; // r15
  _QWORD *v7; // r14
  __int64 v9; // rbx
  __int64 v10; // r12
  _QWORD *v11; // r15
  char v12; // r14
  __int64 v13; // r14
  int v14; // ecx
  unsigned int v15; // eax
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // r14
  char v19; // bl
  __int64 v20; // rsi
  __int64 v21; // rdi
  int v22; // eax
  __int64 v23; // rdx
  __m128i v24; // xmm2
  _QWORD *v25; // rdx
  __int64 v26; // rax
  char v27; // dl
  __int64 v28; // r9
  __int64 v29; // r12
  int v30; // eax
  _BOOL4 v31; // eax
  __int64 v32; // rax
  const __m128i *v33; // r14
  int v34; // esi
  __m128i *v35; // rax
  int v36; // eax
  char v37; // r8
  __int64 v38; // rax
  __int64 v39; // r12
  int v40; // eax
  unsigned int v41; // eax
  int v42; // eax
  __m128i v43; // xmm5
  int v44; // eax
  int v45; // eax
  __m128i v46; // xmm6
  int v47; // eax
  int v48; // eax
  __int64 v49; // rax
  _BOOL4 v50; // eax
  __int64 v51; // rax
  _BOOL4 v52; // eax
  int v53; // eax
  _BOOL4 v54; // eax
  __int64 v55; // rax
  __m128i v56; // xmm6
  __int64 v57; // rdi
  __m128i *v58; // rax
  __m128i *v59; // r12
  const char *v60; // r13
  char *v61; // rax
  char v62; // al
  char *v63; // rax
  const char *v64; // rsi
  char *v65; // r12
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 *v71; // r9
  int v72; // eax
  __int64 v73; // rax
  int v74; // r8d
  __int64 v75; // rdi
  __int64 v76; // rdi
  char v79; // [rsp+26h] [rbp-10Ah]
  int v81; // [rsp+30h] [rbp-100h]
  _BOOL4 v82; // [rsp+30h] [rbp-100h]
  char v85; // [rsp+48h] [rbp-E8h]
  int v86; // [rsp+48h] [rbp-E8h]
  int v87; // [rsp+48h] [rbp-E8h]
  int v88; // [rsp+50h] [rbp-E0h]
  bool v89; // [rsp+54h] [rbp-DCh]
  char v90; // [rsp+54h] [rbp-DCh]
  _QWORD *v92; // [rsp+60h] [rbp-D0h]
  __m128i *v93; // [rsp+68h] [rbp-C8h]
  int v94; // [rsp+68h] [rbp-C8h]
  __int64 v95; // [rsp+70h] [rbp-C0h]
  __int64 *v96; // [rsp+78h] [rbp-B8h]
  char v97; // [rsp+8Bh] [rbp-A5h] BYREF
  unsigned int v98; // [rsp+8Ch] [rbp-A4h] BYREF
  __m128i v99; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v100; // [rsp+A0h] [rbp-90h]
  const char *v101; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v102; // [rsp+B8h] [rbp-78h]
  __int64 v103; // [rsp+C0h] [rbp-70h]
  __m128i v104; // [rsp+D0h] [rbp-60h] BYREF
  __m128i v105; // [rsp+E0h] [rbp-50h] BYREF
  __m128i v106[4]; // [rsp+F0h] [rbp-40h] BYREF

  v6 = a6;
  if ( !a4 )
  {
    v92 = 0;
LABEL_31:
    v25 = qword_4D03C68;
    if ( qword_4D03C68 )
      qword_4D03C68 = (_QWORD *)*qword_4D03C68;
    else
      v25 = (_QWORD *)sub_823970(152);
    *v25 = 0;
    v25[18] = 0;
    memset(
      (void *)((unsigned __int64)(v25 + 1) & 0xFFFFFFFFFFFFFFF8LL),
      0,
      8LL * (((unsigned int)v25 - (((_DWORD)v25 + 8) & 0xFFFFFFF8) + 152) >> 3));
    v25[14] = v6;
    v25[6] = a2;
    v25[15] = v92;
    *((_BYTE *)v25 + 144) = a1;
    *v25 = *a5;
    *a5 = v25;
    return;
  }
  v7 = 0;
  v96 = a4;
  v95 = 0;
  v88 = 0x2000000;
  v92 = 0;
  v79 = (a1 - 16) & 0xEE;
  while ( 1 )
  {
    v89 = 0;
    v9 = v96[3];
    v10 = *(_QWORD *)(v9 + 8);
    v93 = (__m128i *)(v9 + 8);
    if ( *(_BYTE *)(v9 + 25) == 1 )
    {
      v89 = !sub_6ED0A0(v9 + 8);
      v11 = qword_4D03C60;
      if ( qword_4D03C60 )
      {
LABEL_5:
        qword_4D03C60 = (_QWORD *)*v11;
        goto LABEL_6;
      }
    }
    else
    {
      v11 = qword_4D03C60;
      if ( qword_4D03C60 )
        goto LABEL_5;
    }
    v11 = (_QWORD *)sub_823970(104);
LABEL_6:
    sub_82D850((__int64)v11);
    if ( v92 )
      *v7 = v11;
    else
      v92 = v11;
    v12 = a2[v95];
    if ( !a6 )
    {
      v36 = sub_8E31E0(v10);
      v37 = v12;
      if ( v36 )
      {
        v38 = *a4;
        if ( !*a4 )
          goto LABEL_57;
        if ( *(_BYTE *)(v38 + 8) == 3 )
        {
          v38 = sub_6BBB10(a4);
          v37 = v12;
        }
        if ( v96 != a4 )
          v38 = (__int64)a4;
        v39 = *(_QWORD *)(*(_QWORD *)(v38 + 24) + 8LL);
        v90 = v37;
        v40 = sub_827590(v39, v37);
        v37 = v90;
        if ( v40 )
        {
          if ( v12 != 69 && v12 != 83 )
          {
            v48 = sub_8D28F0(v39);
            v37 = v90;
            if ( v48 )
            {
              v49 = sub_8D6540(v39);
              v37 = v90;
              v39 = v49;
            }
          }
        }
        else
        {
LABEL_57:
          v39 = 0;
        }
        v41 = sub_8274A0(v37);
        v21 = v9 + 8;
        v20 = v39;
        v42 = sub_840360(v93->m128i_i64, v39, v41, a3, 1, a1 != 35 && a1 != 36, 0, 0, v88, (__int64)&v104, &v98, 0);
        if ( !(v98 | v42) )
          goto LABEL_40;
        *((_DWORD *)v11 + 2) = 4;
        *((__m128i *)v11 + 3) = _mm_loadu_si128(&v104);
        *((__m128i *)v11 + 4) = _mm_loadu_si128(&v105);
        v43 = _mm_loadu_si128(v106);
        v11[5] = v39;
        *((__m128i *)v11 + 5) = v43;
        goto LABEL_22;
      }
      v21 = sub_6EEB30(v10, (__int64)v93);
      v20 = (unsigned int)v12;
      if ( !(unsigned int)sub_827590(v21, v12) )
        goto LABEL_40;
      if ( HIDWORD(qword_4D0495C) || unk_4D04760 && a1 == 43 && v12 == 68 )
      {
        *((_BYTE *)v11 + 84) |= 0x20u;
        v47 = 2;
LABEL_77:
        *((_DWORD *)v11 + 2) = v47;
        *((_BYTE *)v11 + 17) = v89;
        goto LABEL_22;
      }
      v58 = sub_73D720(*(const __m128i **)(v9 + 8));
      v59 = v58;
      if ( v12 == 66 )
      {
        if ( !(unsigned int)sub_8D29A0(v58) )
        {
          *((_BYTE *)v11 + 84) |= 0x20u;
          if ( (unsigned int)sub_8D2E30(v59) || (unsigned int)sub_8D3D10(v59) )
          {
            *((_BYTE *)v11 + 85) |= 1u;
            v47 = 2;
          }
          else
          {
            v47 = 2;
          }
          goto LABEL_77;
        }
        goto LABEL_116;
      }
      if ( (v12 & 0xD7) == 0x41 )
      {
        if ( !qword_4D0495C )
          goto LABEL_122;
        if ( v12 != 68 )
        {
          if ( (unsigned int)sub_8D2930(v58) )
            goto LABEL_130;
          goto LABEL_116;
        }
LABEL_106:
        if ( !(unsigned int)sub_8D2930(v58) )
          goto LABEL_116;
        v12 = 73;
        if ( !(unsigned int)sub_8D2870(v59) )
        {
LABEL_132:
          v75 = sub_6E8540((__int64)v93);
          if ( v59 != (__m128i *)v75 && !(unsigned int)sub_8DED30(v75, v59, 1) )
          {
            *((_BYTE *)v11 + 84) |= 0x60u;
            v47 = 1;
            goto LABEL_77;
          }
          goto LABEL_116;
        }
      }
      else
      {
        if ( v12 != 68 )
          goto LABEL_116;
        if ( qword_4D0495C )
          goto LABEL_106;
LABEL_122:
        if ( !(unsigned int)sub_8D2930(v58) )
          goto LABEL_116;
        if ( v12 == 68 )
        {
          if ( !(unsigned int)sub_8D3990(v59) )
          {
            v73 = sub_6E8540((__int64)v93);
            v74 = sub_8D3990(v73);
            v47 = 2;
            if ( v74 )
            {
              *((_BYTE *)v11 + 84) |= 0x40u;
              v47 = 1;
            }
            *((_BYTE *)v11 + 84) |= 0x20u;
            goto LABEL_77;
          }
          goto LABEL_116;
        }
LABEL_130:
        if ( !(unsigned int)sub_8D2870(v59) )
        {
          if ( (v12 & 0xF7) == 0x41 )
            goto LABEL_132;
LABEL_116:
          v47 = 0;
          goto LABEL_77;
        }
      }
      if ( !qword_4D0495C )
      {
        *((_BYTE *)v11 + 84) |= 0x40u;
        if ( dword_4F077C4 == 2
          && unk_4F07778 > 201401
          && !dword_4F077BC
          && (!(_DWORD)qword_4F077B4 || qword_4F077A0 > 0x1869Fu) )
        {
          while ( v59[8].m128i_i8[12] == 12 )
            v59 = (__m128i *)v59[10].m128i_i64[0];
          if ( (v59[10].m128i_i8[1] & 0x14) == 4 )
          {
            if ( (v12 & 0xF7) != 0x41
              || (v76 = sub_8D6540(v59), v59 == (__m128i *)v76)
              || (unsigned int)sub_8DED30(v76, v59, 1) )
            {
              *((_BYTE *)v11 + 84) |= 0x80u;
            }
          }
        }
        *((_BYTE *)v11 + 84) |= 0x20u;
        v47 = 1;
        goto LABEL_77;
      }
      goto LABEL_116;
    }
    if ( v12 != 79 )
    {
      if ( v12 == 67 )
      {
        if ( (const __m128i *)v10 == a6 || (unsigned int)sub_8DED30(v10, a6, 3) )
        {
          v13 = 0;
        }
        else if ( !(unsigned int)sub_8D3A70(v10) || (v13 = sub_8D5CE0(v10, a6)) == 0 )
        {
          v20 = 0;
          v21 = v9 + 8;
          v45 = sub_836C50(v93, 0, a6, 0, 1u, 1u, 0, 0, v88, (__int64)&v104, 0, &v98, 0);
          if ( !(v98 | v45) )
            goto LABEL_40;
          *((_DWORD *)v11 + 2) = 4;
          *((__m128i *)v11 + 3) = _mm_loadu_si128(&v104);
          *((__m128i *)v11 + 4) = _mm_loadu_si128(&v105);
          v46 = _mm_loadu_si128(v106);
          v11[4] = a6;
          *((__m128i *)v11 + 5) = v46;
          goto LABEL_22;
        }
        if ( (*(_BYTE *)(v10 + 140) & 0xFB) == 8 )
        {
          v20 = v10;
          v21 = (__int64)a6;
          if ( (unsigned int)sub_8D5780(a6, v10) )
            goto LABEL_40;
          if ( (a6[8].m128i_i8[12] & 0xFB) == 8 )
          {
LABEL_16:
            v14 = sub_8D4C10(a6, dword_4F077C4 != 2) & 0xFFFFFF8F;
            if ( (*(_BYTE *)(v10 + 140) & 0xFB) != 8 )
            {
              v15 = 0;
              goto LABEL_18;
            }
LABEL_66:
            v94 = v14;
            v44 = sub_8D4C10(v10, dword_4F077C4 != 2);
            v14 = v94;
            v15 = v44 & 0xFFFFFF8F;
LABEL_18:
            if ( v15 != v14 )
              *((_BYTE *)v11 + 64) |= 0x10u;
            goto LABEL_20;
          }
          if ( (*(_BYTE *)(v10 + 140) & 0xFB) == 8 )
          {
            v14 = 0;
            goto LABEL_66;
          }
        }
        else if ( (a6[8].m128i_i8[12] & 0xFB) == 8 )
        {
          goto LABEL_16;
        }
LABEL_20:
        if ( v13 )
        {
          v16 = *((_BYTE *)v11 + 64);
          *((_BYTE *)v11 + 84) |= 0x20u;
          *((_DWORD *)v11 + 2) = 2;
          v16 |= 0x10u;
          v11[9] = v13;
          *((_BYTE *)v11 + 64) = v16;
          *((_BYTE *)v11 + 64) = (4 * (*(_BYTE *)(v9 + 25) == 1)) | v16 & 0xFB;
        }
        else
        {
          *((_DWORD *)v11 + 2) = 0;
          *((_BYTE *)v11 + 64) = (4 * (*(_BYTE *)(v9 + 25) == 1)) | v11[8] & 0xFB;
        }
        goto LABEL_22;
      }
      goto LABEL_26;
    }
    if ( a2[1] == 77 )
      break;
LABEL_26:
    v18 = (__int64)a6;
    if ( !(unsigned int)sub_8D3A70(v10) )
      goto LABEL_34;
    v19 = 0;
LABEL_28:
    v20 = v18;
    v21 = (__int64)v93;
    v22 = sub_840360(v93->m128i_i64, v18, 0, a3, 1, 1, 0, 0, v88, (__int64)&v104, &v98, 0);
    if ( !(v98 | v22) )
      goto LABEL_40;
    *((_DWORD *)v11 + 2) = 4;
    *((__m128i *)v11 + 3) = _mm_loadu_si128(&v104);
    *((__m128i *)v11 + 4) = _mm_loadu_si128(&v105);
    v24 = _mm_loadu_si128(v106);
    v11[4] = v18;
    *((_BYTE *)v11 + 18) = v19;
    *((__m128i *)v11 + 5) = v24;
    v17 = *v96;
    if ( !*v96 )
      goto LABEL_30;
LABEL_23:
    if ( *(_BYTE *)(v17 + 8) == 3 )
    {
      v17 = sub_6BBB10(v96);
      if ( !v17 )
      {
LABEL_30:
        v6 = a6;
        goto LABEL_31;
      }
    }
    ++v95;
    v7 = v11;
    v96 = (__int64 *)v17;
    a3 = 0;
  }
  v33 = (const __m128i *)sub_8D4890(a6);
  if ( (unsigned int)sub_8D2E30(v10) && (v57 = sub_8D46C0(v10), (*(_BYTE *)(v57 + 140) & 0xFB) == 8) )
    v34 = sub_8D4C10(v57, dword_4F077C4 != 2);
  else
    v34 = 0;
  v35 = sub_73C570(v33, v34);
  v18 = sub_72D2E0(v35);
  if ( (unsigned int)sub_8D3A70(v10) )
  {
    v88 = (int)sub_2000100;
    v19 = 1;
    goto LABEL_28;
  }
LABEL_34:
  v26 = sub_6EEB30(v10, (__int64)v93);
  v27 = *(_BYTE *)(v9 + 24);
  v28 = v9 + 152;
  v29 = v26;
  LODWORD(v20) = v27 == 2;
  if ( *(_BYTE *)(v9 + 25) != 1 )
    goto LABEL_35;
  v82 = v27 == 2;
  v85 = *(_BYTE *)(v9 + 24);
  v50 = sub_6ED0A0((__int64)v93);
  v27 = v85;
  LODWORD(v20) = v82;
  v28 = v9 + 152;
  if ( v50 )
    goto LABEL_35;
  if ( !HIDWORD(qword_4D0495C) )
  {
    v30 = qword_4D0495C;
    if ( dword_4F077BC | (unsigned int)qword_4D0495C )
      goto LABEL_37;
    v51 = sub_6ED2B0((__int64)v93);
    v27 = v85;
    LODWORD(v20) = v82;
    v28 = v9 + 152;
    if ( v51 )
    {
      v28 = v51;
      LODWORD(v20) = 1;
      if ( qword_4D0495C )
        goto LABEL_89;
      goto LABEL_39;
    }
LABEL_35:
    if ( !HIDWORD(qword_4D0495C) )
    {
      v30 = qword_4D0495C;
LABEL_37:
      if ( !v30 )
        goto LABEL_39;
    }
  }
  LODWORD(v20) = 0;
  if ( v27 != 2 )
    goto LABEL_39;
LABEL_89:
  v86 = v28;
  v52 = sub_712570(v28);
  LODWORD(v28) = v86;
  LODWORD(v20) = 1;
  if ( !v52
    || (v21 = v18, v53 = sub_8D2E30(v18), v20 = 1, LODWORD(v28) = v86, !v53)
    && (v21 = v18, v72 = sub_8D3D10(v18), LODWORD(v28) = v86, !v72) )
  {
LABEL_39:
    v81 = v28;
    v31 = sub_6EB660((__int64)v93);
    v21 = v29;
    v20 = (unsigned int)v20;
    if ( !(unsigned int)sub_8E1010(
                          v29,
                          v20,
                          (*(_BYTE *)(v9 + 27) & 0x10) != 0,
                          v31,
                          0,
                          v81,
                          v18,
                          0,
                          0,
                          1,
                          0,
                          (__int64)&v99,
                          0) )
    {
LABEL_40:
      if ( *((_DWORD *)v11 + 2) == 7 )
        goto LABEL_41;
      goto LABEL_22;
    }
    *((_DWORD *)v11 + 2) = ((unsigned __int8)v99.m128i_i8[12] >> 4) & 2;
LABEL_97:
    v55 = v100;
    v56 = _mm_loadu_si128(&v99);
    v11[4] = v18;
    v11[11] = v55;
    *(__m128i *)(v11 + 9) = v56;
    *((_BYTE *)v11 + 17) = v89;
LABEL_22:
    v17 = *v96;
    if ( !*v96 )
      goto LABEL_30;
    goto LABEL_23;
  }
  if ( (*(_BYTE *)(v9 + 27) & 0x40) != 0 )
  {
    v21 = (unsigned int)qword_4D0495C;
    if ( !(_DWORD)qword_4D0495C || v79 )
    {
      v87 = v28;
      v54 = sub_6EB660((__int64)v93);
      v20 = 1;
      v21 = v29;
      if ( !(unsigned int)sub_8E1010(
                            v29,
                            1,
                            (*(_BYTE *)(v9 + 27) & 0x10) != 0,
                            v54,
                            0,
                            v87,
                            v18,
                            0,
                            0,
                            1,
                            0,
                            (__int64)&v99,
                            0) )
        goto LABEL_40;
      *((_DWORD *)v11 + 2) = 2 * (HIDWORD(qword_4D0495C) != 0);
      goto LABEL_97;
    }
  }
  *((_DWORD *)v11 + 2) = 7;
LABEL_41:
  v32 = sub_82BD70(v21, v20, v23);
  if ( (*(_BYTE *)(*(_QWORD *)(v32 + 1008) + 8 * (5LL * *(_QWORD *)(v32 + 1024) - 5)) & 1) != 0 )
  {
    v101 = 0;
    v102 = 0;
    v60 = (const char *)qword_4F064C0[a1];
    v103 = 0;
    v101 = (const char *)sub_823970(0);
    v102 = 0;
    sub_84DAC0((__int64 *)&v101, (__int64)&v97, v60, "(");
    v61 = sub_8281B0(*a2);
    --v103;
    sub_84DC00((__int64 *)&v101, v61);
    v62 = a2[1];
    if ( v62 != 59 && v62 )
    {
      --v103;
      sub_84DC00((__int64 *)&v101, ", ");
      v63 = sub_8281B0(a2[1]);
      --v103;
      sub_84DC00((__int64 *)&v101, v63);
    }
    --v103;
    sub_84DC00((__int64 *)&v101, ")");
    v64 = v101;
    v65 = sub_7248C0(0, v101, v103 - 1);
    v67 = sub_82BD70(0, v64, v66);
    sub_67E8D0(
      0xD00u,
      dword_4F07508,
      (__int64)v65,
      v95 + 1,
      (_QWORD *)(*(_QWORD *)(v67 + 1008) + 8 * (5LL * *(_QWORD *)(v67 + 1024) - 5) + 16));
    sub_823A00((__int64)v101, v102, v68, v69, v70, v71);
  }
  sub_82D8A0(v92);
}
