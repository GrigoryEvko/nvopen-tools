// Function: sub_7DCA00
// Address: 0x7dca00
//
_DWORD *__fastcall sub_7DCA00(__int64 a1, int a2, int a3)
{
  unsigned __int8 v5; // bl
  __int64 v6; // rax
  __int64 v7; // rcx
  const __m128i *v8; // rdi
  const __m128i *i; // rax
  bool v10; // zf
  __int64 v11; // r13
  _BYTE *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  _QWORD *v17; // r13
  __int64 v18; // rax
  const char *v19; // r14
  size_t v20; // rax
  char *v21; // r15
  unsigned __int64 v22; // r14
  _QWORD *v23; // rax
  const __m128i *v24; // rdi
  __m128i *v25; // r14
  __int64 v26; // rdi
  char *v27; // rax
  __int64 v28; // rax
  __int64 *v29; // rsi
  __int64 v30; // r15
  const __m128i *v31; // rax
  __m128i *v32; // rax
  _BYTE *v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // rax
  int v38; // edi
  __int64 v40; // rax
  char *v41; // rax
  __int64 v42; // rax
  __int64 *v43; // rsi
  __int64 v44; // r15
  __int64 v45; // rax
  __m128i *v46; // r14
  __int8 v47; // al
  __m128i *j; // rdi
  unsigned __int64 v49; // rsi
  _BYTE *v50; // rbx
  __m128i *k; // rdi
  __int64 v52; // rax
  const __m128i *v53; // rax
  __m128i *v54; // rax
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  _QWORD *v59; // r14
  __int64 v60; // rax
  int v61; // r9d
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 **v64; // rax
  __int64 *v65; // rbx
  _BYTE *v66; // r12
  __int64 v67; // rax
  const __m128i *v68; // rax
  __m128i *v69; // rax
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // r8
  _BYTE *v74; // rbx
  __int64 v75; // rax
  __int64 v76; // rax
  const __m128i *v77; // rax
  __m128i *v78; // rax
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 *v85; // rax
  __int64 v86; // rdx
  __int64 v87; // r14
  __int64 v88; // r15
  _QWORD *v89; // rax
  _QWORD *v90; // r13
  __int64 v91; // r14
  _BYTE *v92; // r12
  __int64 v93; // rax
  const __m128i *v94; // rax
  __m128i *v95; // rax
  __int64 v96; // rsi
  __int64 v97; // rdx
  __int64 v98; // rcx
  __int64 v99; // r8
  _BYTE *v100; // r15
  char v101; // al
  __int64 v102; // rsi
  __int64 *v103; // rax
  __int64 m; // rdx
  _QWORD *v105; // rbx
  __int64 v106; // rax
  __int64 v107; // r8
  _BYTE *v108; // r12
  _BYTE *v109; // rbx
  _QWORD *v110; // rax
  __int64 v111; // rbx
  char v112; // al
  int v113; // ebx
  char v114; // r9
  __int64 v115; // rax
  char v116; // r9
  __int64 v117; // rax
  __int64 v118; // [rsp+8h] [rbp-78h]
  __int64 v119; // [rsp+10h] [rbp-70h]
  __int64 v120; // [rsp+10h] [rbp-70h]
  _QWORD *v121; // [rsp+18h] [rbp-68h]
  __int16 v122; // [rsp+22h] [rbp-5Eh]
  int v123; // [rsp+24h] [rbp-5Ch]
  __int64 *v124; // [rsp+28h] [rbp-58h]
  _QWORD *v125; // [rsp+28h] [rbp-58h]
  __int64 v126; // [rsp+28h] [rbp-58h]
  unsigned int v127; // [rsp+30h] [rbp-50h]
  __int64 v128; // [rsp+30h] [rbp-50h]
  _BYTE *v129; // [rsp+30h] [rbp-50h]
  __int64 v130; // [rsp+30h] [rbp-50h]
  __int64 v131; // [rsp+38h] [rbp-48h]
  int v132; // [rsp+44h] [rbp-3Ch] BYREF
  const __m128i *v133; // [rsp+48h] [rbp-38h] BYREF

  v5 = *(_BYTE *)(a1 + 140) - 9;
  v127 = sub_7DB6D0(a1);
  v6 = sub_7DB910(v127, a1);
  v7 = *(_QWORD *)(a1 + 152);
  v8 = (const __m128i *)v6;
  v131 = v7;
  v123 = dword_4F07508[0];
  v122 = dword_4F07508[1];
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a1 + 64);
  if ( v5 > 2u )
  {
    if ( a2 )
    {
LABEL_7:
      v10 = *(_QWORD *)(v131 + 8) == 0;
      *(_BYTE *)(v131 + 136) = 2;
      if ( !v10 )
        *(_BYTE *)(v131 + 88) = *(_BYTE *)(v131 + 88) & 0x8F | 0x10;
      if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u )
      {
        v83 = sub_7E3BE0(a1);
        if ( v83 )
        {
          if ( (*(_BYTE *)(v83 + 174) & 1) != 0 )
            *(_BYTE *)(v131 + 174) |= 1u;
        }
      }
      *(_BYTE *)(v131 + 168) &= 0xF8u;
      goto LABEL_11;
    }
  }
  else
  {
    for ( i = *(const __m128i **)(v7 + 120); i[8].m128i_i8[12] == 12; i = (const __m128i *)i[10].m128i_i64[0] )
      ;
    if ( v8 != i )
      *(_QWORD *)(v7 + 120) = sub_73C570(v8, 1);
    if ( a2 )
      goto LABEL_7;
    *(_BYTE *)(v131 + 88) |= 4u;
    *(_BYTE *)(v131 + 136) = 0;
    if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u )
    {
      v84 = sub_7E3BE0(a1);
      if ( v84 )
      {
        if ( (*(_BYTE *)(v84 + 174) & 1) != 0 )
          *(_BYTE *)(v131 + 174) |= 1u;
      }
    }
  }
  if ( a3 )
    sub_7E4C10(v131);
LABEL_11:
  sub_7296C0(&v132);
  v11 = qword_4D03EE0[v127];
  if ( !v11 )
  {
    v87 = qword_4F18960[v127];
    v88 = *(_QWORD *)(v87 + 8);
    *(_QWORD *)(v87 + 8) = *(&off_4B6D480 + (int)v127);
    if ( v127 )
    {
      if ( v127 - 1 > 9 )
        goto LABEL_115;
      v89 = qword_4D04998;
    }
    else
    {
      if ( !dword_4F06900 )
        goto LABEL_112;
      v89 = qword_4D049B8;
    }
    if ( v89 )
    {
      v126 = *(_QWORD *)(v87 + 40);
      *(_QWORD *)(v87 + 40) = *(_QWORD *)(v89[11] + 128LL);
      v11 = sub_7E32B0(v87, 0, 0);
      sub_7E3260(v11, v87, 0, 0);
      qword_4D03EE0[v127] = v11;
      *(_QWORD *)(v87 + 8) = v88;
      *(_QWORD *)(v87 + 40) = v126;
      goto LABEL_12;
    }
LABEL_112:
    v11 = sub_7E32B0(v87, 0, 0);
    sub_7E3260(v11, v87, 0, 0);
    qword_4D03EE0[v127] = v11;
    *(_QWORD *)(v87 + 8) = v88;
  }
LABEL_12:
  v12 = sub_724D50(6);
  sub_72D510(v11, (__int64)v12, 1);
  *(_BYTE *)(v11 + 88) |= 4u;
  *((_QWORD *)v12 + 24) = 2 * sub_7E1340();
  v13 = sub_7E1DC0();
  sub_70FEE0((__int64)v12, v13, v14, v15, v16);
  v17 = sub_724D50(10);
  v18 = qword_4F18960[0];
  v17[22] = v12;
  v17[16] = v18;
  v19 = (const char *)sub_80F740(a1);
  v20 = strlen(v19);
  v21 = (char *)sub_724830(v20 + 1);
  strcpy(v21, v19);
  v22 = strlen(v21) + 1;
  v133 = (const __m128i *)sub_724DC0();
  sub_724C70((__int64)v133, 2);
  v124 = (__int64 *)v133;
  v23 = sub_73CA60(v22);
  v24 = v133;
  v124[16] = (__int64)v23;
  v24[11].m128i_i64[0] = v22;
  v24[11].m128i_i64[1] = (__int64)v21;
  v25 = sub_740630(v24);
  sub_73C570((const __m128i *)v25[8].m128i_i64[0], 1);
  v26 = a1;
  if ( !(unsigned int)sub_8D96C0(a1)
    || (*(_BYTE *)(a1 + 89) & 1) != 0 && (v40 = sub_72B7F0(a1), v26 = a1, (unsigned int)sub_736A50(v40)) )
  {
    v41 = (char *)sub_80FBA0(v26);
    v42 = sub_7E2190(v41);
    *(_BYTE *)(v42 + 89) |= 8u;
    v43 = (__int64 *)v133;
    v44 = v42;
    sub_72D510(v42, (__int64)v133, 1);
    *(_BYTE *)(v44 + 177) = 1;
    *(_QWORD *)(v44 + 184) = v25;
    *(_BYTE *)(v44 + 168) = sub_8DD330(a1, v43) & 7 | *(_BYTE *)(v44 + 168) & 0xF8;
    sub_7E4C10(v44);
    if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u )
    {
      v45 = sub_7E3BE0(a1);
      if ( v45 )
      {
        if ( (*(_BYTE *)(v45 + 174) & 1) != 0 )
          *(_BYTE *)(v44 + 174) |= 1u;
      }
    }
  }
  else
  {
    v27 = (char *)sub_80FBA0(v26);
    v28 = sub_7E2190(v27);
    *(_BYTE *)(v28 + 89) |= 8u;
    v29 = (__int64 *)v133;
    v30 = v28;
    sub_72D510(v28, (__int64)v133, 1);
    *(_BYTE *)(v30 + 177) = 1;
    *(_QWORD *)(v30 + 184) = v25;
    *(_BYTE *)(v30 + 168) = sub_8DD330(a1, v29) & 7 | *(_BYTE *)(v30 + 168) & 0xF8;
  }
  v31 = (const __m128i *)sub_72BA30(0);
  v32 = sub_73C570(v31, 1);
  v33 = (_BYTE *)sub_72D2E0(v32);
  sub_70FEE0((__int64)v133, (__int64)v33, v34, v35, v36);
  v37 = sub_724E50((__int64 *)&v133, v33);
  *((_QWORD *)v12 + 15) = v37;
  v17[23] = v37;
  *(_QWORD *)v131 = 0;
  v125 = sub_724D50(10);
  v125[16] = *(_QWORD *)(v131 + 120);
  if ( v127 > 7 )
  {
    if ( v127 - 9 > 1 )
      goto LABEL_115;
    if ( *(_BYTE *)(a1 + 140) == 6 )
      v46 = (__m128i *)sub_8D46C0(a1);
    else
      v46 = (__m128i *)sub_8D4870(a1);
    v47 = v46[8].m128i_i8[12];
    if ( dword_4D0425C && unk_4D04250 <= 0x765Bu && v47 == 8 )
    {
      v128 = 0;
    }
    else
    {
      if ( (v47 & 0xFB) != 8 )
      {
        v128 = 0;
LABEL_31:
        if ( (unsigned int)sub_8D2E30(a1) && (unsigned int)sub_7DB020(a1) )
        {
          v128 |= 8uLL;
        }
        else if ( (unsigned int)sub_8D3D10(a1) )
        {
          v61 = sub_7DB020(a1);
          v62 = v128 | 0x10;
          if ( !v61 )
            v62 = v128;
          v128 = v62;
        }
        if ( (unsigned int)sub_8D2310(v46) )
        {
          for ( j = v46; j[8].m128i_i8[12] == 12; j = (__m128i *)j[10].m128i_i64[0] )
            ;
          if ( (unsigned int)sub_8D76D0(j) )
          {
            while ( v46[8].m128i_i8[12] == 12 )
              v46 = (__m128i *)v46[10].m128i_i64[0];
            v128 |= 0x40uLL;
            v46 = sub_73C240(v46);
          }
        }
        v49 = v128;
        v129 = sub_724D50(1);
        sub_72BBE0((__int64)v129, v49, 6u);
        v50 = sub_724D50(6);
        for ( k = sub_73D4C0(v46, dword_4F077C4 == 2); k[8].m128i_i8[12] == 12; k = (__m128i *)k[10].m128i_i64[0] )
          ;
        v52 = sub_7DC650((__int64)k);
        sub_72D510(v52, (__int64)v50, 1);
        v53 = (const __m128i *)sub_7DBE60();
        v54 = sub_73C570(v53, 1);
        v55 = sub_72D2E0(v54);
        sub_70FEE0((__int64)v50, v55, v56, v57, v58);
        v17[15] = v129;
        *((_QWORD *)v129 + 15) = v50;
        v59 = sub_724D50(10);
        v60 = sub_7DB910(8u, 0);
        v59[23] = v50;
        v59[16] = v60;
        v59[22] = v17;
        v125[22] = v59;
        if ( (unsigned int)sub_8D2E30(a1) )
        {
          v125[23] = v59;
        }
        else if ( *(_BYTE *)(a1 + 140) == 13 )
        {
          v74 = sub_724D50(6);
          v75 = sub_8D4890(a1);
          v76 = sub_7DC650(v75);
          sub_72D510(v76, (__int64)v74, 1);
          v77 = (const __m128i *)sub_7DB910(5u, 0);
          v78 = sub_73C570(v77, 1);
          v79 = sub_72D2E0(v78);
          sub_70FEE0((__int64)v74, v79, v80, v81, v82);
          v59[15] = v74;
          v125[23] = v74;
        }
        goto LABEL_19;
      }
      v112 = sub_8D4C10(v46, dword_4F077C4 != 2);
      v128 = v112 & 1;
      v113 = v112 & 1;
      if ( (v46[8].m128i_i8[12] & 0xFB) != 8 )
        goto LABEL_31;
      v114 = sub_8D4C10(v46, dword_4F077C4 != 2);
      v115 = v113 | 2u;
      if ( (v114 & 2) == 0 )
        v115 = v128;
      v128 = v115;
      if ( (v46[8].m128i_i8[12] & 0xFB) != 8 )
        goto LABEL_31;
    }
    v116 = sub_8D4C10(v46, dword_4F077C4 != 2);
    v117 = v128 | 4;
    if ( (v116 & 4) == 0 )
      v117 = v128;
    v128 = v117;
    goto LABEL_31;
  }
  if ( v127 <= 5 )
  {
    if ( v127 - 1 <= 4 )
    {
      v125[22] = v17;
      v125[23] = v17;
      goto LABEL_19;
    }
LABEL_115:
    sub_721090();
  }
  v121 = sub_724D50(10);
  v63 = sub_7DB910(5u, 0);
  v121[22] = v17;
  v121[16] = v63;
  v121[23] = v17;
  v125[22] = v121;
  v64 = *(__int64 ***)(a1 + 168);
  if ( v127 == 6 )
  {
    v65 = *v64;
    if ( !*v64 )
    {
LABEL_116:
      sub_724D50(6);
      BUG();
    }
    while ( (v65[12] & 1) == 0 )
    {
      v65 = (__int64 *)*v65;
      if ( !v65 )
        goto LABEL_116;
    }
    v66 = sub_724D50(6);
    v67 = sub_7DC650(v65[5]);
    sub_72D510(v67, (__int64)v66, 1);
    v68 = (const __m128i *)sub_7DB910(5u, 0);
    v69 = sub_73C570(v68, 1);
    v70 = sub_72D2E0(v69);
    sub_70FEE0((__int64)v66, v70, v71, v72, v73);
    v121[15] = v66;
    v125[23] = v66;
  }
  else
  {
    v85 = *v64;
    if ( v85 )
    {
      v86 = 0;
      do
      {
        if ( (v85[12] & 2) != 0 )
        {
          if ( *(_QWORD *)v85[14] )
            v86 |= 2uLL;
        }
        else if ( (v85[12] & 4) != 0 )
        {
          v86 |= 1uLL;
        }
        v85 = (__int64 *)*v85;
      }
      while ( v85 && v86 != 3 );
      v119 = v86;
    }
    else
    {
      v119 = 0;
    }
    v90 = sub_724D50(10);
    v91 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 8LL);
    if ( v91 )
    {
      v130 = 0;
      do
      {
        ++v130;
        v92 = sub_724D50(6);
        v93 = sub_7DC650(*(_QWORD *)(v91 + 40));
        sub_72D510(v93, (__int64)v92, 1);
        v94 = (const __m128i *)sub_7DB910(5u, 0);
        v95 = sub_73C570(v94, 1);
        v96 = sub_72D2E0(v95);
        sub_70FEE0((__int64)v92, v96, v97, v98, v99);
        v100 = sub_724D50(1);
        v101 = *(_BYTE *)(v91 + 96) & 2;
        if ( v101 )
        {
          v111 = *(_QWORD *)(v91 + 144);
          v102 = v111 * sub_7E1340();
          v101 = *(_BYTE *)(v91 + 96) & 2;
        }
        else
        {
          v102 = *(_QWORD *)(v91 + 104);
        }
        v10 = v101 == 0;
        v103 = *(__int64 **)(v91 + 112);
        for ( m = !v10; (v103[3] & 1) == 0; v103 = (__int64 *)*v103 )
          ;
        if ( !*((_BYTE *)v103 + 25) )
          m |= 2uLL;
        sub_72BAF0((__int64)v100, m | (v102 << 8), 7u);
        *((_QWORD *)v92 + 15) = v100;
        v105 = sub_724D50(10);
        v106 = sub_7DBFA0();
        v105[22] = v92;
        v105[16] = v106;
        v105[23] = v100;
        if ( v90[22] )
          *(_QWORD *)(v90[23] + 120LL) = v105;
        else
          v90[22] = v105;
        v90[23] = v105;
        v91 = *(_QWORD *)(v91 + 8);
      }
      while ( v91 );
      v107 = v130;
    }
    else
    {
      v130 = 0;
      v107 = 0;
    }
    v118 = v107;
    v108 = sub_724D50(1);
    sub_72BAF0((__int64)v108, v119, 6u);
    v109 = sub_724D50(1);
    sub_72BAF0((__int64)v109, v118, 6u);
    v120 = sub_7DBFA0();
    v110 = sub_7259C0(8);
    v110[20] = v120;
    v90[16] = v110;
    v110[22] = v130;
    sub_8D6090(v90[16]);
    v121[15] = v108;
    *((_QWORD *)v108 + 15) = v109;
    *((_QWORD *)v109 + 15) = v90;
    v125[23] = v90;
  }
LABEL_19:
  v38 = v132;
  *(_QWORD *)(v131 + 184) = v125;
  *(_BYTE *)(v131 + 177) = 1;
  sub_729730(v38);
  dword_4F07508[0] = v123;
  LOWORD(dword_4F07508[1]) = v122;
  return dword_4F07508;
}
