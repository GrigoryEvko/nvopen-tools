// Function: sub_6A2D50
// Address: 0x6a2d50
//
__int64 __fastcall sub_6A2D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v7; // rbx
  _BYTE *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdx
  _QWORD *v12; // rdi
  int i; // r15d
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  bool v17; // r13
  __int64 v18; // rsi
  _BYTE *v19; // r12
  __int64 v20; // rax
  __int64 v21; // rdi
  char v22; // cl
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rdi
  bool v26; // zf
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  char v31; // al
  int v32; // eax
  __int64 v33; // r13
  _BYTE *v34; // rdi
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r12
  __int64 v39; // rax
  int v40; // eax
  __int64 v41; // rdi
  __int64 v42; // r14
  __int64 v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rcx
  __int64 v50; // rsi
  __int64 v51; // r15
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // rdi
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdi
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  char v66; // al
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 j; // rax
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // r9
  _QWORD *v75; // r15
  __int64 v76; // [rsp-10h] [rbp-3A0h]
  __int64 v77; // [rsp-8h] [rbp-398h]
  _BOOL4 v78; // [rsp+4h] [rbp-38Ch]
  int v79; // [rsp+8h] [rbp-388h]
  __int16 v80; // [rsp+Eh] [rbp-382h]
  __int64 v81; // [rsp+10h] [rbp-380h]
  __int64 v82; // [rsp+30h] [rbp-360h]
  _BOOL4 v83; // [rsp+38h] [rbp-358h]
  bool v84; // [rsp+38h] [rbp-358h]
  unsigned int v85; // [rsp+3Ch] [rbp-354h]
  unsigned int v86; // [rsp+48h] [rbp-348h]
  __int64 v87; // [rsp+48h] [rbp-348h]
  __int64 v88; // [rsp+48h] [rbp-348h]
  __int64 v89; // [rsp+50h] [rbp-340h]
  __int64 v90; // [rsp+50h] [rbp-340h]
  __int64 v91; // [rsp+50h] [rbp-340h]
  unsigned int v92; // [rsp+58h] [rbp-338h]
  _QWORD *v93; // [rsp+58h] [rbp-338h]
  __int64 v94; // [rsp+58h] [rbp-338h]
  _QWORD *v95; // [rsp+58h] [rbp-338h]
  unsigned int v96; // [rsp+64h] [rbp-32Ch] BYREF
  __int64 v97; // [rsp+68h] [rbp-328h] BYREF
  __int64 v98; // [rsp+70h] [rbp-320h] BYREF
  __int64 v99; // [rsp+78h] [rbp-318h] BYREF
  __int64 v100; // [rsp+80h] [rbp-310h] BYREF
  __int64 v101; // [rsp+88h] [rbp-308h] BYREF
  __int64 v102; // [rsp+90h] [rbp-300h] BYREF
  __int64 v103; // [rsp+98h] [rbp-2F8h] BYREF
  _BYTE v104[32]; // [rsp+A0h] [rbp-2F0h] BYREF
  _BYTE v105[160]; // [rsp+C0h] [rbp-2D0h] BYREF
  _QWORD v106[20]; // [rsp+160h] [rbp-230h] BYREF
  _BYTE v107[68]; // [rsp+200h] [rbp-190h] BYREF
  __int64 v108; // [rsp+244h] [rbp-14Ch]

  v7 = (_BYTE *)a1;
  if ( a1 )
  {
    v8 = v107;
    sub_6F8810(
      a1,
      (unsigned int)&v96,
      (unsigned int)v107,
      (unsigned int)&v97,
      (unsigned int)&v99,
      (unsigned int)v106,
      (__int64)&v101);
    a6 = v96;
    a1 = v76;
    a5 = v77;
    a4 = *(unsigned int *)(*(_QWORD *)v7 + 44LL);
    v79 = *(_DWORD *)(*(_QWORD *)v7 + 44LL);
    v80 = *(_WORD *)(*(_QWORD *)v7 + 48LL);
    if ( !v96 )
      v101 = v108;
  }
  else
  {
    v8 = v107;
    v99 = *(_QWORD *)&dword_4F063F8;
  }
  v92 = 0;
  v100 = v99;
  if ( unk_4F04C50 )
  {
    v9 = *(_QWORD *)(unk_4F04C50 + 32LL);
    if ( v9 )
    {
      if ( (*(_BYTE *)(v9 + 198) & 0x10) != 0 )
      {
        a1 = 3645;
        sub_6851A0(0xE3Du, &v100, (__int64)"typeid");
        v92 = 1;
      }
    }
  }
  v10 = dword_4D04324;
  if ( dword_4D04324 )
  {
    a1 = (__int64)&v100;
    v10 = 878;
    sub_684AB0(&v100, 0x36Eu);
  }
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
  {
    if ( (unsigned int)sub_6E5430(a1, v10, a3, a4, a5, a6) )
    {
      v10 = (__int64)&v100;
      sub_6851C0(0x39u, &v100);
    }
    if ( unk_4F07320 )
    {
      v92 = 1;
      if ( qword_4F06CE0 )
        goto LABEL_13;
    }
LABEL_122:
    v92 = 1;
    goto LABEL_21;
  }
  if ( !unk_4F07320 )
  {
    v11 = v92;
    if ( !v92 )
      goto LABEL_17;
LABEL_74:
    v92 = 1;
    goto LABEL_21;
  }
  if ( !qword_4F06CE0 )
  {
    if ( !v92 )
      goto LABEL_20;
    goto LABEL_122;
  }
LABEL_13:
  if ( dword_4F06900 )
  {
    sub_8865A0("type_info");
  }
  else
  {
    v10 = 0;
    sub_879550("type_info");
  }
  if ( dword_4F077C4 == 2 )
  {
    v75 = qword_4F06CE0;
    if ( (unsigned int)sub_8D23B0(qword_4F06CE0) )
      sub_8AE000(v75);
  }
  v11 = v92;
  if ( v92 )
    goto LABEL_74;
LABEL_17:
  v12 = qword_4F06CE0;
  if ( qword_4F06CE0 )
  {
    if ( (_DWORD)qword_4F077B4 )
    {
      if ( *(_DWORD *)(*qword_4F06CE0 + 40LL) != -1 )
        goto LABEL_20;
    }
    else if ( !(unsigned int)sub_8D23B0(qword_4F06CE0) )
    {
      goto LABEL_20;
    }
    if ( (unsigned int)sub_6E5430(v12, v10, v11, a4, a5, a6) )
    {
      sub_6851C0(0x2B5u, &v100);
      goto LABEL_21;
    }
  }
LABEL_20:
  v92 = 0;
LABEL_21:
  v78 = 0;
  v83 = 0;
  v89 = 0;
  v81 = a2;
  for ( i = 0; ; i = 1 )
  {
    v82 = unk_4F074B0;
    if ( i || v7 )
    {
      v18 = (__int64)v105;
      v14 = *(unsigned __int8 *)(qword_4D03C50 + 16LL);
      sub_6E2140(v14, v105, 0, 0, v7);
      v68 = qword_4D03C50;
      *(_BYTE *)(qword_4D03C50 + 17LL) |= 4u;
      v17 = (*(_BYTE *)(v68 + 19) & 0x20) != 0;
      if ( v7 )
      {
        v85 = 0;
        goto LABEL_96;
      }
    }
    else
    {
      v14 = 5;
      sub_6E2140(5, v105, 0, 0, 0);
      v17 = (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x20) != 0;
    }
    v85 = 0;
    if ( !i )
    {
      sub_7B8B50(v14, v105, v15, v16);
      v85 = dword_4F06650[0];
      sub_7BDB60(1);
    }
    v18 = 125;
    sub_7BE280(27, 125, 0, 0);
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    ++*(_QWORD *)(qword_4D03C50 + 40LL);
    if ( (unsigned int)sub_679C10(5u) )
      break;
    v19 = (_BYTE *)qword_4F06BC0;
    if ( *(_BYTE *)qword_4F06BC0 == 4 )
      qword_4F06BC0 = *(_QWORD *)(qword_4F06BC0 + 32LL);
    sub_6E1DD0(&v98);
    sub_6E1E00(*(unsigned __int8 *)(v98 + 16), v106, 1, 0);
    v20 = qword_4D03C50;
    *(_BYTE *)(qword_4D03C50 + 22LL) |= 1u;
    v21 = v98;
    *(_BYTE *)(v20 + 17) |= 4u;
    sub_6E2170(v21);
    v18 = 0;
    v89 = sub_6A2C00(1, 0);
    v22 = *(_BYTE *)(qword_4D03C50 + 17LL);
    v78 = (v22 & 0x10) != 0;
    v83 = (v22 & 8) != 0;
    v23 = *(_QWORD *)(qword_4D03C50 + 88LL);
    v103 = *(_QWORD *)(qword_4D03C50 + 96LL);
    v102 = v23;
    sub_6E2B30(1, 0);
    sub_6E1DF0(v98);
    if ( *v19 == 4 )
      qword_4F06BC0 = v19;
    v96 = 0;
    v24 = *(_QWORD *)(v89 + 24);
    v8 = (_BYTE *)(v24 + 8);
    v101 = *(_QWORD *)(v24 + 76);
LABEL_33:
    sub_6F6C80(v8);
    sub_831BB0(v8);
    v25 = *(_QWORD *)v8;
    v26 = v8[17] == 1;
    v97 = *(_QWORD *)v8;
    if ( v26 )
    {
      if ( (unsigned int)sub_8D3E60() )
        goto LABEL_35;
      v25 = v97;
    }
    if ( !(unsigned int)sub_8DD3B0(v25) )
      goto LABEL_101;
LABEL_35:
    v31 = v8[16];
    if ( v31 == 1 )
    {
      v25 = *((_QWORD *)v8 + 18);
    }
    else
    {
      if ( v31 != 2 )
        goto LABEL_37;
      v25 = *((_QWORD *)v8 + 36);
      if ( v25 )
        goto LABEL_100;
      if ( v8[317] != 12 || v8[320] != 1 )
        goto LABEL_37;
      v25 = sub_72E9A0(v8 + 144);
    }
    if ( !v25 )
    {
      if ( !v83 )
        goto LABEL_38;
      goto LABEL_114;
    }
LABEL_100:
    v18 = 1;
    if ( (*(_BYTE *)(sub_6E36E0(v25, 1) + 26) & 2) != 0 )
    {
LABEL_101:
      v84 = v17;
      v33 = v81;
      v86 = word_4D04898 != 0;
      if ( v89 )
      {
        if ( !i )
          goto LABEL_103;
        goto LABEL_108;
      }
LABEL_45:
      v34 = v8;
      v18 = 0;
      v38 = sub_6F6F40(v8, 0);
      goto LABEL_46;
    }
LABEL_37:
    if ( !v83 )
      goto LABEL_38;
LABEL_114:
    if ( (unsigned int)sub_6E5430(v25, v18, v27, v28, v29, v30) )
    {
      v18 = (__int64)&v102;
      sub_6851C0(0xF5u, &v102);
    }
LABEL_38:
    v26 = (unsigned int)sub_6E9250(&v99) == 0;
    v32 = 1;
    if ( v26 )
      v32 = v92;
    v92 = v32;
    if ( v8[16] == 1 && !(unsigned int)sub_8DD3B0(v97) )
    {
      v18 = 1;
      if ( sub_82F150(v8, 1) )
      {
        if ( !(unsigned int)sub_6FE220(v8) )
        {
          v84 = v17;
          v86 = 0;
          v33 = v81;
          if ( !v89 )
          {
            i = 0;
            goto LABEL_45;
          }
LABEL_103:
          v69 = *(_QWORD *)(v89 + 32);
          if ( v69 )
          {
            for ( j = *(_QWORD *)(v69 + 24); j; j = *(_QWORD *)(j + 32) )
              *(_QWORD *)(j + 24) = 0;
            *(_QWORD *)(v89 + 32) = 0;
          }
          i = 0;
          *(_BYTE *)(v89 + 9) &= ~1u;
LABEL_108:
          sub_832D70(v89, v18);
          sub_6E1990(v89);
          goto LABEL_45;
        }
      }
    }
    if ( i || v7 || unk_4F074B0 != v82 )
    {
      v84 = v17;
      i = 1;
      v86 = 0;
      v33 = v81;
      if ( v89 )
        goto LABEL_108;
      goto LABEL_45;
    }
    sub_6E1990(v89);
    sub_6E2B30(v89, v18);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    --*(_QWORD *)(qword_4D03C50 + 40LL);
    sub_7ADF70(v104, 0);
    sub_7AE700(unk_4F061C0 + 24LL, v85, dword_4F06650[0], 0, v104);
    sub_7BC000(v104);
    sub_7BDC00();
  }
  v96 = 1;
  v101 = *(_QWORD *)&dword_4F063F8;
  sub_65CD60(&v97);
LABEL_96:
  if ( !v96 )
    goto LABEL_33;
  v34 = (_BYTE *)v97;
  v84 = v17;
  v38 = 0;
  v33 = v81;
  v86 = sub_8D32E0(v97);
  if ( v86 )
  {
    v34 = (_BYTE *)v97;
    v86 = 0;
    v97 = sub_8D46C0(v97);
  }
LABEL_46:
  if ( v85 )
    sub_7BDC00();
  if ( !i )
  {
    if ( word_4D04898 )
    {
      v36 = qword_4D03C50;
      v35 = 32 * (unsigned int)v84;
      *(_BYTE *)(qword_4D03C50 + 19LL) = (32 * v84) | *(_BYTE *)(qword_4D03C50 + 19LL) & 0xDF;
    }
    if ( !unk_4D041C4 && v78 )
    {
      v92 = 1;
      if ( (unsigned int)sub_6E5430(v34, v18, v35, v36, v37, v78) )
      {
        v18 = (__int64)&v103;
        sub_6851C0(0x6F1u, &v103);
      }
    }
  }
  if ( dword_4F04C44 == -1
    && (v39 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v39 + 6) & 6) == 0)
    && *(_BYTE *)(v39 + 4) != 12
    || !(unsigned int)sub_8DC060(v97) )
  {
    v40 = sub_8D3410(v97);
    v41 = v97;
    if ( v40 )
    {
      v18 = dword_4F077C4 == 2;
      v41 = sub_73D4C0(v97, v18);
    }
    while ( *(_BYTE *)(v41 + 140) == 12 )
      v41 = *(_QWORD *)(v41 + 160);
    v97 = v41;
    if ( dword_4F077C4 != 2 )
      goto LABEL_61;
LABEL_86:
    if ( (unsigned int)sub_8D23B0(v41) )
      sub_8AE000(v41);
    v41 = v97;
    if ( (unsigned int)sub_8D3A70(v97) )
      goto LABEL_89;
LABEL_62:
    if ( !dword_4D047EC )
      goto LABEL_63;
    v41 = v97;
    if ( !(unsigned int)sub_8DD010(v97) )
      goto LABEL_63;
    if ( (unsigned int)sub_6E5430(v41, v18, v71, v72, v73, v74) )
    {
      v18 = (__int64)&v101;
      v41 = 975;
      sub_6851C0(0x3CFu, &v101);
    }
LABEL_91:
    if ( v7 )
    {
      sub_6E2B30(v41, v18);
      goto LABEL_93;
    }
    v92 = 1;
LABEL_139:
    v18 = 18;
    v41 = 28;
    v79 = qword_4F063F0;
    v80 = WORD2(qword_4F063F0);
    sub_7BE280(28, 18, 0, 0);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    --*(_QWORD *)(qword_4D03C50 + 40LL);
    goto LABEL_64;
  }
  v97 = sub_8D2220(v97);
  v41 = v97;
  if ( dword_4F077C4 == 2 )
    goto LABEL_86;
LABEL_61:
  if ( !(unsigned int)sub_8D3A70(v41) )
    goto LABEL_62;
LABEL_89:
  v41 = v97;
  if ( (unsigned int)sub_8D23B0(v97) )
  {
    v18 = v97;
    v41 = (__int64)&v101;
    sub_6E5F60(&v101, v97, 8);
    goto LABEL_91;
  }
LABEL_63:
  if ( !v7 )
    goto LABEL_139;
LABEL_64:
  sub_6E2B30(v41, v18);
  if ( v92 )
  {
LABEL_93:
    sub_6E6260(v33);
    goto LABEL_83;
  }
  v42 = v97;
  if ( (dword_4F04C44 != -1
     || (v43 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v43 + 6) & 6) != 0)
     || *(_BYTE *)(v43 + 4) == 12)
    && ((unsigned int)sub_8DBE70(v97) || (unsigned int)sub_8DC060(v42)) )
  {
    v59 = sub_73C750(0);
    v94 = v59;
    v49 = sub_8D46C0(v59);
    if ( !v86 )
      goto LABEL_77;
    v106[0] = sub_724DC0(v59, v18, v59, v49, v60, v61);
    sub_724C70(v106[0], 12);
    sub_7249B0(v106[0], 9);
    v56 = v106[0];
    *(_QWORD *)(v106[0] + 184LL) = v42;
    if ( v38 )
      *(_QWORD *)(v56 + 192) = v38;
    *(_QWORD *)(v56 + 128) = v94;
LABEL_72:
    v57 = sub_73A720(v56);
    v58 = sub_73DCD0(v57);
    sub_6E7150(v58, v33);
    sub_724E30(v106);
  }
  else
  {
    v44 = sub_73C750(0);
    v45 = sub_8D46C0(v44);
    v18 = v86;
    v49 = v45;
    if ( v86 )
    {
      v87 = v45;
      v106[0] = sub_724DC0(v44, v18, v46, v45, v47, v48);
      sub_73C780(v42, 0, v106[0]);
      v50 = v106[0];
      v90 = v106[0];
      v51 = sub_726700(11);
      v93 = (_QWORD *)sub_726700(22);
      *v93 = sub_72CBE0(22, v50, v52, v53, v54, v55);
      v93[7] = v42;
      *(_QWORD *)(v51 + 56) = v93;
      if ( v38 )
        v93[2] = v38;
      *(_BYTE *)(v51 + 64) &= ~1u;
      *(_BYTE *)(v51 + 25) |= 1u;
      *(_QWORD *)v51 = v87;
      *(_QWORD *)(v90 + 144) = v51;
      v56 = v106[0];
      goto LABEL_72;
    }
LABEL_77:
    v88 = v49;
    v91 = sub_726700(11);
    v95 = (_QWORD *)sub_726700(22);
    *v95 = sub_72CBE0(22, v18, v62, v63, v64, v65);
    v95[7] = v42;
    *(_QWORD *)(v91 + 56) = v95;
    if ( v38 )
      v95[2] = v38;
    v66 = *(_BYTE *)(v91 + 64);
    *(_BYTE *)(v91 + 25) |= 1u;
    *(_QWORD *)v91 = v88;
    *(_BYTE *)(v91 + 64) = i | v66 & 0xFE;
    sub_6E7150(v91, v33);
  }
  if ( !v7 || !v7[56] )
    sub_8DCE90(v42);
LABEL_83:
  *(_DWORD *)(v33 + 68) = v100;
  *(_WORD *)(v33 + 72) = WORD2(v100);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(v33 + 68);
  *(_DWORD *)(v33 + 76) = v79;
  *(_WORD *)(v33 + 80) = v80;
  unk_4F061D8 = *(_QWORD *)(v33 + 76);
  sub_6E3280(v33, &v99);
  sub_6E3BA0(v33, &v99, 0, &v101);
  return sub_6E26D0(2, v33);
}
