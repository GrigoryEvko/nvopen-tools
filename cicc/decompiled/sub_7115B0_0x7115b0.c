// Function: sub_7115B0
// Address: 0x7115b0
//
__int64 __fastcall sub_7115B0(
        const __m128i *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        unsigned int a8,
        unsigned int a9,
        int a10,
        int a11,
        _DWORD *a12,
        _DWORD *a13,
        _DWORD *a14)
{
  int v14; // r14d
  __m128i *v16; // rdi
  __int64 i; // rbx
  char v18; // al
  __int64 j; // r12
  __m128i *v20; // rdi
  unsigned int v21; // r10d
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r10
  char v33; // al
  char v34; // al
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  int v38; // eax
  char v39; // al
  __int64 v40; // rdi
  __int64 v41; // rax
  char k; // dl
  __int64 v43; // rax
  char m; // cl
  char v45; // al
  __m128i *v46; // r15
  __int64 v47; // rdi
  __int64 v48; // rax
  unsigned int *v49; // rax
  __int64 v50; // r8
  __int64 v51; // rcx
  __int64 v52; // r8
  char v53; // al
  unsigned __int8 v54; // al
  __int64 v55; // rax
  char n; // dl
  unsigned __int8 v57; // cl
  unsigned __int8 v58; // dl
  unsigned __int8 v59; // r9
  char v60; // al
  __m128i *v61; // r15
  int v62; // eax
  FILE *v63; // r15
  unsigned int v64; // eax
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // rax
  __int64 v70; // rdi
  __int64 v71; // rdx
  char v72; // al
  int v73; // eax
  char v74; // al
  __int64 v75; // rcx
  _QWORD *v76; // r8
  _QWORD *v77; // r15
  __int64 v78; // rax
  int v79; // eax
  __int64 v80; // rbx
  char v81; // al
  __int64 v82; // rcx
  __int64 v83; // r8
  __int64 v84; // rcx
  __int64 v85; // [rsp+0h] [rbp-90h]
  _QWORD *v86; // [rsp+8h] [rbp-88h]
  __int64 v87; // [rsp+10h] [rbp-80h]
  __int64 v88; // [rsp+18h] [rbp-78h]
  __m128i *v89; // [rsp+18h] [rbp-78h]
  __int64 v90; // [rsp+20h] [rbp-70h]
  __int64 v91; // [rsp+20h] [rbp-70h]
  __int64 v92; // [rsp+28h] [rbp-68h]
  __int64 v93; // [rsp+28h] [rbp-68h]
  _QWORD *v94; // [rsp+28h] [rbp-68h]
  __int64 v95; // [rsp+28h] [rbp-68h]
  __int64 v96; // [rsp+28h] [rbp-68h]
  char v97; // [rsp+28h] [rbp-68h]
  int v98; // [rsp+30h] [rbp-60h]
  __int64 v99; // [rsp+30h] [rbp-60h]
  unsigned int *v100; // [rsp+30h] [rbp-60h]
  unsigned __int8 v101; // [rsp+30h] [rbp-60h]
  unsigned __int8 v102; // [rsp+30h] [rbp-60h]
  int v103; // [rsp+38h] [rbp-58h]
  unsigned int v104; // [rsp+3Ch] [rbp-54h]
  char v105; // [rsp+4Bh] [rbp-45h] BYREF
  unsigned int v106; // [rsp+4Ch] [rbp-44h] BYREF
  int v107; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v108; // [rsp+54h] [rbp-3Ch] BYREF
  __m128i *v109; // [rsp+58h] [rbp-38h] BYREF

  v14 = a4;
  v104 = a3;
  v103 = a5;
  v98 = a6;
  v107 = 0;
  if ( a13 )
    *a13 = 0;
  v106 = 0;
  v105 = 5;
  *a12 = 0;
  v109 = (__m128i *)sub_724DC0(a1, a2, a3, a4, a5, a6);
  sub_724C70(v109, 0);
  if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) && !(dword_4F077BC | v104) )
  {
    v16 = v109;
    v109[10].m128i_i8[9] |= 4u;
  }
  else
  {
    v16 = v109;
    v109[10].m128i_i8[9] = a1[10].m128i_i8[9] & 4 | v109[10].m128i_i8[9] & 0xFB;
  }
  v16[8].m128i_i64[0] = a2;
  for ( i = a1[8].m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v18 = *(_BYTE *)(a2 + 140);
  for ( j = a2; v18 == 12; v18 = *(_BYTE *)(j + 140) )
    j = *(_QWORD *)(j + 160);
  if ( v18 == 0 || a1[10].m128i_i8[13] == 0 )
    goto LABEL_10;
  if ( (unsigned int)sub_8D97B0(i) )
  {
    v16 = v109;
LABEL_10:
    sub_72C970(v16);
    v20 = v109;
    goto LABEL_11;
  }
  if ( dword_4F077C4 != 2 )
  {
LABEL_30:
    if ( i != j && !(unsigned int)sub_8D97D0(i, j, 0, v27, v28) )
    {
      LOBYTE(v29) = 0;
      v30 = qword_4D03C50;
      if ( !qword_4D03C50 )
        goto LABEL_77;
      goto LABEL_33;
    }
    goto LABEL_101;
  }
  if ( a1[10].m128i_i8[13] == 12 )
  {
    v29 = 1;
  }
  else
  {
    if ( !dword_4F07588 )
      goto LABEL_30;
    LOBYTE(v29) = (unsigned int)sub_8DBE70(j) != 0;
  }
  if ( i != j )
  {
    v97 = v29;
    v73 = sub_8D97D0(i, j, 0, v29, v28);
    LOBYTE(v29) = v97;
    if ( !v73 )
    {
      v30 = qword_4D03C50;
      if ( !qword_4D03C50 )
        goto LABEL_35;
LABEL_33:
      if ( (*(_BYTE *)(v30 + 17) & 2) != 0 || v103 || (*(_BYTE *)(v30 + 18) & 8) != 0 )
        goto LABEL_35;
      goto LABEL_72;
    }
  }
  if ( v104 || !(_BYTE)v29 )
  {
LABEL_101:
    sub_72A510(a1, v109);
    v20 = v109;
    v109[8].m128i_i64[0] = a2;
    goto LABEL_11;
  }
  if ( !qword_4D03C50 )
    goto LABEL_36;
  LOBYTE(v29) = (v103 == 0) & ((*(_BYTE *)(qword_4D03C50 + 17LL) >> 1) ^ 1);
  if ( !(_BYTE)v29 || (*(_BYTE *)(qword_4D03C50 + 18LL) & 8) != 0 )
    goto LABEL_36;
LABEL_72:
  v34 = *(_BYTE *)(j + 140);
  if ( v34 != 19 && v34 != 6 || a1[10].m128i_i8[13] != 1 )
    goto LABEL_75;
LABEL_35:
  if ( (_BYTE)v29 )
  {
LABEL_36:
    sub_70FDD0((__int64)a1, (__int64)v109, j, v104 == 0);
    v20 = v109;
    goto LABEL_11;
  }
LABEL_77:
  if ( (unsigned int)sub_8D29A0(j) )
  {
    if ( sub_70FCE0((__int64)a1) )
    {
      sub_724A80(v109, 1);
      v38 = sub_711520((__int64)a1, 1, v35, v36, v37);
      sub_620D80((__m128i *)v109[11].m128i_i16, v38 == 0);
      v20 = v109;
      goto LABEL_11;
    }
LABEL_75:
    *a12 = 1;
LABEL_76:
    v20 = v109;
    goto LABEL_11;
  }
  if ( (unsigned int)sub_8D2660(j) )
  {
    sub_724A80(v109, 1);
    sub_620D80((__m128i *)v109[11].m128i_i16, 0);
    v20 = v109;
    v109[10].m128i_i8[8] |= 8u;
    goto LABEL_11;
  }
  if ( dword_4D047EC && !v104 && (unsigned int)sub_8E3210(j) )
    goto LABEL_75;
  v39 = *(_BYTE *)(i + 140);
  if ( *(_BYTE *)(j + 140) == 15 )
  {
    if ( v39 != 15 )
      goto LABEL_75;
    v41 = *(_QWORD *)(j + 160);
    for ( k = *(_BYTE *)(v41 + 140); k == 12; k = *(_BYTE *)(v41 + 140) )
      v41 = *(_QWORD *)(v41 + 160);
    v43 = *(_QWORD *)(i + 160);
    for ( m = *(_BYTE *)(v43 + 140); m == 12; m = *(_BYTE *)(v43 + 140) )
      v43 = *(_QWORD *)(v43 + 160);
    if ( m != k )
      goto LABEL_75;
    v99 = sub_8D4620(j);
    if ( v99 != sub_8D4620(i) )
      goto LABEL_75;
    goto LABEL_101;
  }
  if ( v39 == 15 )
    goto LABEL_75;
  if ( a1[10].m128i_i8[13] == 6 )
  {
LABEL_115:
    sub_710F60(a1, v109, a8, a9, v104, v98, a10, 0, a12, a14, &v106, &v105);
    v20 = v109;
    goto LABEL_11;
  }
  v40 = v109[8].m128i_i64[0];
  if ( (unsigned int)sub_8D23B0(v40) )
  {
    v63 = (FILE *)v109[8].m128i_i64[0];
    v64 = sub_67F240();
    sub_685A50(v64, a14, v63, 8u);
    v20 = v109;
    *a12 = 1;
    goto LABEL_11;
  }
  switch ( *(_BYTE *)(i + 140) )
  {
    case 0:
      v20 = v109;
      v109[8].m128i_i64[0] = a1[8].m128i_i64[0];
      goto LABEL_11;
    case 2:
      v54 = *(_BYTE *)(j + 140);
      if ( v54 == 6 )
        goto LABEL_141;
      if ( v54 <= 6u )
      {
        if ( v54 != 2 )
        {
          if ( (unsigned __int8)(v54 - 3) <= 2u )
          {
            v55 = v109[8].m128i_i64[0];
            for ( n = *(_BYTE *)(v55 + 140); n == 12; n = *(_BYTE *)(v55 + 140) )
              v55 = *(_QWORD *)(v55 + 160);
            v57 = *(_BYTE *)(v55 + 160);
            v106 = 0;
            v105 = 5;
            v101 = v57;
            if ( n == 5 )
              v58 = 4;
            else
              v58 = 2 * (n == 4) + 3;
            v93 = v55;
            sub_724A80(v109, v58);
            v59 = v101;
            v60 = *(_BYTE *)(v93 + 140);
            if ( v60 == 5 )
            {
              v61 = (__m128i *)v109[11].m128i_i64[0];
              sub_70B680(v101, 0, (__m128i *)v61[1].m128i_i8, &v108);
              v59 = v101;
            }
            else
            {
              v61 = v109 + 11;
              if ( v60 == 4 )
              {
                sub_70B680(v101, 0, v61, &v108);
LABEL_131:
                if ( v108 )
                {
                  v106 = 220;
                  v105 = 8;
                }
                goto LABEL_76;
              }
            }
            v102 = v59;
            v62 = sub_620E90((__int64)a1);
            sub_622780((__m128i *)&a1[11], v62, (__int64)v61, v102, &v108);
            goto LABEL_131;
          }
LABEL_104:
          sub_721090(v40);
        }
        goto LABEL_119;
      }
      if ( v54 != 13 )
        goto LABEL_104;
      goto LABEL_140;
    case 3:
    case 4:
    case 5:
      v45 = *(_BYTE *)(j + 140);
      if ( v45 == 2 )
      {
        sub_7103C0((__int64)a1, (__int64)v109, &v106, (unsigned __int8 *)&v105, &v107, v14);
        v20 = v109;
      }
      else
      {
        if ( (unsigned __int8)(v45 - 3) > 2u )
          goto LABEL_104;
        sub_70F9A0(a1, (__int64)v109, &v106, &v105, &v107);
        v20 = v109;
      }
      goto LABEL_11;
    case 6:
      goto LABEL_115;
    case 9:
    case 0xA:
    case 0xB:
      v20 = v109;
      v109[8].m128i_i64[0] = a2;
      goto LABEL_11;
    case 0xD:
      v46 = v109;
      v105 = 5;
      v47 = a1[8].m128i_i64[0];
      v106 = 0;
      v48 = v109[8].m128i_i64[0];
      *a12 = 0;
      v92 = v48;
      v49 = 0;
      if ( a13 )
        v49 = &v106;
      v100 = v49;
      v90 = sub_8D4890(v47);
      v50 = sub_8D4890(v92);
      if ( a10 )
      {
        if ( (a1[10].m128i_i8[8] & 0x40) == 0 && a1[11].m128i_i64[0] )
          goto LABEL_75;
        sub_72A510(a1, v46);
        sub_70C9E0((__int64)v46, v92, v104, v51, v52);
        v46[10].m128i_i8[8] |= 0x40u;
        v20 = v109;
LABEL_11:
        if ( (v20[10].m128i_i8[9] & 4) != 0 )
          goto LABEL_12;
        goto LABEL_53;
      }
      if ( (a1[10].m128i_i8[8] & 0x40) != 0 )
        goto LABEL_75;
      if ( v90 == v50 || (v88 = v50, (unsigned int)sub_8DED30(v90, v50, 1)) )
      {
        sub_72A510(a1, v46);
        sub_70C9E0((__int64)v46, v92, v104, v65, v66);
        v20 = v109;
        goto LABEL_11;
      }
      v31 = sub_8D5CE0(v90, v88);
      if ( v31 )
      {
        v32 = v46[8].m128i_i64[0];
        *a12 = 0;
        if ( v100 )
        {
          *v100 = 0;
          v33 = *(_BYTE *)(v31 + 96);
          if ( (v33 & 4) != 0 )
          {
            *v100 = 286;
            goto LABEL_45;
          }
        }
        else
        {
          v33 = *(_BYTE *)(v31 + 96);
          if ( (v33 & 4) != 0 )
          {
            sub_685360(0x11Eu, a14, *(_QWORD *)(v31 + 40));
            goto LABEL_45;
          }
        }
        if ( (v33 & 2) == 0
          && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v31 + 112) + 8LL) + 16LL) + 96LL) & 2) == 0
          || qword_4D0495C )
        {
          v91 = v32;
          v94 = (_QWORD *)v31;
          sub_72A510(a1, v46);
          *a12 = 0;
          if ( (unsigned int)sub_737660(v46) )
            sub_70C9E0((__int64)v46, v91, 0, v67, v68);
          else
            sub_70D630((__int64)v46, v91, v94, 1, 0, a12);
          goto LABEL_46;
        }
        if ( v100 )
        {
          *v100 = 916;
        }
        else
        {
          v95 = *(_QWORD *)(v31 + 40);
          v69 = sub_8D4890(a1[8].m128i_i64[0]);
          sub_6861A0(0x394u, a14, v69, v95);
        }
LABEL_45:
        sub_72C970(v46);
        goto LABEL_46;
      }
      v40 = v88;
      v96 = sub_8D5CE0(v88, v90);
      if ( !v96 )
        goto LABEL_104;
      v70 = v46[8].m128i_i64[0];
      *a12 = 0;
      if ( v100 )
      {
        *v100 = 0;
        v71 = sub_8D4890(v70);
        v72 = *(_BYTE *)(v96 + 96);
        if ( (v72 & 4) != 0 )
        {
          *v100 = 287;
          goto LABEL_45;
        }
        if ( (v72 & 2) != 0
          || (v75 = *(_QWORD *)(v96 + 112), v76 = *(_QWORD **)(v75 + 8), (*(_BYTE *)(v76[2] + 96LL) & 2) != 0) )
        {
          *v100 = 804;
          goto LABEL_45;
        }
      }
      else
      {
        v71 = sub_8D4890(v70);
        v74 = *(_BYTE *)(v96 + 96);
        if ( (v74 & 4) != 0 )
        {
          sub_6861A0(0x11Fu, a14, v71, *(_QWORD *)(v96 + 40));
          goto LABEL_45;
        }
        if ( (v74 & 2) != 0
          || (v75 = *(_QWORD *)(v96 + 112), v76 = *(_QWORD **)(v75 + 8), (*(_BYTE *)(v76[2] + 96LL) & 2) != 0) )
        {
          sub_6861A0(0x324u, a14, v71, *(_QWORD *)(v96 + 40));
          goto LABEL_45;
        }
      }
      if ( dword_4F07588 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0 || !a8 )
        goto LABEL_197;
      v89 = v46;
      v77 = v76;
      v87 = i;
      v86 = *(_QWORD **)(v75 + 16);
      break;
    case 0xE:
      goto LABEL_75;
    case 0x13:
      v53 = *(_BYTE *)(j + 140);
      switch ( v53 )
      {
        case 6:
LABEL_141:
          sub_70CD50((__int64)a1, (__int64)v109, v104, &v106, &v105);
          v20 = v109;
          break;
        case 13:
LABEL_140:
          sub_70DAE0((__int64)v109, v104);
          v20 = v109;
          break;
        case 2:
LABEL_119:
          sub_710080(a1, (__int64)v109, v104, &v106, &v105);
          v20 = v109;
          goto LABEL_11;
        default:
          goto LABEL_104;
      }
      goto LABEL_11;
    default:
      goto LABEL_104;
  }
  while ( 1 )
  {
    if ( (_QWORD *)*v86 == v77 )
    {
      v46 = v89;
      i = v87;
      goto LABEL_197;
    }
    v80 = v77[2];
    v81 = *(_BYTE *)(v80 + 96);
    if ( (v81 & 2) != 0 )
    {
      if ( (v81 & 1) == 0 || (v78 = *(_QWORD *)(v80 + 112), *(_QWORD *)v78) )
      {
        v79 = sub_87DE40(v77[2], v71);
        goto LABEL_182;
      }
    }
    else
    {
      v78 = *(_QWORD *)(v80 + 112);
    }
    if ( !*(_BYTE *)(v78 + 25) || (v85 = v71, (v79 = sub_87D890(v71)) != 0) )
    {
      LOBYTE(v79) = 1;
    }
    else if ( *(_BYTE *)(*(_QWORD *)(v80 + 112) + 25LL) == 1 )
    {
      LOBYTE(v79) = (unsigned int)sub_87D970(v85) != 0;
    }
    v79 = (unsigned __int8)v79;
LABEL_182:
    if ( !v79
      && (*(_BYTE *)(*(_QWORD *)(v80 + 112) + 25LL) != 1
       || !dword_4F077BC
       || qword_4F077A8 > 0x9DCFu
       || !(unsigned int)sub_87E070(v80, v96)) )
    {
      break;
    }
    v71 = *(_QWORD *)(v80 + 40);
    v77 = (_QWORD *)*v77;
  }
  v84 = v80;
  v46 = v89;
  i = v87;
  if ( v100 )
  {
    if ( sub_67D3C0((int *)0x500, 7, a14) )
      *v100 = 1280;
  }
  else
  {
    sub_685260(7u, 0x500u, a14, *(_QWORD *)(v84 + 40));
  }
LABEL_197:
  sub_72A510(a1, v46);
  *a12 = 0;
  if ( (unsigned int)sub_737660(v46) )
    sub_70C9E0((__int64)v46, v70, v104, v82, v83);
  else
    sub_70D630((__int64)v46, v70, (_QWORD *)v96, 0, v104, a12);
LABEL_46:
  if ( !v100 )
    goto LABEL_76;
  if ( !v106 )
  {
    v20 = v109;
    if ( (v109[10].m128i_i8[9] & 4) != 0 )
      goto LABEL_13;
    goto LABEL_53;
  }
  v20 = v109;
  v105 = 8;
  if ( (v109[10].m128i_i8[9] & 4) == 0 )
  {
LABEL_53:
    if ( (unsigned int)sub_8D2930(j) && (unsigned int)sub_8D2D50(i)
      || dword_4F077C4 != 2 && (unsigned int)sub_8D4C80(j) && (unsigned int)sub_8D2930(i) )
    {
      v20 = v109;
    }
    else
    {
      v20 = v109;
      if ( !dword_4F077C0 || qword_4F077A8 > 0x9E33u || (*(_BYTE *)(i + 140) & 0xFB) != 2 || *(_BYTE *)(j + 140) != 6 )
        v109[10].m128i_i8[9] |= 4u;
    }
LABEL_12:
    v21 = v106;
    if ( v106 )
      goto LABEL_50;
LABEL_13:
    if ( v107 && !v14 )
    {
      *a12 = 1;
      if ( a11 )
        goto LABEL_16;
    }
    else if ( a11 )
    {
      goto LABEL_16;
    }
    goto LABEL_24;
  }
  v21 = v106;
LABEL_50:
  sub_70CE90((int *)v21, v105, v14, v103, a12, a13, a14, (__int64)v20);
  if ( v105 != 8 )
  {
    v20 = v109;
    goto LABEL_13;
  }
  v20 = v109;
  v107 = 0;
  if ( !a11 )
    goto LABEL_24;
LABEL_16:
  v22 = a1[9].m128i_i64[0];
  if ( !v22 || (unsigned __int8)v105 > 7u || *a12 )
  {
LABEL_24:
    v20[9].m128i_i64[0] = 0;
    goto LABEL_25;
  }
  if ( !v104 )
    goto LABEL_23;
  v23 = v20[8].m128i_i64[0];
  v24 = a1[8].m128i_i64[0];
  if ( v24 == v23 )
    goto LABEL_103;
  if ( (unsigned int)sub_8D97D0(a1[8].m128i_i64[0], v23, 0, v104, v24) )
  {
    v20 = v109;
    v22 = a1[9].m128i_i64[0];
LABEL_103:
    v20[9].m128i_i64[0] = v22;
    goto LABEL_25;
  }
  v22 = a1[9].m128i_i64[0];
  v20 = v109;
LABEL_23:
  v25 = sub_73DBF0(5, v20[8].m128i_i64[0], v22);
  *(_BYTE *)(v25 + 27) = (2 * (v104 & 1)) | *(_BYTE *)(v25 + 27) & 0xFD;
  v20 = v109;
  *(_BYTE *)(v25 + 58) = (2 * (a10 & 1)) | *(_BYTE *)(v25 + 58) & 0xFD;
  v20[9].m128i_i64[0] = v25;
LABEL_25:
  sub_72A510(v20, a1);
  return sub_724E30(&v109);
}
