// Function: sub_76EF80
// Address: 0x76ef80
//
char __fastcall sub_76EF80(_QWORD *a1, __m128i *a2, _DWORD *a3)
{
  _QWORD *v4; // r12
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rcx
  __int64 *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rdi
  int v14; // r9d
  __int16 v15; // r10
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r14
  __int64 v23; // r14
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // r13
  int v31; // eax
  int v32; // r11d
  int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // r12
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  _QWORD *v42; // rax
  _QWORD *v43; // rcx
  _QWORD *v44; // rdx
  __int64 v45; // rcx
  __int64 i; // r8
  __int64 v47; // rax
  int v48; // r15d
  __int64 v49; // r13
  __int64 v50; // rax
  __int64 v51; // r14
  __int64 v52; // rax
  _QWORD *v53; // rax
  __int64 v54; // rax
  _QWORD *v55; // rax
  char v56; // r8
  __m128i v57; // xmm1
  const __m128i *v58; // rax
  const __m128i *v59; // rdi
  __int16 v60; // r12
  __int16 v61; // r14
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rcx
  __int64 v66; // r8
  __m128i *v67; // rbx
  __int64 v68; // rcx
  __int64 v69; // r8
  _QWORD *v70; // rax
  _QWORD *v71; // r13
  __int64 v72; // rsi
  __int64 v73; // rdx
  __int64 v74; // rcx
  void *v75; // rdi
  __int64 v76; // r14
  int v77; // eax
  __int64 v78; // rax
  __int64 v79; // rsi
  _QWORD *v80; // rax
  __int64 v82; // [rsp+8h] [rbp-108h]
  __int64 v83; // [rsp+8h] [rbp-108h]
  __int64 v84; // [rsp+10h] [rbp-100h]
  __int64 v85; // [rsp+10h] [rbp-100h]
  __int64 v86; // [rsp+10h] [rbp-100h]
  __m128i *v87; // [rsp+20h] [rbp-F0h]
  __int64 v88; // [rsp+30h] [rbp-E0h]
  __int64 v89; // [rsp+38h] [rbp-D8h]
  const __m128i *v90; // [rsp+40h] [rbp-D0h]
  _BOOL4 v91; // [rsp+54h] [rbp-BCh]
  _QWORD *v92; // [rsp+58h] [rbp-B8h]
  unsigned int v93; // [rsp+68h] [rbp-A8h]
  __m128i *v94; // [rsp+68h] [rbp-A8h]
  __int64 v95; // [rsp+68h] [rbp-A8h]
  int v96; // [rsp+70h] [rbp-A0h]
  __int64 v97; // [rsp+70h] [rbp-A0h]
  int v98; // [rsp+70h] [rbp-A0h]
  __int32 v99; // [rsp+70h] [rbp-A0h]
  __int16 v100; // [rsp+78h] [rbp-98h]
  unsigned int v101; // [rsp+78h] [rbp-98h]
  _QWORD *v102; // [rsp+78h] [rbp-98h]
  __int64 v103; // [rsp+78h] [rbp-98h]
  __int32 v104; // [rsp+78h] [rbp-98h]
  __int64 v105; // [rsp+78h] [rbp-98h]
  int v106; // [rsp+88h] [rbp-88h] BYREF
  int v107; // [rsp+8Ch] [rbp-84h] BYREF
  int v108; // [rsp+90h] [rbp-80h] BYREF
  int v109; // [rsp+94h] [rbp-7Ch] BYREF
  unsigned __int64 v110; // [rsp+98h] [rbp-78h] BYREF
  _OWORD v111[2]; // [rsp+A0h] [rbp-70h] BYREF
  __m128i v112; // [rsp+C0h] [rbp-50h] BYREF
  __m128i v113[4]; // [rsp+D0h] [rbp-40h] BYREF

  v4 = a1;
  v110 = 0;
  v106 = 0;
  if ( a3 )
    *a3 = 0;
  v6 = a1[9];
  v7 = sub_72B0F0(v6, 0);
  v8 = v7;
  if ( !v7 )
    return v7;
  if ( *(char *)(v7 + 192) >= 0 )
    goto LABEL_5;
  v7 = (__int64)&dword_4F077C4;
  if ( dword_4F077C4 != 2 )
  {
    v7 = (__int64)&dword_4F07588;
    if ( dword_4F07588 )
    {
      LOBYTE(v7) = dword_4F04C64;
      if ( dword_4F04C64 != -1 )
      {
        v9 = 776LL * dword_4F04C64;
        v10 = (__int64 *)(qword_4F04C68[0] + v9 + 240);
        v11 = qword_4F04C68[0] + v9 - 536 - 776LL * (unsigned int)dword_4F04C64;
        while ( 1 )
        {
          v7 = *v10;
          if ( *v10 )
            break;
LABEL_15:
          v10 -= 97;
          if ( (__int64 *)v11 == v10 )
            goto LABEL_16;
        }
        while ( !*(_DWORD *)(v7 + 16) || v8 != *(_QWORD *)(v7 + 24) )
        {
          v7 = *(_QWORD *)v7;
          if ( !v7 )
            goto LABEL_15;
        }
        goto LABEL_5;
      }
    }
  }
LABEL_16:
  if ( (*(_BYTE *)(v8 + 198) & 0x10) != 0 )
  {
    if ( (*((_BYTE *)a1 + 25) & 4) == 0 )
      goto LABEL_18;
    LODWORD(v7) = sub_8D2600(*a1);
    if ( !(_DWORD)v7 )
      goto LABEL_5;
    if ( (*(_BYTE *)(v8 + 198) & 0x10) != 0 )
    {
LABEL_18:
      if ( (*(_BYTE *)(v8 - 8) & 0x10) != 0 )
      {
LABEL_5:
        *(_BYTE *)(v8 + 204) |= 1u;
        return v7;
      }
    }
  }
  v12 = sub_72B0F0(a1[9], 0);
  v7 = dword_4F04C58;
  if ( dword_4F04C58 == -1 )
  {
    if ( !qword_4F04C50 )
      goto LABEL_5;
    v13 = *(_QWORD *)(qword_4F04C50 + 32LL);
  }
  else
  {
    v13 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216);
  }
  if ( !v13 || !v12 )
    goto LABEL_5;
  if ( (*(_BYTE *)(v13 + 193) & 0x10) != 0
    || (*(_QWORD *)(v13 + 200) & 0x8000001000000LL) == 0x8000000000000LL && (*(_BYTE *)(v13 + 192) & 2) == 0 )
  {
    if ( (*(_BYTE *)(v12 + 193) & 0x10) == 0
      && ((*(_QWORD *)(v12 + 200) & 0x8000001000000LL) != 0x8000000000000LL || (*(_BYTE *)(v12 + 192) & 2) != 0) )
    {
      goto LABEL_5;
    }
  }
  else if ( (*(_BYTE *)(v12 + 193) & 0x10) == 0
         && ((*(_QWORD *)(v12 + 200) & 0x8000001000000LL) != 0x8000000000000LL || (*(_BYTE *)(v12 + 192) & 2) != 0) )
  {
    v14 = dword_4F07508[0];
    v15 = dword_4F07508[1];
    *(_QWORD *)dword_4F07508 = *(_QWORD *)((char *)v4 + 28);
    v96 = v14;
    v100 = v15;
    LODWORD(v7) = sub_825950(v13, v12, 0, 0);
    dword_4F07508[0] = v96;
    LOWORD(dword_4F07508[1]) = v100;
    if ( !(_DWORD)v7 )
      goto LABEL_5;
    v7 = dword_4F04C58;
  }
  if ( (_DWORD)v7 == -1 )
  {
    LOBYTE(v7) = qword_4F04C50;
    if ( !qword_4F04C50 )
      goto LABEL_5;
    v7 = *(_QWORD *)(qword_4F04C50 + 32LL);
  }
  else
  {
    v7 = *(_QWORD *)(qword_4F04C68[0] + 776 * v7 + 216);
  }
  if ( !v7 )
    goto LABEL_5;
  LOBYTE(v7) = *(_BYTE *)(v8 + 197) ^ *(_BYTE *)(v7 + 197);
  if ( (v7 & 0x18) != 0 )
    goto LABEL_5;
  LOBYTE(v7) = *(_BYTE *)(v8 + 203);
  if ( (v7 & 2) == 0 )
    goto LABEL_5;
  v108 = 1;
  v107 = 0;
  *(_BYTE *)(v8 + 203) = v7 & 0xFD;
  v92 = (_QWORD *)sub_72B840(v8);
  qword_4F08048 = (__int64)v92;
  if ( a2 )
  {
    v90 = (const __m128i *)sub_726B30(11);
    sub_7E1740(v90);
  }
  else
  {
    sub_7E1790(v111);
    v90 = 0;
  }
  v16 = *(_QWORD *)(v6 + 16);
  v112.m128i_i64[0] = 0;
  v97 = v16;
  v91 = (*((_BYTE *)v4 + 60) & 2) != 0;
  v89 = v92[4];
  v101 = 1;
  v17 = *(_QWORD *)(sub_72B840(v89) + 80);
  if ( *(_BYTE *)(v17 + 40) == 11 )
  {
    v47 = *(_QWORD *)(v17 + 72);
    if ( v47 )
    {
      if ( *(_BYTE *)(v47 + 40) == 8 && !*(_QWORD *)(v47 + 16) )
      {
        v20 = *(_QWORD *)(v47 + 48);
        if ( v20 )
        {
          if ( *(_BYTE *)(v20 + 24) == 1 && *(_BYTE *)(v20 + 56) == 73 )
          {
            v76 = *(_QWORD *)(v20 + 72);
            v105 = *(_QWORD *)(v47 + 48);
            v77 = sub_731770(v76, 0, v18, v19, v20, v21);
            v20 = v105;
            if ( !v77 )
              v20 = *(_QWORD *)(v76 + 16);
          }
          v101 = sub_731770(v20, 0, v18, v19, v20, v21);
        }
        else
        {
          v101 = 0;
        }
      }
    }
  }
  if ( v97 )
  {
    v22 = v97;
    do
    {
      if ( (unsigned int)sub_731770(v22, 0, v18, v19, v20, v21) )
      {
        v93 = 1;
        v101 = 1;
        goto LABEL_42;
      }
      v22 = *(_QWORD *)(v22 + 16);
    }
    while ( v22 );
    v93 = 0;
  }
  else
  {
    v93 = 0;
  }
LABEL_42:
  v23 = v92[5];
  if ( !v23 )
    goto LABEL_81;
  v88 = v8;
  v24 = v97;
  v87 = a2;
  do
  {
    v25 = sub_76D880(v23, v91, v112.m128i_i64);
    *(_QWORD *)(v25 + 32) = v24;
    v30 = v25;
    if ( (*(_BYTE *)(v23 + 88) & 4) != 0 )
    {
      v31 = sub_7E6B40(v24, v93, v101, 1, &v109);
      if ( (*(_BYTE *)(v23 + 171) & 0x30) != 0 )
      {
        v98 = 0;
        if ( (*(_BYTE *)(v23 + 172) & 1) == 0 || *(_BYTE *)(v89 + 174) != 1 || (v98 = 1, !v109) )
        {
          v32 = 0;
          goto LABEL_52;
        }
      }
      else
      {
        v98 = 0;
      }
      if ( v31 | v101 )
      {
        if ( !v31 )
          goto LABEL_71;
      }
      else if ( (*(_BYTE *)(v23 + 171) & 0x40) != 0 || !*(_QWORD *)(v23 + 8) )
      {
        goto LABEL_71;
      }
      if ( !(unsigned int)sub_8D3A70(*(_QWORD *)(v23 + 120)) || (unsigned int)sub_7E2090(v24) )
      {
        for ( i = *(_QWORD *)v24; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        if ( !dword_4F068C4
          || (v85 = i, !sub_730740(v24))
          || !(unsigned int)sub_8D2E30(v85)
          || !(unsigned int)sub_8D2E30(**(_QWORD **)(v24 + 72))
          || (v63 = sub_8D46C0(v85), !(unsigned int)sub_8D2310(v63))
          || (v83 = v85,
              v86 = sub_8D46C0(**(_QWORD **)(v24 + 72)),
              v64 = sub_8D46C0(v83),
              (unsigned int)sub_8D97D0(v64, v86, 0, v65, v66)) )
        {
          *(_DWORD *)(v30 + 16) = 2;
          *(_QWORD *)(v30 + 24) = v24;
          goto LABEL_46;
        }
        goto LABEL_163;
      }
LABEL_71:
      v32 = 1;
      if ( *(_BYTE *)(v24 + 24) != 1
        || *(_BYTE *)(v24 + 56) != 91
        || (v84 = *(_QWORD *)(v24 + 72),
            v82 = *(_QWORD *)(v84 + 16),
            v33 = sub_7E6B40(v82, v93, v101, 1, &v109),
            v32 = 1,
            !v33) )
      {
LABEL_52:
        if ( !*(_DWORD *)(v30 + 16) )
          sub_76D960(v30, v98, v32);
        goto LABEL_46;
      }
      if ( (unsigned int)sub_8D3A70(*(_QWORD *)(v23 + 120)) && !(unsigned int)sub_7E2090(v82) )
      {
        v32 = 1;
        goto LABEL_52;
      }
      *(_DWORD *)(v30 + 16) = 2;
      *(_QWORD *)(v30 + 24) = v82;
      if ( (unsigned int)sub_731770(v84, 0, v34, v35, v36, v37) )
      {
        v32 = 1;
        *(_QWORD *)(v30 + 40) = v84;
        goto LABEL_52;
      }
LABEL_163:
      v32 = 1;
      goto LABEL_52;
    }
    if ( (unsigned int)sub_731770(v24, 0, v26, v27, v28, v29) )
      *(_QWORD *)(v30 + 40) = v24;
LABEL_46:
    v23 = *(_QWORD *)(v23 + 112);
    v24 = *(_QWORD *)(v24 + 16);
  }
  while ( v23 );
  v8 = v88;
  a2 = v87;
LABEL_81:
  if ( v92[15] )
  {
    v102 = v4;
    v38 = v92[15];
    do
    {
      if ( (*(_BYTE *)(v38 + 88) & 4) != 0 )
      {
        v39 = sub_76D880(v38, v91, v112.m128i_i64);
        sub_76D960(v39, (*(_BYTE *)(v38 + 173) & 0x20) != 0, (*(_BYTE *)(v38 + 173) & 0x10) != 0);
      }
      v38 = *(_QWORD *)(v38 + 112);
    }
    while ( v38 );
    v4 = v102;
  }
  sub_76DBF0(v92[10], (__int64)v111, &v110, &v106, &v108, &v107);
  if ( v107 )
  {
    *(_BYTE *)(v8 + 204) |= 1u;
    goto LABEL_89;
  }
  if ( !qword_4F08050 )
    goto LABEL_148;
  v103 = v8;
  v48 = v107;
  v94 = a2;
  v49 = qword_4F08050;
  while ( 2 )
  {
    if ( *(_BYTE *)(v49 + 56) )
    {
      if ( *(_DWORD *)(v49 + 16) != 1 )
      {
        v50 = *(_QWORD *)(v49 + 48);
        if ( !v50 )
          goto LABEL_131;
        *(_DWORD *)(v49 + 16) = 1;
        *(_QWORD *)(v49 + 24) = v50;
        *(_QWORD *)(v49 + 48) = 0;
      }
      if ( *(_QWORD *)(v49 + 32) )
      {
LABEL_119:
        if ( v48 )
        {
LABEL_120:
          v48 = 1;
LABEL_121:
          v51 = *(_QWORD *)(v49 + 24);
          if ( !*(_BYTE *)(v49 + 59) )
          {
            sub_7E7A90(*(_QWORD *)(v49 + 24), 0, 0);
            if ( *(_BYTE *)(v49 + 58) )
              sub_7E2130(v51);
          }
          v52 = *(_QWORD *)(v49 + 32);
          if ( v52 )
          {
            *(_QWORD *)(v52 + 16) = 0;
            v53 = (_QWORD *)sub_7E6AB0(v51, *(_QWORD *)(v49 + 32), &v112);
            if ( v53 )
            {
              *v53 = *(_QWORD *)dword_4D03F38;
              v40 = *(_QWORD *)dword_4D03F38;
              v53[1] = *(_QWORD *)dword_4D03F38;
            }
            *(_BYTE *)(v51 + 173) |= 8u;
          }
LABEL_126:
          v54 = *(_QWORD *)(v49 + 40);
          if ( !v54 )
            goto LABEL_129;
          goto LABEL_127;
        }
        goto LABEL_137;
      }
LABEL_141:
      if ( !*(_QWORD *)(v49 + 40) )
      {
        if ( *(_DWORD *)(v49 + 16) != 1 )
          goto LABEL_129;
        goto LABEL_121;
      }
LABEL_134:
      if ( v48 )
      {
        if ( *(_DWORD *)(v49 + 16) != 1 )
        {
          v54 = *(_QWORD *)(v49 + 40);
LABEL_127:
          *(_QWORD *)(v54 + 16) = 0;
          v55 = (_QWORD *)sub_7E69E0(*(_QWORD *)(v49 + 40), &v112);
          if ( v55 )
          {
            *v55 = *(_QWORD *)dword_4D03F38;
            v40 = *(_QWORD *)dword_4D03F38;
            v55[1] = *(_QWORD *)dword_4D03F38;
          }
          goto LABEL_129;
        }
        goto LABEL_120;
      }
LABEL_137:
      if ( v90 )
        sub_7E1740(v90);
      else
        sub_7E1790(&v112);
      v48 = 1;
      if ( *(_DWORD *)(v49 + 16) != 1 )
        goto LABEL_126;
      goto LABEL_120;
    }
LABEL_131:
    if ( !*(_QWORD *)(v49 + 32) )
      goto LABEL_141;
    if ( *(_DWORD *)(v49 + 16) == 1 )
      goto LABEL_119;
    if ( *(_QWORD *)(v49 + 40) )
      goto LABEL_134;
LABEL_129:
    v49 = *(_QWORD *)v49;
    if ( v49 )
      continue;
    break;
  }
  v56 = v48;
  v8 = v103;
  a2 = v94;
  if ( !v90 && (v56 & 1) != 0 )
  {
    sub_7E25D0(*((_QWORD *)&v111[0] + 1), &v112, v40, v41);
    v57 = _mm_loadu_si128(v113);
    v111[0] = _mm_loadu_si128(&v112);
    v111[1] = v57;
  }
LABEL_148:
  if ( a2 )
  {
    v58 = v90;
    do
    {
      v59 = v58;
      v58 = (const __m128i *)v58[4].m128i_i64[1];
    }
    while ( v58 && v58[2].m128i_i8[8] == 11 && !v58[1].m128i_i64[0] && !*(_QWORD *)(v59[5].m128i_i64[0] + 8) );
    v60 = a2->m128i_i16[2];
    v95 = a2[1].m128i_i64[1];
    v61 = a2->m128i_i16[6];
    v99 = a2->m128i_i32[2];
    v104 = a2->m128i_i32[0];
    sub_732B40(v59, a2);
    a2->m128i_i16[2] = v60;
    a2->m128i_i16[6] = v61;
    a2->m128i_i32[0] = v104;
    a2[1].m128i_i64[1] = v95;
    v62 = a2[5].m128i_i64[0];
    a2->m128i_i32[2] = v99;
    *(_DWORD *)v62 = v104;
    *(_WORD *)(v62 + 4) = v60;
    *a3 = 1;
  }
  else
  {
    v67 = (__m128i *)*((_QWORD *)&v111[0] + 1);
    if ( (unsigned int)sub_8D2600(*v4) )
    {
      v67 = (__m128i *)sub_73E130(v67, *v4);
    }
    else if ( *v4 != v67->m128i_i64[0] && !(unsigned int)sub_8D97D0(*v4, v67->m128i_i64[0], 1, v68, v69) )
    {
      v70 = sub_724DC0();
      v71 = (_QWORD *)*v4;
      v112.m128i_i64[0] = (__int64)v70;
      if ( (unsigned int)sub_8D3A70(v71) )
      {
        v78 = sub_72D2E0(v71);
        v79 = v112.m128i_i64[0];
        sub_72BB40(v78, (const __m128i *)v112.m128i_i64[0]);
        v80 = sub_73A720((const __m128i *)v112.m128i_i64[0], v79);
        v75 = sub_73DCD0(v80);
      }
      else
      {
        v72 = v112.m128i_i64[0];
        sub_72BB40((__int64)v71, (const __m128i *)v112.m128i_i64[0]);
        v75 = sub_73A720((const __m128i *)v112.m128i_i64[0], v72);
      }
      sub_7E25D0(v75, v111, v73, v74);
      v67 = (__m128i *)*((_QWORD *)&v111[0] + 1);
      sub_724E30((__int64)&v112);
    }
    sub_730620((__int64)v4, v67);
  }
LABEL_89:
  v42 = (_QWORD *)qword_4F08050;
  if ( qword_4F08050 )
  {
    v43 = qword_4D03E98;
    while ( 1 )
    {
      v44 = (_QWORD *)*v42;
      *v42 = v43;
      v45 = v42[1];
      qword_4D03E98 = v42;
      *(_QWORD *)(v45 + 264) = 0;
      v43 = v42;
      if ( !v44 )
        break;
      v42 = v44;
    }
  }
  qword_4F08050 = 0;
  LOBYTE(v7) = (2 * (v108 & 1)) | *(_BYTE *)(v8 + 203) & 0xFD;
  *(_BYTE *)(v8 + 203) = v7;
  if ( v107 && *(_QWORD *)v8 )
  {
    if ( (v7 & 2) != 0 )
      LOBYTE(v7) = sub_685480(0x2A6u, *(_QWORD *)v8);
    else
      LOBYTE(v7) = sub_685460(v106 == 0 ? 679 : 2321, (FILE *)(*(_QWORD *)v8 + 48LL), *(_QWORD *)v8);
  }
  qword_4F08048 = 0;
  return v7;
}
