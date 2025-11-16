// Function: sub_27E95E0
// Address: 0x27e95e0
//
__int64 __fastcall sub_27E95E0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r15
  __int64 v3; // r14
  __int64 v4; // rdi
  __int64 v5; // rdx
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  unsigned int v8; // r13d
  __int64 v10; // rax
  _QWORD *v11; // rbx
  unsigned __int64 v12; // rax
  char v13; // dl
  unsigned int v14; // r12d
  unsigned __int8 v15; // al
  _BYTE *v16; // rax
  __int64 v17; // rax
  unsigned int v18; // eax
  _BYTE *v19; // rax
  unsigned __int8 v20; // dl
  _BYTE *v21; // rsi
  _BYTE *v22; // r9
  char v23; // al
  _QWORD *v24; // r10
  _BYTE *v25; // r8
  const __m128i *v26; // r13
  const __m128i *v27; // rax
  __int64 v28; // rax
  __int64 v29; // r13
  __m128i *v30; // rax
  __m128i *v31; // rbx
  __int64 v32; // rsi
  __m128i *v33; // r13
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  _QWORD *v37; // rbx
  _BYTE *v38; // r8
  const __m128i *v39; // r12
  __int64 v40; // rax
  __int64 v41; // rax
  unsigned int v42; // eax
  unsigned int v43; // r13d
  __int64 v44; // rax
  unsigned __int64 v45; // rax
  int v46; // edx
  _QWORD *v47; // r12
  _QWORD *v48; // rax
  unsigned int v49; // eax
  unsigned int i; // r14d
  const __m128i *v51; // rsi
  __int64 v52; // rbx
  __int64 v53; // rdi
  __int64 v54; // rsi
  unsigned __int8 *v55; // rsi
  __int64 v56; // rax
  _QWORD *v57; // rdi
  _BYTE *v58; // rcx
  unsigned __int8 *v59; // rsi
  const __m128i *v60; // rax
  __int64 *v61; // rdi
  unsigned __int64 v62; // rax
  _QWORD *v63; // rax
  _BYTE *v64; // rbx
  unsigned int v65; // r13d
  bool v66; // al
  unsigned __int64 v67; // rsi
  unsigned int v68; // ebx
  __int64 v69; // r9
  __int64 v70; // r13
  __int64 v71; // r12
  __int64 v72; // rax
  __int64 v73; // r8
  __int64 v74; // rdx
  __int64 v75; // rcx
  const __m128i *v76; // rax
  __int64 v77; // rax
  __int64 v78; // r15
  const __m128i *v79; // rax
  const __m128i *v80; // rdx
  unsigned __int64 v81; // rax
  unsigned __int64 v82; // r15
  char v83; // al
  _BYTE v84[12]; // [rsp+14h] [rbp-16Ch]
  _BYTE *v85; // [rsp+20h] [rbp-160h]
  __int64 v86; // [rsp+28h] [rbp-158h]
  _QWORD *v87; // [rsp+28h] [rbp-158h]
  _QWORD *v88; // [rsp+28h] [rbp-158h]
  unsigned int v89; // [rsp+28h] [rbp-158h]
  _BYTE *v90; // [rsp+30h] [rbp-150h]
  _BYTE *v91; // [rsp+30h] [rbp-150h]
  _BYTE *v92; // [rsp+30h] [rbp-150h]
  _QWORD *v93; // [rsp+30h] [rbp-150h]
  unsigned __int8 *v94; // [rsp+38h] [rbp-148h]
  __int64 *v95; // [rsp+40h] [rbp-140h]
  unsigned __int8 *v96; // [rsp+40h] [rbp-140h]
  unsigned __int8 v97; // [rsp+40h] [rbp-140h]
  __int64 v98; // [rsp+40h] [rbp-140h]
  int v99; // [rsp+40h] [rbp-140h]
  unsigned __int8 *src; // [rsp+48h] [rbp-138h]
  _BYTE *srca; // [rsp+48h] [rbp-138h]
  unsigned __int64 v102; // [rsp+58h] [rbp-128h] BYREF
  unsigned __int64 v103; // [rsp+60h] [rbp-120h] BYREF
  __int64 v104; // [rsp+68h] [rbp-118h] BYREF
  __m128i v105; // [rsp+70h] [rbp-110h] BYREF
  _QWORD v106[2]; // [rsp+80h] [rbp-100h] BYREF
  __int64 v107; // [rsp+90h] [rbp-F0h]
  __int64 v108; // [rsp+98h] [rbp-E8h]
  __int64 v109; // [rsp+A0h] [rbp-E0h]
  __m128i *v110; // [rsp+B0h] [rbp-D0h] BYREF
  __m128i *v111; // [rsp+B8h] [rbp-C8h]
  __int64 v112; // [rsp+C0h] [rbp-C0h] BYREF
  int v113; // [rsp+C8h] [rbp-B8h]
  char v114; // [rsp+CCh] [rbp-B4h]
  _QWORD v115[22]; // [rsp+D0h] [rbp-B0h] BYREF

  v2 = (_QWORD *)a2;
  v3 = a1;
  v4 = *(_QWORD *)(a1 + 48);
  if ( *(_BYTE *)(v4 + 560) )
  {
    v5 = *(unsigned int *)(v4 + 588);
    if ( (_DWORD)v5 != *(_DWORD *)(v4 + 592) )
    {
      if ( *(_BYTE *)(v4 + 596) )
      {
        v6 = *(_QWORD **)(v4 + 576);
        v7 = &v6[v5];
        if ( v6 != v7 )
        {
          while ( a2 != *v6 )
          {
            if ( v7 == ++v6 )
              goto LABEL_11;
          }
          return 0;
        }
      }
      else if ( sub_C8CA60(v4 + 568, a2) )
      {
        return 0;
      }
    }
  }
LABEL_11:
  v10 = *(_QWORD *)(a2 + 16);
  if ( v10 )
  {
    while ( (unsigned __int8)(**(_BYTE **)(v10 + 24) - 30) > 0xAu )
    {
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        goto LABEL_55;
    }
  }
  else
  {
LABEL_55:
    v28 = *(_QWORD *)(*(_QWORD *)(a2 + 72) + 80LL);
    if ( !v28 || a2 != v28 - 24 )
      return 0;
  }
  v8 = sub_27DCB90(v3, a2);
  if ( (_BYTE)v8 )
    return 1;
  if ( !(_BYTE)qword_4FFDD08 )
    goto LABEL_15;
  v37 = *(_QWORD **)(*(_QWORD *)(a2 + 72) + 40LL);
  v105.m128i_i64[0] = (__int64)v106;
  v38 = (_BYTE *)v37[29];
  v39 = (const __m128i *)v37[30];
  if ( &v38[(_QWORD)v39] && !v38 )
LABEL_72:
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v110 = (__m128i *)v37[30];
  if ( (unsigned __int64)v39 > 0xF )
  {
    srca = v38;
    v56 = sub_22409D0((__int64)&v105, (unsigned __int64 *)&v110, 0);
    v38 = srca;
    v105.m128i_i64[0] = v56;
    v57 = (_QWORD *)v56;
    v106[0] = v110;
    goto LABEL_118;
  }
  if ( v39 != (const __m128i *)1 )
  {
    if ( !v39 )
    {
      v40 = (__int64)v106;
      goto LABEL_76;
    }
    v57 = v106;
LABEL_118:
    a2 = (__int64)v38;
    memcpy(v57, v38, (size_t)v39);
    v39 = v110;
    v40 = v105.m128i_i64[0];
    goto LABEL_76;
  }
  LOBYTE(v106[0]) = *v38;
  v40 = (__int64)v106;
LABEL_76:
  v105.m128i_i64[1] = (__int64)v39;
  v39->m128i_i8[v40] = 0;
  v107 = v37[33];
  v108 = v37[34];
  v109 = v37[35];
  if ( (unsigned int)(v107 - 42) <= 1 )
  {
    if ( (_QWORD *)v105.m128i_i64[0] != v106 )
    {
      a2 = v106[0] + 1LL;
      j_j___libc_free_0(v105.m128i_u64[0]);
    }
    goto LABEL_16;
  }
  if ( (_QWORD *)v105.m128i_i64[0] != v106 )
    j_j___libc_free_0(v105.m128i_u64[0]);
LABEL_15:
  a2 = (__int64)v2;
  if ( (unsigned __int8)sub_27E0610(v3, (__int64)v2) )
    return 1;
LABEL_16:
  if ( *(_BYTE *)(v3 + 89) )
  {
    a2 = (__int64)v2;
    if ( (unsigned __int8)sub_27E2460(v3, v2) )
      return 1;
  }
  v11 = v2 + 6;
  v12 = v2[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 + 6 == (_QWORD *)v12 )
    goto LABEL_206;
  if ( !v12 )
    goto LABEL_206;
  v94 = (unsigned __int8 *)(v12 - 24);
  if ( (unsigned int)*(unsigned __int8 *)(v12 - 24) - 30 > 0xA )
    goto LABEL_206;
  v13 = *(_BYTE *)(v12 - 24);
  if ( v13 != 31 )
  {
    if ( v13 == 32 )
    {
      v14 = 0;
      src = **(unsigned __int8 ***)(v12 - 32);
      goto LABEL_23;
    }
    if ( v13 == 33 && (*(_DWORD *)(v12 - 20) & 0x7FFFFFF) != 1 )
    {
      v14 = 1;
      src = sub_BD3990(**(unsigned __int8 ***)(v12 - 32), a2);
      goto LABEL_23;
    }
    return 0;
  }
  if ( (*(_DWORD *)(v12 - 20) & 0x7FFFFFF) == 1 )
    return 0;
  v14 = 0;
  src = *(unsigned __int8 **)(v12 - 120);
LABEL_23:
  v15 = *src;
  if ( *src <= 0x1Cu )
    goto LABEL_85;
  v95 = *(__int64 **)(v3 + 16);
  v16 = (_BYTE *)sub_AA4E30((__int64)v2);
  v17 = sub_97D880((__int64)src, v16, v95);
  if ( v17 )
  {
    v96 = (unsigned __int8 *)v17;
    sub_BD84D0((__int64)src, v17);
    LOBYTE(v18) = sub_F50EE0(src, *(__int64 **)(v3 + 16));
    v8 = v18;
    if ( (_BYTE)v18 )
    {
      sub_B43D60(src);
      src = v96;
    }
    else
    {
      src = v96;
      v8 = 1;
    }
  }
  v15 = *src;
  if ( *src <= 0x1Cu )
  {
LABEL_85:
    if ( (unsigned __int8)(v15 - 12) <= 1u )
    {
      src = 0;
LABEL_87:
      v42 = sub_27DC010((__int64)v2);
      v110 = 0;
      v111 = 0;
      v43 = v42;
      v44 = v2[6];
      v112 = 0;
      v45 = v44 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v11 == (_QWORD *)v45 )
      {
        v47 = 0;
        goto LABEL_91;
      }
      if ( v45 )
      {
        v46 = *(unsigned __int8 *)(v45 - 24);
        v47 = 0;
        v48 = (_QWORD *)(v45 - 24);
        if ( (unsigned int)(v46 - 30) < 0xB )
          v47 = v48;
LABEL_91:
        v49 = sub_B46E30((__int64)v47);
        sub_F58D10((const __m128i **)&v110, v49);
        v99 = sub_B46E30((__int64)v47);
        if ( v99 )
        {
          v86 = v3;
          for ( i = 0; i != v99; ++i )
          {
            if ( v43 != i )
            {
              v52 = sub_B46EC0((__int64)v47, i);
              sub_AA5980(v52, (__int64)v2, 1u);
              v105.m128i_i64[0] = (__int64)v2;
              v51 = v111;
              v105.m128i_i64[1] = v52 | 4;
              if ( v111 == (__m128i *)v112 )
              {
                sub_F38BA0((const __m128i **)&v110, v111, &v105);
              }
              else
              {
                if ( v111 )
                {
                  *v111 = _mm_loadu_si128(&v105);
                  v51 = v111;
                }
                v111 = (__m128i *)&v51[1];
              }
            }
          }
          v3 = v86;
        }
        v29 = sub_B46EC0((__int64)v47, v43);
        v30 = (__m128i *)sub_BD2C40(72, 1u);
        v31 = v30;
        if ( v30 )
          sub_B4C8F0((__int64)v30, v29, 1u, (__int64)(v47 + 3), 0);
        v32 = v47[6];
        v33 = v31 + 3;
        v105.m128i_i64[0] = v32;
        if ( v32 )
        {
          sub_B96E90((__int64)&v105, v32, 1);
          if ( v33 == &v105 )
          {
            if ( v105.m128i_i64[0] )
              sub_B91220((__int64)&v105, v105.m128i_i64[0]);
            goto LABEL_65;
          }
          v54 = v31[3].m128i_i64[0];
          if ( !v54 )
          {
LABEL_113:
            v55 = (unsigned __int8 *)v105.m128i_i64[0];
            v31[3].m128i_i64[0] = v105.m128i_i64[0];
            if ( v55 )
              sub_B976B0((__int64)&v105, v55, (__int64)v31[3].m128i_i64);
            goto LABEL_65;
          }
        }
        else if ( v33 == &v105 || (v54 = v31[3].m128i_i64[0]) == 0 )
        {
LABEL_65:
          sub_B43D60(v47);
          sub_FFDB80(*(_QWORD *)(v3 + 48), (unsigned __int64 *)v110, v111 - v110, v34, v35, v36);
          if ( src )
            sub_B43D60(src);
          if ( v110 )
            j_j___libc_free_0((unsigned __int64)v110);
          return 1;
        }
        sub_B91220((__int64)v31[3].m128i_i64, v54);
        goto LABEL_113;
      }
LABEL_206:
      BUG();
    }
  }
  else if ( v15 == 96 && (unsigned int)**((unsigned __int8 **)src - 4) - 12 <= 1 )
  {
    v41 = *((_QWORD *)src + 2);
    if ( v41 )
    {
      if ( !*(_QWORD *)(v41 + 8) )
        goto LABEL_87;
    }
  }
  if ( sub_27DB860(src, v14) )
  {
    sub_F5CD10((__int64)v2, 1, 0, *(_QWORD *)(v3 + 48));
    v53 = sub_27DD130((__int64 *)v3);
    if ( v53 )
      sub_FF0C10(v53, (__int64)v2);
    return 1;
  }
  if ( *src <= 0x1Cu )
  {
    if ( !(unsigned __int8)sub_27E9540(v3, (__int64)src, v2, v14, v94) )
      return v8;
    return 1;
  }
  v97 = *src;
  v19 = (_BYTE *)sub_986580((__int64)v2);
  v20 = v97;
  v21 = v19;
  if ( v97 == 96 )
  {
    v20 = **((_BYTE **)src - 4);
    v98 = *((_QWORD *)src - 4);
    if ( v20 <= 0x1Cu )
    {
      if ( *v19 != 32 )
      {
        v22 = (_BYTE *)*((_QWORD *)src - 4);
        goto LABEL_39;
      }
      goto LABEL_34;
    }
  }
  else
  {
    v98 = (__int64)src;
  }
  if ( (unsigned __int8)(v20 - 82) <= 1u )
  {
    v58 = *(_BYTE **)(v98 - 32);
    if ( *v58 <= 0x15u )
    {
      v59 = (unsigned __int8 *)sub_22CF7C0(
                                 *(__int64 **)(v3 + 32),
                                 *(_WORD *)(v98 + 2) & 0x3F,
                                 *(_QWORD *)(v98 - 64),
                                 (__int64)v58,
                                 (__int64)v19,
                                 0);
      if ( v59 && (unsigned __int8)sub_27DB9F0(v98, v59, (__int64)v2) || sub_27DDE90(v3, v98, (__int64)v2) )
        return 1;
      v21 = (_BYTE *)sub_986580((__int64)v2);
      if ( *v21 != 32 )
      {
LABEL_35:
        v22 = (_BYTE *)v98;
        v23 = *(_BYTE *)v98;
        if ( *(_BYTE *)v98 <= 0x1Cu )
          goto LABEL_39;
        goto LABEL_36;
      }
      goto LABEL_34;
    }
  }
  if ( *v19 == 32 )
  {
LABEL_34:
    if ( !(unsigned __int8)sub_27DDD70((__int64 *)v3, (__int64)v21, (__int64)v2) )
      goto LABEL_35;
    return 1;
  }
  v23 = *(_BYTE *)v98;
LABEL_36:
  v22 = (_BYTE *)v98;
  if ( (unsigned __int8)(v23 - 82) <= 1u && **(_BYTE **)(v98 - 32) <= 0x15u )
    v22 = *(_BYTE **)(v98 - 64);
LABEL_39:
  v24 = *(_QWORD **)(v2[9] + 40LL);
  v110 = (__m128i *)&v112;
  v25 = (_BYTE *)v24[29];
  v26 = (const __m128i *)v24[30];
  if ( &v25[(_QWORD)v26] && !v25 )
    goto LABEL_72;
  v105.m128i_i64[0] = v24[30];
  if ( (unsigned __int64)v26 > 0xF )
  {
    v85 = v25;
    v87 = v24;
    v91 = v22;
    v60 = (const __m128i *)sub_22409D0((__int64)&v110, (unsigned __int64 *)&v105, 0);
    v22 = v91;
    v24 = v87;
    v110 = (__m128i *)v60;
    v61 = (__int64 *)v60;
    v25 = v85;
    v112 = v105.m128i_i64[0];
    goto LABEL_142;
  }
  if ( v26 != (const __m128i *)1 )
  {
    if ( !v26 )
    {
      v27 = (const __m128i *)&v112;
      goto LABEL_44;
    }
    v61 = &v112;
LABEL_142:
    v88 = v24;
    v92 = v22;
    memcpy(v61, v25, (size_t)v26);
    v26 = (const __m128i *)v105.m128i_i64[0];
    v27 = v110;
    v22 = v92;
    v24 = v88;
    goto LABEL_44;
  }
  LOBYTE(v112) = *v25;
  v27 = (const __m128i *)&v112;
LABEL_44:
  v111 = (__m128i *)v26;
  v26->m128i_i8[(_QWORD)v27] = 0;
  v115[0] = v24[33];
  v115[1] = v24[34];
  v115[2] = v24[35];
  if ( (unsigned int)(LODWORD(v115[0]) - 42) > 1 )
  {
    if ( v110 != (__m128i *)&v112 )
    {
      v90 = v22;
      j_j___libc_free_0((unsigned __int64)v110);
      v22 = v90;
    }
    if ( *v22 == 61 && (unsigned __int8)sub_27E8080((__int64 *)v3, (__int64)v22) )
      return 1;
  }
  else if ( v110 != (__m128i *)&v112 )
  {
    j_j___libc_free_0((unsigned __int64)v110);
  }
  if ( *src != 84 || v2 != *((_QWORD **)src + 5) || *(_BYTE *)sub_986580((__int64)v2) != 31 )
    goto LABEL_49;
  v62 = v2[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v11 == (_QWORD *)v62 || !v62 || (unsigned int)*(unsigned __int8 *)(v62 - 24) - 30 > 0xA )
    goto LABEL_206;
  if ( *(_BYTE *)(v62 - 24) != 31 || !(unsigned __int8)sub_BC8C50(v62 - 24, &v102, &v103) || !(v102 + v103) )
    goto LABEL_49;
  v89 = v14;
  *(_DWORD *)&v84[8] = 0;
  v93 = v2;
  *(_QWORD *)v84 = *((_DWORD *)src + 1) & 0x7FFFFFF;
  while ( 1 )
  {
    if ( *(_DWORD *)v84 == *(_DWORD *)&v84[4] )
    {
LABEL_200:
      v14 = v89;
      v2 = v93;
      goto LABEL_49;
    }
    v63 = (_QWORD *)(*((_QWORD *)src - 1) + 32LL * *(_QWORD *)&v84[4]);
    v64 = (_BYTE *)*v63;
    if ( *(_BYTE *)*v63 == 17 && sub_BCAC40(*((_QWORD *)v64 + 1), 1) )
      break;
LABEL_199:
    ++*(_QWORD *)&v84[4];
  }
  v65 = *((_DWORD *)v64 + 8);
  if ( v65 <= 0x40 )
    v66 = *((_QWORD *)v64 + 3) == 1;
  else
    v66 = v65 - 1 == (unsigned int)sub_C444A0((__int64)(v64 + 24));
  v67 = v103 + v102;
  if ( v66 )
    v68 = sub_F02DD0(v102, v67);
  else
    v68 = sub_F02DD0(v103, v67);
  v70 = (__int64)v93;
  v71 = *(_QWORD *)(*((_QWORD *)src - 1) + 32LL * *((unsigned int *)src + 18) + 8LL * *(_QWORD *)&v84[4]);
  v110 = 0;
  v112 = 16;
  v111 = (__m128i *)v115;
  v113 = 0;
  v114 = 1;
  while ( 1 )
  {
    v72 = *(_QWORD *)(v71 + 48);
    v73 = v71 + 48;
    v74 = v72 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v72 & 0xFFFFFFFFFFFFFFF8LL) == v71 + 48 )
      goto LABEL_206;
    if ( !v74 )
      goto LABEL_206;
    v75 = (unsigned int)*(unsigned __int8 *)(v74 - 24) - 30;
    if ( (unsigned int)v75 > 0xA )
      goto LABEL_206;
    if ( *(_BYTE *)(v74 - 24) == 31 )
    {
      v74 = *(_DWORD *)(v74 - 20) & 0x7FFFFFF;
      if ( (_DWORD)v74 == 3 )
      {
        if ( !v114 )
        {
          _libc_free((unsigned __int64)v111);
          v72 = *(_QWORD *)(v71 + 48);
          v73 = v71 + 48;
        }
        v81 = v72 & 0xFFFFFFFFFFFFFFF8LL;
        v82 = v81;
        if ( v81 == v73 || !v81 || (unsigned int)*(unsigned __int8 *)(v81 - 24) - 30 > 0xA )
          goto LABEL_206;
        if ( *(_BYTE *)(v81 - 24) != 31 )
          goto LABEL_200;
        if ( !(unsigned __int8)sub_BC8C50(v81 - 24, &v104, &v105) )
        {
          sub_F02DB0(&v110, 0x32u, 0x64u);
          if ( (unsigned int)v110 > v68 )
          {
            if ( *(_QWORD *)(v82 - 56) == v70 )
            {
              LODWORD(v110) = v68;
              HIDWORD(v110) = 0x80000000 - v68;
            }
            else
            {
              LODWORD(v110) = 0x80000000 - v68;
              HIDWORD(v110) = v68;
            }
            v83 = sub_BC87E0(v82 - 24);
            sub_BC8EC0(v82 - 24, (unsigned int *)&v110, 2, v83);
          }
        }
        goto LABEL_199;
      }
    }
    if ( !v114 )
    {
LABEL_185:
      sub_C8CC70((__int64)&v110, v71, v74, v75, v73, v69);
      goto LABEL_170;
    }
    v76 = v111;
    v75 = HIDWORD(v112);
    v74 = (__int64)&v111->m128i_i64[HIDWORD(v112)];
    if ( v111 == (__m128i *)v74 )
    {
LABEL_186:
      if ( HIDWORD(v112) >= (unsigned int)v112 )
        goto LABEL_185;
      ++HIDWORD(v112);
      *(_QWORD *)v74 = v71;
      v110 = (__m128i *)((char *)v110 + 1);
    }
    else
    {
      while ( v71 != v76->m128i_i64[0] )
      {
        v76 = (const __m128i *)((char *)v76 + 8);
        if ( (const __m128i *)v74 == v76 )
          goto LABEL_186;
      }
    }
LABEL_170:
    v77 = sub_AA54C0(v71);
    v78 = v77;
    if ( !v77 )
      goto LABEL_176;
    if ( v114 )
      break;
    if ( sub_C8CA60((__int64)&v110, v77) )
      goto LABEL_176;
LABEL_184:
    v70 = v71;
    v71 = v78;
  }
  v79 = v111;
  v80 = (__m128i *)((char *)v111 + 8 * HIDWORD(v112));
  if ( v111 == v80 )
    goto LABEL_184;
  while ( v78 != v79->m128i_i64[0] )
  {
    v79 = (const __m128i *)((char *)v79 + 8);
    if ( v80 == v79 )
      goto LABEL_184;
  }
LABEL_176:
  v14 = v89;
  v2 = v93;
  if ( !v114 )
    _libc_free((unsigned __int64)v111);
LABEL_49:
  if ( (unsigned __int8)sub_27E9540(v3, (__int64)src, v2, v14, v94) )
    return 1;
  if ( *(_BYTE *)v98 == 84 && v2 == *(_QWORD **)(v98 + 40) && *(_BYTE *)sub_986580((__int64)v2) == 31 )
    return sub_27E5040(v3, v98);
  if ( *src == 59 && v2 == *((_QWORD **)src + 5) && *(_BYTE *)sub_986580((__int64)v2) == 31 )
    return sub_27E5160(v3, (__int64)src);
  return sub_27DD230((__int64 *)v3, (__int64)v2);
}
