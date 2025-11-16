// Function: sub_170D400
// Address: 0x170d400
//
__int64 __fastcall sub_170D400(const __m128i *a1, __int64 a2, __int64 a3, __m128i a4, double a5, __m128i a6)
{
  __int64 v6; // r14
  int v7; // ecx
  __int64 v8; // rcx
  unsigned __int8 *v9; // r15
  unsigned __int8 *v10; // r13
  __int64 v11; // rbx
  int v12; // eax
  __int64 v13; // r12
  int v14; // edx
  unsigned int v15; // r13d
  __int64 v17; // r15
  int v18; // eax
  __int64 v19; // rax
  __int64 *v20; // rsi
  __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r13
  char v27; // al
  int v28; // ebx
  int v29; // eax
  __int64 v30; // rsi
  __int64 v31; // rbx
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  __int64 v34; // rsi
  __int64 v35; // rdx
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rcx
  unsigned __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // r12
  int v43; // eax
  unsigned int v44; // r12d
  int v45; // eax
  unsigned int v46; // eax
  __int64 v47; // rcx
  __int64 v48; // r11
  char v49; // al
  __int64 v50; // rdx
  __int64 v51; // r9
  __int64 v52; // rsi
  __int64 v53; // rax
  __int64 v54; // r12
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // rdi
  __int64 v60; // r15
  unsigned __int8 *v61; // rsi
  __m128i v62; // xmm6
  _QWORD *v63; // rax
  unsigned __int8 *v64; // r8
  unsigned __int8 *v65; // rdx
  __int64 v66; // r13
  _QWORD *v67; // rax
  unsigned __int8 *v68; // rdx
  __int64 v69; // r13
  _QWORD *v70; // rax
  __int64 v71; // rcx
  unsigned __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // rdx
  unsigned __int64 v75; // rax
  __int64 v76; // rax
  unsigned __int8 *v77; // rdx
  __m128i v78; // xmm6
  __m128i v79; // xmm4
  __int64 v80; // r13
  _QWORD *v81; // rax
  __int64 v82; // rcx
  unsigned __int64 v83; // rdx
  __int64 v84; // rdx
  __int64 v85; // rcx
  unsigned __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // rcx
  unsigned __int64 v89; // rdx
  __int64 v90; // rdx
  __int64 v91; // rcx
  unsigned __int64 v92; // rdx
  __int64 v93; // rcx
  __int64 v94; // rcx
  unsigned __int64 v95; // r12
  __int64 v96; // rax
  unsigned __int8 v97; // al
  __int64 v98; // r15
  unsigned __int8 v99; // al
  __int64 v100; // rdx
  __int64 v101; // rcx
  unsigned __int64 v102; // rdx
  __int64 v103; // rcx
  __int64 v104; // rdx
  unsigned __int64 v105; // rax
  __int64 v106; // rax
  int v107; // eax
  __int64 v108; // rax
  __int64 v109; // rax
  __int64 v110; // [rsp+0h] [rbp-A0h]
  __int64 v111; // [rsp+0h] [rbp-A0h]
  unsigned __int8 v113; // [rsp+20h] [rbp-80h]
  int v114; // [rsp+20h] [rbp-80h]
  __int64 *v115; // [rsp+20h] [rbp-80h]
  unsigned int v116; // [rsp+28h] [rbp-78h]
  char v117; // [rsp+2Fh] [rbp-71h]
  char v118; // [rsp+3Fh] [rbp-61h] BYREF
  __m128i v119; // [rsp+40h] [rbp-60h] BYREF
  __m128 v120; // [rsp+50h] [rbp-50h]
  __int64 v121; // [rsp+60h] [rbp-40h]

  v6 = a2;
  v113 = 0;
  v116 = *(unsigned __int8 *)(a2 + 16) - 24;
  v7 = *(unsigned __int8 *)(a2 + 16);
  v117 = *(_BYTE *)(a2 + 16);
  while ( 1 )
  {
    v8 = (unsigned int)(v7 - 24);
    if ( (unsigned int)v8 > 0x1C || ((1LL << v8) & 0x1C019800) == 0 )
      goto LABEL_4;
    v42 = *(_QWORD *)(v6 - 48);
    v43 = *(unsigned __int8 *)(v42 + 16);
    if ( (unsigned __int8)v43 <= 0x17u )
    {
      if ( (_BYTE)v43 == 17 )
      {
        v44 = 3;
      }
      else
      {
        v44 = 2;
        if ( (unsigned __int8)v43 <= 0x10u )
          v44 = (_BYTE)v43 != 9;
      }
    }
    else if ( (unsigned int)(v43 - 60) <= 0xC
           || sub_15FB6B0(*(_QWORD *)(v6 - 48), a2, a3, v8)
           || (a2 = 0, sub_15FB6D0(v42, 0, a3, v8))
           || (v59 = v42, v44 = 5, sub_15FB730(v59, 0, a3, v8)) )
    {
      v44 = 4;
    }
    v10 = *(unsigned __int8 **)(v6 - 24);
    v45 = v10[16];
    if ( (unsigned __int8)v45 > 0x17u )
    {
      if ( (unsigned int)(v45 - 60) <= 0xC
        || sub_15FB6B0(*(_QWORD *)(v6 - 24), a2, a3, v8)
        || sub_15FB6D0((__int64)v10, 0, v55, v56)
        || sub_15FB730((__int64)v10, 0, v57, v58) )
      {
        v46 = 4;
      }
      else
      {
        v46 = 5;
      }
LABEL_63:
      if ( v44 < v46 )
        v113 = sub_15FB800(v6) ^ 1;
LABEL_4:
      v9 = *(unsigned __int8 **)(v6 - 48);
      v10 = *(unsigned __int8 **)(v6 - 24);
      v11 = 0;
      v12 = v9[16];
      if ( (unsigned __int8)v12 > 0x17u && (unsigned int)(v12 - 35) < 0x12 )
        v11 = *(_QWORD *)(v6 - 48);
      v13 = 0;
      if ( (unsigned __int8)(v10[16] - 35) <= 0x11u )
        v13 = *(_QWORD *)(v6 - 24);
      goto LABEL_9;
    }
    if ( (_BYTE)v45 == 17 )
    {
      v46 = 3;
      goto LABEL_63;
    }
    if ( (unsigned __int8)v45 > 0x10u )
    {
      v46 = 2;
      goto LABEL_63;
    }
    if ( (_BYTE)v45 != 9 )
    {
      v46 = 1;
      goto LABEL_63;
    }
    v11 = *(_QWORD *)(v6 - 48);
    v107 = *(unsigned __int8 *)(v11 + 16);
    v9 = (unsigned __int8 *)v11;
    if ( (unsigned __int8)v107 <= 0x17u )
    {
      if ( !(unsigned __int8)sub_15F34B0(v6) )
        return v113;
      v11 = 0;
      v13 = 0;
      goto LABEL_14;
    }
    if ( (unsigned int)(v107 - 35) <= 0x11 )
    {
      if ( !(unsigned __int8)sub_15F34B0(v6) )
        return v113;
      v13 = 0;
      goto LABEL_11;
    }
    v11 = 0;
    v13 = 0;
LABEL_9:
    if ( !(unsigned __int8)sub_15F34B0(v6) )
      return v113;
    if ( !v11 )
    {
LABEL_12:
      if ( !v13 || v117 != *(_BYTE *)(v13 + 16) )
        goto LABEL_14;
LABEL_95:
      a2 = (__int64)v9;
      v65 = *(unsigned __int8 **)(v13 - 48);
      v66 = *(_QWORD *)(v13 - 24);
      v121 = v6;
      a6 = _mm_loadu_si128(a1 + 168);
      a4 = _mm_loadu_si128(a1 + 167);
      v119 = a4;
      v120 = (__m128)a6;
      v67 = sub_13E1140(v116, v9, v65, &v119);
      if ( !v67 )
        goto LABEL_96;
      if ( *(_QWORD *)(v6 - 48) )
      {
        v101 = *(_QWORD *)(v6 - 40);
        v102 = *(_QWORD *)(v6 - 32) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v102 = v101;
        if ( v101 )
        {
          a2 = *(_QWORD *)(v101 + 16) & 3LL;
          *(_QWORD *)(v101 + 16) = a2 | v102;
        }
      }
      *(_QWORD *)(v6 - 48) = v67;
      v103 = v67[1];
      *(_QWORD *)(v6 - 40) = v103;
      if ( v103 )
      {
        a2 = (v6 - 40) | *(_QWORD *)(v103 + 16) & 3LL;
        *(_QWORD *)(v103 + 16) = a2;
      }
      *(_QWORD *)(v6 - 32) = *(_QWORD *)(v6 - 32) & 3LL | (unsigned __int64)(v67 + 1);
      v67[1] = v6 - 48;
      if ( *(_QWORD *)(v6 - 24) )
      {
        v104 = *(_QWORD *)(v6 - 16);
        v105 = *(_QWORD *)(v6 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v105 = v104;
        if ( v104 )
          *(_QWORD *)(v104 + 16) = *(_QWORD *)(v104 + 16) & 3LL | v105;
      }
      *(_QWORD *)(v6 - 24) = v66;
      if ( v66 )
      {
        v106 = *(_QWORD *)(v66 + 8);
        *(_QWORD *)(v6 - 16) = v106;
        if ( v106 )
        {
          a2 = v6 - 16;
          *(_QWORD *)(v106 + 16) = (v6 - 16) | *(_QWORD *)(v106 + 16) & 3LL;
        }
        *(_QWORD *)(v6 - 8) = (v66 + 8) | *(_QWORD *)(v6 - 8) & 3LL;
        *(_QWORD *)(v66 + 8) = v6 - 24;
      }
LABEL_55:
      sub_1704560((_BYTE *)v6);
      goto LABEL_56;
    }
LABEL_11:
    if ( v117 != *(_BYTE *)(v11 + 16) )
      goto LABEL_12;
    v60 = *(_QWORD *)(v11 - 48);
    v61 = *(unsigned __int8 **)(v11 - 24);
    v121 = v6;
    v62 = _mm_loadu_si128(a1 + 168);
    v119 = _mm_loadu_si128(a1 + 167);
    v120 = (__m128)v62;
    v63 = sub_13E1140(v116, v61, v10, &v119);
    v64 = v61;
    if ( !v63 )
      break;
    if ( *(_QWORD *)(v6 - 48) )
    {
      v88 = *(_QWORD *)(v6 - 40);
      v89 = *(_QWORD *)(v6 - 32) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v89 = v88;
      if ( v88 )
        *(_QWORD *)(v88 + 16) = *(_QWORD *)(v88 + 16) & 3LL | v89;
    }
    *(_QWORD *)(v6 - 48) = v60;
    if ( v60 )
    {
      v90 = *(_QWORD *)(v60 + 8);
      *(_QWORD *)(v6 - 40) = v90;
      if ( v90 )
        *(_QWORD *)(v90 + 16) = (v6 - 40) | *(_QWORD *)(v90 + 16) & 3LL;
      *(_QWORD *)(v6 - 32) = (v60 + 8) | *(_QWORD *)(v6 - 32) & 3LL;
      *(_QWORD *)(v60 + 8) = v6 - 48;
    }
    a2 = v6 - 24;
    if ( *(_QWORD *)(v6 - 24) )
    {
      v91 = *(_QWORD *)(v6 - 16);
      v92 = *(_QWORD *)(v6 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v92 = v91;
      if ( v91 )
        *(_QWORD *)(v91 + 16) = *(_QWORD *)(v91 + 16) & 3LL | v92;
    }
    *(_QWORD *)(v6 - 24) = v63;
    v93 = v63[1];
    *(_QWORD *)(v6 - 16) = v93;
    if ( v93 )
      *(_QWORD *)(v93 + 16) = (v6 - 16) | *(_QWORD *)(v93 + 16) & 3LL;
    v94 = *(_QWORD *)(v6 - 8) & 3LL;
    *(_QWORD *)(v6 - 8) = v94 | (unsigned __int64)(v63 + 1);
    v63[1] = a2;
    v95 = *(unsigned __int8 *)(v6 + 16);
    if ( (unsigned __int8)v95 > 0x2Fu )
      goto LABEL_55;
    v96 = 0x80A800000000LL;
    if ( !_bittest64(&v96, v95) || (*(_BYTE *)(v6 + 17) & 4) == 0 || (((_BYTE)v95 - 35) & 0xFD) != 0 )
      goto LABEL_55;
    v97 = v64[16];
    v98 = (__int64)(v64 + 24);
    if ( v97 != 13 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v64 + 8LL) != 16 )
        goto LABEL_55;
      if ( v97 > 0x10u )
        goto LABEL_55;
      v108 = sub_15A1020(v64, a2, *(_QWORD *)v64, v94);
      if ( !v108 || *(_BYTE *)(v108 + 16) != 13 )
        goto LABEL_55;
      v98 = v108 + 24;
    }
    v99 = v10[16];
    v100 = (__int64)(v10 + 24);
    if ( v99 != 13 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) != 16 )
        goto LABEL_55;
      if ( v99 > 0x10u )
        goto LABEL_55;
      v109 = sub_15A1020(v10, a2, *(_QWORD *)v10, v94);
      if ( !v109 || *(_BYTE *)(v109 + 16) != 13 )
        goto LABEL_55;
      v100 = v109 + 24;
    }
    v118 = 0;
    a2 = v98;
    if ( (_BYTE)v95 == 35 )
    {
      sub_16A7290((__int64)&v119, v98, v100, &v118);
      if ( v119.m128i_i32[2] > 0x40u )
      {
LABEL_145:
        if ( v119.m128i_i64[0] )
          j_j___libc_free_0_0(v119.m128i_i64[0]);
      }
    }
    else
    {
      sub_16A7620((__int64)&v119, v98, v100, &v118);
      if ( v119.m128i_i32[2] > 0x40u )
        goto LABEL_145;
    }
    if ( v118 || !sub_15F2380(v11) )
      goto LABEL_55;
    *(_BYTE *)(v6 + 17) &= 1u;
    a2 = 1;
    sub_15F2330(v6, 1);
LABEL_56:
    v113 = 1;
    v7 = *(unsigned __int8 *)(v6 + 16);
  }
  if ( v13 && v117 == *(_BYTE *)(v13 + 16) )
  {
    v9 = *(unsigned __int8 **)(v6 - 48);
    goto LABEL_95;
  }
LABEL_96:
  if ( !(unsigned __int8)sub_15F34B0(v6) )
    return v113;
LABEL_14:
  v14 = *(unsigned __int8 *)(v6 + 16);
  v15 = v14 - 24;
  if ( (unsigned int)(v14 - 24) <= 0x1C && ((1LL << v15) & 0x1C019800) != 0 )
  {
    v17 = *(_QWORD *)(v6 - 48);
    v18 = *(unsigned __int8 *)(v17 + 16);
    if ( (unsigned __int8)v18 > 0x17u && (unsigned int)(v18 - 60) <= 0xC )
    {
      v47 = *(_QWORD *)(v17 + 8);
      if ( v47 )
      {
        if ( !*(_QWORD *)(v47 + 8) && v18 == 61 && (unsigned int)(v14 - 50) <= 2 )
        {
          v48 = *(_QWORD *)(v17 - 24);
          v49 = *(_BYTE *)(v48 + 16);
          if ( (unsigned __int8)(v49 - 35) <= 0x11u )
          {
            v50 = *(_QWORD *)(v48 + 8);
            if ( v50 )
            {
              if ( !*(_QWORD *)(v50 + 8) && *(_BYTE *)(v6 + 16) == v49 )
              {
                v51 = *(_QWORD *)(v6 - 24);
                if ( *(_BYTE *)(v51 + 16) <= 0x10u )
                {
                  v52 = *(_QWORD *)(v48 - 24);
                  v111 = *(_QWORD *)(v17 - 24);
                  if ( *(_BYTE *)(v52 + 16) <= 0x10u )
                  {
                    v115 = *(__int64 **)(v6 - 24);
                    v53 = sub_15A46C0(37, (__int64 ***)v52, *(__int64 ***)v51, 0);
                    v54 = sub_15A2A30(
                            (__int64 *)v15,
                            v115,
                            v53,
                            0,
                            0,
                            *(double *)a4.m128i_i64,
                            a5,
                            *(double *)a6.m128i_i64);
                    sub_1593B40((_QWORD *)(v17 - 24), *(_QWORD *)(v111 - 48));
                    a2 = v54;
                    sub_1593B40((_QWORD *)(v6 - 24), v54);
                    goto LABEL_56;
                  }
                }
              }
            }
          }
        }
      }
    }
    if ( v11 )
    {
      if ( v117 == *(_BYTE *)(v11 + 16) )
      {
        v68 = *(unsigned __int8 **)(v11 - 48);
        a2 = *(_QWORD *)(v6 - 24);
        a6 = _mm_loadu_si128(a1 + 168);
        a4 = _mm_loadu_si128(a1 + 167);
        v69 = *(_QWORD *)(v11 - 24);
        v121 = v6;
        v119 = a4;
        v120 = (__m128)a6;
        v70 = sub_13E1140(v116, (unsigned __int8 *)a2, v68, &v119);
        if ( v70 )
        {
          if ( *(_QWORD *)(v6 - 48) )
          {
            v71 = *(_QWORD *)(v6 - 40);
            v72 = *(_QWORD *)(v6 - 32) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v72 = v71;
            if ( v71 )
            {
              a2 = *(_QWORD *)(v71 + 16) & 3LL;
              *(_QWORD *)(v71 + 16) = a2 | v72;
            }
          }
          *(_QWORD *)(v6 - 48) = v70;
          v73 = v70[1];
          *(_QWORD *)(v6 - 40) = v73;
          if ( v73 )
          {
            a2 = (v6 - 40) | *(_QWORD *)(v73 + 16) & 3LL;
            *(_QWORD *)(v73 + 16) = a2;
          }
          *(_QWORD *)(v6 - 32) = *(_QWORD *)(v6 - 32) & 3LL | (unsigned __int64)(v70 + 1);
          v70[1] = v6 - 48;
          if ( *(_QWORD *)(v6 - 24) )
          {
            v74 = *(_QWORD *)(v6 - 16);
            v75 = *(_QWORD *)(v6 - 8) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v75 = v74;
            if ( v74 )
              *(_QWORD *)(v74 + 16) = *(_QWORD *)(v74 + 16) & 3LL | v75;
          }
          *(_QWORD *)(v6 - 24) = v69;
          if ( v69 )
          {
            v76 = *(_QWORD *)(v69 + 8);
            *(_QWORD *)(v6 - 16) = v76;
            if ( v76 )
            {
              a2 = v6 - 16;
              *(_QWORD *)(v76 + 16) = (v6 - 16) | *(_QWORD *)(v76 + 16) & 3LL;
            }
            *(_QWORD *)(v6 - 8) = (v69 + 8) | *(_QWORD *)(v6 - 8) & 3LL;
            *(_QWORD *)(v69 + 8) = v6 - 24;
          }
          goto LABEL_55;
        }
      }
    }
    if ( !v13 )
      return v113;
    if ( v117 == *(_BYTE *)(v13 + 16) )
    {
      a2 = *(_QWORD *)(v13 - 24);
      v77 = *(unsigned __int8 **)(v6 - 48);
      v78 = _mm_loadu_si128(a1 + 168);
      v79 = _mm_loadu_si128(a1 + 167);
      v80 = *(_QWORD *)(v13 - 48);
      v121 = v6;
      v119 = v79;
      v120 = (__m128)v78;
      v81 = sub_13E1140(v116, (unsigned __int8 *)a2, v77, &v119);
      if ( v81 )
      {
        if ( *(_QWORD *)(v6 - 48) )
        {
          v82 = *(_QWORD *)(v6 - 40);
          v83 = *(_QWORD *)(v6 - 32) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v83 = v82;
          if ( v82 )
          {
            a2 = *(_QWORD *)(v82 + 16) & 3LL;
            *(_QWORD *)(v82 + 16) = a2 | v83;
          }
        }
        *(_QWORD *)(v6 - 48) = v80;
        if ( v80 )
        {
          v84 = *(_QWORD *)(v80 + 8);
          a2 = v80 + 8;
          *(_QWORD *)(v6 - 40) = v84;
          if ( v84 )
            *(_QWORD *)(v84 + 16) = (v6 - 40) | *(_QWORD *)(v84 + 16) & 3LL;
          *(_QWORD *)(v6 - 32) = a2 | *(_QWORD *)(v6 - 32) & 3LL;
          *(_QWORD *)(v80 + 8) = v6 - 48;
        }
        if ( *(_QWORD *)(v6 - 24) )
        {
          v85 = *(_QWORD *)(v6 - 16);
          v86 = *(_QWORD *)(v6 - 8) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v86 = v85;
          if ( v85 )
          {
            a2 = *(_QWORD *)(v85 + 16) & 3LL;
            *(_QWORD *)(v85 + 16) = a2 | v86;
          }
        }
        *(_QWORD *)(v6 - 24) = v81;
        v87 = v81[1];
        *(_QWORD *)(v6 - 16) = v87;
        if ( v87 )
        {
          a2 = (v6 - 16) | *(_QWORD *)(v87 + 16) & 3LL;
          *(_QWORD *)(v87 + 16) = a2;
        }
        *(_QWORD *)(v6 - 8) = *(_QWORD *)(v6 - 8) & 3LL | (unsigned __int64)(v81 + 1);
        v81[1] = v6 - 24;
        goto LABEL_55;
      }
    }
    if ( !v11 )
      return v113;
    if ( v117 != *(_BYTE *)(v11 + 16) )
      return v113;
    if ( v117 != *(_BYTE *)(v13 + 16) )
      return v113;
    v19 = *(_QWORD *)(v11 + 8);
    if ( !v19 )
      return v113;
    if ( *(_QWORD *)(v19 + 8) )
      return v113;
    v20 = *(__int64 **)(v11 - 48);
    if ( !v20 )
      return v113;
    v21 = *(_QWORD *)(v11 - 24);
    if ( *(_BYTE *)(v21 + 16) > 0x10u )
      return v113;
    v22 = *(_QWORD *)(v13 + 8);
    if ( !v22 )
      return v113;
    if ( *(_QWORD *)(v22 + 8) )
      return v113;
    v23 = *(_QWORD *)(v13 - 48);
    if ( !v23 )
      return v113;
    v110 = *(_QWORD *)(v13 - 24);
    if ( *(_BYTE *)(v110 + 16) > 0x10u )
      return v113;
    v120.m128_i16[0] = 257;
    v24 = sub_15FB440(v116, v20, v23, (__int64)&v119, 0);
    v25 = *(_QWORD *)v24;
    v26 = v24;
    v27 = *(_BYTE *)(*(_QWORD *)v24 + 8LL);
    if ( v27 == 16 )
      v27 = *(_BYTE *)(**(_QWORD **)(v25 + 16) + 8LL);
    if ( (unsigned __int8)(v27 - 1) <= 5u || *(_BYTE *)(v26 + 16) == 76 )
    {
      v114 = sub_15F24E0(v6);
      v28 = sub_15F24E0(v11) & v114;
      v29 = sub_15F24E0(v13);
      sub_15F2440(v26, v28 & v29);
    }
    v30 = *(_QWORD *)(v6 + 48);
    v119.m128i_i64[0] = v30;
    if ( v30 )
    {
      v31 = v26 + 48;
      sub_1623A60((__int64)&v119, v30, 2);
      v32 = *(_QWORD *)(v26 + 48);
      if ( !v32 )
        goto LABEL_41;
    }
    else
    {
      v32 = *(_QWORD *)(v26 + 48);
      v31 = v26 + 48;
      if ( !v32 )
      {
LABEL_43:
        sub_157E9D0(*(_QWORD *)(v6 + 40) + 40LL, v26);
        v34 = *(_QWORD *)(v6 + 24);
        *(_QWORD *)(v26 + 32) = v6 + 24;
        v34 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v26 + 24) = v34 | *(_QWORD *)(v26 + 24) & 7LL;
        *(_QWORD *)(v34 + 8) = v26 + 24;
        *(_QWORD *)(v6 + 24) = *(_QWORD *)(v6 + 24) & 7LL | (v26 + 24);
        sub_170B990(a1->m128i_i64[0], v26);
        sub_164B7C0(v26, v13);
        if ( *(_QWORD *)(v6 - 48) )
        {
          v35 = *(_QWORD *)(v6 - 40);
          v36 = *(_QWORD *)(v6 - 32) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v36 = v35;
          if ( v35 )
            *(_QWORD *)(v35 + 16) = *(_QWORD *)(v35 + 16) & 3LL | v36;
        }
        *(_QWORD *)(v6 - 48) = v26;
        v37 = *(_QWORD *)(v26 + 8);
        *(_QWORD *)(v6 - 40) = v37;
        if ( v37 )
          *(_QWORD *)(v37 + 16) = (v6 - 40) | *(_QWORD *)(v37 + 16) & 3LL;
        *(_QWORD *)(v6 - 32) = (v26 + 8) | *(_QWORD *)(v6 - 32) & 3LL;
        *(_QWORD *)(v26 + 8) = v6 - 48;
        a2 = v21;
        v38 = sub_15A2A30(
                (__int64 *)v116,
                (__int64 *)v21,
                v110,
                0,
                0,
                *(double *)a4.m128i_i64,
                a5,
                *(double *)a6.m128i_i64);
        if ( *(_QWORD *)(v6 - 24) )
        {
          v39 = *(_QWORD *)(v6 - 16);
          v40 = *(_QWORD *)(v6 - 8) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v40 = v39;
          if ( v39 )
          {
            a2 = *(_QWORD *)(v39 + 16) & 3LL;
            *(_QWORD *)(v39 + 16) = a2 | v40;
          }
        }
        *(_QWORD *)(v6 - 24) = v38;
        if ( v38 )
        {
          v41 = *(_QWORD *)(v38 + 8);
          a2 = v38 + 8;
          *(_QWORD *)(v6 - 16) = v41;
          if ( v41 )
            *(_QWORD *)(v41 + 16) = (v6 - 16) | *(_QWORD *)(v41 + 16) & 3LL;
          *(_QWORD *)(v6 - 8) = a2 | *(_QWORD *)(v6 - 8) & 3LL;
          *(_QWORD *)(v38 + 8) = v6 - 24;
        }
        goto LABEL_55;
      }
    }
    sub_161E7C0(v31, v32);
LABEL_41:
    v33 = (unsigned __int8 *)v119.m128i_i64[0];
    *(_QWORD *)(v26 + 48) = v119.m128i_i64[0];
    if ( v33 )
      sub_1623210((__int64)&v119, v33, v31);
    goto LABEL_43;
  }
  return v113;
}
