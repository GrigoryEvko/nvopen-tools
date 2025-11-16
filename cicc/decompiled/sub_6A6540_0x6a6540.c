// Function: sub_6A6540
// Address: 0x6a6540
//
__int64 __fastcall sub_6A6540(_QWORD *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *v6; // r14
  __int64 v7; // rbx
  char v8; // r13
  __int64 v9; // rdx
  unsigned __int8 v10; // al
  __int64 v11; // rsi
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // rdx
  __int64 v17; // rcx
  _QWORD *v18; // rdi
  unsigned __int16 v19; // r15
  int v20; // eax
  __int64 v21; // r13
  __int32 v22; // eax
  __int64 v23; // rdi
  char v24; // al
  __int64 v25; // r15
  __int64 v26; // rsi
  __int64 v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rax
  __int64 v35; // rax
  char v36; // cl
  __int64 v37; // rax
  char v38; // dl
  __int64 v39; // rbx
  unsigned int v40; // edi
  int v41; // ebx
  _QWORD *v42; // rbx
  int v44; // eax
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rax
  __int64 v50; // rsi
  __int64 *v51; // rax
  int v52; // esi
  __int64 v53; // rbx
  unsigned __int16 v54; // ax
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // rbx
  __int64 v60; // rdi
  _QWORD *v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  _QWORD *v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // rax
  __int64 v68; // r14
  unsigned int *v69; // rsi
  __int64 v70; // rdi
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // r8
  __int64 v75; // r9
  __m128i v76; // xmm1
  __m128i v77; // xmm2
  __m128i v78; // xmm3
  unsigned __int64 v79; // rcx
  __int64 v80; // rax
  __int64 v81; // r12
  char v82; // al
  char v83; // al
  const __m128i *v84; // rax
  __int8 v85; // dl
  _BYTE *v86; // rax
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // rdi
  __int64 v90; // rax
  _BYTE *v91; // rax
  __int64 v92; // rax
  __m128i *v93; // rdi
  const __m128i *v94; // rsi
  __int64 i; // rcx
  _QWORD *v96; // [rsp-10h] [rbp-300h]
  __int64 v97; // [rsp-8h] [rbp-2F8h]
  __int32 v98; // [rsp+10h] [rbp-2E0h]
  __m128i *v99; // [rsp+10h] [rbp-2E0h]
  unsigned int v100; // [rsp+18h] [rbp-2D8h]
  __int64 v101; // [rsp+18h] [rbp-2D8h]
  bool v102; // [rsp+21h] [rbp-2CFh]
  __int16 v103; // [rsp+22h] [rbp-2CEh]
  int v104; // [rsp+24h] [rbp-2CCh]
  unsigned int v105; // [rsp+24h] [rbp-2CCh]
  unsigned int v106; // [rsp+28h] [rbp-2C8h]
  int v107; // [rsp+30h] [rbp-2C0h]
  int v108; // [rsp+38h] [rbp-2B8h]
  unsigned int v109; // [rsp+48h] [rbp-2A8h] BYREF
  unsigned int v110; // [rsp+4Ch] [rbp-2A4h] BYREF
  __int64 v111; // [rsp+50h] [rbp-2A0h] BYREF
  __int64 v112; // [rsp+58h] [rbp-298h] BYREF
  _QWORD *v113; // [rsp+60h] [rbp-290h] BYREF
  _QWORD *v114; // [rsp+68h] [rbp-288h] BYREF
  _QWORD *v115; // [rsp+70h] [rbp-280h] BYREF
  __int64 v116; // [rsp+78h] [rbp-278h] BYREF
  _OWORD v117[4]; // [rsp+80h] [rbp-270h] BYREF
  _BYTE v118[160]; // [rsp+C0h] [rbp-230h] BYREF
  unsigned int v119[17]; // [rsp+160h] [rbp-190h] BYREF
  __int64 v120; // [rsp+1A4h] [rbp-14Ch]
  __int64 v121; // [rsp+1ACh] [rbp-144h]

  v6 = a2;
  v7 = (__int64)a1;
  v109 = 0;
  v111 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v8 = (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0;
  v102 = (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0;
  if ( !dword_4D04408 )
  {
    if ( a1 )
    {
      v9 = *a1;
      v10 = *(_BYTE *)(*a1 + 24LL);
LABEL_17:
      v100 = v10;
      if ( v10 != 2 )
      {
LABEL_21:
        sub_6F8810(
          (_DWORD)a1,
          (unsigned int)&v109,
          (unsigned int)v119,
          (unsigned int)&v112,
          (unsigned int)&v114,
          (unsigned int)v118,
          (__int64)&v116);
        v25 = v112;
        v104 = v109 == 0;
        v103 = *(_WORD *)(*a1 + 48LL);
        v98 = *(_DWORD *)(*a1 + 44LL);
        v115 = v114;
        sub_68B050((unsigned int)dword_4F04C64, (__int64)&v110, &v113);
        v26 = (__int64)v118;
        sub_6E2140(5, v118, 0, 0, a1);
        *(_BYTE *)(qword_4D03C50 + 18LL) |= 0x20u;
        goto LABEL_22;
      }
      v24 = *(_BYTE *)(*(_QWORD *)(v9 + 56) + 176LL);
      if ( v24 == 5 )
      {
        v100 = 12;
        goto LABEL_21;
      }
      if ( v24 == 6 )
      {
        v100 = 15;
        goto LABEL_21;
      }
LABEL_119:
      sub_721090(a1);
    }
    goto LABEL_111;
  }
  if ( a1 )
  {
    v9 = *a1;
    v10 = *(_BYTE *)(*a1 + 24LL);
    if ( v10 == 13 )
    {
      LODWORD(v112) = 0;
      v115 = *(_QWORD **)&dword_4F077C8;
      sub_6E2140(5, v118, 0, 0, a1);
      v11 = 0;
      *(_BYTE *)(qword_4D03C50 + 18LL) |= 0x20u;
      v12 = a1[6];
      v108 = *(_DWORD *)(v12 + 92);
      v13 = sub_6E3DA0(*a1, 0);
      v16 = *(_QWORD **)(v13 + 68);
      v114 = v16;
      v17 = *(unsigned int *)(v13 + 76);
      v18 = *(_QWORD **)(v13 + 128);
      v19 = *(_WORD *)(v13 + 80);
      v106 = *(_DWORD *)(v13 + 76);
      if ( v18 )
      {
        if ( (*(_DWORD *)(v7 + 40) & 0x2040) != 0 )
        {
          v84 = *(const __m128i **)(*(_QWORD *)v7 + 80LL);
          *v6 = _mm_loadu_si128(v84);
          v6[1] = _mm_loadu_si128(v84 + 1);
          v6[2] = _mm_loadu_si128(v84 + 2);
          v6[3] = _mm_loadu_si128(v84 + 3);
          v6[4] = _mm_loadu_si128(v84 + 4);
          v6[5] = _mm_loadu_si128(v84 + 5);
          v6[6] = _mm_loadu_si128(v84 + 6);
          v6[7] = _mm_loadu_si128(v84 + 7);
          v6[8] = _mm_loadu_si128(v84 + 8);
          v85 = v84[1].m128i_i8[0];
          if ( v85 != 2 )
            goto LABEL_201;
          v6[9] = _mm_loadu_si128(v84 + 9);
          v6[10] = _mm_loadu_si128(v84 + 10);
          v6[11] = _mm_loadu_si128(v84 + 11);
          v6[12] = _mm_loadu_si128(v84 + 12);
          v6[13] = _mm_loadu_si128(v84 + 13);
          v6[14] = _mm_loadu_si128(v84 + 14);
          v6[15] = _mm_loadu_si128(v84 + 15);
          v6[16] = _mm_loadu_si128(v84 + 16);
          v6[17] = _mm_loadu_si128(v84 + 17);
          v6[18] = _mm_loadu_si128(v84 + 18);
          v6[19] = _mm_loadu_si128(v84 + 19);
          v6[20] = _mm_loadu_si128(v84 + 20);
          v6[21] = _mm_loadu_si128(v84 + 21);
          goto LABEL_203;
        }
        *(_DWORD *)(*(_QWORD *)(v7 + 48) + 92LL) = 0;
        v20 = sub_869530(
                (_DWORD)v18,
                *(_QWORD *)(v7 + 32),
                *(_QWORD *)(v7 + 24),
                (unsigned int)&v113,
                *(_DWORD *)(v7 + 40),
                *(_QWORD *)(v7 + 48),
                (__int64)v119);
        v11 = v119[0];
        v16 = v96;
        v17 = v97;
        if ( v119[0] )
          *(_BYTE *)(v7 + 56) = 1;
        v21 = 0;
        if ( v20 )
        {
          do
          {
            v11 = 1;
            ++v21;
            sub_867630(v113, 1);
            v18 = v113;
          }
          while ( (unsigned int)sub_866C00(v113) );
          goto LABEL_10;
        }
      }
      else
      {
        v119[0] = 1;
        *(_BYTE *)(v7 + 56) = 1;
      }
      if ( (*(_BYTE *)(v7 + 41) & 0x40) != 0 )
      {
        v21 = 0;
        if ( *(_DWORD *)(*(_QWORD *)(v7 + 48) + 92LL) )
        {
          v84 = *(const __m128i **)(*(_QWORD *)v7 + 80LL);
          *v6 = _mm_loadu_si128(v84);
          v6[1] = _mm_loadu_si128(v84 + 1);
          v6[2] = _mm_loadu_si128(v84 + 2);
          v6[3] = _mm_loadu_si128(v84 + 3);
          v6[4] = _mm_loadu_si128(v84 + 4);
          v6[5] = _mm_loadu_si128(v84 + 5);
          v6[6] = _mm_loadu_si128(v84 + 6);
          v6[7] = _mm_loadu_si128(v84 + 7);
          v6[8] = _mm_loadu_si128(v84 + 8);
          v85 = v84[1].m128i_i8[0];
          if ( v85 != 2 )
          {
LABEL_201:
            if ( v85 == 5 || v85 == 1 )
              v6[9].m128i_i64[0] = v84[9].m128i_i64[0];
            goto LABEL_203;
          }
          v93 = v6 + 9;
          v94 = v84 + 9;
          for ( i = 52; i; --i )
          {
            v93->m128i_i32[0] = v94->m128i_i32[0];
            v94 = (const __m128i *)((char *)v94 + 4);
            v93 = (__m128i *)((char *)v93 + 4);
          }
LABEL_203:
          sub_6F4B70(v6);
          *(_DWORD *)(v12 + 92) = v108;
          goto LABEL_12;
        }
      }
      else
      {
        v21 = 0;
      }
LABEL_10:
      *(_DWORD *)(v12 + 92) = v108;
      goto LABEL_11;
    }
    goto LABEL_17;
  }
  v54 = word_4F06418[0];
  if ( word_4F06418[0] == 99 )
  {
    if ( (unsigned __int16)sub_7BE840(0, 0) == 76 )
    {
      v11 = (__int64)v118;
      v60 = 5;
      LODWORD(v112) = 0;
      v115 = *(_QWORD **)&dword_4F077C8;
      sub_6E2140(5, v118, 0, 0, 0);
      *(_BYTE *)(qword_4D03C50 + 18LL) |= 0x20u;
      if ( dword_4F04C64 == -1
        || (v61 = qword_4F04C68, (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 1) == 0) )
      {
        if ( (unsigned int)sub_6E5430(5, v118, v61, v62, v63, v64) )
        {
          v60 = 2014;
          v11 = (__int64)&dword_4F063F8;
          sub_6851C0(0x7DEu, &dword_4F063F8);
        }
        v107 = 1;
      }
      else
      {
        v107 = 0;
      }
      v114 = *(_QWORD **)&dword_4F063F8;
      sub_7B8B50(v60, v11, v61, v62);
      v18 = &v113;
      v105 = sub_869470(&v113);
      if ( v105 )
      {
        v105 = 0;
        v21 = 0;
        v99 = v6;
        do
        {
          ++v21;
          if ( dword_4F04C64 != -1
            && (v65 = qword_4F04C68, (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 1) != 0) )
          {
            sub_867610();
          }
          else
          {
            sub_7B8B50(v18, v11, v65, v66);
          }
          sub_7BE280(27, 125, 0, 0);
          ++*(_BYTE *)(qword_4F061C8 + 36LL);
          ++*(_QWORD *)(qword_4D03C50 + 40LL);
          if ( word_4F06418[0] == 1 )
          {
            v69 = 0;
            v70 = (__int64)&qword_4D04A00;
            v115 = *(_QWORD **)&dword_4F063F8;
            v116 = qword_4F063F0;
            v71 = sub_7D5DD0(&qword_4D04A00, 0);
            v7 = v71;
            if ( v71 )
            {
              if ( *(_BYTE *)(v71 + 80) != 17 )
              {
                v69 = &dword_4F063F8;
                v70 = v71;
                sub_6E50B0(v71, &dword_4F063F8);
                if ( *(_BYTE *)(v7 + 80) == 8 )
                {
                  v70 = v105;
                  if ( !v105 )
                  {
                    v76 = _mm_loadu_si128((const __m128i *)&unk_4D04A10);
                    v105 = 1;
                    v77 = _mm_loadu_si128(&xmmword_4D04A20);
                    v78 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
                    v117[0] = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
                    v117[1] = v76;
                    v117[2] = v77;
                    v117[3] = v78;
                  }
                }
              }
              v79 = (unsigned int)dword_4F04C44;
              if ( dword_4F04C44 != -1
                || (v69 = (unsigned int *)qword_4F04C68,
                    v80 = qword_4F04C68[0] + 776LL * dword_4F04C64,
                    (*(_BYTE *)(v80 + 6) & 6) != 0)
                || *(_BYTE *)(v80 + 4) == 12 )
              {
                if ( v113 && !v113[2] )
                {
                  if ( dword_4F04C64 != -1
                    && (v69 = (unsigned int *)qword_4F04C68,
                        v86 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64),
                        (v86[7] & 1) != 0)
                    && (dword_4F04C44 != -1 || (v86[6] & 6) != 0 || v86[4] == 12)
                    && (v70 = v7, (unsigned int)sub_8782F0(v7)) )
                  {
                    if ( dword_4F04C64 != -1 )
                    {
                      v79 = (unsigned __int64)qword_4F04C68;
                      v91 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
                      if ( (v91[7] & 1) != 0 && (dword_4F04C44 != -1 || (v91[6] & 6) != 0 || v91[4] == 12) )
                      {
                        v69 = &dword_4F063F8;
                        v70 = v7;
                        sub_867130(v7, &dword_4F063F8, 0, 0);
                      }
                    }
                  }
                  else
                  {
                    if ( (unsigned int)sub_6E5430(v70, v69, v72, v79, v74, v75) )
                    {
                      v69 = &dword_4F063F8;
                      sub_6851C0(0x7DDu, &dword_4F063F8);
                    }
                    v70 = (__int64)v113;
                    sub_866870(v113);
                    v107 = 1;
                  }
                }
              }
            }
            else
            {
              v107 = 1;
              if ( (unsigned int)sub_6E5430(&qword_4D04A00, 0, v72, v73, v74, v75) )
              {
                v69 = &dword_4F063F8;
                v70 = 2013;
                sub_6851C0(0x7DDu, &dword_4F063F8);
              }
            }
            sub_7B8B50(v70, v69, v72, v79);
          }
          else
          {
            sub_6E5F20(40);
            v107 = 1;
          }
          sub_7BE5B0(28, 18, 0, 0);
          v11 = 1;
          --*(_BYTE *)(qword_4F061C8 + 36LL);
          --*(_QWORD *)(qword_4D03C50 + 40LL);
          v67 = sub_867630(v113, 1);
          v18 = v113;
          v68 = v67;
        }
        while ( (unsigned int)sub_866C00(v113) );
        v101 = v68;
        v6 = v99;
      }
      else
      {
        v101 = 0;
        v21 = 0;
      }
      v17 = dword_4F063F8;
      v19 = word_4F063FC[0];
      v106 = dword_4F063F8;
      if ( word_4F06418[0] == 28 )
        sub_7B8B50(v18, v11, v65, dword_4F063F8);
      if ( v107 )
      {
        sub_6E6260(v6);
        goto LABEL_12;
      }
      if ( unk_4F04C48 == -1
        || (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 6) & 0x10) == 0
        || dword_4F04C64 <= 0
        || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 - 770) & 2) == 0 )
      {
        if ( dword_4F04C44 == -1
          && (v16 = qword_4F04C68, v88 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v88 + 6) & 6) == 0)
          && *(_BYTE *)(v88 + 4) != 12
          || !v101 && !(unsigned int)sub_867AA0() )
        {
LABEL_11:
          *(_QWORD *)v119 = sub_724DC0(v18, v11, v16, v17, v14, v15);
          sub_72BBE0(*(_QWORD *)v119, v21, unk_4F06A51);
          sub_6E6A50(*(_QWORD *)v119, v6);
          sub_724E30(v119);
LABEL_12:
          v22 = (int)v114;
          v6[5].m128i_i16[0] = v19;
          v6[4].m128i_i32[1] = v22;
          v6[4].m128i_i16[4] = WORD2(v114);
          *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)v6[4].m128i_i64 + 4);
          v6[4].m128i_i32[3] = v106;
          unk_4F061D8 = *(__int64 *)((char *)&v6[4].m128i_i64[1] + 4);
          sub_6E3280(v6, &v114);
          sub_6E3BA0(v6, &v114, 0, 0);
          v23 = (unsigned int)v112;
          if ( (_DWORD)v112 )
            sub_729730((unsigned int)v112);
          sub_6E2B30(v23, &v114);
          return sub_724E30(&v111);
        }
      }
      if ( *(_BYTE *)(v7 + 80) == 7 )
      {
        v89 = *(_QWORD *)(*(_QWORD *)(v7 + 88) + 48LL);
        if ( v89 )
        {
          v90 = sub_72B840(v89);
          sub_7296F0(*(unsigned int *)(v90 + 240), &v112);
        }
      }
      v81 = sub_726700(13);
      a1 = (_QWORD *)unk_4F06A51;
      *(_QWORD *)v81 = sub_72BA30(unk_4F06A51);
      v82 = *(_BYTE *)(v7 + 80);
      if ( v82 == 3 || dword_4F077C4 == 2 && (unsigned __int8)(v82 - 4) <= 2u )
      {
        *(_BYTE *)(v81 + 56) = 1;
        *(_QWORD *)(v81 + 64) = *(_QWORD *)(v7 + 88);
      }
      else if ( v82 == 19 )
      {
        v92 = *(_QWORD *)(*(_QWORD *)(v7 + 88) + 104LL);
        *(_WORD *)(v81 + 56) = 256;
        *(_QWORD *)(v81 + 64) = v92;
      }
      else
      {
        *(_BYTE *)(v81 + 56) = 0;
        v83 = *(_BYTE *)(v7 + 80);
        switch ( v83 )
        {
          case 7:
            sub_6F8E70(*(_QWORD *)(v7 + 88), &v115, &v116, v119, 0);
            break;
          case 18:
            sub_688B20((__int64)v119, v7);
            break;
          case 2:
            sub_6F35D0(v7, v119);
            break;
          default:
            if ( v83 != 8 || (v105 & 1) == 0 )
              goto LABEL_119;
            sub_6EAEF0(v117, &v115, &v116, v119);
            break;
        }
        LODWORD(v120) = (_DWORD)v115;
        WORD2(v120) = WORD2(v115);
        *(_QWORD *)dword_4F07508 = v120;
        v121 = v116;
        unk_4F061D8 = v116;
        sub_6E3280(v119, 0);
        *(_QWORD *)(v81 + 64) = sub_6F6F40(v119, 0);
      }
      sub_6E70E0(v81, v6);
      if ( v101 )
        v6[8].m128i_i64[0] = v101;
      sub_6F4B70(v6);
      v6[8].m128i_i64[0] = 0;
      goto LABEL_12;
    }
LABEL_111:
    v54 = word_4F06418[0];
  }
  v100 = 3 * (v54 == 284) + 12;
  v114 = *(_QWORD **)&dword_4F063F8;
  v115 = *(_QWORD **)&dword_4F063F8;
  sub_68B050((unsigned int)dword_4F04C64, (__int64)&v110, &v113);
  sub_6E2140(5, v118, 0, 0, 0);
  *(_BYTE *)(qword_4D03C50 + 18LL) |= 0x20u;
  sub_7B8B50(5, v118, v55, v56);
  if ( word_4F06418[0] != 27 )
  {
    a1 = (_QWORD *)v109;
    if ( v109 )
    {
      v116 = *(_QWORD *)&dword_4F063F8;
      goto LABEL_119;
    }
    v26 = 0;
    sub_69ED20((__int64)v119, 0, 18, 0);
    goto LABEL_110;
  }
  *(_QWORD *)&v117[0] = *(_QWORD *)&dword_4F063F8;
  sub_7B8B50(5, v118, v57, v58);
  if ( (unsigned int)sub_679C10(0x405u) )
  {
    v109 = 1;
  }
  else if ( !v109 )
  {
    v26 = 0;
    sub_69ED20((__int64)v119, 0, 18, 8);
    v120 = *(_QWORD *)&v117[0];
LABEL_110:
    v25 = 0;
    v104 = 1;
    v98 = v121;
    v103 = WORD2(v121);
    goto LABEL_22;
  }
  v116 = *(_QWORD *)&dword_4F063F8;
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  ++*(_QWORD *)(qword_4D03C50 + 40LL);
  v112 = sub_65CF50(v8);
  v98 = qword_4F063F0;
  v103 = WORD2(qword_4F063F0);
  sub_7BE280(28, 18, 0, 0);
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  --*(_QWORD *)(qword_4D03C50 + 40LL);
  v26 = dword_4D0477C;
  if ( dword_4D0477C && word_4F06418[0] == 73 )
  {
    v26 = (__int64)v117;
    sub_68D9C0((__int64)&v112, (__int64)v117, &v116, 0, 0, (__int64)v6, 0);
    v25 = v6->m128i_i64[0];
    v112 = v6->m128i_i64[0];
  }
  else
  {
    v25 = v112;
  }
  v104 = 0;
LABEL_22:
  if ( v109 )
  {
    LODWORD(v27) = sub_8D32E0(v25);
    if ( (_DWORD)v27 )
    {
      if ( dword_4F04C44 == -1
        && (v28 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v28 + 6) & 6) == 0)
        && *(_BYTE *)(v28 + 4) != 12
        || !(unsigned int)sub_8DBE70(v25) )
      {
        v25 = sub_8D46C0(v25);
      }
      LODWORD(v27) = 0;
    }
    if ( dword_4F077C4 != 2 )
      goto LABEL_31;
  }
  else
  {
    v26 = 39;
    sub_6F69D0(v119, 39);
    if ( (unsigned int)sub_6ECD10(v119) && dword_4F077C4 != 1 )
    {
      v26 = (__int64)v119;
      sub_6E68E0(71, v119);
    }
    sub_831BB0(v119);
    v25 = *(_QWORD *)v119;
    v112 = *(_QWORD *)v119;
    v116 = v120;
    v27 = (unsigned int)sub_696840((__int64)v119) != 0;
    if ( dword_4F077C4 != 2 )
      goto LABEL_31;
  }
  if ( (unsigned int)sub_8D23B0(v25) )
    sub_8AE000(v25);
  if ( dword_4F077C4 == 2 )
  {
    if ( dword_4F04C44 != -1
      || (v49 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v49 + 6) & 6) != 0)
      || *(_BYTE *)(v49 + 4) == 12 )
    {
      if ( (unsigned int)sub_8DBE70(v25) )
      {
LABEL_72:
        LODWORD(v27) = 1;
        goto LABEL_34;
      }
    }
  }
LABEL_31:
  v29 = v25;
  if ( (unsigned int)sub_8D2310(v25) )
  {
    if ( HIDWORD(qword_4F077B4) )
    {
      v34 = sub_72BA30(0);
      v109 = 1;
      v25 = v34;
      if ( dword_4F077BC )
      {
        v26 = 56;
        if ( (unsigned int)sub_6E53E0(5, 56, &v116) )
        {
          v26 = (__int64)&v116;
          sub_684B30(0x38u, &v116);
        }
      }
      goto LABEL_34;
    }
    if ( (unsigned int)sub_6E5430(v25, v26, v30, v31, v32, v33) )
    {
      v26 = (__int64)&v116;
      v29 = 56;
      sub_6851C0(0x38u, &v116);
    }
LABEL_95:
    v25 = sub_72C930(v29);
    goto LABEL_34;
  }
  if ( !(unsigned int)sub_8D23B0(v25) )
  {
    if ( !(unsigned int)sub_8D2BE0(v25) )
      goto LABEL_34;
    v29 = (__int64)&v116;
    v26 = v25;
    sub_6E5FC0(&v116, v25);
    goto LABEL_95;
  }
  if ( !HIDWORD(qword_4F077B4) || !(unsigned int)sub_8D2600(v25) )
  {
    if ( !dword_4F077BC
      || qword_4F077A8 > 0x76BFu
      || dword_4F04C44 == -1
      && (v87 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v87 + 6) & 6) == 0)
      && *(_BYTE *)(v87 + 4) != 12
      || unk_4D047C8
      || !(unsigned int)sub_8D3A70(v25) )
    {
      v26 = v25;
      sub_6E5F60(&v116, v25, 8);
      v25 = sub_72C930(&v116);
      goto LABEL_34;
    }
    goto LABEL_72;
  }
  if ( dword_4F077BC )
  {
    v26 = v25;
    sub_6E5F60(&v116, v25, 5);
  }
  v25 = sub_72BA30(0);
LABEL_34:
  if ( dword_4F04C44 != -1
    || (v35 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v35 + 6) & 6) != 0)
    || *(_BYTE *)(v35 + 4) == 12 )
  {
    if ( (unsigned int)sub_8DC060(v25) )
      LODWORD(v27) = 1;
    if ( !dword_4D047EC )
      goto LABEL_38;
  }
  else if ( !dword_4D047EC )
  {
    goto LABEL_38;
  }
  v44 = sub_8D4070(v25);
  if ( (v27 & 1) == 0 && v44 )
  {
    if ( v102 )
    {
      if ( (unsigned int)sub_6E5430(v25, v26, v45, v46, v47, v48) )
        sub_6851C0(0x1Cu, &v115);
      v41 = 0;
      sub_6E6260(v6);
    }
    else
    {
      v41 = 1;
      sub_688D10(v100, v109, v112, v119, (__int64)v6);
      v104 = (v109 != 0) & (unsigned __int8)v104;
    }
    goto LABEL_47;
  }
LABEL_38:
  v36 = *(_BYTE *)(v25 + 140);
  if ( v36 == 12 )
  {
    v37 = v25;
    do
    {
      v37 = *(_QWORD *)(v37 + 160);
      v38 = *(_BYTE *)(v37 + 140);
    }
    while ( v38 == 12 );
  }
  else
  {
    v38 = *(_BYTE *)(v25 + 140);
  }
  if ( v38 )
  {
    if ( (_DWORD)v27 )
    {
      sub_724C70(v111, 12);
      sub_7249B0(v111, (unsigned int)(v100 == 15) + 5);
      v39 = v111;
      v40 = v109;
      *(_QWORD *)(v111 + 184) = v25;
      if ( !v40 )
      {
        sub_6F40C0(v119);
        v59 = v111;
        v104 = 0;
        *(_QWORD *)(v59 + 192) = sub_6F6F40(v119, 0);
        v39 = v111;
      }
      *(_QWORD *)(v39 + 128) = sub_72BA30(unk_4F06A51);
      goto LABEL_46;
    }
    if ( v100 == 15 )
    {
      v50 = sub_8D4B80(v25);
    }
    else if ( v36 == 12 )
    {
      v50 = sub_8D4A00(v25);
    }
    else if ( dword_4F077C0 && (v36 == 1 || v36 == 7) )
    {
      v50 = 1;
    }
    else
    {
      v50 = *(_QWORD *)(v25 + 128);
    }
    sub_72BBE0(v111, v50, unk_4F06A51);
    if ( !*(_BYTE *)(qword_4D03C50 + 16LL) )
    {
      v51 = *(__int64 **)(qword_4D03C50 + 72LL);
      if ( !v51 )
      {
        v51 = *(__int64 **)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 304);
        if ( !v51 )
          goto LABEL_46;
      }
      while ( (*((_BYTE *)v51 + 29) & 0x10) == 0 )
      {
        v51 = (__int64 *)*v51;
        if ( !v51 )
          goto LABEL_46;
      }
    }
    sub_729730(v110);
    v52 = v109;
    if ( !v109 && unk_4F07270 == unk_4F073B8 && (unk_4F04C50 || (v52 = dword_4F04C38) != 0) )
    {
      v109 = 1;
      v52 = 1;
    }
    v53 = v111;
    *(_QWORD *)(v53 + 144) = sub_688D10(v100, v52, v112, v119, 0);
    LODWORD(v53) = v109;
    sub_7296F0((unsigned int)dword_4F04C64, &v110);
    v104 = ((_DWORD)v53 != 0) & (unsigned __int8)v104;
    goto LABEL_46;
  }
  sub_72C970(v111);
LABEL_46:
  v41 = 0;
  sub_6E6A50(v111, v6);
LABEL_47:
  if ( v104 )
    sub_6E24C0();
  v6[4].m128i_i32[1] = (int)v115;
  v6[4].m128i_i16[4] = WORD2(v115);
  *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)v6[4].m128i_i64 + 4);
  v6[4].m128i_i32[3] = v98;
  v6[5].m128i_i16[0] = v103;
  unk_4F061D8 = *(__int64 *)((char *)&v6[4].m128i_i64[1] + 4);
  sub_6E3280(v6, &v114);
  sub_6E3BA0(v6, &v114, 0, &v116);
  sub_6E2B30(v6, &v114);
  if ( dword_4F077C4 == 2
    && (unk_4F07778 > 201102 || dword_4F07774)
    && *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u
    && v41
    && (unsigned int)sub_6E91E0(28, &v115) )
  {
    sub_6E6840(v6);
  }
  v42 = v113;
  sub_729730(v110);
  qword_4F06BC0 = v42;
  return sub_724E30(&v111);
}
