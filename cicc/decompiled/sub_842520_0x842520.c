// Function: sub_842520
// Address: 0x842520
//
__int64 __fastcall sub_842520(_QWORD *a1, __int64 a2, _BYTE *a3, unsigned int a4, unsigned int a5, unsigned int a6)
{
  __int64 v8; // r12
  unsigned int v9; // ebx
  char v10; // al
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 i; // rdx
  __int64 v14; // r8
  __int64 v15; // rcx
  _BOOL4 v16; // eax
  bool v17; // r15
  const __m128i *v18; // r15
  __int64 v19; // rdi
  _DWORD *v20; // rsi
  unsigned int v21; // edi
  _QWORD *v22; // r15
  const __m128i *v23; // rbx
  __m128i *v24; // r15
  const __m128i *v25; // rax
  __int64 v26; // rax
  char j; // dl
  const __m128i *v28; // rax
  __int64 v29; // r9
  __int64 v31; // rdi
  const __m128i *v32; // rdi
  __int64 v33; // rdi
  unsigned int v34; // r9d
  __int64 v35; // rdx
  const __m128i *v36; // rax
  int v37; // eax
  bool v38; // zf
  unsigned int v39; // r12d
  int v40; // edx
  int v41; // eax
  _DWORD *v42; // rsi
  __int64 v43; // rax
  char v44; // al
  char v45; // di
  int v46; // eax
  char v47; // al
  const __m128i *v48; // rax
  int v49; // edi
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  unsigned int v61; // r8d
  int v62; // r12d
  unsigned int v63; // r12d
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  int v68; // eax
  _DWORD *v69; // rdx
  __int64 v70; // rdx
  const __m128i *v71; // r13
  __int64 v72; // r9
  int v73; // edi
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  int v78; // edi
  int v79; // esi
  char v80; // r12
  int v81; // edi
  __int64 v82; // [rsp-8h] [rbp-258h]
  unsigned int v83; // [rsp+Ch] [rbp-244h]
  _BOOL4 v84; // [rsp+10h] [rbp-240h]
  int v86; // [rsp+14h] [rbp-23Ch]
  __m128i *v87; // [rsp+18h] [rbp-238h]
  int v88; // [rsp+20h] [rbp-230h]
  int v89; // [rsp+28h] [rbp-228h]
  unsigned int v90; // [rsp+2Ch] [rbp-224h]
  int v91; // [rsp+30h] [rbp-220h]
  char v92; // [rsp+30h] [rbp-220h]
  char v93; // [rsp+37h] [rbp-219h]
  const __m128i *v94; // [rsp+38h] [rbp-218h]
  unsigned int v96; // [rsp+44h] [rbp-20Ch]
  __int64 v97; // [rsp+48h] [rbp-208h]
  unsigned int v98; // [rsp+5Ch] [rbp-1F4h] BYREF
  int v99; // [rsp+60h] [rbp-1F0h] BYREF
  unsigned int v100; // [rsp+64h] [rbp-1ECh] BYREF
  int v101; // [rsp+68h] [rbp-1E8h] BYREF
  int v102; // [rsp+6Ch] [rbp-1E4h] BYREF
  int v103; // [rsp+70h] [rbp-1E0h] BYREF
  unsigned int v104; // [rsp+74h] [rbp-1DCh] BYREF
  const __m128i *v105; // [rsp+78h] [rbp-1D8h] BYREF
  __int64 v106; // [rsp+80h] [rbp-1D0h] BYREF
  __int64 v107; // [rsp+88h] [rbp-1C8h] BYREF
  _BYTE v108[48]; // [rsp+90h] [rbp-1C0h] BYREF
  _OWORD v109[9]; // [rsp+C0h] [rbp-190h] BYREF
  __m128i v110; // [rsp+150h] [rbp-100h]
  __m128i v111; // [rsp+160h] [rbp-F0h]
  __m128i v112; // [rsp+170h] [rbp-E0h]
  __m128i v113; // [rsp+180h] [rbp-D0h]
  __m128i v114; // [rsp+190h] [rbp-C0h]
  __m128i v115; // [rsp+1A0h] [rbp-B0h]
  __m128i v116; // [rsp+1B0h] [rbp-A0h]
  __m128i v117; // [rsp+1C0h] [rbp-90h]
  __m128i v118; // [rsp+1D0h] [rbp-80h]
  __m128i v119; // [rsp+1E0h] [rbp-70h]
  __m128i v120; // [rsp+1F0h] [rbp-60h]
  __m128i v121; // [rsp+200h] [rbp-50h]
  __m128i v122; // [rsp+210h] [rbp-40h]

  v8 = a2;
  v9 = a5;
  v94 = (const __m128i *)*a1;
  v98 = 0;
  v90 = a5 & 1;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v89 = a5 & 0x20;
  v105 = 0;
  v106 = 0;
  v96 = sub_8D3110(a2);
  v109[0] = _mm_loadu_si128((const __m128i *)a1);
  v109[1] = _mm_loadu_si128((const __m128i *)a1 + 1);
  v10 = *((_BYTE *)a1 + 16);
  v109[2] = _mm_loadu_si128((const __m128i *)a1 + 2);
  v109[3] = _mm_loadu_si128((const __m128i *)a1 + 3);
  v109[4] = _mm_loadu_si128((const __m128i *)a1 + 4);
  v109[5] = _mm_loadu_si128((const __m128i *)a1 + 5);
  v109[6] = _mm_loadu_si128((const __m128i *)a1 + 6);
  v109[7] = _mm_loadu_si128((const __m128i *)a1 + 7);
  v109[8] = _mm_loadu_si128((const __m128i *)a1 + 8);
  if ( v10 == 2 )
  {
    v110 = _mm_loadu_si128((const __m128i *)a1 + 9);
    v111 = _mm_loadu_si128((const __m128i *)a1 + 10);
    v112 = _mm_loadu_si128((const __m128i *)a1 + 11);
    v113 = _mm_loadu_si128((const __m128i *)a1 + 12);
    v114 = _mm_loadu_si128((const __m128i *)a1 + 13);
    v115 = _mm_loadu_si128((const __m128i *)a1 + 14);
    v116 = _mm_loadu_si128((const __m128i *)a1 + 15);
    v117 = _mm_loadu_si128((const __m128i *)a1 + 16);
    v118 = _mm_loadu_si128((const __m128i *)a1 + 17);
    v119 = _mm_loadu_si128((const __m128i *)a1 + 18);
    v120 = _mm_loadu_si128((const __m128i *)a1 + 19);
    v121 = _mm_loadu_si128((const __m128i *)a1 + 20);
    v122 = _mm_loadu_si128((const __m128i *)a1 + 21);
    goto LABEL_78;
  }
  if ( v10 != 5 )
  {
    if ( v10 != 1 )
    {
      if ( !a3 || (a3[16] & 0x28) != 0x20 )
      {
LABEL_6:
        if ( (dword_4F04C44 != -1
           || (v11 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v11 + 6) & 6) != 0)
           || *(_BYTE *)(v11 + 4) == 12)
          && ((unsigned int)sub_8DBE70(a2) || (unsigned int)sub_8DBE70(v94)) )
        {
          v104 = 1;
          v91 = 0;
          v88 = 0;
        }
        else
        {
          v35 = a2;
          a2 = 0;
          v91 = 0;
          v88 = sub_831CF0((__int64)a1, 0, v35, 0, v9, &v102, &v103, &v101, &v99, &v104, &v106);
          if ( !v88 )
          {
            if ( (*(_BYTE *)(qword_4D03C50 + 16LL) > 3u || (a2 = word_4D04898) != 0)
              && (unsigned int)sub_8E31E0(*a1)
              && (a2 = v8, v46 = sub_8413E0((__m128i *)a1, v8, v9, 0, (__int64)v108, &v100, &v105), v100 | v46) )
            {
              if ( (v108[17] & 1) != 0 )
              {
                v104 = 1;
                a3 = v108;
                v91 = 1;
              }
              else
              {
                v91 = 1;
                a3 = v108;
              }
            }
            else
            {
              v91 = 0;
            }
          }
        }
        goto LABEL_10;
      }
LABEL_87:
      if ( (a3[17] & 1) != 0 )
        v104 = 1;
      v91 = 1;
      v88 = 0;
LABEL_10:
      v12 = sub_8D46C0(v8);
      v15 = *((unsigned __int8 *)a1 + 17);
      v97 = v12;
      v93 = v15;
      if ( (_BYTE)v15 == 2 )
      {
        if ( !v89 )
        {
          v84 = 1;
          v87 = (__m128i *)v12;
          goto LABEL_17;
        }
        v84 = 1;
        v17 = 0;
      }
      else
      {
        v16 = sub_6ED0A0((__int64)a1);
        v84 = v16;
        v15 = *((unsigned __int8 *)a1 + 17);
        v17 = !v16 && v96 != 0;
        v93 = *((_BYTE *)a1 + 17);
        if ( !v89 )
        {
          v87 = (__m128i *)v97;
LABEL_13:
          if ( v17 && v93 != 3 && v88 && *((_BYTE *)a1 + 16) )
          {
            v36 = (const __m128i *)*a1;
            for ( i = *(unsigned __int8 *)(*a1 + 140LL); (_BYTE)i == 12; i = v36[8].m128i_u8[12] )
              v36 = (const __m128i *)v36[10].m128i_i64[0];
            if ( (_BYTE)i )
            {
              if ( !dword_4F077BC || qword_4F077A8 > 0x9E33u )
                sub_721090();
              a2 = (v9 & 0x40) == 0 ? 1768 : 2209;
              sub_6E5C80(4, a2, (_DWORD *)a1 + 17);
            }
          }
LABEL_17:
          if ( dword_4F04C58 != -1 )
          {
            i = (__int64)qword_4F04C68;
            if ( (*(_BYTE *)(*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216) + 198LL) & 0x10) != 0
              && ((unsigned int)sub_8D2FF0(*a1, a2) || (unsigned int)sub_8D3030(*a1)) )
            {
              a2 = (__int64)a1 + 68;
              sub_6851C0(0xDE7u, (_DWORD *)a1 + 17);
              sub_6E6840((__int64)a1);
            }
          }
          if ( v91 && !v104 )
          {
            if ( v105 )
            {
              if ( (unsigned int)sub_6E5430() )
              {
                v18 = v105;
                if ( v105 )
                {
                  while ( 1 )
                  {
                    v19 = v18->m128i_i64[1];
                    if ( v19 )
                    {
                      if ( (unsigned __int8)sub_877F80(v19) == 1 )
                        break;
                    }
                    v18 = (const __m128i *)v18->m128i_i64[0];
                    if ( !v18 )
                      goto LABEL_27;
                  }
                  v20 = (_DWORD *)a1 + 17;
                  v21 = 348;
                }
                else
                {
LABEL_27:
                  v20 = (_DWORD *)a1 + 17;
                  v21 = 417;
                }
                v22 = sub_67DAA0(v21, v20, (__int64)v94, v97);
                a2 = 0;
                sub_82E650(v105->m128i_i64, 0, 0, 0, v22);
                sub_685910((__int64)v22, 0);
              }
              if ( v105 )
              {
                v83 = v9;
                v23 = v105;
                do
                {
                  v24 = (__m128i *)v23;
                  v23 = (const __m128i *)v23->m128i_i64[0];
                  sub_725130((__int64 *)v24[2].m128i_i64[1]);
                  sub_82D8A0((_QWORD *)v24[7].m128i_i64[1]);
                  v24->m128i_i64[0] = (__int64)qword_4D03C68;
                  qword_4D03C68 = v24->m128i_i64;
                }
                while ( v23 );
                v9 = v83;
              }
              goto LABEL_33;
            }
            if ( v100 )
            {
              sub_6E6000();
LABEL_33:
              sub_6E6840((__int64)a1);
              goto LABEL_34;
            }
            a2 = v97;
            sub_845370(a1, v97, a3);
            v37 = sub_8D4D20(v8);
            v38 = *((_BYTE *)a1 + 17) == 2;
            v101 = v37;
            v84 = v91;
            if ( !v38 )
              v84 = sub_6ED0A0((__int64)a1);
            v88 = v91;
          }
LABEL_34:
          if ( !*((_BYTE *)a1 + 16) )
            goto LABEL_54;
          v25 = (const __m128i *)*a1;
          for ( i = *(unsigned __int8 *)(*a1 + 140LL); (_BYTE)i == 12; i = v25[8].m128i_u8[12] )
            v25 = (const __m128i *)v25[10].m128i_i64[0];
          if ( !(_BYTE)i )
            goto LABEL_54;
          v26 = v97;
          for ( j = *(_BYTE *)(v97 + 140); j == 12; j = *(_BYTE *)(v26 + 140) )
            v26 = *(_QWORD *)(v26 + 160);
          if ( !j )
            goto LABEL_120;
          if ( v104 )
          {
            if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
            {
              if ( v96 )
              {
                sub_6F3DD0((__int64)a1, 0, 1, v15, v96, v104);
                goto LABEL_45;
              }
              sub_6F8020((const __m128i *)a1);
              v47 = *((_BYTE *)a1 + 17);
              if ( v47 == 1 )
              {
                if ( !sub_6ED0A0((__int64)a1) )
                  goto LABEL_54;
                v47 = *((_BYTE *)a1 + 17);
              }
              i = *((unsigned __int8 *)a1 + 16);
              if ( v47 == 3 )
              {
                if ( (_BYTE)i != 4 )
                {
                  if ( (_BYTE)i == 3 )
                    sub_6F3BA0((__m128i *)a1, 0);
                  goto LABEL_54;
                }
                goto LABEL_150;
              }
              if ( (_BYTE)i )
              {
LABEL_150:
                v48 = (const __m128i *)*a1;
                for ( i = *(unsigned __int8 *)(*a1 + 140LL); (_BYTE)i == 12; i = v48[8].m128i_u8[12] )
                  v48 = (const __m128i *)v48[10].m128i_i64[0];
                if ( (_BYTE)i )
                  sub_6E68E0(0x9Eu, (__int64)a1);
              }
            }
            else
            {
              sub_6F3DD0((__int64)a1, v96 == 0, v96, v15, v14, v104);
              if ( v96 )
              {
LABEL_45:
                if ( (*((_BYTE *)a1 + 17) == 2 || sub_6ED0A0((__int64)a1))
                  && !(unsigned int)sub_8D3A70(*a1)
                  && !(unsigned int)sub_8D3D40(*a1)
                  && *((_BYTE *)a1 + 16) )
                {
                  v28 = (const __m128i *)*a1;
                  for ( i = *(unsigned __int8 *)(*a1 + 140LL); (_BYTE)i == 12; i = v28[8].m128i_u8[12] )
                    v28 = (const __m128i *)v28[10].m128i_i64[0];
                  if ( (_BYTE)i )
                    sub_845EB0((_DWORD)a1, v97, v8, (_DWORD)a3, v9, a6, (__int64)&v98);
                }
              }
            }
LABEL_54:
            v29 = v90;
            if ( v90 )
            {
              sub_844910(a1, (v9 & 0x200) != 0);
            }
            else if ( (v9 & 2) != 0 )
            {
              v14 = v104;
              if ( !v104
                && *((_BYTE *)a1 + 16) == 1
                && ((unsigned int)sub_6EFD60(a1[18], &v107, i, v15, v104, v90) || (unsigned int)sub_6EFE40(
                                                                                                  a1[18],
                                                                                                  &v107)) )
              {
                v39 = (_DWORD)v107 == 0 ? 836 : 430;
                if ( sub_6E53E0(5, v39, (_DWORD *)a1 + 17) )
                  sub_684B30(v39, (_DWORD *)a1 + 17);
              }
            }
            if ( !a4 && (*((_BYTE *)a1 + 16) != 2 || !(unsigned int)sub_8D32E0(a1[34])) )
            {
              if ( !v104
                || v93 != 3
                || !(unsigned int)sub_8D3EA0(v97)
                || *((_BYTE *)a1 + 16) != 1
                || (v43 = a1[18], *(_BYTE *)(v43 + 24) != 1)
                || *(_BYTE *)(v43 + 56) != 3 )
              {
                sub_6FF600((unsigned __int8 *)a1, v96, i, v15, v14, v29);
              }
            }
            return sub_6E4BC0((__int64)a1, (__int64)v109);
          }
          if ( !v88 )
          {
            if ( v99 && !v96 && (unsigned int)sub_8D3A70(v97) )
            {
LABEL_191:
              v86 = qword_4D0495C | HIDWORD(qword_4D0495C);
              if ( qword_4D0495C )
                v86 = sub_8319F0((__int64)a1, 0);
              if ( v99 | v91 || dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) || dword_4D04460 )
                goto LABEL_199;
              if ( dword_4F077BC )
              {
                if ( qword_4F077A8 > 0x9D6Bu )
                {
LABEL_199:
                  if ( *((_BYTE *)a1 + 17) == 2 )
                  {
                    v71 = (const __m128i *)*a1;
                    if ( *(_BYTE *)(*a1 + 140LL) == 12 )
                    {
                      do
                        v71 = (const __m128i *)v71[10].m128i_i64[0];
                      while ( v71[8].m128i_i8[12] == 12 );
                    }
                    if ( !(unsigned int)sub_8D32E0(v71) )
                    {
                      if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v71) )
                        sub_8AE000(v71);
                      if ( (v71[8].m128i_i8[13] & 0x20) == 0 )
                        sub_6F9770((__int64)a1, v96, v50, v51, v52, v72);
                    }
                  }
                  sub_831820((__m128i *)a1, (__int64)v87, v50, v51, v52);
                  if ( !v99 )
                  {
                    if ( v101 || !v84 )
                      goto LABEL_54;
                    if ( qword_4D0495C )
                      v68 = (v86 != 0) | (unsigned __int8)v90 ^ 1;
                    else
                      v68 = unk_4D04950;
                    v69 = (_DWORD *)a1 + 17;
                    if ( v68 )
                    {
                      if ( v103 )
                        goto LABEL_269;
                    }
                    else
                    {
                      if ( v103 )
                      {
                        if ( !(unk_4D04950 | (unsigned int)qword_4D0495C | HIDWORD(qword_4D0495C)) )
                        {
                          sub_6E5C80(8, 0x2A2u, v69);
                          goto LABEL_120;
                        }
LABEL_269:
                        sub_6E5C80(5, 0x2A2u, v69);
                        goto LABEL_54;
                      }
                      if ( !dword_4D0435C )
                      {
                        sub_6E5C80(8, 0x1CDu, v69);
                        goto LABEL_120;
                      }
                    }
                    sub_6E5C80(5, 0x1CDu, v69);
                    goto LABEL_54;
                  }
                  if ( !(unsigned int)sub_6E5430() )
                    goto LABEL_120;
                  v42 = (_DWORD *)a1 + 17;
                  if ( v89 )
                  {
                    v70 = v97;
                    if ( *(_BYTE *)(v97 + 140) == 12 )
                    {
                      do
                        v70 = *(_QWORD *)(v70 + 160);
                      while ( *(_BYTE *)(v70 + 140) == 12 );
                    }
                    else
                    {
                      v70 = v97;
                    }
                    sub_685360(0x174u, v42, v70);
                    goto LABEL_120;
                  }
                  goto LABEL_119;
                }
              }
              else if ( !dword_4D04964 )
              {
                goto LABEL_199;
              }
              sub_83E750(v94, 0, (FILE *)((char *)a1 + 68));
              goto LABEL_199;
            }
LABEL_115:
            v40 = HIDWORD(qword_4D0495C);
            v41 = qword_4D0495C;
            if ( !qword_4D0495C || v90 )
            {
              if ( v99 )
              {
                if ( !(unsigned int)sub_6E5430() )
                {
LABEL_120:
                  sub_6E6840((__int64)a1);
                  goto LABEL_54;
                }
                v42 = (_DWORD *)a1 + 17;
LABEL_119:
                sub_6861A0(0x1B1u, v42, v8, (__int64)v94);
                goto LABEL_120;
              }
              v92 = 0;
            }
            else
            {
              v92 = 1;
            }
            if ( !v96 )
            {
LABEL_160:
              if ( !(v41 | unk_4D04950 | v101 | v40) )
              {
                if ( v84 )
                {
                  v49 = -(v103 == 0);
                  LOBYTE(v49) = v49 & 0x2B;
                  sub_6E68E0(v49 + 674, (__int64)a1);
                  goto LABEL_54;
                }
                if ( (unsigned int)sub_6E5430() )
                {
                  v78 = -(v103 == 0);
                  LOBYTE(v78) = v78 & 0x11;
                  sub_6861A0(v78 + 673, (_DWORD *)a1 + 17, v8, (__int64)v94);
                }
                goto LABEL_120;
              }
              if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
              {
                sub_6E68E0(0x1CBu, (__int64)a1);
                goto LABEL_54;
              }
              if ( !(_DWORD)qword_4F077B4 || (v61 = v9, !qword_4F077A0) )
                v61 = v9 & 0xFFFFFBFF;
              sub_845EB0((_DWORD)a1, v97, v8, (_DWORD)a3, v61, a6, (__int64)&v98);
              v15 = v98;
              i = v82;
              if ( v98 )
                goto LABEL_54;
              if ( v101 )
              {
                sub_6E5C80(4, 0x154u, (_DWORD *)a1 + 17);
                goto LABEL_54;
              }
              if ( HIDWORD(qword_4D0495C) )
              {
                if ( v92
                  || (_DWORD)qword_4D0495C && qword_4F04C50
                  || (unsigned int)sub_8319F0((__int64)a1, &v107) && *(_BYTE *)(*(_QWORD *)(v107 + 56) + 48LL) == 5 )
                {
LABEL_181:
                  v62 = -(v103 == 0);
                  LOBYTE(v62) = v62 & 0xA;
                  v63 = v62 + 672;
                  if ( sub_6E53E0(5, v63, (_DWORD *)a1 + 17) )
                    sub_684B30(v63, (_DWORD *)a1 + 17);
                  goto LABEL_54;
                }
              }
              else
              {
                if ( !(_DWORD)qword_4D0495C )
                {
                  v79 = -(v103 == 0);
                  LOBYTE(v79) = v79 & 0xA;
                  v80 = unk_4F07470;
                  sub_6E5C80(unk_4F07470, v79 + 672, (_DWORD *)a1 + 17);
                  if ( v80 != 8 )
                    goto LABEL_54;
                  goto LABEL_261;
                }
                if ( v92 || qword_4F04C50 )
                  goto LABEL_181;
              }
              if ( v84 )
              {
                v73 = -(v103 == 0);
                LOBYTE(v73) = v73 & 0x2B;
                sub_6E68E0(v73 + 674, (__int64)a1);
              }
              else
              {
                if ( (unsigned int)sub_6E5430() )
                {
                  v81 = -(v103 == 0);
                  LOBYTE(v81) = v81 & 0x11;
                  sub_6861A0(v81 + 673, (_DWORD *)a1 + 17, v8, (__int64)v94);
                }
                sub_6E6840((__int64)a1);
              }
LABEL_261:
              v98 = 1;
              goto LABEL_54;
            }
            if ( (unsigned int)sub_8D2310(v97) )
            {
              if ( v84 )
              {
                if ( (unsigned int)sub_6E5430() )
                  sub_6851C0(0x7Eu, (_DWORD *)a1 + 17);
                goto LABEL_120;
              }
            }
            else if ( v84 )
            {
LABEL_224:
              v40 = HIDWORD(qword_4D0495C);
              v41 = qword_4D0495C;
              goto LABEL_160;
            }
            if ( v93 != 3 && ((unsigned int)sub_8DF8D0(v97, v94) || (unsigned int)sub_82EAE0()) )
            {
              if ( (unsigned int)sub_6E5430() )
                sub_6851C0(0x6E8u, (_DWORD *)a1 + 17);
              goto LABEL_120;
            }
            goto LABEL_224;
          }
          v44 = *((_BYTE *)a1 + 17);
          if ( v44 != 1 )
          {
LABEL_133:
            if ( v44 == 3 )
            {
              if ( !v106 )
              {
                if ( (v9 & 4) != 0
                  && *((_BYTE *)a1 + 16) == 1
                  && !(unsigned int)sub_696840((__int64)a1)
                  && (unsigned int)sub_829FF0((__m128i *)a1, a2, v64, v65, v66, v67)
                  && sub_6E53E0(7, a6, (_DWORD *)a1 + 17) )
                {
                  sub_684AA0(7u, 0xAECu, (_DWORD *)a1 + 17);
                }
                goto LABEL_136;
              }
            }
            else if ( !v106 )
            {
              if ( (unsigned int)sub_8D3A70(v97) )
                goto LABEL_191;
              if ( !sub_6ED0A0((__int64)a1) )
              {
                if ( !v84 || !(unsigned int)sub_8D3410(v97) )
                  goto LABEL_115;
                if ( v90 )
                {
                  a2 = (v9 >> 9) & 1;
                  sub_844910(a1, a2);
                }
                sub_6FB030((__int64)a1, a2, v53, v54, v55, v56);
                sub_6F9270((const __m128i *)a1, a2, v57, v58, v59, v60);
              }
              sub_6F7690((const __m128i *)a1, (__int64)v87);
              goto LABEL_54;
            }
            sub_82F430(v106, a1[17], v109, 0, 0, 0, 1, 0, (__m128i *)a1, (int *)&v107, 0);
LABEL_136:
            if ( !(unsigned int)sub_8DBCE0(*a1, v97) )
            {
              v45 = 7;
              if ( dword_4F077BC )
                v45 = (_DWORD)qword_4F077B4 == 0 ? 5 : 7;
              sub_6E5C80(v45, 0x343u, (_DWORD *)a1 + 17);
            }
            goto LABEL_54;
          }
          if ( sub_6ED0A0((__int64)a1) )
          {
            v44 = *((_BYTE *)a1 + 17);
            goto LABEL_133;
          }
          if ( (v9 & 4) != 0 )
          {
            if ( *((_BYTE *)a1 + 16) == 1
              && !(unsigned int)sub_696840((__int64)a1)
              && (unsigned int)sub_829FF0((__m128i *)a1, a2, v74, v75, v76, v77) )
            {
              if ( sub_6E53E0(7, a6, (_DWORD *)a1 + 17) )
              {
                sub_684AA0(7u, 0xAECu, (_DWORD *)a1 + 17);
                if ( sub_67D370((int *)0xAEC, 7u, (_DWORD *)a1 + 17) )
                  goto LABEL_274;
              }
              goto LABEL_236;
            }
            if ( !(unsigned int)sub_6FE220((__int64)a1, a2) )
            {
              if ( (unsigned int)sub_8D3A70(v97) )
              {
                if ( sub_8D5CE0(v94, v97) )
                {
                  if ( sub_6E53E0(7, a6, (_DWORD *)a1 + 17) )
                  {
                    sub_686040(7u, a6, (_DWORD *)a1 + 17, (__int64)v94, v8);
                    if ( sub_67D370((int *)a6, 7u, (_DWORD *)a1 + 17) )
LABEL_274:
                      sub_6E6260(a1);
                  }
                }
              }
LABEL_236:
              sub_6F7690((const __m128i *)a1, (__int64)v87);
              if ( v102 )
                sub_6E5A30(a1[11], 32, 4128);
              goto LABEL_54;
            }
          }
          else if ( !(unsigned int)sub_6FE220((__int64)a1, a2) )
          {
            goto LABEL_236;
          }
          if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0 )
          {
            if ( dword_4F06970 )
              sub_6E68E0(0x11Cu, (__int64)a1);
            else
              sub_69D070(0x11Cu, (_DWORD *)a1 + 17);
          }
          else
          {
            sub_6E5C80(4, 0x11Cu, (_DWORD *)a1 + 17);
          }
          goto LABEL_236;
        }
      }
      a2 = (__int64)v94;
      if ( v94[8].m128i_i8[12] == 12 )
      {
        do
          a2 = *(_QWORD *)(a2 + 160);
        while ( *(_BYTE *)(a2 + 140) == 12 );
      }
      v31 = v97;
      if ( *(_BYTE *)(v97 + 140) == 12 )
      {
        do
          v31 = *(_QWORD *)(v31 + 160);
        while ( *(_BYTE *)(v31 + 140) == 12 );
      }
      else
      {
        v31 = v97;
      }
      v87 = (__m128i *)v94;
      if ( !(unsigned int)sub_8D0520(v31, a2) )
      {
        a2 = 0;
        if ( (v94[8].m128i_i8[12] & 0xFB) == 8 )
          a2 = (unsigned int)sub_8D4C10(v94, dword_4F077C4 != 2);
        v32 = (const __m128i *)v97;
        if ( *(_BYTE *)(v97 + 140) == 12 )
        {
          do
            v32 = (const __m128i *)v32[10].m128i_i64[0];
          while ( v32[8].m128i_i8[12] == 12 );
        }
        else
        {
          v32 = (const __m128i *)v97;
        }
        v87 = sub_73C570(v32, a2);
      }
      goto LABEL_13;
    }
    v110.m128i_i64[0] = a1[18];
LABEL_78:
    if ( !a3 || (a3[16] & 0x28) != 0x20 )
      goto LABEL_6;
    goto LABEL_87;
  }
  v33 = a1[18];
  v110.m128i_i64[0] = a1[18];
  if ( a3 && (a3[16] & 0x28) == 0x20 )
    goto LABEL_87;
  v34 = v9 | 0x8000;
  if ( !a4 )
    v34 = v9;
  sub_839D30(v33, (__m128i *)a2, 0, 0, 0, v34, 1, 1, a4, (__int64)a1, 0, 0);
  return sub_6E4BC0((__int64)a1, (__int64)v109);
}
