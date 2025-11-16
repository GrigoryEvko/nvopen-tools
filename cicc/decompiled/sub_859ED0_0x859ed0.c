// Function: sub_859ED0
// Address: 0x859ed0
//
_DWORD *__fastcall sub_859ED0(unsigned __int64 a1, unsigned int *a2)
{
  unsigned int v2; // r12d
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned int *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r9
  unsigned __int64 v10; // rdi
  __int64 v11; // rsi
  int v12; // r13d
  __int64 v13; // r8
  __int64 v14; // rdx
  unsigned int v15; // r14d
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _DWORD *result; // rax
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rcx
  unsigned __int16 v24; // ax
  __int64 v25; // rdx
  bool v26; // cf
  bool v27; // zf
  const char *v28; // rdi
  __int64 v29; // r9
  bool v30; // cf
  bool v31; // zf
  __int64 v32; // r8
  __int64 *v33; // rax
  _DWORD *v34; // rdx
  _BOOL8 v35; // rcx
  __m128i *v36; // r9
  __int64 *v37; // r10
  __int64 v38; // r8
  char v39; // al
  unsigned __int8 v40; // al
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // r9
  char v46; // al
  int v47; // r8d
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // rdx
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  const char *v57; // r9
  __int64 v58; // r8
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r9
  __int64 v63; // r8
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // rdx
  unsigned __int8 v69; // al
  bool v70; // cf
  bool v71; // zf
  __int64 v72; // rax
  unsigned __int8 v73; // dl
  unsigned __int8 v74; // dl
  __int64 v75; // rcx
  char *v76; // rdi
  __int64 v77; // rdx
  char v78; // al
  __int64 v79; // rax
  char v80; // al
  __int64 v81; // rdi
  __int64 v82; // rdx
  char v83; // al
  __int64 v84; // rcx
  __int64 v85; // rdx
  int v86; // eax
  const char *v87; // rcx
  char v88; // al
  char v89; // al
  int v90; // eax
  __int64 *v91; // r10
  unsigned int v92; // r8d
  int v93; // [rsp+Ch] [rbp-D4h]
  __int64 *v94; // [rsp+10h] [rbp-D0h]
  int v95; // [rsp+18h] [rbp-C8h]
  unsigned int v96; // [rsp+1Ch] [rbp-C4h]
  unsigned int v97; // [rsp+1Ch] [rbp-C4h]
  unsigned int v98; // [rsp+1Ch] [rbp-C4h]
  unsigned int v99; // [rsp+20h] [rbp-C0h]
  int v100; // [rsp+20h] [rbp-C0h]
  __int64 *v101; // [rsp+20h] [rbp-C0h]
  unsigned int v102; // [rsp+20h] [rbp-C0h]
  __int64 *v103; // [rsp+20h] [rbp-C0h]
  int v104; // [rsp+28h] [rbp-B8h]
  int v105; // [rsp+28h] [rbp-B8h]
  const char *v106; // [rsp+28h] [rbp-B8h]
  __int64 *v107; // [rsp+28h] [rbp-B8h]
  __int64 *v108; // [rsp+28h] [rbp-B8h]
  const char *v109; // [rsp+28h] [rbp-B8h]
  char v110; // [rsp+30h] [rbp-B0h]
  int v111; // [rsp+30h] [rbp-B0h]
  int v112; // [rsp+30h] [rbp-B0h]
  __int32 v113; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v114; // [rsp+30h] [rbp-B0h]
  const char *v115; // [rsp+30h] [rbp-B0h]
  const char *v116; // [rsp+30h] [rbp-B0h]
  __int64 *v117; // [rsp+30h] [rbp-B0h]
  __int64 *v118; // [rsp+30h] [rbp-B0h]
  unsigned int v119; // [rsp+38h] [rbp-A8h]
  unsigned __int16 v120; // [rsp+3Ch] [rbp-A4h]
  __int16 v121; // [rsp+3Eh] [rbp-A2h]
  int v122; // [rsp+40h] [rbp-A0h]
  __int64 v123; // [rsp+44h] [rbp-9Ch]
  int v124; // [rsp+4Ch] [rbp-94h]
  unsigned int v125; // [rsp+5Ch] [rbp-84h] BYREF
  __int64 v126; // [rsp+60h] [rbp-80h] BYREF
  __int64 v127; // [rsp+68h] [rbp-78h] BYREF
  __m128i v128[7]; // [rsp+70h] [rbp-70h] BYREF

  v2 = 14;
  v125 = 0;
  v123 = qword_4D03D1C;
  v122 = dword_4D03D08;
  v124 = dword_4F07508[0];
  v121 = dword_4F07508[1];
  v119 = dword_4F061D8;
  v120 = unk_4F061DC;
  v127 = *(_QWORD *)&dword_4F063F8;
  dword_4D03D18 = 1;
  qword_4D03D1C = 0x100000000LL;
  dword_4D03D08 = 0;
  dword_4D03CC0[0] = 0;
  sub_7B8190();
  dword_4D03CF8 = 1;
  ++*(_BYTE *)(qword_4F061C8 + 18LL);
  dword_4D03CE0 = 0;
  dword_4D03CF4 = 1;
  sub_7B8B50(a1, a2, v3, v4, v5, v6);
  dword_4D03CF8 = 0;
  qword_4F5FCC0 = *(_QWORD *)&dword_4F063F8;
  if ( word_4F06418[0] != 10 )
  {
    v2 = 15;
    if ( word_4F06418[0] != 13 )
    {
      v2 = 22;
      if ( word_4F06418[0] == 1 )
      {
        v2 = sub_855880();
        v10 = (unsigned int)dword_4D03C84;
        if ( !dword_4D03C84 )
          goto LABEL_5;
LABEL_29:
        sub_852DF0();
        goto LABEL_5;
      }
    }
  }
  v10 = (unsigned int)dword_4D03C84;
  if ( dword_4D03C84 )
    goto LABEL_29;
LABEL_5:
  v11 = dword_4D03C98[0];
  if ( dword_4D03C98[0]
    && (qword_4F064B0[11] & 1) == 0
    && (v7 = &dword_4D03CA0, v8 = dword_4D03CA0, *((_DWORD *)qword_4F064B0 + 20) == dword_4D03CA0)
    && WORD2(v127) == unk_4D03CA4 )
  {
    v12 = 1;
  }
  else
  {
    v12 = dword_4F5FCB8;
    if ( !dword_4F5FCB8 )
      goto LABEL_9;
    v12 = 0;
  }
  if ( v2 == 8 || v2 == 20 )
  {
    v12 = 0;
    dword_4F5FCB8 = 0;
    unk_4D03C80 = 1;
    v13 = dword_4D03CB0[0];
    if ( dword_4D03CB0[0] )
      goto LABEL_15;
    goto LABEL_10;
  }
LABEL_9:
  v13 = dword_4D03CB0[0];
  if ( !dword_4D03CB0[0] )
  {
LABEL_10:
    v14 = v2;
    switch ( v2 )
    {
      case 0u:
        v112 = v13;
        v46 = sub_7AFE70();
        v47 = v112;
        if ( v46 )
        {
          if ( v46 != 1 )
            goto LABEL_111;
          goto LABEL_153;
        }
        v68 = qword_4F06408 + 1LL;
        v69 = *(_BYTE *)(qword_4F06408 + 1LL);
        if ( v69 != 32 )
          goto LABEL_137;
        do
        {
          do
            v69 = *(_BYTE *)++v68;
          while ( v69 == 32 );
LABEL_137:
          ;
        }
        while ( v69 == 9 );
        v70 = v69 < 0x21u;
        v71 = v69 == 33;
        if ( v69 == 33 )
        {
          v72 = v68 + 1;
          v73 = *(_BYTE *)(v68 + 1);
          if ( v73 == 9 || (v70 = v73 < 0x20u, v71 = v73 == 32) )
          {
            do
            {
              do
                v74 = *(_BYTE *)++v72;
              while ( v74 == 32 );
              v70 = v74 < 9u;
              v71 = v74 == 9;
            }
            while ( v74 == 9 );
          }
          v47 = 1;
        }
        else
        {
          v72 = v68;
        }
        v75 = 7;
        v76 = "defined";
        v11 = v72;
        do
        {
          if ( !v75 )
            break;
          v70 = *(_BYTE *)v11 < (unsigned __int8)*v76;
          v71 = *(_BYTE *)v11++ == (unsigned __int8)*v76++;
          --v75;
        }
        while ( v71 );
        if ( (!v70 && !v71) != v70 )
          goto LABEL_153;
        v77 = v72 + 7;
        v78 = *(_BYTE *)(v72 + 7);
        if ( v78 == 9 || v78 == 32 )
        {
          do
          {
            do
              v78 = *(_BYTE *)++v77;
            while ( v78 == 32 );
          }
          while ( v78 == 9 );
        }
        if ( v78 != 40 )
          goto LABEL_153;
        v80 = *(_BYTE *)(v77 + 1);
        v81 = v77 + 1;
        if ( v80 != 32 )
          goto LABEL_164;
        do
        {
          do
            v80 = *(_BYTE *)++v81;
          while ( v80 == 32 );
LABEL_164:
          ;
        }
        while ( v80 == 9 );
        if ( (v80 & 0xDF) == 0 || (v80 & 0xDF) == 9 )
        {
          v82 = v81;
          v11 = 0;
        }
        else
        {
          v82 = v81;
          do
            v80 = *(_BYTE *)++v82;
          while ( (v80 & 0xDF) != 0 && (v80 & 0xDF) != 9 );
          v11 = v82 - v81;
          if ( v80 == 32 || v80 == 9 )
          {
            do
            {
              do
                v80 = *(_BYTE *)++v82;
              while ( v80 == 9 );
            }
            while ( v80 == 32 );
          }
        }
        if ( v80 != 41 )
          goto LABEL_153;
        v83 = *(_BYTE *)(v82 + 1);
        v84 = v82 + 1;
        if ( v83 == 32 || v83 == 9 )
        {
          do
          {
            do
              v83 = *(_BYTE *)++v84;
            while ( v83 == 9 );
          }
          while ( v83 == 32 );
        }
        if ( v83 || *(_BYTE *)(v84 + 1) != 2 || (v105 = v47, !(unsigned int)sub_822080(v81, v11, &v126, v128)) )
        {
LABEL_153:
          sub_7AFEC0(2);
          goto LABEL_111;
        }
        sub_7AFEC0(3);
        v85 = qword_4F064B0[12];
        if ( v105 )
          *(_BYTE *)(v85 + 8) |= 8u;
        else
          *(_BYTE *)(v85 + 8) |= 4u;
        *(_QWORD *)(qword_4F064B0[12] + 16LL) = *(_QWORD *)(v128[0].m128i_i64[0] + 8);
LABEL_111:
        v10 = (unsigned __int64)v128;
        sub_855790(v128, (unsigned int *)v11);
        v113 = v128[0].m128i_i32[0];
        sub_855540((__int64)v128, v11, v48, v49, v50, v51);
        if ( !v113 )
        {
          v10 = 1;
          sub_856950(1u, v11, v52, v23, v21, v22);
        }
        goto LABEL_33;
      case 1u:
        v10 = 1;
        sub_856C60(1);
        goto LABEL_33;
      case 2u:
        v10 = 0;
        sub_856C60(0);
        goto LABEL_33;
      case 3u:
      case 5u:
      case 6u:
        v10 = 1;
        sub_856D20(1u, (unsigned int *)dword_4D03C98[0]);
        v23 = v2;
        if ( ((1LL << v2) & 0x100197) == 0 )
          goto LABEL_31;
        goto LABEL_33;
      case 4u:
        v10 = 1;
        sub_856E70(1u);
        goto LABEL_33;
      case 7u:
        sub_855E20(v10, (unsigned int *)dword_4D03C98[0], v2, v8, v13, v9);
        goto LABEL_33;
      case 8u:
        v11 = (__int64)&v125;
        v10 = 0;
        sub_8574B0(0, &v125);
        goto LABEL_33;
      case 9u:
        sub_8200E0(v10, (unsigned int *)dword_4D03C98[0], v2, v8, v13, v9);
        goto LABEL_31;
      case 0xAu:
        if ( (unsigned __int16)sub_7B8B50(v10, (unsigned int *)dword_4D03C98[0], v2, v8, v13, v9) == 1 )
        {
          v57 = qword_4F06410;
          v58 = qword_4F06400;
          v128[0].m128i_i64[0] = qword_4F06400;
          if ( unk_4F061E4 )
          {
            v79 = sub_7B3EE0((unsigned __int8 *)qword_4F06410, v128);
            v58 = v128[0].m128i_i64[0];
            v57 = (const char *)v79;
          }
          if ( dword_4D04788 && v58 == 11 )
          {
            if ( !memcmp(v57, "__VA_ARGS__", 0xBu) )
            {
              v115 = v57;
              sub_6851C0(0x3C9u, dword_4F07508);
              v58 = v128[0].m128i_i64[0];
              v57 = v115;
            }
          }
          else if ( unk_4D041B8 && v58 == 10 && !memcmp(v57, "__VA_OPT__", 0xAu) )
          {
            v116 = v57;
            sub_6851C0(0xB7Bu, dword_4F07508);
            v58 = v128[0].m128i_i64[0];
            v57 = v116;
          }
          v11 = v58;
          v10 = (unsigned __int64)v57;
          v59 = sub_87A510(v57, v58, &qword_4D04A00);
          v63 = v59;
          if ( v59 )
          {
            if ( (**(_BYTE **)(v59 + 88) & 2) != 0 && !HIDWORD(qword_4F077B4) )
            {
              v11 = 45;
              v10 = 7;
              sub_684AC0(7u, 0x2Du);
            }
            else
            {
              v11 = v59;
              v114 = v59;
              sub_8767A0(4, v59, &dword_4F063F8, 1);
              v10 = v114;
              sub_881DB0(v114);
            }
          }
          sub_7B8B50(v10, (unsigned int *)v11, v60, v61, v63, v62);
          if ( word_4F06418[0] != 10 )
            sub_855DA0(v10, v11, v64, v65, v66, v67);
        }
        else
        {
          v10 = 40;
          sub_6851D0(0x28u);
          dword_4D03CE0 = 1;
        }
        goto LABEL_31;
      case 0xBu:
        v10 = 0;
        sub_858F80(0, (unsigned int *)dword_4D03C98[0]);
        goto LABEL_31;
      case 0xCu:
        sub_7BC390();
        sub_685220(0x23u, (__int64)qword_4F06460);
      case 0xDu:
        v99 = v13;
        v10 = (unsigned __int64)v128;
        v111 = dword_4D0493C;
        v33 = sub_8579E0(v128, (unsigned int *)dword_4D03C98[0], v2, v8, v13, (__int64)v128);
        v36 = v128;
        v37 = v33;
        if ( dword_4D0493C && dword_4D04944 )
        {
          v38 = v99;
          if ( v33 )
          {
            v39 = *((_BYTE *)v33 + 8);
            switch ( v39 )
            {
              case 22:
                v10 = 0;
                v108 = v37;
                dword_4D03CF4 = 0;
                unk_4D03CF0 = 1;
                sub_857860(0, (unsigned int *)v11, (__int64)v34, v35, v99, (__int64)v128);
                v37 = v108;
                v38 = v99;
                v39 = *((_BYTE *)v108 + 8);
                goto LABEL_82;
              case 37:
                v10 = 0;
                v117 = v37;
                sub_8819D0(0, v11, v34, v35, v99, v128);
                v91 = v117;
                v92 = v99;
                if ( *((_BYTE *)v117 + 8) != 28 )
                  goto LABEL_31;
                break;
              case 38:
                v10 = 0;
                v118 = v37;
                sub_8860A0(0, v11, v34, v35, v99, v128);
                v91 = v118;
                v92 = v99;
                if ( *((_BYTE *)v118 + 8) != 28 )
                {
LABEL_31:
                  if ( (unsigned __int8)sub_7AFE70() <= 1u )
                  {
                    v10 = 2;
                    sub_7AFEC0(2);
                  }
LABEL_33:
                  v24 = word_4F06418[0];
                  if ( word_4F06418[0] != 10 )
                  {
                    v25 = (unsigned int)dword_4D03CE0;
                    if ( !dword_4D03CE0 )
                    {
                      v11 = 14;
                      v10 = 7;
                      sub_684AA0(7u, 0xEu, &dword_4F063F8);
                      v24 = word_4F06418[0];
                    }
                    if ( (unsigned __int16)(v24 - 9) > 1u )
                    {
                      do
                        sub_7B8B50(v10, (unsigned int *)v11, v25, v23, v21, v22);
                      while ( (unsigned __int16)(word_4F06418[0] - 9) > 1u );
                    }
                  }
                  goto LABEL_22;
                }
                break;
              default:
LABEL_82:
                if ( v39 != 28 )
                  goto LABEL_83;
                if ( qword_4F06400 != 3 )
                  goto LABEL_83;
                v10 = (unsigned __int64)"GCC";
                v97 = v38;
                v101 = v37;
                v11 = (__int64)qword_4F06410;
                v106 = qword_4F06410;
                v86 = strncmp("GCC", qword_4F06410, 3u);
                v87 = v106;
                v37 = v101;
                v38 = v97;
                if ( v86 )
                  goto LABEL_83;
                goto LABEL_191;
            }
            if ( qword_4F06400 == 3 )
            {
              v10 = (unsigned __int64)"GCC";
              v98 = v92;
              v103 = v91;
              v11 = (__int64)qword_4F06410;
              v109 = qword_4F06410;
              v111 = strncmp("GCC", qword_4F06410, 3u);
              if ( !v111 )
              {
                v87 = v109;
                v37 = v103;
                LODWORD(v38) = v98;
LABEL_191:
                v88 = v87[3];
                v10 = (unsigned __int64)(v87 + 3);
                if ( v88 == 32 || v88 == 9 )
                {
                  do
                  {
                    do
                      v89 = *(_BYTE *)++v10;
                    while ( v89 == 9 );
                  }
                  while ( v89 == 32 );
                }
                v11 = (__int64)"system_header";
                v102 = v38;
                v107 = v37;
                v90 = strcmp((const char *)v10, "system_header");
                v37 = v107;
                v38 = v102;
                if ( !v90 )
                {
                  if ( !v111 )
                  {
LABEL_90:
                    dword_4D03CF4 = 1;
                    goto LABEL_31;
                  }
                  v38 = 1;
                  goto LABEL_84;
                }
LABEL_83:
                if ( !v111 )
                  goto LABEL_31;
LABEL_84:
                v34 = &dword_4D03D0C;
                v104 = qword_4D03D1C;
                v100 = dword_4D03D08;
                v96 = HIDWORD(qword_4D03D1C);
                v95 = dword_4D03D0C;
                v40 = *((_BYTE *)v37 + 17);
                LODWORD(qword_4D03D1C) = (v40 & 0x20) != 0;
                v35 = (v40 & 0x40) != 0;
                dword_4D03D08 = (v40 & 0x40) != 0;
                v11 = (__int64)&qword_4D03D1C + 4;
                dword_4D03D0C = (v40 & 0x40) != 0;
                HIDWORD(qword_4D03D1C) = v40 >> 7;
                dword_4D03CF4 = 0;
                unk_4D03CF0 = 1;
                if ( (unsigned __int16)(word_4F06418[0] - 9) <= 1u )
                {
                  unk_4D03CF0 = 0;
                  goto LABEL_88;
                }
                v94 = v37;
                v93 = v38;
                do
LABEL_86:
                  sub_7B8B50(v10, (unsigned int *)v11, (__int64)v34, v35, v38, (__int64)v36);
                while ( (unsigned __int16)(word_4F06418[0] - 9) > 1u );
                LODWORD(v38) = v93;
                unk_4D03CF0 = 0;
                if ( !v94 )
                {
LABEL_89:
                  if ( !(_DWORD)v38 )
                    goto LABEL_31;
                  goto LABEL_90;
                }
LABEL_88:
                LODWORD(qword_4D03D1C) = v104;
                dword_4D03D08 = v100;
                v11 = v96;
                HIDWORD(qword_4D03D1C) = v96;
                dword_4D03D0C = v95;
                goto LABEL_89;
              }
            }
            goto LABEL_31;
          }
          if ( !v111 )
            goto LABEL_31;
          dword_4D03CF4 = 0;
          unk_4D03CF0 = 1;
          if ( (unsigned __int16)(word_4F06418[0] - 9) > 1u )
          {
            v95 = 0;
            v96 = 0;
            v100 = 0;
            v104 = 0;
            v94 = 0;
            v93 = v38;
            goto LABEL_86;
          }
        }
        else
        {
          if ( v33 && (unsigned __int8)(*((_BYTE *)v33 + 8) - 37) <= 1u || !v111 )
          {
            v11 = (__int64)&v127;
            v10 = (unsigned __int64)v33;
            sub_8578F0((__int64)v33, (unsigned int *)&v127, (unsigned int *)v128, 0, 0, (__int64)v128);
            goto LABEL_31;
          }
          v10 = (unsigned __int64)v33;
          dword_4D03CF4 = 0;
          unk_4D03CF0 = 1;
          sub_8578F0((__int64)v33, (unsigned int *)&v127, (unsigned int *)v128, 0, 0, (__int64)v128);
          v11 = (__int64)&dword_4D03CF4;
          unk_4D03CF0 = 1;
          dword_4D03CF4 = 0;
          while ( (unsigned __int16)(word_4F06418[0] - 9) > 1u )
            sub_7B8B50(v10, &dword_4D03CF4, v53, v54, v55, v56);
        }
LABEL_104:
        unk_4D03CF0 = 0;
        goto LABEL_31;
      case 0xEu:
        goto LABEL_31;
      case 0xFu:
        if ( dword_4D04964 )
        {
          v11 = 518;
          sub_684AC0(unk_4F07471, 0x206u);
        }
        v10 = 1;
        sub_858F80(1u, (unsigned int *)v11);
        goto LABEL_31;
      case 0x10u:
        if ( dword_4D04964 )
        {
          v11 = 518;
          v10 = unk_4F07471;
          sub_684AC0(unk_4F07471, 0x206u);
        }
        v45 = (unsigned int)dword_4D0493C;
        if ( !dword_4D0493C )
        {
          v11 = (__int64)&v127;
          v10 = qword_4D03D40[21];
          sub_856400(v10, &v127, &dword_4F063F8, 0, 0);
          goto LABEL_31;
        }
        dword_4D03CF4 = 0;
        unk_4D03CF0 = 1;
        while ( (unsigned __int16)(word_4F06418[0] - 9) > 1u )
          sub_7B8B50(v10, (unsigned int *)v11, v14, v8, v13, v45);
        goto LABEL_104;
      case 0x11u:
        v32 = (unsigned int)dword_4D04964;
        if ( dword_4D04964 )
        {
          v11 = 518;
          v10 = unk_4F07471;
          sub_684AC0(unk_4F07471, 0x206u);
        }
        sub_821840(v10, (unsigned int *)v11, v14, v8, v32, v9);
        goto LABEL_31;
      case 0x12u:
        v10 = (unsigned int)dword_4D04964;
        if ( dword_4D04964 )
        {
          v11 = 518;
          v10 = unk_4F07471;
          sub_684AC0(unk_4F07471, 0x206u);
        }
        sub_821AA0(v10, (unsigned int *)v11, v14, v8, v13, v9);
        goto LABEL_31;
      case 0x14u:
        if ( dword_4D04964 )
          sub_684AC0(unk_4F07471, 0x206u);
        v11 = (__int64)&v125;
        v10 = 1;
        sub_8574B0(1u, &v125);
        goto LABEL_33;
      case 0x15u:
        sub_7BC390();
        v10 = 1105;
        v11 = (__int64)qword_4F06460;
        sub_685190(0x451u, (__int64)qword_4F06460);
        while ( (unsigned __int16)(word_4F06418[0] - 9) > 1u )
          sub_7B8B50(0x451u, (unsigned int *)v11, v41, v42, v43, v44);
        goto LABEL_31;
      case 0x16u:
        v11 = 11;
        v10 = 7;
        sub_684AC0(7u, 0xBu);
        dword_4D03CE0 = 1;
        goto LABEL_31;
      default:
        sub_721090();
    }
  }
LABEL_15:
  v15 = *((_DWORD *)qword_4F064B0 + 20);
  sub_7B8B50(v10, (unsigned int *)dword_4D03C98[0], (__int64)v7, v8, v13, v9);
  if ( v2 == 13 )
  {
    v30 = qword_4F06400 < 7;
    v31 = qword_4F06400 == 7;
    if ( qword_4F06400 == 7 )
    {
      v17 = 7;
      v11 = (__int64)"hdrstop";
      v10 = (unsigned __int64)qword_4F06410;
      do
      {
        if ( !v17 )
          break;
        v30 = *(_BYTE *)v11 < *(_BYTE *)v10;
        v31 = *(_BYTE *)v11++ == *(_BYTE *)v10++;
        --v17;
      }
      while ( v31 );
      if ( (!v30 && !v31) == v30 )
      {
        v19 = dword_4D03C90;
        if ( !dword_4D03C90 )
        {
          if ( !dword_4D03C94 )
            sub_852770();
          goto LABEL_22;
        }
        v110 = 1;
        if ( word_4F06418[0] == 10 )
        {
          if ( !dword_4D03CB0[0] )
            goto LABEL_22;
          goto LABEL_21;
        }
        goto LABEL_18;
      }
    }
  }
  if ( dword_4D03C90 )
  {
    v110 = 0;
    if ( word_4F06418[0] == 10 )
    {
      if ( !dword_4D03CB0[0] )
        goto LABEL_22;
      goto LABEL_60;
    }
    do
LABEL_18:
      sub_7B8B50(v10, (unsigned int *)v11, v16, v17, v18, v19);
    while ( word_4F06418[0] != 10 );
    if ( !dword_4D03CB0[0] )
      goto LABEL_22;
    if ( v110 )
    {
LABEL_21:
      dword_4D03C84 = 1;
      goto LABEL_22;
    }
LABEL_60:
    if ( v15 != dword_4D03C88 || WORD2(v127) != unk_4D03C8C )
      goto LABEL_22;
    goto LABEL_21;
  }
  if ( !dword_4D03C94 )
  {
    if ( v2 != 13 )
      goto LABEL_50;
    v26 = qword_4F06400 < 6;
    v27 = qword_4F06400 == 6;
    if ( qword_4F06400 != 6 )
      goto LABEL_50;
    v17 = 6;
    v11 = (__int64)"no_pch";
    v28 = qword_4F06410;
    do
    {
      if ( !v17 )
        break;
      v26 = *(_BYTE *)v11 < *v28;
      v27 = *(_BYTE *)v11++ == *v28++;
      --v17;
    }
    while ( v27 );
    if ( (!v26 && !v27) == v26 )
    {
      unk_4D03CA8 = 1;
    }
    else
    {
LABEL_50:
      sub_856220(0, v11, v16, v17, v18, v19);
      sub_852630(2, v2, (const char *)qword_4F5FC08, &v127, v15, v29);
    }
  }
LABEL_22:
  --*(_BYTE *)(qword_4F061C8 + 18LL);
  sub_7B8260();
  dword_4D03D18 = 0;
  qword_4D03D1C = v123;
  dword_4D03D08 = v122;
  if ( dword_4D03CC0[0] )
    sub_7B0600((unsigned int)v123);
  dword_4F07508[0] = v124;
  LOWORD(dword_4F07508[1]) = v121;
  sub_854730();
  result = &dword_4F061D8;
  dword_4F061D8 = v119;
  unk_4F061DC = v120;
  if ( dword_4F5FCB8 | v12 && v2 != 8 && v2 != 20 )
  {
    dword_4F5FCB8 = 0;
    sub_852780(v120, (__int64 *)v119);
    return sub_852D60();
  }
  return result;
}
