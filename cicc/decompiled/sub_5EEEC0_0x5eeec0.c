// Function: sub_5EEEC0
// Address: 0x5eeec0
//
__int64 __fastcall sub_5EEEC0(__int64 a1, int a2)
{
  unsigned __int8 v3; // al
  __int64 v4; // rcx
  __int64 v5; // rax
  char v6; // dl
  __int64 v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rdx
  _BOOL8 v11; // rsi
  unsigned __int16 v12; // ax
  int v13; // r13d
  _BOOL4 v14; // ebx
  int v15; // r12d
  unsigned __int16 v16; // ax
  __int64 v17; // rbx
  int v18; // r13d
  __int64 v19; // r12
  unsigned __int64 v20; // rsi
  __int64 v21; // rdi
  int v22; // r13d
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r15
  __int64 v26; // rdx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // r13
  __int64 v32; // r15
  char v33; // al
  char v34; // al
  unsigned __int64 v35; // r13
  unsigned int v36; // r15d
  __int64 v37; // rdx
  __int64 v38; // r8
  __int64 v39; // rax
  __m128i v40; // xmm2
  __m128i v41; // xmm3
  __int64 v42; // rax
  char v43; // dl
  unsigned __int64 v44; // rcx
  __int64 v45; // rax
  __int64 **v46; // r13
  __int64 v47; // rax
  __int64 *v48; // rax
  __int64 *v49; // rdx
  __int64 v50; // rsi
  __int64 *v51; // rdi
  _QWORD **v52; // rax
  __int64 v53; // rax
  char i; // dl
  unsigned __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // r13
  char v58; // r14
  __int64 v59; // r15
  __int64 v60; // rbx
  __int64 v61; // r14
  __int64 v62; // r15
  __int64 v63; // r13
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 v66; // rdi
  __int64 v67; // rax
  __int64 *v68; // rbx
  __int64 v69; // rdi
  __int64 v70; // rax
  __int64 *v71; // rcx
  __int64 v72; // rdx
  char v73; // cl
  __int64 v74; // rax
  char v75; // al
  __m128i v76; // xmm5
  __m128i v77; // xmm6
  __m128i v78; // xmm7
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rdi
  __int64 v84; // rdx
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rdx
  __int64 v88; // [rsp+8h] [rbp-2C8h]
  int v89; // [rsp+14h] [rbp-2BCh]
  int v91; // [rsp+20h] [rbp-2B0h]
  unsigned int v92; // [rsp+24h] [rbp-2ACh]
  char v93; // [rsp+30h] [rbp-2A0h]
  bool v94; // [rsp+35h] [rbp-29Bh]
  bool v95; // [rsp+36h] [rbp-29Ah]
  unsigned __int8 v96; // [rsp+37h] [rbp-299h]
  __int64 v97; // [rsp+38h] [rbp-298h]
  __int64 v98; // [rsp+40h] [rbp-290h]
  __m128i *v99; // [rsp+48h] [rbp-288h]
  __int64 v100; // [rsp+50h] [rbp-280h]
  int v101; // [rsp+58h] [rbp-278h]
  int v102; // [rsp+5Ch] [rbp-274h]
  int v103; // [rsp+60h] [rbp-270h] BYREF
  int v104; // [rsp+64h] [rbp-26Ch] BYREF
  __int64 v105; // [rsp+68h] [rbp-268h] BYREF
  __int64 v106; // [rsp+70h] [rbp-260h] BYREF
  __int64 v107; // [rsp+78h] [rbp-258h] BYREF
  __int64 v108; // [rsp+80h] [rbp-250h] BYREF
  __int64 v109; // [rsp+88h] [rbp-248h] BYREF
  __int64 v110; // [rsp+90h] [rbp-240h] BYREF
  __int64 v111; // [rsp+98h] [rbp-238h] BYREF
  __int64 v112[4]; // [rsp+A0h] [rbp-230h] BYREF
  _OWORD v113[33]; // [rsp+C0h] [rbp-210h] BYREF

  v3 = *(_BYTE *)(a1 + 12);
  v4 = *(_QWORD *)a1;
  v107 = 0;
  v96 = v3;
  v98 = v4;
  v103 = 0;
  v104 = 0;
  v5 = 776LL * dword_4F04C64;
  LOBYTE(v4) = *(_BYTE *)(v5 + qword_4F04C68[0] + 7);
  v6 = v4 & 0xFD;
  v95 = (v4 & 2) != 0;
  LODWORD(v4) = dword_4D0489C & 1;
  v7 = (unsigned int)(2 * v4);
  v8 = (unsigned int)(4 * v4);
  *(_BYTE *)(v5 + qword_4F04C68[0] + 7) = v7 | v6;
  v9 = qword_4F04C68[0] + v5;
  v10 = (unsigned int)v8 | *(_BYTE *)(v9 + 7) & 0xFB;
  v11 = (*(_BYTE *)(v9 + 7) & 4) != 0;
  v94 = (*(_BYTE *)(v9 + 7) & 4) != 0;
  *(_BYTE *)(v9 + 7) = v8 | *(_BYTE *)(v9 + 7) & 0xFB;
  v99 = 0;
  if ( unk_4D0418C )
  {
    v7 = 1;
    v99 = (__m128i *)sub_5CC190(1);
  }
  ++*(_BYTE *)(unk_4F061C8 + 83LL);
  v110 = *(_QWORD *)&dword_4F063F8;
  v12 = word_4F06418[0];
  if ( word_4F06418[0] != 179 )
  {
    v102 = 0;
    goto LABEL_12;
  }
  v111 = qword_4F063F0;
  sub_7B8B50(v7, v11, v10, v8);
  v13 = unk_4D04400;
  if ( unk_4D04400 )
  {
    v15 = sub_869470(&v106);
    ++*(_BYTE *)(unk_4F061C8 + 75LL);
    if ( !(unsigned int)sub_65B9A0(&v104) )
    {
      v102 = 1;
      v14 = v15 == 0;
      if ( !unk_4D04324 )
        goto LABEL_7;
      goto LABEL_38;
    }
    v13 = 1;
    sub_867030(v106);
LABEL_173:
    memset(v113, 0, 0x1D8u);
    *((_QWORD *)&v113[9] + 1) = v113;
    if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
      BYTE2(v113[11]) |= 1u;
    BYTE9(v113[7]) |= 0x40u;
    *((_QWORD *)&v113[1] + 1) = v110;
    if ( a2 )
      BYTE14(v113[7]) |= 4u;
    sub_65F0A0(v113, &v111);
    v19 = *(_QWORD *)&v113[0];
    goto LABEL_179;
  }
  v102 = sub_65B9A0(&v104);
  if ( v102 )
    goto LABEL_173;
  v14 = 0;
  v15 = 1;
  if ( !unk_4D04324 )
    goto LABEL_7;
LABEL_38:
  sub_684AB0(&v110, 880);
LABEL_7:
  v16 = word_4F06418[0];
  if ( dword_4F077C4 == 2 )
  {
    if ( word_4F06418[0] == 1 && (unk_4D04A11 & 2) != 0 || (unsigned int)sub_7C0F00(0, 0) )
      goto LABEL_9;
    v16 = word_4F06418[0];
  }
  else if ( word_4F06418[0] == 1 )
  {
    goto LABEL_9;
  }
  if ( v16 == 183 )
  {
LABEL_9:
    if ( v15 || (v102 & 1) == 0 )
    {
      v12 = word_4F06418[0];
LABEL_12:
      v91 = 0;
      v17 = 0;
      v18 = 0;
      v19 = 0;
      v101 = 0;
      v89 = 0;
      v97 = 0;
      while ( 1 )
      {
        if ( v12 == 183 )
        {
          v20 = (unsigned __int64)v112;
          v21 = (__int64)&v108;
          sub_671BC0(&v108, v112, 1, 0, 0, 0);
          v53 = v108;
          for ( i = *(_BYTE *)(v108 + 140); i == 12; i = *(_BYTE *)(v53 + 140) )
            v53 = *(_QWORD *)(v53 + 160);
          if ( !i )
          {
            v103 = 1;
            v22 = 0;
            v91 = 1;
            goto LABEL_22;
          }
          v91 = 1;
          v22 = v103;
        }
        else
        {
          if ( dword_4F077C4 == 2 )
          {
            if ( (v12 != 1 || (unk_4D04A11 & 2) == 0) && !(unsigned int)sub_7C0F00(3, 0) )
            {
LABEL_68:
              v20 = (unsigned __int64)&dword_4F063F8;
              v21 = 40;
              sub_6851C0(40, &dword_4F063F8);
              v103 = 1;
              goto LABEL_21;
            }
          }
          else if ( v12 != 1 )
          {
            goto LABEL_68;
          }
          if ( !v99 || !sub_736C60(83, v99) )
          {
            if ( !v104 && (!v18 || !unk_4D0418C) )
              goto LABEL_54;
            sub_7ADF70(v112, 0);
            sub_7AE360(v112);
            sub_7B8B50(v112, 0, v28, v29);
            v30 = sub_5CC190(23);
            v31 = v30;
            if ( !v30 )
            {
              sub_7BC000(v112);
LABEL_54:
              v20 = 9;
              v21 = 3;
              sub_7BF130(3, 9, &v103);
              v22 = v103;
              goto LABEL_55;
            }
            v32 = sub_736C60(83, v30);
            sub_5CC9F0(v31);
            sub_7BC000(v112);
            if ( !v32 )
              goto LABEL_54;
          }
          v20 = 9;
          v21 = 524291;
          if ( !sub_7BF130(524291, 9, &v103) )
          {
            v103 = 1;
            goto LABEL_21;
          }
          v22 = v103;
        }
LABEL_55:
        if ( v22 )
          goto LABEL_21;
        v100 = 0;
        v19 = unk_4D04A18;
        v109 = qword_4D04A08;
        if ( unk_4D04A18 )
        {
          v33 = *(_BYTE *)(unk_4D04A18 + 80LL);
          v100 = unk_4D04A18;
          if ( v33 == 16 )
          {
            v100 = **(_QWORD **)(unk_4D04A18 + 88LL);
            v33 = *(_BYTE *)(v100 + 80);
          }
          if ( v33 == 24 )
            v100 = *(_QWORD *)(v100 + 88);
        }
        if ( (unk_4D04A12 & 2) == 0 )
        {
          v20 = (unsigned __int64)dword_4F07508;
          v21 = 738;
          sub_6851C0(738, dword_4F07508);
          v103 = 1;
          goto LABEL_22;
        }
        if ( (*(_BYTE *)(unk_4D04A18 + 82LL) & 4) != 0 )
        {
          v20 = unk_4D04A18;
          v21 = 266;
          sub_6854E0(266, unk_4D04A18);
          v103 = 1;
          goto LABEL_22;
        }
        if ( (unk_4D04A12 & 1) != 0 )
        {
          v20 = (unsigned __int64)dword_4F07508;
          v21 = 753;
          sub_6851C0(753, dword_4F07508);
          v103 = 1;
          goto LABEL_22;
        }
        v34 = sub_877F80(unk_4D04A18);
        if ( dword_4D0489C )
        {
          if ( v34 == 1 || (unk_4D04A11 & 0x10) != 0 || *(_BYTE *)(v19 + 80) == 3 && *(_BYTE *)(v19 + 104) )
          {
            v62 = 0;
            if ( (unk_4D04A12 & 2) != 0 )
              v62 = xmmword_4D04A20.m128i_i64[0];
            v63 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 168LL) + 8LL);
            if ( !(unsigned int)sub_8D3A70(v62) )
            {
              v83 = sub_8D2220(v62);
              v62 = sub_7CFE40(v83);
            }
            while ( v63 )
            {
              v66 = *(_QWORD *)(v63 + 40);
              if ( v66 == v62 || (unsigned int)sub_8D97D0(v66, v62, 0, v64, v65) )
              {
                v68 = *(__int64 **)(*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 184) + 176LL);
                if ( v68 )
                  goto LABEL_196;
                v86 = sub_726DD0();
                *(_BYTE *)(v86 + 40) |= 6u;
                v17 = v86;
                *(_QWORD *)(v86 + 48) = v62;
LABEL_201:
                *(_BYTE *)(v17 + 16) = 37;
                *(_QWORD *)(v17 + 24) = v63;
                goto LABEL_202;
              }
              v63 = *(_QWORD *)(v63 + 8);
            }
            if ( unk_4F04C48 == -1 || (v67 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v67 + 6) & 6) == 0) )
            {
              v20 = 2474;
              v17 = 0;
              v25 = 0;
              v21 = dword_4F077BC == 0 ? 7 : 5;
              sub_684AA0(v21, 2474, &qword_4D04A08);
              sub_854B40();
              goto LABEL_26;
            }
            v68 = *(__int64 **)(*(_QWORD *)(v67 + 184) + 176LL);
            if ( v68 )
            {
              v63 = 0;
              do
              {
LABEL_196:
                if ( (v68[5] & 4) != 0 )
                {
                  v69 = v68[6];
                  if ( v62 == v69 || (unsigned int)sub_8D97D0(v69, v62, 0, v64, v65) )
                  {
                    v71 = v68 + 1;
                    v20 = 2540;
                    v17 = 0;
                    v25 = 0;
                    v21 = 8;
                    sub_6854F0(8, 2540, &v110, v71);
                    goto LABEL_26;
                  }
                }
                v68 = (__int64 *)*v68;
              }
              while ( v68 );
              v70 = sub_726DD0();
              *(_BYTE *)(v70 + 40) |= 6u;
              v17 = v70;
              *(_QWORD *)(v70 + 48) = v62;
              if ( !v63 )
                goto LABEL_255;
              goto LABEL_201;
            }
            v85 = sub_726DD0();
            *(_BYTE *)(v85 + 40) |= 6u;
            v17 = v85;
            *(_QWORD *)(v85 + 48) = v62;
LABEL_255:
            *(_BYTE *)(v17 + 16) = 6;
            *(_QWORD *)(v17 + 24) = v62;
LABEL_202:
            *(_QWORD *)(v17 + 8) = v110;
            sub_733230(v17, (unsigned int)dword_4F04C64);
            v21 = dword_4F04C3C;
            if ( !dword_4F04C3C )
            {
              v21 = v17;
              sub_8699D0(v17, 29, 0);
            }
            v25 = 0;
            *(_DWORD *)(a1 + 8) |= 0x800002u;
            sub_854AB0();
            v20 = dword_4F077BC;
            if ( dword_4F077BC )
            {
              if ( dword_4F077C4 != 2 || unk_4F07778 <= 201102 && (v24 = dword_4F07774) == 0 )
              {
                v21 = (__int64)&v110;
                v20 = 2511;
                v25 = 0;
                sub_684B40(&v110, 2511);
              }
            }
            goto LABEL_26;
          }
LABEL_77:
          v21 = v19;
          if ( (unsigned __int8)sub_877F80(v19) != 2 )
            goto LABEL_78;
          goto LABEL_86;
        }
        if ( v34 != 1 )
          goto LABEL_77;
LABEL_86:
        v36 = 7 - ((unk_4D04964 == 0) - 1);
        sub_684AA0(v36, 1012, &v109);
        v20 = v36;
        v21 = 1012;
        if ( (unsigned int)sub_67D370(1012, v36, &v109) )
        {
          v103 = 1;
          goto LABEL_22;
        }
        v101 = 1;
LABEL_78:
        if ( v103 )
          goto LABEL_22;
        v35 = 0;
        if ( (unk_4D04A12 & 2) != 0 )
          v35 = xmmword_4D04A20.m128i_i64[0];
        if ( !unk_4D04868 || (v21 = v35, !(unsigned int)sub_8D2870(v35)) )
        {
          v21 = v35;
          if ( !(unsigned int)sub_8DD3B0(v35) )
          {
            v39 = v98;
            if ( *(_BYTE *)(v98 + 140) == 12 )
            {
              do
                v39 = *(_QWORD *)(v39 + 160);
              while ( *(_BYTE *)(v39 + 140) == 12 );
            }
            else
            {
              v39 = v98;
            }
            if ( ((*(_BYTE *)(v39 + 177) & 0x20) == 0
               || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v39 + 96LL) + 180LL) & 2) == 0)
              && !(unsigned int)sub_867AA0() )
            {
              goto LABEL_95;
            }
          }
          if ( (*(_BYTE *)(v35 + 141) & 0x20) != 0 && !dword_4F077BC
            || v35 == v98
            || v98 && dword_4F07588 && (v45 = *(_QWORD *)(v98 + 32), *(_QWORD *)(v35 + 32) == v45) && v45 )
          {
LABEL_95:
            v21 = v98;
            v20 = v35;
            v97 = sub_8D5CE0(v98, v35);
            if ( !v97 )
            {
              v20 = (unsigned __int64)dword_4F07508;
              v21 = 264;
              v22 = 0;
              sub_6851C0(264, dword_4F07508);
              v103 = 1;
              goto LABEL_22;
            }
            if ( (*(_BYTE *)(v97 + 96) & 1) == 0
              && !(dword_4F077BC | (unsigned int)qword_4D0495C | HIDWORD(qword_4D0495C)) )
            {
              v20 = v100;
              v21 = *(_QWORD *)(v98 + 168);
              sub_5E8E30(v21, v100, &v103);
            }
          }
          else if ( !v101 )
          {
            v89 = 1;
            v97 = sub_725160(v35, v20, v37, v98, v38);
            *(_QWORD *)(v97 + 40) = *(_QWORD *)(v19 + 64);
            *(_QWORD *)(v97 + 56) = v98;
          }
          v22 = v103;
          if ( !v103 )
          {
            v21 = v98;
            if ( (unsigned int)sub_8D3B10(v98) )
            {
              v20 = (unsigned __int64)&v110;
              v21 = 1021;
              sub_6851C0(1021, &v110);
              v103 = 1;
              goto LABEL_22;
            }
            v22 = v103;
            if ( !v103 )
            {
              if ( v101 )
              {
LABEL_84:
                sub_854AB0();
                goto LABEL_23;
              }
              v21 = (__int64)v113;
              v20 = 64;
              v40 = _mm_loadu_si128(&xmmword_4F06660[2]);
              v41 = _mm_loadu_si128(&xmmword_4F06660[3]);
              v113[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
              *((_QWORD *)&v113[0] + 1) = v109;
              v113[2] = v40;
              *(_QWORD *)&v113[0] = unk_4D04A00;
              v113[3] = v41;
              sub_7CFB70(v113, 64);
              v42 = *((_QWORD *)&v113[1] + 1);
              if ( *((_QWORD *)&v113[1] + 1) )
              {
                v43 = *(_BYTE *)(*((_QWORD *)&v113[1] + 1) + 80LL);
                if ( v43 == 16 )
                {
                  v42 = **(_QWORD **)(*((_QWORD *)&v113[1] + 1) + 88LL);
                  v43 = *(_BYTE *)(v42 + 80);
                }
                if ( v43 == 24 )
                  v42 = *(_QWORD *)(v42 + 88);
                if ( v42 != v100 && v42 != 0 )
                {
                  v44 = *(unsigned __int8 *)(v42 + 80);
                  if ( (_BYTE)v44 != 3 || !*(_BYTE *)(v42 + 104) )
                  {
                    v20 = *(unsigned __int8 *)(v100 + 80);
                    v21 = (unsigned int)(v20 - 4);
                    if ( (unsigned __int8)(v20 - 4) <= 2u || (_BYTE)v20 == 3 && (v21 = v100, *(_BYTE *)(v100 + 104)) )
                    {
                      v21 = (unsigned int)(v44 - 4);
                      if ( (unsigned __int8)(v44 - 4) <= 2u )
                        goto LABEL_231;
                      if ( (_BYTE)v44 == 3 && (*(_BYTE *)(v42 + 104) || v42 == v100 || v42 == 0) )
                      {
LABEL_115:
                        if ( (unsigned __int8)v20 > 0x14u )
                          goto LABEL_116;
LABEL_231:
                        v80 = 1182720;
                        if ( _bittest64(&v80, v20)
                          || (_BYTE)v20 == 2
                          && (v20 = v100, (v87 = *(_QWORD *)(v100 + 88)) != 0)
                          && *(_BYTE *)(v87 + 173) == 12 )
                        {
                          if ( !v103 )
                          {
                            if ( (unsigned __int8)v44 <= 0x14u )
                            {
                              v81 = 1182720;
                              if ( _bittest64(&v81, v44)
                                || (_BYTE)v44 == 2 && (v82 = *(_QWORD *)(v42 + 88)) != 0 && *(_BYTE *)(v82 + 173) == 12 )
                              {
LABEL_119:
                                if ( *(_QWORD *)&v113[0] != **(_QWORD **)v98 )
                                {
                                  v22 = 0;
                                  sub_854AB0();
                                  goto LABEL_23;
                                }
                                v20 = (unsigned __int64)&v109;
                                v21 = 280;
                                v22 = 0;
                                sub_6851C0(280, &v109);
                                v103 = 1;
                                goto LABEL_22;
                              }
                            }
                            goto LABEL_116;
                          }
                        }
                        else
                        {
LABEL_116:
                          v103 = 1;
                        }
                        v20 = (unsigned __int64)&v109;
                        v21 = 101;
                        sub_6851A0(101, &v109, *(_QWORD *)(unk_4D04A00 + 8LL));
                      }
                    }
                    else if ( (unsigned __int8)(v44 - 4) > 2u && ((_BYTE)v44 != 3 || !*(_BYTE *)(v42 + 104)) )
                    {
                      goto LABEL_115;
                    }
                  }
                }
              }
              if ( !v103 )
                goto LABEL_119;
            }
          }
LABEL_21:
          v22 = 0;
          goto LABEL_22;
        }
        v22 = 1;
        if ( !v103 )
          goto LABEL_84;
LABEL_22:
        sub_854B40();
LABEL_23:
        v25 = 0;
        if ( v103 )
          goto LABEL_26;
        if ( v22 )
        {
          v20 = v100;
          v21 = (__int64)&unk_4D04A00;
          sub_65A3F0(&unk_4D04A00, v100, v98, v96, 0);
          goto LABEL_26;
        }
        if ( v101 )
          goto LABEL_26;
        v46 = *(__int64 ***)(*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 184) + 176LL);
        if ( !v46 )
        {
LABEL_150:
          v55 = *(unsigned __int8 *)(v19 + 80);
          v105 = 0;
          if ( (_BYTE)v55 == 16 )
          {
            v100 = **(_QWORD **)(v19 + 88);
            v55 = *(unsigned __int8 *)(v100 + 80);
          }
          else
          {
            v100 = v19;
          }
          if ( (_BYTE)v55 == 24 )
          {
            v100 = *(_QWORD *)(v100 + 88);
            v55 = *(unsigned __int8 *)(v100 + 80);
          }
          if ( (unsigned __int8)v55 <= 0x14u && (v72 = 1182720, _bittest64(&v72, v55)) )
          {
            v56 = *((_QWORD *)&v113[1] + 1);
            v105 = *((_QWORD *)&v113[1] + 1);
            if ( *((_QWORD *)&v113[1] + 1) )
              goto LABEL_213;
LABEL_216:
            if ( (_BYTE)v55 == 17 )
            {
              v92 = 1;
              v57 = *(_QWORD *)(v100 + 88);
              goto LABEL_158;
            }
          }
          else if ( (*(_BYTE *)(v100 + 84) & 2) != 0 )
          {
            v56 = *((_QWORD *)&v113[1] + 1);
            v105 = *((_QWORD *)&v113[1] + 1);
            if ( *((_QWORD *)&v113[1] + 1) )
            {
LABEL_213:
              v73 = *(_BYTE *)(v56 + 80);
              if ( v73 == 2 )
              {
                v84 = *(_QWORD *)(v56 + 88);
                if ( !v84 || *(_BYTE *)(v84 + 173) != 12 )
                  goto LABEL_216;
              }
              else if ( (unsigned __int8)(v73 - 4) > 2u && (v73 != 3 || !*(_BYTE *)(v56 + 104)) )
              {
                goto LABEL_216;
              }
              v105 = 0;
              goto LABEL_216;
            }
          }
          v92 = 0;
          v57 = v19;
LABEL_158:
          if ( unk_4F04C44 != -1
            || (v74 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v74 + 6) & 6) != 0)
            || *(_BYTE *)(v74 + 4) == 12
            || (v75 = *(_BYTE *)(v57 + 80), (unsigned __int8)(v75 - 4) <= 2u)
            || v75 == 3 && *(_BYTE *)(v57 + 104) )
          {
            v58 = v96;
            v59 = 0;
          }
          else
          {
            v112[0] = 0;
            v76 = _mm_loadu_si128((const __m128i *)&unk_4D04A10);
            v77 = _mm_loadu_si128(&xmmword_4D04A20);
            v78 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
            v113[0] = _mm_loadu_si128((const __m128i *)&unk_4D04A00);
            v113[1] = v76;
            v113[2] = v77;
            v113[3] = v78;
            if ( (v76.m128i_i8[1] & 0x40) == 0 )
            {
              LOBYTE(v113[1]) &= ~0x80u;
              *((_QWORD *)&v113[1] + 1) = 0;
            }
            v79 = sub_7D2AC0(v113, *(_QWORD *)(v97 + 40), 4098);
            if ( v79 )
            {
              v58 = v96;
              if ( (*(_BYTE *)(v79 + 80) & 0xEF) != 3 )
              {
                sub_5EE990(v79, v79, v112, (_QWORD *)v97, v89, v98, &v107, v96, v99, 0);
                v59 = v107;
                if ( v107 )
                {
                  if ( v91 )
                    *(_BYTE *)(v107 + 41) |= 8u;
                  goto LABEL_160;
                }
              }
            }
            else
            {
              v58 = v96;
            }
            v59 = 0;
          }
LABEL_160:
          v88 = v17;
          v60 = v57;
          v93 = v58;
          v61 = v59;
          do
          {
            v20 = v19;
            v21 = v60;
            sub_5EE990(v60, v19, &v105, (_QWORD *)v97, v89, v98, &v107, v93, v99, *(_QWORD *)(v60 + 8) != 0);
            v24 = v92;
            if ( !v61 )
              v61 = v107;
            if ( !v92 )
              break;
            v60 = *(_QWORD *)(v60 + 8);
          }
          while ( v60 );
          v17 = v88;
          v25 = v61;
          if ( v61 )
            *(_BYTE *)(v61 + 41) |= 1u;
          goto LABEL_26;
        }
        while ( 1 )
        {
          v48 = v46[6];
          v49 = *(__int64 **)(v19 + 64);
          if ( v48 == v49 || v48 && v49 && dword_4F07588 && (v47 = v48[4], v49[4] == v47) && v47 )
          {
            v50 = *((unsigned __int8 *)v46 + 16);
            v51 = v46[3];
            v52 = (_QWORD **)((_BYTE)v50 == 37 ? v51[5] : sub_72A270(v51, v50));
            if ( **v52 == *(_QWORD *)v19 )
              break;
          }
          v46 = (__int64 **)*v46;
          if ( !v46 )
            goto LABEL_150;
        }
        v21 = 5;
        if ( unk_4D04964 )
          v21 = unk_4F07471;
        v20 = 941;
        v25 = 0;
        sub_6853B0(v21, 941, &v110, v19);
LABEL_26:
        sub_7B8B50(v21, v20, v23, v24);
        sub_650A90(v25);
        if ( !v102 )
          goto LABEL_35;
        if ( sub_867630(v106, 0) )
        {
          if ( v107 )
            *(_BYTE *)(v107 + 40) |= 0x80u;
          if ( v17 )
          {
            *(_BYTE *)(v17 + 40) |= 0x80u;
            v17 = 0;
          }
        }
        if ( !(unsigned int)sub_866C00(v106) && (!(unsigned int)sub_7BE800(67) || !(unsigned int)sub_869470(&v106)) )
          goto LABEL_34;
        v18 = v102;
        v12 = word_4F06418[0];
      }
    }
    v13 = v102;
    v19 = 0;
LABEL_179:
    if ( !v13 )
      goto LABEL_35;
    goto LABEL_34;
  }
  if ( !v14 || v16 != 75 )
    sub_6851D0(40);
  sub_854B40();
  v19 = 0;
  if ( v102 )
  {
    sub_867030(v106);
LABEL_34:
    --*(_BYTE *)(unk_4F061C8 + 75LL);
  }
LABEL_35:
  v26 = 776LL * dword_4F04C64;
  *(_BYTE *)(qword_4F04C68[0] + v26 + 7) = *(_BYTE *)(qword_4F04C68[0] + v26 + 7) & 0xFD | (2 * v95);
  *(_BYTE *)(qword_4F04C68[0] + v26 + 7) = *(_BYTE *)(qword_4F04C68[0] + v26 + 7) & 0xFB | (4 * v94);
  --*(_BYTE *)(unk_4F061C8 + 83LL);
  sub_7BE280(75, 65, 0, 0);
  return v19;
}
