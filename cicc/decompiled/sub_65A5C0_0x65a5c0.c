// Function: sub_65A5C0
// Address: 0x65a5c0
//
__int64 __fastcall sub_65A5C0(__int64 a1, int a2)
{
  int i; // ebx
  unsigned __int16 v3; // ax
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // rax
  char j; // dl
  __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r13
  unsigned __int64 v13; // xmm1_8
  __m128i v14; // xmm2
  __m128i v15; // xmm3
  __int64 v16; // r12
  char v17; // al
  __int64 v18; // r11
  __int64 *v19; // rdx
  int v20; // eax
  __int64 *v21; // r15
  char v22; // al
  int v23; // eax
  int v24; // eax
  __int64 v25; // r14
  __int64 v26; // rax
  __int64 *v27; // r9
  __int64 v28; // r13
  __int64 v29; // rbx
  __int64 v30; // r14
  __int64 v31; // rsi
  char v32; // bl
  unsigned int v33; // r13d
  int v34; // r13d
  int v35; // ebx
  __int64 *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // r13
  __int64 v41; // rbx
  __int64 *v42; // rax
  __int64 result; // rax
  char v44; // dl
  __m128i v45; // xmm4
  __m128i v46; // xmm5
  __m128i v47; // xmm6
  __m128i v48; // xmm7
  __int64 v49; // rax
  __int64 v50; // r14
  int v51; // eax
  int v52; // ecx
  __int64 v53; // r10
  char v54; // dl
  char v55; // dl
  __int64 v56; // rdi
  int v57; // eax
  __int64 v58; // rax
  __int64 v59; // r8
  char v60; // r9
  __int64 v61; // rax
  bool v62; // cf
  bool v63; // zf
  __int64 v64; // rcx
  char *v65; // rdi
  __int64 v66; // r15
  __int64 v67; // rax
  unsigned __int64 v68; // rax
  __int64 v69; // r13
  int v70; // eax
  __int64 v71; // rax
  int v72; // r8d
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // r13
  __int64 v76; // [rsp+8h] [rbp-128h]
  __int64 v77; // [rsp+8h] [rbp-128h]
  __int64 v78; // [rsp+10h] [rbp-120h]
  __int64 v79; // [rsp+10h] [rbp-120h]
  __int64 v80; // [rsp+10h] [rbp-120h]
  __int64 v81; // [rsp+10h] [rbp-120h]
  __int64 v82; // [rsp+18h] [rbp-118h]
  __int64 v83; // [rsp+20h] [rbp-110h]
  __int64 *v85; // [rsp+30h] [rbp-100h]
  __int64 *v86; // [rsp+30h] [rbp-100h]
  __int64 v87; // [rsp+30h] [rbp-100h]
  __int64 v88; // [rsp+38h] [rbp-F8h]
  _BOOL4 v89; // [rsp+38h] [rbp-F8h]
  int v90; // [rsp+40h] [rbp-F0h]
  int v91; // [rsp+44h] [rbp-ECh]
  unsigned int v93; // [rsp+54h] [rbp-DCh] BYREF
  __int64 *v94; // [rsp+58h] [rbp-D8h] BYREF
  __int64 v95; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v96; // [rsp+68h] [rbp-C8h] BYREF
  __int64 v97; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v98; // [rsp+78h] [rbp-B8h] BYREF
  __m128i v99; // [rsp+80h] [rbp-B0h] BYREF
  unsigned __int64 v100; // [rsp+90h] [rbp-A0h]
  __int64 v101; // [rsp+98h] [rbp-98h]
  __m128i v102; // [rsp+A0h] [rbp-90h]
  __m128i v103; // [rsp+B0h] [rbp-80h]
  __m128i v104; // [rsp+C0h] [rbp-70h] BYREF
  __m128i v105; // [rsp+D0h] [rbp-60h]
  __m128i v106; // [rsp+E0h] [rbp-50h]
  __m128i v107; // [rsp+F0h] [rbp-40h]

  v94 = 0;
  v93 = 0;
  v97 = 0;
  v83 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( unk_4D04324 )
    sub_684AB0(a1 + 24, 880);
  ++*(_BYTE *)(qword_4F061C8 + 83LL);
  v90 = unk_4D04400;
  if ( unk_4D04400 )
  {
    v72 = sub_869470(&v96);
    result = qword_4F061C8;
    v44 = *(_BYTE *)(qword_4F061C8 + 75LL);
    *(_BYTE *)(qword_4F061C8 + 75LL) = v44 + 1;
    if ( !v72 )
      goto LABEL_100;
    v90 = 1;
  }
  for ( i = 0; ; i = 1 )
  {
    v3 = word_4F06418[0];
    if ( dword_4F077C4 != 2 )
    {
      if ( word_4F06418[0] != 1 )
        break;
      goto LABEL_56;
    }
    if ( word_4F06418[0] == 1 && (unk_4D04A11 & 2) != 0 )
      goto LABEL_54;
    v63 = (unsigned int)sub_7C0F00(0, 0) == 0;
    v3 = word_4F06418[0];
    if ( v63 )
      break;
    if ( word_4F06418[0] == 183 )
      goto LABEL_8;
    if ( dword_4F077C4 == 2 )
    {
      if ( word_4F06418[0] == 1 )
      {
LABEL_54:
        if ( (unk_4D04A11 & 2) != 0 )
          goto LABEL_56;
      }
      if ( !(unsigned int)sub_7C0F00(1, 0) )
        goto LABEL_92;
      goto LABEL_56;
    }
    if ( word_4F06418[0] != 1 )
    {
LABEL_92:
      v5 = (__int64)&dword_4F063F8;
      v4 = 40;
      sub_6851C0(40, &dword_4F063F8);
LABEL_11:
      v93 = 1;
LABEL_12:
      sub_854B40();
      v9 = v93;
      if ( v93 )
        goto LABEL_36;
      goto LABEL_13;
    }
LABEL_56:
    v31 = *(_QWORD *)(a1 + 184);
    if ( v31 && sub_736C60(83, v31) )
    {
LABEL_58:
      if ( dword_4F077C4 == 2 )
      {
        v32 = 1;
        v33 = 524289;
        if ( unk_4F07778 > 201401 && (_DWORD)qword_4F077B4 )
          goto LABEL_212;
      }
      else
      {
        v32 = 1;
        v33 = 524289;
      }
      goto LABEL_60;
    }
    if ( a2 || i && unk_4D0418C )
    {
      sub_7ADF70(&v104, 0);
      sub_7AE360(&v104);
      sub_7B8B50(&v104, 0, v37, v38);
      v39 = sub_5CC190(23);
      v40 = v39;
      if ( v39 )
      {
        v41 = sub_736C60(83, v39);
        sub_5CC9F0(v40);
        sub_7BC000(&v104);
        if ( v41 )
          goto LABEL_58;
      }
      else
      {
        sub_7BC000(&v104);
      }
    }
    if ( dword_4F077C4 == 2 )
    {
      if ( unk_4F07778 <= 201401 || !(_DWORD)qword_4F077B4 )
      {
        v5 = 0;
        v4 = 1;
        v94 = (__int64 *)sub_7BF130(1, 0, &v93);
LABEL_62:
        v34 = v93;
        goto LABEL_63;
      }
      v32 = 0;
      v33 = 1;
LABEL_212:
      if ( (unsigned int)sub_729F80(dword_4F063F8) )
      {
        if ( dword_4F04C64 != -1 )
        {
          v73 = qword_4F04C68[0] + 776LL * dword_4F04C64;
          if ( *(_BYTE *)(v73 + 4) == 4 )
          {
            v74 = *(_QWORD *)(v73 + 224);
            if ( v74 )
            {
              if ( (*(_BYTE *)(v74 + 124) & 0x10) != 0 )
                v33 |= 0x8000000u;
            }
          }
        }
      }
LABEL_60:
      v5 = 0;
      v4 = v33;
      v94 = (__int64 *)sub_7BF130(v33, 0, &v93);
      if ( !v94 && v32 )
        goto LABEL_11;
      goto LABEL_62;
    }
    v5 = 0;
    v4 = 1;
    v42 = (__int64 *)sub_7BF130(1, 0, &v93);
    v34 = v93;
    v94 = v42;
LABEL_63:
    if ( v34 )
      goto LABEL_12;
    v35 = unk_4D04868;
    v36 = v94;
    if ( unk_4D04868 )
    {
      if ( !v94 )
        goto LABEL_107;
      v35 = 0;
      if ( *((_BYTE *)v94 + 80) != 2 )
        goto LABEL_67;
      v75 = v94[11];
      v4 = v75;
      v35 = sub_72AE00(v75);
      if ( v35 )
      {
        v35 = 1;
      }
      else if ( *(_BYTE *)(v75 + 173) == 12 )
      {
        v35 = *(_BYTE *)(v75 + 176) == 2;
      }
      if ( v93 )
        goto LABEL_168;
      v36 = v94;
    }
    if ( !v36 )
    {
      v34 = v35;
LABEL_107:
      v4 = 20;
      v35 = v34;
      v5 = *(_QWORD *)(qword_4D04A00 + 8);
      sub_6851F0(20, v5);
      v93 = 1;
      goto LABEL_168;
    }
LABEL_67:
    if ( (unk_4D04A10 & 0x8001) == 0 && !unk_4D0479C )
    {
      v5 = (__int64)dword_4F07508;
      v4 = 727;
      sub_6851C0(727, dword_4F07508);
      v93 = 1;
      goto LABEL_168;
    }
    if ( (unk_4D04A12 & 2) == 0 || (v35 & 1) != 0 )
    {
      if ( (unk_4D04A12 & 1) != 0 )
      {
        v5 = (__int64)dword_4F07508;
        v4 = 753;
        sub_6851C0(753, dword_4F07508);
        v93 = 1;
        goto LABEL_168;
      }
      if ( *((_BYTE *)v36 + 80) != 23 )
      {
        sub_854AB0();
        goto LABEL_74;
      }
      v5 = (__int64)&qword_4D04A08;
      v4 = 728;
      sub_6851C0(728, &qword_4D04A08);
      v93 = 1;
LABEL_168:
      sub_854B40();
LABEL_74:
      if ( v93 )
        goto LABEL_36;
      if ( v35 )
      {
        v5 = (__int64)v94;
        v4 = (__int64)&qword_4D04A00;
        sub_65A3F0((__int64)&qword_4D04A00, v94, 0, 0, 0);
        goto LABEL_36;
      }
      goto LABEL_13;
    }
    v5 = (__int64)dword_4F07508;
    v4 = 754;
    sub_6851C0(754, dword_4F07508);
    v93 = 1;
    sub_854B40();
    if ( v93 )
      goto LABEL_36;
LABEL_13:
    v10 = xmmword_4D04A20.m128i_i64[0];
    if ( (unk_4D04A12 & 2) == 0 )
    {
      if ( xmmword_4D04A20.m128i_i64[0] )
      {
        v11 = *(_QWORD *)(v83 + 184);
        if ( v11 && *(_BYTE *)(v11 + 28) == 3 )
        {
          v69 = *(_QWORD *)(v11 + 32);
          if ( (*(_BYTE *)(xmmword_4D04A20.m128i_i64[0] + 124) & 1) != 0 )
          {
            if ( v69 == sub_735B70(xmmword_4D04A20.m128i_i64[0]) )
              goto LABEL_114;
          }
          else if ( v69 == xmmword_4D04A20.m128i_i64[0] )
          {
            goto LABEL_114;
          }
        }
      }
      else
      {
        v8 = (unsigned int)dword_4F04C64;
        if ( !dword_4F04C64 )
        {
LABEL_112:
          if ( unk_4D04A11 >= 0 )
          {
            if ( unk_4D042AC )
              goto LABEL_36;
LABEL_114:
            v5 = (__int64)dword_4F07508;
            v4 = 737;
            sub_684B30(737, dword_4F07508);
            goto LABEL_36;
          }
          v10 = 0;
        }
      }
      v12 = 0;
      goto LABEL_18;
    }
    v5 = (unsigned int)dword_4F04C64;
    if ( !dword_4F04C64 && !xmmword_4D04A20.m128i_i64[0] )
      goto LABEL_112;
    v12 = xmmword_4D04A20.m128i_i64[0];
    v10 = 0;
LABEL_18:
    v5 = 64;
    v13 = _mm_loadu_si128((const __m128i *)&unk_4D04A10).m128i_u64[0];
    v14 = _mm_loadu_si128(&xmmword_4D04A20);
    v15 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
    v99 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
    v100 = v13;
    LOWORD(v100) = unk_4D04A10 & 0xBF7F;
    v101 = 0;
    v102 = v14;
    v103 = v15;
    sub_7CFB70(&v99, 64);
    v16 = v101;
    if ( v101 )
    {
      v17 = *(_BYTE *)(v101 + 80);
      v18 = v101;
      if ( v17 == 16 )
      {
        v18 = **(_QWORD **)(v101 + 88);
        v17 = *(_BYTE *)(v18 + 80);
      }
      if ( v17 == 24 )
        v18 = *(_QWORD *)(v18 + 88);
      v19 = v94;
      v95 = 0;
      v20 = *((unsigned __int8 *)v94 + 80);
      v21 = v94;
      v8 = (unsigned int)(v20 - 10);
      if ( (unsigned __int8)(v20 - 10) <= 1u )
      {
        v91 = 0;
LABEL_173:
        v68 = *(unsigned __int8 *)(v18 + 80);
        if ( (unsigned __int8)v68 <= 0x14u )
        {
          v8 = 1182720;
          if ( _bittest64(&v8, v68) )
            v95 = v101;
        }
        goto LABEL_176;
      }
    }
    else
    {
      v21 = v94;
      v18 = 0;
      v95 = 0;
      v20 = *((unsigned __int8 *)v94 + 80);
      v19 = v94;
      v8 = (unsigned int)(v20 - 10);
      if ( (unsigned __int8)(v20 - 10) <= 1u )
      {
        v91 = 0;
        goto LABEL_28;
      }
    }
    v91 = 0;
    if ( (_BYTE)v20 != 20 )
    {
      if ( (_BYTE)v20 != 17 )
        goto LABEL_26;
      v91 = 1;
      v19 = (__int64 *)v19[11];
      v94 = v19;
    }
    if ( v101 )
      goto LABEL_173;
LABEL_176:
    LOBYTE(v20) = *((_BYTE *)v19 + 80);
    v21 = v19;
LABEL_26:
    if ( (_BYTE)v20 == 16 )
    {
      v21 = *(__int64 **)v19[11];
      LOBYTE(v20) = *((_BYTE *)v21 + 80);
    }
LABEL_28:
    if ( (_BYTE)v20 == 24 )
      v21 = (__int64 *)v21[11];
    if ( !v101
      || v95
      || !*(_DWORD *)(v101 + 48)
      || (v22 = *(_BYTE *)(v83 + 4), (unsigned __int8)(v22 - 3) > 1u) && v22
      || (v5 = v18, v4 = (__int64)v21, v88 = v18, v23 = sub_7D0550(v21, v18, 0, 0), v18 = v88, !v23) )
    {
      v24 = *((unsigned __int8 *)v21 + 80);
      v9 = (unsigned int)(v24 - 4);
      if ( (unsigned __int8)(v24 - 4) <= 2u )
      {
        v89 = 0;
        v25 = 0;
        goto LABEL_43;
      }
      if ( (_BYTE)v24 == 3 && *((_BYTE *)v21 + 104) )
      {
        v89 = 0;
        v25 = 0;
        if ( v18 )
          goto LABEL_192;
LABEL_152:
        v59 = v21[11];
LABEL_153:
        v60 = *(_BYTE *)(v59 + 140);
        if ( v60 == 12 )
        {
          v61 = v59;
          do
          {
            v61 = *(_QWORD *)(v61 + 160);
            v9 = *(unsigned __int8 *)(v61 + 140);
          }
          while ( (_BYTE)v9 == 12 );
        }
        else
        {
          v9 = *(unsigned __int8 *)(v59 + 140);
        }
        if ( (_BYTE)v9 )
        {
          if ( !dword_4F04C5C )
          {
            if ( v10 )
            {
              v62 = *(_QWORD *)(unk_4D049B8 + 88LL) < v10;
              v63 = *(_QWORD *)(unk_4D049B8 + 88LL) == v10;
              if ( *(_QWORD *)(unk_4D049B8 + 88LL) == v10 )
              {
                v64 = 7;
                v65 = "size_t";
                v5 = *(_QWORD *)(*v21 + 8);
                do
                {
                  if ( !v64 )
                    break;
                  v62 = *(_BYTE *)v5 < (unsigned __int8)*v65;
                  v63 = *(_BYTE *)v5++ == (unsigned __int8)*v65++;
                  --v64;
                }
                while ( v63 );
                if ( (!v62 && !v63) == v62 )
                {
                  if ( v60 == 12 )
                  {
                    do
                      v59 = *(_QWORD *)(v59 + 160);
                    while ( *(_BYTE *)(v59 + 140) == 12 );
                  }
                  v87 = v59;
                  v66 = sub_7259C0(12);
                  v67 = sub_7259C0(*(unsigned __int8 *)(v87 + 140));
                  *(_QWORD *)(v66 + 160) = v67;
                  sub_73C230(v87, v67);
                  v5 = 6;
                  *(_QWORD *)(v66 + 8) = "size_t";
                  sub_7604D0(v66, 6);
                  *(_BYTE *)(v66 + 141) |= 1u;
                  sub_7E1CA0(v66);
                }
              }
            }
          }
        }
        goto LABEL_45;
      }
      v45 = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
      v98 = 0;
      v46 = _mm_loadu_si128((const __m128i *)&unk_4D04A10);
      v47 = _mm_loadu_si128(&xmmword_4D04A20);
      v48 = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
      v82 = v97;
      v104 = v45;
      v105 = v46;
      v106 = v47;
      v107 = v48;
      if ( (v46.m128i_i8[1] & 0x40) != 0 )
      {
        v78 = v18;
        if ( v10 )
          goto LABEL_125;
LABEL_200:
        v5 = (__int64)&v104;
        v71 = sub_7D4600(unk_4F07288, &v104, 524290);
        v18 = v78;
        v50 = v71;
      }
      else
      {
        v105.m128i_i8[0] &= ~0x80u;
        v105.m128i_i64[1] = 0;
        v78 = v18;
        if ( !v10 )
          goto LABEL_200;
LABEL_125:
        v5 = v10;
        v49 = sub_7D4A40(&v104, v10, 524290);
        v18 = v78;
        v50 = v49;
      }
      if ( !v50 )
        goto LABEL_179;
      v51 = *(unsigned __int8 *)(v50 + 80);
      v9 = (unsigned int)(v51 - 4);
      if ( (unsigned __int8)(v51 - 4) <= 2u )
      {
        if ( v16 )
        {
          v9 = qword_4F04C68[0] + 776LL * dword_4F04C64;
          goto LABEL_130;
        }
LABEL_235:
        v53 = 0;
LABEL_147:
        v81 = v18;
        sub_650720(v50, &v98, v53, v10, 0, &v97, 0, 0, 0, 0);
        if ( (v105.m128i_i8[1] & 0x40) == 0 )
        {
          v105.m128i_i8[0] &= ~0x80u;
          v105.m128i_i64[1] = 0;
        }
        v5 = 66;
        v58 = sub_7CFB70(&v104, 66);
        v25 = v97;
        v18 = v81;
        v89 = v58 != 0;
        if ( !v97 )
        {
LABEL_150:
          LOBYTE(v24) = *((_BYTE *)v21 + 80);
          v25 = 0;
          if ( !v18 )
            goto LABEL_151;
LABEL_44:
          if ( (_BYTE)v24 != 3 )
          {
LABEL_45:
            v8 = a1;
            v4 = (__int64)v94;
            *(_QWORD *)a1 = v94;
            *(_QWORD *)(a1 + 48) = *(_QWORD *)dword_4F07508;
            if ( !v4 )
              goto LABEL_36;
            v26 = v12;
            v27 = &v97;
            v28 = v10;
            v29 = v25;
            v30 = v26;
            while ( 2 )
            {
              if ( !v91 )
              {
                v5 = (__int64)&v95;
                v86 = v27;
                sub_650720(v4, &v95, v16, v28, v30, v27, 0, v89, *(__m128i **)(a1 + 184), 0);
                v27 = v86;
                if ( v29 )
                  goto LABEL_51;
                goto LABEL_77;
              }
              v5 = (__int64)&v95;
              v85 = v27;
              sub_650720(v4, &v95, v16, v28, v30, v27, 1, v89, *(__m128i **)(a1 + 184), *(_QWORD *)(v4 + 8) != 0);
              v27 = v85;
              if ( !v29 )
              {
LABEL_77:
                v29 = v97;
                if ( v97 )
                  *(_BYTE *)(v97 + 41) |= 1u;
                if ( !v91 )
                {
LABEL_51:
                  v94 = 0;
                  goto LABEL_36;
                }
              }
              v4 = v94[1];
              v94 = (__int64 *)v4;
              if ( !v4 )
                goto LABEL_36;
              continue;
            }
          }
LABEL_192:
          v70 = *(unsigned __int8 *)(v18 + 80);
          v9 = (unsigned int)(v70 - 4);
          if ( (unsigned __int8)(v70 - 4) > 2u && ((_BYTE)v70 != 3 || !*(_BYTE *)(v18 + 104)) )
          {
            if ( (v100 & 0x4000) == 0 )
            {
              LOBYTE(v100) = v100 & 0x7F;
              v101 = 0;
            }
            v5 = 66;
            v18 = sub_7CFB70(&v99, 66);
            if ( !v18 )
            {
              if ( *((_BYTE *)v21 + 80) != 3 )
                goto LABEL_45;
              goto LABEL_152;
            }
          }
          v59 = *(_QWORD *)(v18 + 88);
          v5 = v21[11];
          if ( *((_BYTE *)v21 + 80) != 3 )
          {
            if ( v5 == v59 )
            {
              v89 = 1;
              goto LABEL_45;
            }
            goto LABEL_195;
          }
          if ( v5 == v59 )
          {
            v89 = 1;
            goto LABEL_153;
          }
LABEL_195:
          v63 = (unsigned int)sub_8D97D0(*(_QWORD *)(v18 + 88), v5, 0, v8, v59) == 0;
          LOBYTE(v24) = *((_BYTE *)v21 + 80);
          if ( !v63 )
          {
            v89 = 1;
            if ( (_BYTE)v24 != 3 )
              goto LABEL_45;
            goto LABEL_152;
          }
LABEL_151:
          if ( (_BYTE)v24 != 3 )
            goto LABEL_45;
          goto LABEL_152;
        }
LABEL_180:
        if ( v25 == v82 )
          goto LABEL_150;
        *(_BYTE *)(v25 + 41) |= 1u;
        LOBYTE(v24) = *((_BYTE *)v21 + 80);
LABEL_43:
        if ( v18 )
          goto LABEL_44;
        goto LABEL_151;
      }
      if ( (_BYTE)v51 != 3 || !*(_BYTE *)(v50 + 104) )
        goto LABEL_179;
      if ( !v16 )
        goto LABEL_235;
      v9 = 776LL * dword_4F04C64 + qword_4F04C68[0];
LABEL_130:
      v52 = *(unsigned __int8 *)(v16 + 80);
      if ( (_BYTE)v52 == 3 )
      {
LABEL_233:
        v53 = v16;
      }
      else
      {
        v5 = (__int64)&dword_4F077C4;
        if ( dword_4F077C4 == 2 )
        {
          v8 = (unsigned int)(v52 - 4);
          if ( (unsigned __int8)v8 <= 2u )
          {
            if ( (_BYTE)v51 != 19 )
              goto LABEL_233;
LABEL_179:
            v25 = v97;
            v89 = 0;
            if ( !v97 )
              goto LABEL_150;
            goto LABEL_180;
          }
        }
        if ( (v105.m128i_i8[1] & 0x40) == 0 )
        {
          v105.m128i_i8[0] &= ~0x80u;
          v105.m128i_i64[1] = 0;
        }
        v5 = 66;
        v79 = v18;
        v76 = v9;
        sub_7CFB70(&v104, 66);
        LOBYTE(v51) = *(_BYTE *)(v50 + 80);
        v18 = v79;
        if ( (_BYTE)v51 == 19 )
          goto LABEL_179;
        v53 = v105.m128i_i64[1];
        v9 = v76;
        if ( !v105.m128i_i64[1] )
          goto LABEL_235;
      }
      v54 = *(_BYTE *)(v9 + 4);
      if ( (unsigned __int8)(v54 - 3) > 1u && v54 )
        goto LABEL_147;
      v55 = *(_BYTE *)(v53 + 80);
      v5 = v53;
      if ( v55 == 16 )
      {
        v5 = **(_QWORD **)(v53 + 88);
        v55 = *(_BYTE *)(v5 + 80);
      }
      if ( v55 == 24 )
        v5 = *(_QWORD *)(v5 + 88);
      v56 = v50;
      if ( (_BYTE)v51 == 16 )
        v56 = **(_QWORD **)(v50 + 88);
      if ( *(_BYTE *)(v56 + 80) == 24 )
        v56 = *(_QWORD *)(v56 + 88);
      v77 = v53;
      v80 = v18;
      v57 = sub_7D0550(v56, v5, 0, 0);
      v18 = v80;
      v53 = v77;
      if ( !v57 )
        goto LABEL_147;
      goto LABEL_179;
    }
LABEL_36:
    sub_7B8B50(v4, v5, v9, v8);
    sub_650A90(v97);
    if ( v90 )
    {
      if ( sub_867630(v96, 0) && v97 )
        *(_BYTE *)(v97 + 40) |= 0x80u;
      if ( !(unsigned int)sub_866C00(v96) && (!(unsigned int)sub_7BE800(67) || !(unsigned int)sub_869470(&v96)) )
      {
        result = qword_4F061C8;
        v44 = *(_BYTE *)(qword_4F061C8 + 75LL) - 1;
        goto LABEL_100;
      }
    }
    else
    {
      if ( word_4F06418[0] != 67 )
        goto LABEL_221;
      if ( (unsigned int)sub_7BE800(67) )
        sub_869470(&v96);
    }
  }
  if ( v3 == 183 )
  {
LABEL_8:
    v4 = (__int64)&v104;
    v5 = (__int64)&v94;
    sub_671BC0(&v104, &v94, 1, 0, 0, 0);
    v6 = v104.m128i_i64[0];
    for ( j = *(_BYTE *)(v104.m128i_i64[0] + 140); j == 12; j = *(_BYTE *)(v6 + 140) )
      v6 = *(_QWORD *)(v6 + 160);
    if ( !j )
      goto LABEL_11;
    goto LABEL_62;
  }
  sub_6851D0(40);
  sub_854B40();
  if ( !v90 )
  {
LABEL_221:
    result = qword_4F061C8;
    goto LABEL_101;
  }
  sub_867030(v96);
  result = qword_4F061C8;
  v44 = *(_BYTE *)(qword_4F061C8 + 75LL) - 1;
LABEL_100:
  *(_BYTE *)(result + 75) = v44;
LABEL_101:
  --*(_BYTE *)(result + 83);
  return result;
}
