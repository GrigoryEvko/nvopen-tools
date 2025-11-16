// Function: sub_7D5DD0
// Address: 0x7d5dd0
//
__int64 __fastcall sub_7D5DD0(__m128i *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  char v6; // al
  __m128i *v8; // rbx
  int v9; // r12d
  bool v10; // dl
  int v11; // r15d
  __int64 v12; // r9
  __int64 v13; // rax
  int v14; // esi
  _BOOL4 v15; // eax
  __int64 v16; // rax
  __int64 v17; // rdx
  const char *v18; // rsi
  const char *v19; // rdi
  int v20; // eax
  char v21; // al
  __int64 v22; // rbx
  __int64 v23; // r12
  __int64 v24; // rax
  __int64 v25; // r15
  char v26; // di
  __int64 *v27; // rsi
  char v28; // al
  __int64 v29; // rax
  __int64 v30; // r8
  char v31; // dl
  __int64 v32; // r11
  __int64 v33; // rax
  char v34; // dl
  __int64 v35; // rcx
  char v36; // si
  __int8 v37; // al
  __int8 v38; // al
  __int64 v39; // rdx
  __int64 v40; // r8
  int v41; // r10d
  __int64 i; // rax
  __int64 v43; // rcx
  int v44; // esi
  int *j; // rax
  int v46; // ecx
  __int8 v47; // al
  __int8 v48; // al
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rsi
  char v52; // dl
  __int64 v53; // rax
  __int64 v54; // r15
  char v55; // al
  __int64 v56; // rax
  char v57; // cl
  char v58; // di
  char v59; // al
  _BOOL4 v60; // eax
  char v61; // dl
  int v62; // eax
  __int64 v63; // rax
  unsigned int v64; // r13d
  __int64 v65; // rax
  __int64 v66; // rax
  __int8 v67; // dl
  int v68; // [rsp+Ch] [rbp-394h]
  __m128i *v69; // [rsp+10h] [rbp-390h]
  __int64 v70; // [rsp+18h] [rbp-388h]
  __int64 v71; // [rsp+18h] [rbp-388h]
  __int64 v72; // [rsp+20h] [rbp-380h]
  __int64 v73; // [rsp+20h] [rbp-380h]
  __int64 v74; // [rsp+20h] [rbp-380h]
  __int64 v75; // [rsp+20h] [rbp-380h]
  __int64 v76; // [rsp+28h] [rbp-378h]
  int v77; // [rsp+28h] [rbp-378h]
  __int64 v78; // [rsp+28h] [rbp-378h]
  __int64 v79; // [rsp+28h] [rbp-378h]
  int v80; // [rsp+3Ch] [rbp-364h] BYREF
  __int64 v81; // [rsp+40h] [rbp-360h] BYREF
  __int64 v82; // [rsp+48h] [rbp-358h] BYREF
  char v83[64]; // [rsp+50h] [rbp-350h] BYREF
  __int64 v84; // [rsp+90h] [rbp-310h] BYREF
  char v85[16]; // [rsp+98h] [rbp-308h] BYREF
  __int64 v86; // [rsp+A8h] [rbp-2F8h]
  char v87; // [rsp+D0h] [rbp-2D0h]
  char v88; // [rsp+D1h] [rbp-2CFh]
  int v89; // [rsp+100h] [rbp-2A0h] BYREF
  int v90; // [rsp+104h] [rbp-29Ch]
  int v91; // [rsp+108h] [rbp-298h]
  int v92; // [rsp+10Ch] [rbp-294h]
  _BOOL4 v93; // [rsp+110h] [rbp-290h]
  int v94; // [rsp+114h] [rbp-28Ch]
  int v95; // [rsp+118h] [rbp-288h]
  int v96; // [rsp+11Ch] [rbp-284h]
  int v97; // [rsp+120h] [rbp-280h]
  int v98; // [rsp+124h] [rbp-27Ch]
  int v99; // [rsp+128h] [rbp-278h]
  int v100; // [rsp+12Ch] [rbp-274h]
  __m128i v101; // [rsp+130h] [rbp-270h]
  __m128i v102; // [rsp+140h] [rbp-260h]
  __m128i si128; // [rsp+150h] [rbp-250h]
  __m128i v104; // [rsp+160h] [rbp-240h]
  __m128i v105; // [rsp+170h] [rbp-230h]
  __m128i v106; // [rsp+180h] [rbp-220h]
  _QWORD v107[66]; // [rsp+190h] [rbp-210h] BYREF

  v5 = a1[1].m128i_i64[1];
  if ( v5 )
    goto LABEL_2;
  v8 = a1;
  if ( (a1[1].m128i_i8[1] & 0x20) != 0 )
    return v5;
  v9 = a2;
  v89 = a2 & 1;
  si128 = _mm_load_si128((const __m128i *)&xmmword_4F18050);
  v101 = _mm_load_si128((const __m128i *)&xmmword_4F18030);
  v90 = (a2 >> 1) & 1;
  v10 = (a2 & 2) != 0;
  v102 = _mm_load_si128((const __m128i *)&xmmword_4F18040);
  v104 = _mm_load_si128((const __m128i *)&xmmword_4F18060);
  v91 = (a2 >> 9) & 1;
  v105 = _mm_load_si128((const __m128i *)&xmmword_4F18070);
  v106 = _mm_load_si128((const __m128i *)&xmmword_4F18080);
  v92 = (a2 >> 11) & 1;
  v11 = a2 & 4;
  v93 = v11 != 0;
  v94 = (a2 >> 15) & 1;
  v95 = (a2 >> 5) & 1;
  v96 = (a2 >> 13) & 1;
  v97 = (a2 >> 14) & 1;
  v100 = (a2 >> 21) & 1;
  v98 = (a2 >> 17) & 1;
  si128.m128i_i32[3] = v98;
  v99 = (a2 >> 20) & 1;
  v101.m128i_i32[1] = (a2 >> 3) & 1;
  v101.m128i_i32[2] = (a2 >> 7) & 1;
  v12 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( unk_4F04C48 != -1
    && (*(_BYTE *)(v12 + 6) & 0x20) != 0
    && (v13 = *(_QWORD *)(v12 + 624)) != 0
    && (*(_BYTE *)(v13 + 129) & 0x40) != 0 )
  {
    v104.m128i_i32[1] = 1;
    v14 = 1;
  }
  else
  {
    v14 = v104.m128i_i32[1];
  }
  v15 = 0;
  if ( !(dword_4D047C8 | v9 & 0x4020) )
  {
    v15 = 1;
    if ( dword_4F04C44 == -1 && (*(_BYTE *)(v12 + 6) & 6) == 0 )
      v15 = *(_BYTE *)(v12 + 4) == 12;
  }
  v102.m128i_i32[0] = v15;
  if ( dword_4F077C4 != 2 && v10 )
    v105.m128i_i32[3] = 1;
  v16 = a1->m128i_i64[0];
  v17 = a1->m128i_i64[0];
  if ( !v14 )
    goto LABEL_38;
  if ( !dword_4F077BC )
  {
    if ( !(_DWORD)qword_4F077B4 )
      goto LABEL_38;
    goto LABEL_26;
  }
  if ( (_DWORD)qword_4F077B4 )
  {
LABEL_26:
    if ( v16 )
    {
      v18 = *(const char **)(v16 + 8);
      if ( v18 )
      {
        if ( !strcmp(v18, "swap")
          && *(_BYTE *)(v12 + 4) == 1
          && *(_BYTE *)(v12 - 772) == 7
          && *(_BYTE *)(v12 - 1548) == 9 )
        {
          a5 = *(_QWORD *)(v12 - 1184);
          v70 = a5;
          if ( *(_QWORD *)a5 )
          {
            v19 = *(const char **)(*(_QWORD *)a5 + 8LL);
            if ( v19 )
            {
              v72 = v17;
              v76 = qword_4F04C68[0] + 776LL * dword_4F04C64;
              v20 = strcmp(v19, "pair");
              v12 = v76;
              v17 = v72;
              if ( !v20 )
              {
                if ( qword_4D049B8 )
                {
                  if ( (unsigned int)sub_879C10(v70, qword_4D049B8) && sub_729F80(v8->m128i_u32[2]) )
                  {
                    v12 = qword_4F04C68[0] + 776LL * dword_4F04C64;
                    v16 = v8->m128i_i64[0];
                    goto LABEL_105;
                  }
                  v17 = v8->m128i_i64[0];
                  v12 = qword_4F04C68[0] + 776LL * dword_4F04C64;
                }
              }
            }
          }
        }
      }
    }
    goto LABEL_38;
  }
  if ( qword_4F077A8 <= 0x1869Fu )
  {
LABEL_105:
    v106.m128i_i32[0] = *(_DWORD *)(v12 + 420);
    v17 = v16;
  }
LABEL_38:
  v105.m128i_i32[2] = v9;
  if ( unk_4D03F98
    || (v21 = *(_BYTE *)(v12 + 5), v21 < 0)
    || dword_4F077C4 == 2 && v97 | v95 | v101.m128i_i32[2] | v101.m128i_i32[1]
    || *(_QWORD *)(v17 + 32) && (v21 & 4) != 0
    || (v8[1].m128i_i8[0] & 0x10) != 0 )
  {
    v29 = sub_7D5120((__int64)v8, (__int64)&v89, dword_4F04C60, -1, a5);
    v12 = 0;
    v30 = v29;
    if ( dword_4F04C64 != -1 )
      v12 = qword_4F04C68[0] + 776LL * dword_4F04C64;
LABEL_64:
    if ( dword_4D047C0 )
    {
      if ( qword_4F077A8 <= 0x76BFu )
      {
        if ( !v102.m128i_i32[2] )
          goto LABEL_92;
        if ( !v30 )
          goto LABEL_99;
      }
      else
      {
        if ( !v30 )
          goto LABEL_129;
        if ( (*(_BYTE *)(v30 + 82) & 8) == 0 && (v9 & 0x400000) == 0 || !v102.m128i_i32[2] )
          goto LABEL_141;
      }
      v31 = *(_BYTE *)(v30 + 80);
      v32 = v30;
      if ( v31 == 16 )
      {
        v32 = **(_QWORD **)(v30 + 88);
        v31 = *(_BYTE *)(v32 + 80);
      }
      if ( v31 == 24 )
      {
        v32 = *(_QWORD *)(v32 + 88);
        if ( qword_4F077A8 <= 0x76BFu || !v32 )
          goto LABEL_76;
        v31 = *(_BYTE *)(v32 + 80);
      }
      else if ( qword_4F077A8 <= 0x76BFu )
      {
        goto LABEL_76;
      }
      if ( (unsigned __int8)(v31 - 2) > 3u && v31 != 7 )
        goto LABEL_76;
LABEL_141:
      if ( v95 )
        goto LABEL_103;
      goto LABEL_142;
    }
LABEL_92:
    if ( v95 || v101.m128i_i32[0] )
    {
      v37 = v106.m128i_i8[12];
      v8[1].m128i_i64[1] = v30;
      v8[1].m128i_i8[2] = ((v37 & 1) << 6) | v8[1].m128i_i8[2] & 0xBF;
      if ( v30 )
        goto LABEL_144;
      return v5;
    }
LABEL_101:
    if ( dword_4F077C4 != 2 )
    {
LABEL_102:
      if ( !v30 )
        goto LABEL_163;
      goto LABEL_103;
    }
    if ( !v30 )
    {
LABEL_107:
      if ( !unk_4D04950
        || (*(_BYTE *)(v8->m128i_i64[0] + 73) & 4) == 0
        || (v39 = *(_QWORD *)(v8->m128i_i64[0] + 32)) == 0 )
      {
LABEL_180:
        if ( !v102.m128i_i32[1] || v90 )
        {
LABEL_131:
          v47 = v106.m128i_i8[12];
          v8[1].m128i_i64[1] = 0;
          v8[1].m128i_i8[2] = ((v47 & 1) << 6) | v8[1].m128i_i8[2] & 0xBF;
          return v5;
        }
LABEL_182:
        v30 = sub_7D2AC0(v8, (const char *)v104.m128i_i64[1], v9 | 0x900000u);
        goto LABEL_102;
      }
      v40 = 0;
      v41 = -1;
      do
      {
        if ( (unsigned __int8)(*(_BYTE *)(v39 + 80) - 3) <= 3u && (*(_BYTE *)(v39 + 81) & 0x10) != 0 )
        {
          for ( i = *(_QWORD *)(v39 + 64); (*(_BYTE *)(i + 89) & 4) != 0; i = *(_QWORD *)(*(_QWORD *)(i + 40) + 32LL) )
            ;
          v43 = *(_QWORD *)(i + 40);
          if ( !v43 || *(_BYTE *)(v43 + 28) != 3 )
          {
            v44 = *(_DWORD *)(*(_QWORD *)i + 40LL);
            if ( v44 != v41 )
            {
              for ( j = (int *)(qword_4F04C68[0] + 776LL * dword_4F04C64); ; j -= 194 )
              {
                v46 = *j;
                if ( v40 )
                {
                  if ( v44 == v46 )
                    goto LABEL_126;
                  if ( v46 == v41 )
                    goto LABEL_111;
                }
                else if ( v44 == v46 )
                {
LABEL_126:
                  v41 = v44;
                  v40 = v39;
                  goto LABEL_111;
                }
                if ( (int *)qword_4F04C68[0] == j )
                  goto LABEL_111;
              }
            }
            goto LABEL_180;
          }
        }
LABEL_111:
        v39 = *(_QWORD *)(v39 + 8);
      }
      while ( v39 );
      if ( !v40 )
        goto LABEL_180;
      v50 = *(unsigned __int8 *)(v40 + 80);
      v51 = v40;
      v52 = *(_BYTE *)(v40 + 80);
      if ( (_BYTE)v50 == 16 )
      {
        v51 = **(_QWORD **)(v40 + 88);
        v52 = *(_BYTE *)(v51 + 80);
      }
      if ( v52 == 24 )
        v51 = *(_QWORD *)(v51 + 88);
      if ( dword_4F04BA0[v50] != v105.m128i_i32[3] )
      {
        if ( !v102.m128i_i32[1] )
          goto LABEL_131;
LABEL_162:
        if ( !v90 )
          goto LABEL_182;
        goto LABEL_163;
      }
      v59 = *(_BYTE *)(v51 + 83);
      if ( (v59 & 0x40) == 0 && (*(_BYTE *)(v40 + 83) & 0x40) == 0 )
        goto LABEL_203;
      if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || (unsigned __int64)(qword_4F077A8 - 50000LL) > 0x270F )
        goto LABEL_210;
      v61 = *(_BYTE *)(v51 + 80);
      if ( v61 == 20 )
        goto LABEL_203;
      if ( v61 != 17 || (v71 = v12, v75 = v40, v62 = sub_8780F0(v51), v40 = v75, v12 = v71, !(v95 | v62)) )
      {
LABEL_210:
        if ( !v97 )
          goto LABEL_211;
      }
      v59 = *(_BYTE *)(v51 + 83);
LABEL_203:
      if ( v59 >= 0
        || !v12
        || unk_4F04C48 == -1
        || (v63 = *(_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 368), v51 != v63)
        || !v63 )
      {
        v79 = v40;
        v60 = sub_7CF0D0(v40, v51, &v89);
        v30 = v79;
        if ( v60 )
        {
          v8[1].m128i_i8[0] |= 0x80u;
          goto LABEL_103;
        }
      }
LABEL_211:
      if ( !v102.m128i_i32[1] )
        goto LABEL_163;
      goto LABEL_162;
    }
LABEL_103:
    v38 = v106.m128i_i8[12];
    v8[1].m128i_i64[1] = v30;
    v5 = v30;
    v8[1].m128i_i8[2] = ((v38 & 1) << 6) | v8[1].m128i_i8[2] & 0xBF;
    goto LABEL_2;
  }
  if ( !*(_QWORD *)(v17 + 24) )
    goto LABEL_128;
  v69 = v8;
  v22 = *(_QWORD *)(v17 + 24);
  v68 = v9;
  v23 = 0;
  v77 = v11;
  v73 = v12;
  do
  {
    v24 = *(unsigned __int8 *)(v22 + 80);
    v25 = v22;
    v26 = *(_BYTE *)(v22 + 80);
    if ( (_BYTE)v24 == 16 )
    {
      v27 = *(__int64 **)(v22 + 88);
      v25 = *v27;
      v26 = *(_BYTE *)(*v27 + 80);
    }
    if ( v26 == 24 )
      v25 = *(_QWORD *)(v25 + 88);
    if ( v23 && *(_DWORD *)(v23 + 40) != *(_DWORD *)(v22 + 40) )
    {
      v11 = v77;
      v12 = v73;
      v5 = 0;
      v8 = v69;
      v30 = v23;
      v9 = v68;
      goto LABEL_64;
    }
    if ( dword_4F04BA0[v24] == v105.m128i_i32[3] )
    {
      v28 = *(_BYTE *)(v25 + 83);
      if ( (v28 & 0x40) == 0 && (*(_BYTE *)(v22 + 83) & 0x40) == 0 )
        goto LABEL_56;
      if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || (unsigned __int64)(qword_4F077A8 - 50000LL) > 0x270F )
        goto LABEL_135;
      v58 = *(_BYTE *)(v25 + 80);
      if ( v58 == 20 )
      {
LABEL_56:
        if ( v28 >= 0
          || unk_4F04C48 == -1
          || (v49 = *(_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 368), v25 != v49)
          || !v49 )
        {
          if ( sub_7CF0D0(v22, v25, &v89) )
          {
            if ( !v90 || *(_BYTE *)(v25 + 80) != 3 )
            {
              v30 = v22;
              v11 = v77;
              v12 = v73;
              v5 = 0;
              v8 = v69;
              v9 = v68;
              goto LABEL_64;
            }
            v23 = v22;
          }
        }
        goto LABEL_46;
      }
      if ( v58 != 17 || !(unsigned int)sub_8780F0(v25) )
      {
LABEL_135:
        if ( !(v97 | v95) )
          goto LABEL_46;
      }
      v28 = *(_BYTE *)(v25 + 83);
      goto LABEL_56;
    }
LABEL_46:
    v22 = *(_QWORD *)(v22 + 8);
  }
  while ( v22 );
  v11 = v77;
  v12 = v73;
  v5 = 0;
  v8 = v69;
  v30 = v23;
  v9 = v68;
  if ( v30 )
    goto LABEL_64;
LABEL_128:
  if ( !dword_4D047C0 )
    goto LABEL_130;
LABEL_129:
  if ( !v102.m128i_i32[2] )
    goto LABEL_130;
LABEL_99:
  v30 = 0;
  v32 = 0;
LABEL_76:
  v74 = v32;
  v78 = v30;
  si128.m128i_i32[3] = 1;
  v33 = sub_7D5120((__int64)v8, (__int64)&v89, dword_4F04C60, -1, v30);
  v12 = 0;
  v30 = v78;
  if ( dword_4F04C64 != -1 )
    v12 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  si128.m128i_i32[3] = 0;
  if ( v33 )
  {
    v34 = *(_BYTE *)(v33 + 80);
    v35 = v33;
    if ( v34 == 16 )
    {
      v35 = **(_QWORD **)(v33 + 88);
      v34 = *(_BYTE *)(v35 + 80);
    }
    if ( v34 == 24 && (v35 = *(_QWORD *)(v35 + 88)) == 0
      || (*(_BYTE *)(v35 + 81) & 0x10) == 0
      || (*(_BYTE *)(*(_QWORD *)(v35 + 64) + 177LL) & 0x20) == 0 )
    {
      if ( !v78 )
      {
        if ( v95 )
        {
          v67 = v106.m128i_i8[12];
          v8[1].m128i_i64[1] = v33;
          v5 = v33;
          v8[1].m128i_i8[2] = ((v67 & 1) << 6) | v8[1].m128i_i8[2] & 0xBF;
          goto LABEL_2;
        }
        v30 = v33;
LABEL_142:
        if ( v101.m128i_i32[0] )
          goto LABEL_143;
        goto LABEL_101;
      }
      if ( (*(_BYTE *)(v78 + 82) & 8) != 0 )
        goto LABEL_140;
LABEL_86:
      if ( qword_4F077A8 > 0x9CA3u )
      {
        if ( qword_4F077A8 > 0x9E97u )
          goto LABEL_92;
      }
      else
      {
        v36 = *(_BYTE *)(v74 + 80);
        if ( (unsigned __int8)(v36 - 10) <= 1u || v36 == 17 )
          goto LABEL_91;
      }
      if ( v74 == v35 && v74 )
      {
LABEL_91:
        v30 = v33;
        goto LABEL_92;
      }
      if ( v33 && qword_4F077A8 <= 0x9CA3u )
      {
        v57 = *(_BYTE *)(v35 + 80);
        if ( (v57 == 17 || (unsigned __int8)(v57 - 10) <= 1u)
          && v74
          && (unsigned __int8)(*(_BYTE *)(v74 + 80) - 19) > 3u )
        {
          goto LABEL_140;
        }
        if ( v57 == 8 && qword_4F077A8 <= 0x76BFu )
        {
          if ( *(_BYTE *)(v78 + 80) == 8 )
          {
LABEL_140:
            v30 = v33;
            goto LABEL_141;
          }
          goto LABEL_141;
        }
      }
      goto LABEL_92;
    }
  }
  if ( v78 )
  {
    v35 = 0;
    v33 = 0;
    goto LABEL_86;
  }
LABEL_130:
  if ( v95 || v101.m128i_i32[0] )
    goto LABEL_131;
  if ( dword_4F077C4 == 2 )
    goto LABEL_107;
LABEL_163:
  if ( dword_4F077C4 )
    goto LABEL_131;
  if ( dword_4D04964 | v11 )
    goto LABEL_131;
  if ( v9 < 0 )
    goto LABEL_131;
  v53 = sub_879D20(v8, 3, 0, 0, 0, v83);
  v54 = v53;
  if ( !v53 || *(_BYTE *)(v53 + 80) == 15 && *(_BYTE *)(*(_QWORD *)(v53 + 88) + 16LL) )
    goto LABEL_131;
  if ( !sub_7D1FF0(v53, v9) )
    goto LABEL_131;
  sub_6854B0(0x2A4u, v54);
  v55 = *(_BYTE *)(v54 + 80);
  if ( v55 == 15 )
  {
    memset(v107, 0, 0x1D8u);
    v64 = dword_4F04C3C;
    v107[19] = v107;
    v107[3] = *(_QWORD *)&dword_4F063F8;
    if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
      BYTE2(v107[22]) |= 1u;
    v107[0] = sub_885AD0(11, v8, (unsigned int)dword_4F04C64, 0);
    v65 = **(_QWORD **)(v54 + 88);
    WORD2(v107[33]) = 257;
    v107[36] = v65;
    sub_87E3B0(&v84);
    v87 |= 0x40u;
    if ( dword_4D048B8 )
      v86 = v8->m128i_i64[1];
    dword_4F04C3C = 1;
    sub_6523A0((__int64)v8, (__int64)v107, (__int64)&v84, 129, &v80, &v81, &v82, 0);
    dword_4F04C3C = v64;
    if ( dword_4F04C64 == -1
      || (v66 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v66 + 7) & 1) == 0)
      || dword_4F04C44 == -1 && (*(_BYTE *)(v66 + 6) & 2) == 0 )
    {
      if ( (v88 & 8) == 0 )
        sub_87E280(v85);
    }
  }
  else
  {
    if ( v55 != 14 )
      sub_721090();
    memset(v107, 0, 0x1D8u);
    v107[19] = v107;
    v107[3] = *(_QWORD *)&dword_4F063F8;
    if ( dword_4F077BC )
    {
      if ( qword_4F077A8 <= 0x9F5Fu )
        BYTE2(v107[22]) |= 1u;
    }
    v56 = **(_QWORD **)(v54 + 88);
    BYTE5(v107[33]) = 1;
    v107[36] = v56;
    sub_6582F0(v8, (__int64)v107, 0x81u, (int *)&v82, &v84, 0);
  }
  v30 = v107[0];
  *(_BYTE *)(*(_QWORD *)(v107[0] + 88LL) + 88LL) |= 4u;
LABEL_143:
  v48 = v106.m128i_i8[12];
  v8[1].m128i_i64[1] = v30;
  v8[1].m128i_i8[2] = ((v48 & 1) << 6) | v8[1].m128i_i8[2] & 0xBF;
LABEL_144:
  v5 = v30;
LABEL_2:
  v6 = *(_BYTE *)(v5 + 80);
  if ( v6 == 16 )
  {
    v5 = **(_QWORD **)(v5 + 88);
    if ( *(_BYTE *)(v5 + 80) == 24 )
      goto LABEL_7;
LABEL_4:
    if ( *(_QWORD *)(v5 + 72) )
      goto LABEL_5;
    return v5;
  }
  else
  {
    if ( v6 != 24 )
      goto LABEL_4;
LABEL_7:
    v5 = *(_QWORD *)(v5 + 88);
    if ( !*(_QWORD *)(v5 + 72) )
      return v5;
LABEL_5:
    if ( *(char *)(v5 + 81) < 0 )
      return v5;
    return sub_7D2A80(v5);
  }
}
