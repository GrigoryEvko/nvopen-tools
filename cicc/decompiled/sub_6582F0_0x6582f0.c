// Function: sub_6582F0
// Address: 0x6582f0
//
__int64 __fastcall sub_6582F0(__m128i *a1, __int64 a2, unsigned __int64 a3, int *a4, __int64 *a5, __int64 a6)
{
  char v7; // al
  char v8; // r15
  __int16 v9; // r13
  char v10; // al
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r12
  char v15; // al
  unsigned int v16; // edi
  char v17; // r13
  unsigned int v18; // r15d
  __int64 v19; // rcx
  __int64 v20; // rax
  unsigned int v21; // edi
  char v22; // dl
  char v23; // si
  char v24; // cl
  bool v25; // zf
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rdi
  unsigned __int8 v29; // r15
  unsigned int v30; // eax
  __int64 *v31; // rcx
  __int64 v32; // rdx
  const char *v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // rcx
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 *v42; // rdx
  __int64 v43; // rdx
  _BYTE *v44; // rdi
  __int64 result; // rax
  char v46; // al
  __int64 v47; // rax
  char v48; // dl
  __int64 v49; // rdx
  int v50; // eax
  _DWORD *v51; // r11
  size_t v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rax
  char v55; // dl
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // rax
  __int64 v59; // r14
  __int64 v60; // rax
  char v61; // al
  __int64 v62; // rax
  __int64 *v63; // rax
  __int64 v64; // rax
  char i; // dl
  __int64 v66; // rax
  __int64 v67; // r15
  char j; // al
  char v69; // al
  char *v70; // r8
  char v71; // al
  __int64 v72; // rdi
  char v73; // al
  int v74; // eax
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // [rsp+0h] [rbp-140h]
  __int64 *v78; // [rsp+8h] [rbp-138h]
  unsigned int s2; // [rsp+10h] [rbp-130h]
  char *s2a; // [rsp+10h] [rbp-130h]
  unsigned int v82; // [rsp+20h] [rbp-120h]
  int v83; // [rsp+24h] [rbp-11Ch]
  unsigned int v84; // [rsp+24h] [rbp-11Ch]
  unsigned int v85; // [rsp+24h] [rbp-11Ch]
  int v87; // [rsp+40h] [rbp-100h]
  int v88; // [rsp+48h] [rbp-F8h]
  unsigned int v89; // [rsp+4Ch] [rbp-F4h]
  __int64 v90; // [rsp+60h] [rbp-E0h]
  const __m128i *v91; // [rsp+68h] [rbp-D8h]
  __int64 v92; // [rsp+70h] [rbp-D0h]
  unsigned __int64 v93; // [rsp+78h] [rbp-C8h]
  __int64 v94; // [rsp+80h] [rbp-C0h]
  int v96; // [rsp+90h] [rbp-B0h]
  char v97; // [rsp+96h] [rbp-AAh]
  char v98; // [rsp+96h] [rbp-AAh]
  unsigned __int8 v99; // [rsp+97h] [rbp-A9h]
  int v100; // [rsp+98h] [rbp-A8h]
  __int64 v101; // [rsp+98h] [rbp-A8h]
  char v102; // [rsp+A7h] [rbp-99h] BYREF
  __int64 v103; // [rsp+A8h] [rbp-98h] BYREF
  __m128i *v104; // [rsp+B0h] [rbp-90h] BYREF
  __int128 v105; // [rsp+B8h] [rbp-88h]
  __int128 v106; // [rsp+C8h] [rbp-78h]
  __int128 v107; // [rsp+D8h] [rbp-68h]
  __int128 v108; // [rsp+E8h] [rbp-58h]
  __int128 v109; // [rsp+F8h] [rbp-48h]

  v93 = a3;
  v90 = *(_QWORD *)(a2 + 288);
  v7 = *(_BYTE *)(a2 + 269);
  v97 = v7;
  v103 = 0;
  if ( HIDWORD(qword_4F077B4) && v7 == 5 )
  {
    if ( *(_QWORD *)(a2 + 240) )
    {
      v8 = 5;
      if ( dword_4F04C5C == dword_4F04C34 )
      {
        v93 = a3 & 0xFFFFFFFFFFFFFFFDLL;
        v9 = 0;
        v8 = 1;
        v96 = 0;
        goto LABEL_6;
      }
    }
    else
    {
      v8 = 5;
    }
  }
  else
  {
    v8 = v7;
  }
  if ( (a3 & 2) != 0 )
  {
    *(_BYTE *)(a2 + 122) |= 1u;
    v9 = 1;
    v96 = 1;
  }
  else
  {
    v96 = 0;
    v9 = 0;
  }
LABEL_6:
  if ( (a1[1].m128i_i32[0] & 0x12000) == 0x10000 )
  {
    sub_6851C0(891, &a1->m128i_u64[1]);
    *a1 = _mm_loadu_si128(xmmword_4F06660);
    a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    v66 = *(_QWORD *)dword_4F07508;
    a1[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
    a1[1].m128i_i8[1] |= 0x20u;
    a1->m128i_i64[1] = v66;
  }
  if ( (*(_BYTE *)(a2 + 10) & 8) != 0 )
  {
    if ( (*(_BYTE *)(v90 + 140) & 0xFB) == 8 && (sub_8D4C10(v90, dword_4F077C4 != 2) & 1) != 0 )
    {
      v92 = v90;
    }
    else
    {
      v92 = sub_73C570(v90, 1, -1);
      *(_QWORD *)(a2 + 288) = v92;
    }
  }
  else
  {
    v92 = v90;
  }
  v108 = (unsigned __int64)v92;
  v10 = *(_BYTE *)(a2 + 228);
  v107 = 0;
  v104 = a1;
  BYTE4(v107) = v8;
  v105 = 0;
  v106 = 0;
  WORD4(v108) = (8 * v9) | ((v10 & 1) << 8);
  v109 = 0;
  sub_6413B0((__int64)&v104, dword_4F04C5C);
  if ( dword_4F077C4 == 2
    && a1[1].m128i_i64[1]
    && ((a1[1].m128i_i8[2] & 2) == 0 && a1[2].m128i_i64[0] || (a1[1].m128i_i8[0] & 4) != 0) )
  {
    sub_648CF0(&v104, a2);
  }
  else
  {
    sub_644100((__int64)&v104, a2);
  }
  v14 = v105;
  v91 = v104;
  v99 = BYTE4(v107);
  if ( (BYTE9(v108) & 4) != 0 && (_QWORD)v105 )
  {
    v15 = *(_BYTE *)(v105 + 80);
    if ( v15 == 16 )
    {
      v14 = **(_QWORD **)(v105 + 88);
      v15 = *(_BYTE *)(v14 + 80);
    }
    if ( v15 == 24 )
      v14 = *(_QWORD *)(v14 + 88);
  }
  v82 = -1;
  s2 = v107;
  v100 = DWORD2(v109);
  v16 = dword_4F077BC;
  if ( dword_4F077BC && !(_DWORD)qword_4F077B4 && qword_4F077A8 <= 0x9E97u && (BYTE8(v108) & 0x10) != 0 )
  {
    v82 = dword_4F04C34;
    if ( dword_4F04C34 )
    {
      v50 = sub_641310(dword_4F077BC, a2, v11, v12, v13, (unsigned int)qword_4F077B4);
      *v51 = v50;
    }
    else
    {
      v82 = -1;
    }
  }
  v94 = (__int64)&v91->m128i_i64[1];
  if ( !v100 )
  {
    if ( (v91[1].m128i_i8[2] & 2) == 0 )
    {
      v18 = 0;
      v89 = 0;
      v14 = sub_87EF90(7, v91);
      *a5 = 0;
LABEL_196:
      v20 = v103;
      if ( v103 )
        goto LABEL_35;
      v88 = 1;
      v19 = v18;
      v87 = 0;
      goto LABEL_198;
    }
    v17 = 0;
    v18 = 0;
    v89 = 0;
    goto LABEL_30;
  }
  if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 || (v17 = 0, (v89 = dword_4F07590) != 0) )
  {
    v89 = 1;
    v17 = 1;
    if ( !v14 )
    {
LABEL_142:
      v18 = 0;
      goto LABEL_151;
    }
  }
  else if ( !v14 )
  {
    goto LABEL_142;
  }
  v46 = *(_BYTE *)(v14 + 80);
  if ( v16 )
  {
    if ( v46 == 16 )
    {
      v14 = **(_QWORD **)(v14 + 88);
      v46 = *(_BYTE *)(v14 + 80);
    }
    if ( v46 == 24 )
    {
      v14 = *(_QWORD *)(v14 + 88);
      v46 = *(_BYTE *)(v14 + 80);
    }
  }
  if ( v46 != 7 || (v47 = *(_QWORD *)(v14 + 88), (*(_BYTE *)(v47 + 170) & 1) != 0) )
  {
    sub_6854C0(147, v94, v14);
LABEL_150:
    v103 = 0;
    v18 = 1;
    *(_QWORD *)(a2 + 296) = 0;
LABEL_151:
    if ( (v91[1].m128i_i8[2] & 2) == 0 )
    {
LABEL_31:
      v14 = sub_87EF90(7, v91);
      *a5 = 0;
      if ( v17 )
      {
        if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0 && (unsigned int)sub_8DBE70(v92) )
        {
          v20 = v103;
          if ( v103 )
          {
LABEL_35:
            v87 = 0;
            v21 = v18;
            v88 = 1;
LABEL_36:
            v22 = *(_BYTE *)(v20 + 136);
            v23 = v100 == 1;
            v24 = v22 == 2;
            if ( dword_4F077C4 != 2 || *(_BYTE *)(a2 + 268) )
            {
              if ( v23 == v24 )
                goto LABEL_43;
              if ( !v21 )
              {
                v101 = v20;
                sub_6854F0(4, 172, v94, v20 + 64);
                v20 = v101;
                v22 = *(_BYTE *)(v101 + 136);
              }
            }
            else if ( v23 == v24 )
            {
              goto LABEL_43;
            }
            if ( !v22 || (v25 = v99 == 0, v100 = 1, v99 = 2, v25) )
            {
              v100 = 2;
              v99 = 0;
            }
            *(_BYTE *)(v20 + 136) = v99;
LABEL_43:
            if ( ((*(_BYTE *)(v103 + 176) & 8) != 0) != ((*(_QWORD *)(a2 + 8) & 0x400000LL) != 0) )
              sub_6854F0(8, (unsigned int)((*(_BYTE *)(v103 + 176) & 8) != 0) + 2502, v94, v103 + 64);
            if ( DWORD2(v109) != v100 )
            {
              DWORD2(v109) = v100;
              sub_6418E0((__int64)&v104);
            }
            v26 = v103;
            if ( v99 )
            {
              if ( v99 == 1 )
              {
LABEL_51:
                v83 = 1;
                if ( (v93 & 2) != 0 )
                {
                  if ( (*(_BYTE *)(v14 + 81) & 2) == 0 || (v93 & 0x200) == 0 )
                  {
                    if ( (*(_BYTE *)(v26 + 88) & 0x70) == 0x30 )
                    {
                      sub_735DA0(v26, 0, 0);
                      sub_735E40(v26, 0);
                      v26 = v103;
                    }
                    else
                    {
                      v54 = v105;
                      if ( (_QWORD)v105 )
                      {
                        v55 = *(_BYTE *)(v105 + 80);
                        if ( v55 == 16 )
                        {
                          v54 = **(_QWORD **)(v105 + 88);
                          v55 = *(_BYTE *)(v54 + 80);
                        }
                        if ( v55 == 24 )
                          v54 = *(_QWORD *)(v54 + 88);
                      }
                      if ( (BYTE9(v108) & 4) != 0 )
                      {
                        sub_864230(*(_QWORD *)(v54 + 64), 1);
                        v85 = dword_4F04C34;
                        sub_735DA0(v26, (unsigned int)dword_4F04C34, 0);
                        sub_735E40(v26, v85);
                        sub_8642D0();
                      }
                      else
                      {
                        v84 = dword_4F04C34;
                        sub_735DA0(v26, (unsigned int)dword_4F04C34, 0);
                        sub_735E40(v26, v84);
                      }
                      v26 = v103;
                    }
                  }
                  v83 = 1;
                  if ( (*(_BYTE *)(v26 + 172) & 0x10) != 0 && (*(_BYTE *)(a2 + 10) & 0x20) == 0 )
                  {
                    sub_6854F0(8, 3116, a2 + 32, v26 + 64);
                    v26 = v103;
                  }
                }
                goto LABEL_55;
              }
            }
            else
            {
              *(_BYTE *)(v103 + 136) = 0;
            }
            if ( (*(_BYTE *)(v26 + 88) & 4) != 0 )
              *(_BYTE *)(v14 + 81) |= 1u;
            goto LABEL_51;
          }
        }
        else
        {
          v52 = sub_647D10(v91, a2, v92, (__int64)&v104, 0, v18, v18, &v103, 0);
          v19 = (__int64)a5;
          *a5 = v52;
          v20 = v103;
          if ( v103 )
            goto LABEL_35;
        }
        v83 = v18;
        v87 = 0;
        v88 = 1;
        goto LABEL_155;
      }
      goto LABEL_196;
    }
LABEL_30:
    v91[1].m128i_i32[0] &= 0xFFFDFFFE;
    v91[2].m128i_i64[0] = 0;
    goto LABEL_31;
  }
  v48 = *(_BYTE *)(v14 + 81) & 2;
  if ( dword_4F077C4 == 2 )
  {
    if ( v48 && v96 )
    {
      sub_685920(v94, v14, 8);
      v91[1].m128i_i8[1] |= 0x20u;
      v91[1].m128i_i64[1] = 0;
      v73 = *(_BYTE *)(v14 + 80);
      *(_BYTE *)(v14 + 81) |= 1u;
      if ( v73 == 7 || v73 == 9 )
        *(_BYTE *)(*(_QWORD *)(v14 + 88) + 169LL) |= 0x10u;
      goto LABEL_150;
    }
  }
  else if ( v48 && *(_BYTE *)(v47 + 177) && (*(_BYTE *)(a2 + 127) & 4) == 0 )
  {
    v93 &= 0xFFFFFFFFFFFFFDFDLL;
    v96 = 0;
  }
  *(_QWORD *)a2 = v14;
  v20 = *(_QWORD *)(v14 + 88);
  v49 = *(_QWORD *)(v20 + 120);
  v103 = v20;
  *(_QWORD *)(a2 + 296) = v49;
  if ( (*(_BYTE *)(a2 + 125) & 1) != 0 )
  {
    v18 = 0;
    *a5 = 0;
LABEL_138:
    v87 = 1;
    v21 = 0;
    v88 = 0;
    goto LABEL_36;
  }
  *(_QWORD *)(a2 + 288) = v92;
  if ( (unsigned int)sub_646C60(a2) )
  {
    v19 = 0;
    v92 = *(_QWORD *)(a2 + 288);
  }
  else
  {
    if ( v96 )
    {
      v92 = *(_QWORD *)(a2 + 288);
      goto LABEL_150;
    }
    v76 = sub_72C930(a2);
    v19 = 1;
    v92 = v76;
    *(_QWORD *)(a2 + 288) = v76;
  }
  v20 = v103;
  *a5 = 0;
  if ( v20 )
  {
    v18 = v19;
    goto LABEL_138;
  }
  if ( v17 )
  {
    v87 = 1;
    v18 = v19;
    v83 = 0;
    v88 = 0;
LABEL_155:
    v89 = 1;
    v53 = (unsigned int)dword_4F04C34;
    if ( dword_4F04C34 )
    {
      if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 9) & 0xE) != 6 || v99 == 2 )
        v89 = 1;
      else
        v53 = 0;
    }
    goto LABEL_199;
  }
  v88 = 0;
  v18 = 0;
  v87 = 1;
LABEL_198:
  v83 = v18;
  v18 = v19;
  v53 = (unsigned int)dword_4F04C5C;
  v61 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4);
  v89 = ((v61 - 15) & 0xFD) != 0 && v61 != 2;
LABEL_199:
  v62 = sub_735FB0(v92, v99, v53, v19);
  v103 = v62;
  v26 = v62;
  if ( unk_4D03B98 && v99 == 3 && unk_4D03B90 >= 0 && *(int *)(unk_4D03B98 + 176LL * unk_4D03B90 + 152) > 0 )
    *(_BYTE *)(v62 + 156) |= 8u;
  if ( *a5 )
  {
    v63 = *(__int64 **)(*a5 + 88);
    if ( v63[1] )
    {
      if ( v83 )
      {
        v83 = 0;
      }
      else
      {
        v64 = *v63;
        for ( i = *(_BYTE *)(v64 + 140); i == 12; i = *(_BYTE *)(v64 + 140) )
          v64 = *(_QWORD *)(v64 + 160);
        if ( i )
          *(_BYTE *)(v26 + 173) |= 1u;
      }
    }
    else
    {
      *(_BYTE *)(a2 + 127) |= 0x10u;
      v83 = 0;
    }
  }
  else
  {
    *(_BYTE *)(a2 + 127) |= 0x10u;
    v83 = 0;
    if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0 )
      *(_BYTE *)(v62 + 170) |= 0x20u;
  }
LABEL_55:
  *(_QWORD *)(v14 + 88) = v26;
  if ( v88 )
  {
    sub_885A00(v14, s2, v18);
    v26 = v103;
  }
  if ( *a5 && (v27 = *(_QWORD *)(*a5 + 88), !*(_QWORD *)(v27 + 8)) )
  {
    *(_QWORD *)(v27 + 8) = v26;
    if ( *(_QWORD *)v26 )
    {
LABEL_60:
      v28 = v26;
      if ( !v87 )
      {
        sub_641F90((__int64 *)v26, v14);
        v28 = v103;
      }
      goto LABEL_62;
    }
  }
  else if ( *(_QWORD *)v26 )
  {
    goto LABEL_60;
  }
  sub_877D80(v26, v14);
  v28 = v103;
  if ( (BYTE8(v108) & 0x10) != 0 && *qword_4D03FD0 )
  {
    sub_8CFC70(v103);
    v28 = v103;
  }
LABEL_62:
  sub_6581B0(v28, a2, v96);
  *(_QWORD *)a2 = v14;
  sub_644920((_QWORD *)a2, v96);
  if ( !HIDWORD(qword_4F077B4) || !*(_QWORD *)(a2 + 240) )
    goto LABEL_71;
  s2a = *(char **)(a2 + 240);
  v78 = (__int64 *)v103;
  v29 = *(_BYTE *)(*(_QWORD *)v103 + 81LL);
  v77 = a2 + 248;
  v30 = sub_703C10(s2a);
  v31 = v78;
  v32 = v30;
  if ( v97 == 5 )
  {
    if ( !(_BYTE)v30 )
    {
      sub_6851A0(1118, v77, s2a);
      goto LABEL_71;
    }
    v67 = v78[15];
    for ( j = *(_BYTE *)(v67 + 140); j == 12; j = *(_BYTE *)(v67 + 140) )
      v67 = *(_QWORD *)(v67 + 160);
    if ( dword_4F077C4 == 2 && (unsigned __int8)(j - 9) <= 2u )
    {
      v72 = *(_QWORD *)(*(_QWORD *)v67 + 96LL);
      if ( unk_4F07778 > 201102 || dword_4F07774 )
      {
        if ( *(_QWORD *)(v72 + 8) )
        {
          v98 = v32;
          v74 = sub_879360(v72, s2a, v32, v78);
          LOBYTE(v32) = v98;
          v31 = v78;
          if ( v74 )
            goto LABEL_289;
        }
        if ( *(_BYTE *)(v67 + 140) == 12 )
        {
          v75 = v67;
          do
            v75 = *(_QWORD *)(v75 + 160);
          while ( *(_BYTE *)(v75 + 140) == 12 );
          if ( !*(_QWORD *)(*(_QWORD *)(*(_QWORD *)v75 + 96LL) + 24LL) )
            goto LABEL_231;
          do
            v67 = *(_QWORD *)(v67 + 160);
          while ( *(_BYTE *)(v67 + 140) == 12 );
        }
        else if ( !*(_QWORD *)(*(_QWORD *)(*(_QWORD *)v67 + 96LL) + 24LL) )
        {
          goto LABEL_231;
        }
        if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v67 + 96LL) + 177LL) & 2) == 0 )
          goto LABEL_289;
      }
      else if ( *(char *)(v72 + 178) >= 0 )
      {
LABEL_289:
        sub_6851C0(1560, v77);
        goto LABEL_71;
      }
    }
LABEL_231:
    v69 = *((_BYTE *)v31 + 169);
    if ( (v69 & 8) != 0 )
    {
      if ( !v31[18] )
      {
        *((_BYTE *)v31 + 144) = v32;
        *((_BYTE *)v31 + 169) = v69 & 0xF7;
        goto LABEL_71;
      }
    }
    else if ( (_BYTE)v32 == *((_BYTE *)v31 + 144) )
    {
      goto LABEL_71;
    }
LABEL_233:
    sub_684B30(1306, v77);
    goto LABEL_71;
  }
  if ( qword_4F077A8 <= 0x752Fu || (_BYTE)v30 == 58 )
  {
    v33 = (const char *)v78[18];
    if ( ((dword_4F077C0 != 0) & (v29 >> 1)) != 0 && !v96 )
    {
      if ( !v33 || strcmp(v33, s2a) )
        sub_684B30(1609, v77);
    }
    else
    {
      if ( v33 )
      {
        if ( !strcmp(v33, s2a) )
          goto LABEL_71;
        goto LABEL_233;
      }
      v78[18] = (__int64)s2a;
      sub_5D0F70(*v78);
    }
  }
  else
  {
    sub_6851C0(1364, v77);
  }
LABEL_71:
  v102 = *(_BYTE *)(v103 + 168) & 7;
  sub_5D0D60(&v102, 0);
  v34 = v103;
  *(_BYTE *)(v103 + 168) = v102 & 7 | *(_BYTE *)(v103 + 168) & 0xF8;
  if ( v100 )
  {
    if ( v100 == 2 )
    {
      v71 = *(_BYTE *)(v34 + 156);
      if ( (v71 & 1) != 0 && dword_4F04C58 == -1 )
        *(_BYTE *)(v34 + 156) = v71 | 0x10;
    }
    *(_BYTE *)(v26 + 89) &= ~1u;
  }
  if ( dword_4F04C34 && (v88 & v89) != 0 )
  {
    sub_641A90(v14, (_QWORD *)v26);
    sub_649AF0((__int64)&v104, v14, v26, *a5, v94);
    sub_648B20((_BYTE *)a2);
    goto LABEL_192;
  }
  sub_649AF0((__int64)&v104, v14, v26, *a5, v94);
  sub_648B20((_BYTE *)a2);
  if ( v89 )
  {
LABEL_192:
    v36 = v103;
    v37 = v103;
    if ( !unk_4F04C50 )
      goto LABEL_80;
    v38 = *(_QWORD *)(unk_4F04C50 + 32LL);
    if ( !v38 )
      goto LABEL_80;
    goto LABEL_79;
  }
  v36 = v103;
  v35 = *(unsigned __int8 *)(v103 + 156);
  v37 = v103;
  if ( v99 == 1 || (v35 & 1) == 0 )
    goto LABEL_77;
  if ( !unk_4F04C50 )
    goto LABEL_83;
  v38 = *(_QWORD *)(unk_4F04C50 + 32LL);
  if ( !v38 )
    goto LABEL_83;
  if ( (*(_BYTE *)(v38 + 198) & 0x10) != 0 && v99 != 2 )
  {
    v70 = "__constant__";
    if ( (v35 & 4) == 0 )
    {
      v70 = "__managed__";
      if ( (*(_BYTE *)(v103 + 157) & 1) == 0 )
        v70 = "__device__";
    }
    sub_686610(3484, v94, "an automatic", v70);
    v36 = v103;
    v37 = v103;
LABEL_77:
    if ( !unk_4F04C50 )
      goto LABEL_83;
    v38 = *(_QWORD *)(unk_4F04C50 + 32LL);
    if ( !v38 )
      goto LABEL_83;
  }
LABEL_79:
  if ( (*(_BYTE *)(v38 + 198) & 0x10) != 0 )
  {
    if ( (*(_BYTE *)(v36 + 156) & 1) != 0 )
    {
      if ( !unk_4F072F3 || (v89 & 1) == 0 )
        goto LABEL_83;
      goto LABEL_215;
    }
    if ( *(_BYTE *)(v36 + 136) != 2 )
    {
      if ( !unk_4F072F3 || (v89 & 1) == 0 )
        goto LABEL_83;
LABEL_190:
      v37 = v36;
      goto LABEL_83;
    }
    sub_684B00(3512, v36 + 64);
    v36 = v103;
    v37 = v103;
  }
LABEL_80:
  if ( !unk_4F072F3 || (v89 & 1) == 0 || !v36 )
    goto LABEL_83;
  if ( (*(_BYTE *)(v36 + 156) & 1) == 0 )
    goto LABEL_190;
LABEL_215:
  if ( (unsigned int)sub_826000(*(_QWORD *)(v26 + 8), v36, &unk_4F07280, v35) )
    sub_6849F0(7, 3658, v94, *(_QWORD *)(v26 + 8));
  v37 = v103;
LABEL_83:
  if ( *(char *)(v37 + 90) >= 0 )
    sub_8D9350(v92, v94);
  sub_8756F0(v93, v14, v94, *(_QWORD *)(a2 + 352));
  if ( (*(_BYTE *)(*(_QWORD *)v14 + 73LL) & 2) == 0 )
    goto LABEL_96;
  if ( !v87 )
  {
    if ( dword_4F04C5C | dword_4F04C34 && (*(_BYTE *)(v103 + 88) & 0x70) != 0x30
      || strcmp(*(const char **)(v91->m128i_i64[0] + 8), "main") )
    {
      goto LABEL_96;
    }
    if ( dword_4F077C4 != 2 )
    {
      if ( qword_4F077B4 )
      {
        v39 = 5;
        goto LABEL_95;
      }
      if ( !dword_4F077BC || qword_4F077A8 > 0xEA5Fu )
      {
        v39 = 7;
LABEL_95:
        sub_684AA0(v39, 2948, v94);
      }
LABEL_96:
      if ( v96 || !(v87 | v83) )
        goto LABEL_98;
      goto LABEL_99;
    }
    if ( dword_4F077BC )
    {
      if ( (_DWORD)qword_4F077B4 )
        goto LABEL_279;
      if ( qword_4F077A8 <= 0xEA5Fu )
        goto LABEL_96;
    }
    else if ( (_DWORD)qword_4F077B4 )
    {
LABEL_279:
      if ( qword_4F077A0 <= 0x784Fu )
        goto LABEL_96;
    }
    v39 = 7 - ((unsigned int)(dword_4D04964 == 0) - 1);
    goto LABEL_95;
  }
  if ( v96 )
LABEL_98:
    sub_729470(v103, a6);
LABEL_99:
  if ( (BYTE9(v108) & 2) != 0 )
    sub_8642D0();
  v40 = *(_QWORD *)(a2 + 352);
  v41 = dword_4F04C64;
  if ( v40 && !*(_BYTE *)(v40 + 16) )
  {
    v42 = *(__int64 **)(v40 + 8);
    if ( v42 )
      v43 = *v42;
    else
      v43 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 328);
    *(_QWORD *)(a2 + 352) = v43;
  }
  if ( (_DWORD)v41 != -1
    && (*(_BYTE *)(qword_4F04C68[0] + 776 * v41 + 7) & 2) != 0
    && dword_4F077C4 == 2
    && (*(_BYTE *)(v26 - 8) & 1) != 0
    && (v91[1].m128i_i8[2] & 0x40) == 0 )
  {
    v56 = sub_7CAFF0(v91, v26, qword_4F04C68);
    if ( v96 && (v93 & 0x200) == 0 )
    {
      if ( v56 )
        *(_BYTE *)(v56 + 33) |= 0x10u;
LABEL_111:
      v44 = (_BYTE *)v103;
      if ( !*(_QWORD *)(v103 + 256) )
        *(_QWORD *)(v103 + 256) = v90;
      if ( (*(_BYTE *)(a2 + 228) & 2) != 0 )
        v44[90] |= 0x40u;
      if ( v44[136] != 2 )
        goto LABEL_116;
      goto LABEL_172;
    }
  }
  else
  {
    if ( v96 && (v93 & 0x200) == 0 )
      goto LABEL_111;
    v56 = 0;
  }
  v57 = *(_BYTE *)(a2 + 127) & 0x10;
  if ( (*(_BYTE *)(a2 + 228) & 2) != 0 )
    v57 = *(_BYTE *)(a2 + 127) & 0x10 | 0x40u;
  sub_86A3D0(v103, v90, v56, v57, a6);
  v44 = (_BYTE *)v103;
  if ( *(_BYTE *)(v103 + 136) != 2 )
  {
LABEL_116:
    if ( !unk_4D047EC )
      goto LABEL_117;
    goto LABEL_176;
  }
LABEL_172:
  if ( (v44[89] & 1) == 0 )
    goto LABEL_116;
  *(_BYTE *)(*(_QWORD *)(unk_4F04C50 + 32LL) + 206LL) |= 0x20u;
  if ( unk_4D04374 )
    sub_85E9E0(v44, v94);
  sub_85E280(v14, (unsigned int)dword_4F04C5C);
  if ( !unk_4D047EC )
  {
LABEL_117:
    if ( !v96 )
      goto LABEL_118;
    goto LABEL_182;
  }
LABEL_176:
  if ( !(unsigned int)sub_8DD010(v92) || unk_4D03B90 < 0 )
    goto LABEL_117;
  *(_BYTE *)(v103 + 173) |= 2u;
  v58 = sub_86E480(22, v94);
  *(_BYTE *)(v58 + 72) = 0;
  v59 = v58;
  *(_QWORD *)(v58 + 80) = v103;
  if ( unk_4D047E8 && (unsigned int)sub_86D9F0() )
    sub_6851C0(1233, v94);
  if ( !(unsigned int)sub_8D4070(v92) )
    goto LABEL_117;
  if ( !v96 )
    goto LABEL_118;
  *(_BYTE *)(v103 + 173) |= 4u;
  sub_86F5D0(v59);
LABEL_182:
  if ( (unsigned int)sub_8D4E00(v92) )
  {
    v60 = v103;
    *(_BYTE *)(v26 + 88) |= 4u;
    *(_BYTE *)(v14 + 81) |= 1u;
    *(_BYTE *)(v60 + 169) |= 0x10u;
    *(_BYTE *)(v14 + 84) |= 0x80u;
  }
LABEL_118:
  sub_854980(v14, 0);
  *a4 = v100;
  *(_BYTE *)(a2 + 269) = v99;
  result = v82;
  if ( v82 != -1 )
    dword_4F04C34 = v82;
  if ( dword_4F04C58 == -1 )
  {
    result = v103;
    if ( (*(_BYTE *)(v103 + 156) & 7) != 0 )
      return sub_8E3700(*(_QWORD *)(v103 + 120));
  }
  return result;
}
