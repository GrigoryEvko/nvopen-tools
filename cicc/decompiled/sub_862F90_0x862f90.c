// Function: sub_862F90
// Address: 0x862f90
//
_QWORD *__fastcall sub_862F90(int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v6; // rcx
  __int64 *v7; // r14
  unsigned __int64 v8; // rbx
  __int64 v9; // r13
  char v10; // al
  bool v11; // r15
  _BYTE *v12; // r12
  __m128i *v13; // rdi
  __m128i *v14; // rdi
  char v15; // al
  unsigned __int8 v16; // si
  _BYTE *v17; // rdx
  _BYTE *v18; // rax
  int v19; // r14d
  int v20; // edx
  _DWORD *v21; // rax
  char v22; // r14
  __int64 *v23; // rbx
  _BYTE *v24; // r15
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 *v27; // rdi
  __int64 v28; // rsi
  _QWORD *v29; // rax
  _QWORD *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 *i; // rax
  __int64 v34; // rdx
  bool v35; // zf
  int *v36; // rcx
  bool v37; // sf
  int v38; // eax
  __int64 v39; // r12
  __int64 v40; // rdi
  __int64 v41; // rsi
  __int64 v42; // rdx
  unsigned int *v43; // rdx
  __int64 v44; // rax
  __int64 *v45; // rax
  __int64 v46; // r15
  __int64 j; // r12
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r14
  _QWORD *v52; // rax
  __int64 v53; // rax
  _QWORD *result; // rax
  __int64 v55; // r12
  __int64 v56; // rdi
  int v57; // r8d
  __int64 v58; // r12
  unsigned __int8 v59; // di
  __int64 v60; // rax
  _QWORD *v61; // rdi
  __int64 v62; // rax
  char v63; // al
  __int64 v64; // rsi
  _QWORD *v65; // rax
  _QWORD *v66; // rdx
  _QWORD *v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rdi
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  _QWORD *v74; // r15
  _QWORD *v75; // r14
  __int64 v76; // rax
  __int64 **v77; // rdi
  int v78; // edx
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rax
  unsigned __int64 v82; // rdx
  __int64 v83; // rax
  unsigned __int64 v84; // rdx
  bool v85; // [rsp+Bh] [rbp-95h]
  _BYTE *v87; // [rsp+10h] [rbp-90h]
  int v88; // [rsp+18h] [rbp-88h]
  int v89; // [rsp+1Ch] [rbp-84h]
  _BOOL4 v90; // [rsp+1Ch] [rbp-84h]
  __m128i v91; // [rsp+30h] [rbp-70h] BYREF
  char v92; // [rsp+40h] [rbp-60h]
  char v93; // [rsp+41h] [rbp-5Fh]
  __int64 v94; // [rsp+48h] [rbp-58h]

  v6 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v7 = *(__int64 **)(v6 + 24);
  v8 = *(unsigned __int8 *)(v6 + 4);
  v9 = v6;
  if ( !v7 )
    v7 = (__int64 *)(v6 + 32);
  if ( (_BYTE)v8 == 17 )
  {
    v56 = *(_QWORD *)(v6 + 184);
    v87 = *(_BYTE **)(v56 + 32);
    v11 = v87 != 0;
    goto LABEL_159;
  }
  if ( (_BYTE)v8 )
  {
    v10 = *(_BYTE *)(v6 + 4);
  }
  else
  {
    v10 = *(_BYTE *)(v6 + 10) & 4;
    if ( v10 )
    {
      v87 = 0;
      v11 = 0;
      goto LABEL_7;
    }
    v55 = qword_4D04970;
    if ( qword_4D04970 )
    {
      do
      {
        sub_8790E0(v55);
        sub_879210(v55);
        v55 = *(_QWORD *)(v55 + 16);
      }
      while ( v55 );
      v10 = *(_BYTE *)(v9 + 4);
    }
    unk_4D04968 = 1;
  }
  v56 = *(_QWORD *)(v9 + 184);
  v87 = 0;
  v11 = ((v10 - 15) & 0xFD) == 0 || v10 == 2;
  if ( v11 )
  {
    v11 = 0;
LABEL_159:
    v57 = 1;
    if ( (*(_BYTE *)(v9 + 10) & 4) != 0 )
      goto LABEL_105;
  }
  v57 = 0;
LABEL_105:
  sub_861D10(v56, v8, v7, 0, v57, a1);
  v9 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( !*(_BYTE *)(v9 + 4) )
  {
    v58 = qword_4F5FD18;
    if ( qword_4F5FD18 )
    {
      while ( 1 )
      {
        if ( (*(_BYTE *)(*(_QWORD *)(v58 + 8) + 203LL) & 0x20) != 0 )
        {
          if ( dword_4D04964 )
            v59 = unk_4F07471;
          else
            v59 = ((unsigned int)qword_4F077B4 | dword_4F077C0) == 0 ? 7 : 5;
          sub_684AA0(v59, (*(_DWORD *)(v58 + 24) == 0) + 1030, (_DWORD *)(v58 + 16));
        }
        v60 = *(_QWORD *)v58;
        *(_QWORD *)v58 = qword_4F5FD10;
        qword_4F5FD10 = v58;
        if ( !v60 )
          break;
        v58 = v60;
      }
    }
  }
LABEL_7:
  v12 = *(_BYTE **)(v9 + 184);
  if ( v12 && dword_4F077C4 == 2 && ((_BYTE)v8 == 2 || (_BYTE)v8 == 17) )
    nullsub_9();
  v13 = *(__m128i **)(v9 + 440);
  if ( v13 && qword_4F074B0 )
  {
    sub_854000(v13);
    *(_QWORD *)(v9 + 440) = 0;
  }
  v14 = *(__m128i **)(v9 + 432);
  if ( v14 )
    sub_8547F0(v14);
  if ( dword_4F077C4 == 2 )
  {
    if ( *(_QWORD *)(v9 + 536) )
    {
      sub_85FE80(dword_4F04C64, 0, 0);
      a5 = (unsigned int)dword_4F04C64;
      if ( dword_4F04C64 )
        sub_85FE80(*(_DWORD *)(v9 + 552), 1, 0);
      v64 = *(_QWORD *)(v9 + 536);
      if ( v64 )
      {
        v65 = *(_QWORD **)(v9 + 536);
        do
        {
          v66 = v65;
          v65 = (_QWORD *)*v65;
        }
        while ( v65 );
        *v66 = unk_4D049E8;
        unk_4D049E8 = v64;
        *(_QWORD *)(v9 + 536) = 0;
      }
    }
    v67 = *(_QWORD **)(v9 + 248);
    if ( v67 )
    {
      *v67 = 0;
      *(_QWORD *)(*(_QWORD *)(v9 + 248) + 8LL) = 0;
      *(_QWORD *)(*(_QWORD *)(v9 + 248) + 16LL) = 0;
    }
    if ( (_BYTE)v8 == 2 || (_BYTE)v8 == 17 )
    {
      if ( (*(_BYTE *)(v9 + 10) & 4) != 0 )
      {
LABEL_209:
        if ( (_BYTE)v8 != 17 )
          goto LABEL_149;
        v81 = *((_QWORD *)v12 + 4);
        if ( (*(_BYTE *)(v81 + 193) & 0x10) == 0 && (*(_BYTE *)(v81 + 206) & 8) == 0 && (*(_BYTE *)(v9 + 10) & 4) == 0 )
          sub_8752D0(v12);
        goto LABEL_148;
      }
      v80 = qword_4F06BC0;
      if ( *(_BYTE *)qword_4F06BC0 != 5 && *(_BYTE *)qword_4F06BC0 != 2 )
        goto LABEL_206;
    }
    else
    {
      if ( (_BYTE)v8 != 15 )
      {
        if ( (unsigned __int8)v8 > 0xDu )
        {
LABEL_149:
          v69 = *(_QWORD *)(v9 + 608);
          a6 = (__int64 *)qword_4F5FCF0;
          if ( v69 )
          {
            while ( 1 )
            {
              v70 = *(_QWORD *)v69;
              *(_BYTE *)(*(_QWORD *)(v69 + 8) + 83LL) = (16 * (*(_BYTE *)(v69 + 24) & 1))
                                                      | *(_BYTE *)(*(_QWORD *)(v69 + 8) + 83LL) & 0xEF;
              *(_QWORD *)v69 = a6;
              a6 = (__int64 *)v69;
              *(_QWORD *)(v69 + 8) = 0;
              *(_QWORD *)(v69 + 16) = 0;
              qword_4F5FCF0 = v69;
              if ( !v70 )
                break;
              v69 = v70;
            }
          }
          if ( unk_4F04C48 == dword_4F04C64 )
          {
            v71 = *(_QWORD *)(v9 + 368);
            if ( v71 )
            {
              v72 = *(_QWORD *)(v71 + 88);
              if ( *(_QWORD *)(v72 + 48) == *(_QWORD *)(v9 + 360) )
                *(_QWORD *)(v72 + 48) = 0;
            }
          }
          goto LABEL_15;
        }
        v68 = 10946;
        if ( !_bittest64(&v68, v8) )
        {
          if ( (unsigned __int8)(v8 - 3) <= 2u )
          {
            v82 = *(_QWORD *)(v9 + 688);
            v83 = *(_QWORD *)(**(_QWORD **)(v9 + 224) + 96LL);
            if ( v82 > *(_QWORD *)(v83 + 184) )
              *(_QWORD *)(v83 + 184) = v82;
            v84 = *(_QWORD *)(v9 + 696);
            if ( v84 > *(_QWORD *)(v83 + 192) )
              *(_QWORD *)(v83 + 192) = v84;
          }
          goto LABEL_149;
        }
LABEL_148:
        qword_4F06BC0 = *(_QWORD *)(v9 + 496);
        goto LABEL_149;
      }
      if ( (*(_BYTE *)(v9 + 10) & 4) != 0 )
        goto LABEL_149;
      v80 = qword_4F06BC0;
      if ( *(_BYTE *)qword_4F06BC0 != 2 && *(_BYTE *)qword_4F06BC0 != 5 )
        goto LABEL_208;
    }
    qword_4F06BC0 = *(_QWORD *)(v80 + 32);
LABEL_206:
    if ( (_BYTE)v8 == 2 && (a1 & 0x8000) != 0 )
    {
      qword_4F06BC0 = *(_QWORD *)(qword_4F06BC0 + 32LL);
      goto LABEL_149;
    }
LABEL_208:
    sub_733F40();
    goto LABEL_209;
  }
LABEL_15:
  if ( (_BYTE)v8 == 3 || !(_BYTE)v8 )
    *(_BYTE *)(*(_QWORD *)(v9 + 24) + 144LL) |= 1u;
  v15 = *(_BYTE *)(v9 + 4);
  if ( !v15
    || (v16 = *(_BYTE *)(v9 + 10),
        unk_4F06C5A = (v16 >> 4) & 3,
        unk_4F06C59 = v16 >> 6,
        unk_4F06C58 = *(_BYTE *)(v9 + 11) & 3,
        v15 == 17) )
  {
    if ( v12 )
    {
      v62 = *(_QWORD *)(v9 + 328);
      if ( v62 )
      {
        *((_QWORD *)v12 + 32) = v62;
        v63 = *(_BYTE *)(v9 + 4);
        if ( !v63 )
        {
          *(_QWORD *)(qword_4D03FF0 + 136) = *(_QWORD *)(v9 + 336);
          v63 = *(_BYTE *)(v9 + 4);
        }
        *(_QWORD *)(v9 + 328) = 0;
        *(_QWORD *)(v9 + 336) = 0;
        if ( v63 == 17 )
          sub_869840(v12);
      }
      v18 = *(_BYTE **)(v9 + 304);
      if ( !v18 )
        goto LABEL_22;
      goto LABEL_21;
    }
    v12 = *(_BYTE **)(v9 + 304);
    v17 = v12;
    if ( !v12 )
      goto LABEL_22;
    goto LABEL_116;
  }
  v17 = *(_BYTE **)(v9 + 304);
  v18 = v17;
  if ( v17 )
  {
    if ( v12 )
    {
LABEL_21:
      *((_QWORD *)v12 + 20) = v18;
      goto LABEL_22;
    }
LABEL_116:
    if ( (_BYTE)v8 == 1 && *(_BYTE *)(v9 - 772) != 1 )
    {
      v12 = sub_732EF0(v9);
      if ( v12 )
      {
        v18 = *(_BYTE **)(v9 + 304);
        goto LABEL_21;
      }
      v17 = *(_BYTE **)(v9 + 304);
    }
    if ( *(_QWORD *)(v9 - 472) )
    {
      *((_QWORD *)v17 + 1) = *(_QWORD *)(*(_QWORD *)(v9 - 464) + 8LL);
      **(_QWORD **)(v9 - 464) = *(_QWORD *)(v9 + 304);
    }
    else
    {
      *(_QWORD *)(v9 - 472) = v17;
    }
    v12 = 0;
    *(_QWORD *)(v9 - 464) = *(_QWORD *)(v9 + 312);
  }
LABEL_22:
  v19 = *(_DWORD *)(v9 + 192);
  v88 = 1;
  if ( dword_4F073B8[0] != v19 )
  {
    v20 = dword_4F04C64 - 1;
    if ( dword_4F04C64 - 1 < 0 )
    {
LABEL_157:
      v88 = 0;
    }
    else
    {
      v21 = (_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 - 584);
      while ( *v21 != v19 )
      {
        --v20;
        v21 -= 194;
        if ( v20 == -1 )
          goto LABEL_157;
      }
      v88 = 1;
    }
  }
  if ( !*(_QWORD *)(v9 + 240) )
    goto LABEL_42;
  v89 = *(_DWORD *)(v9 + 192);
  v22 = v8;
  v23 = *(__int64 **)(v9 + 240);
  v85 = v11;
  v24 = v12;
  do
  {
    while ( 1 )
    {
      v26 = v23[3];
      v25 = v23[1];
      if ( !*((_DWORD *)v23 + 4) )
        break;
      if ( (*(_BYTE *)(v26 + 207) & 0x20) != 0 )
      {
        if ( (*(_BYTE *)(v26 + 193) & 0x20) != 0 && (!qword_4F04C50 || *(_QWORD *)(qword_4F04C50 + 32LL) != v26) )
        {
          if ( !(unsigned int)sub_8D3EA0(*(_QWORD *)(v25 + 160)) )
          {
LABEL_37:
            v26 = v23[3];
            v25 = v23[1];
            goto LABEL_38;
          }
          v25 = v23[1];
        }
        *(_QWORD *)(v25 + 160) = *(_QWORD *)(*(_QWORD *)(v26 + 152) + 160LL);
        goto LABEL_37;
      }
LABEL_38:
      *(_QWORD *)(v26 + 152) = v25;
      v23 = (__int64 *)*v23;
      if ( !v23 )
        goto LABEL_41;
    }
    *(_QWORD *)(v26 + 120) = v25;
    v23 = (__int64 *)*v23;
  }
  while ( v23 );
LABEL_41:
  LOBYTE(v8) = v22;
  v12 = v24;
  v19 = v89;
  v11 = v85;
LABEL_42:
  if ( v11 )
  {
    v90 = sub_862610(v87);
    if ( v90 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 10) & 8) != 0 )
    {
      if ( !(unsigned int)sub_862E50((__int64)v87) )
        sub_85B450((__int64)v12, 1, 0);
      v90 = 1;
    }
    else
    {
      if ( dword_4D03FE8[0] && (unsigned int)sub_862900((__int64)v87, 1) )
      {
        sub_72AC90();
      }
      else if ( !(unsigned int)sub_862E50((__int64)v87) )
      {
        sub_85B450((__int64)v12, 0, 0);
      }
      sub_732AF0((__int64)v87);
    }
  }
  else
  {
    v90 = 0;
  }
  if ( v12 && *((_DWORD *)v12 + 60) == dword_4F04C64 )
    *((_DWORD *)v12 + 60) = -1;
  if ( !v88 )
  {
    if ( v90 )
    {
      if ( (v87[194] & 2) != 0 )
      {
        sub_734690(v12);
        v87[193] |= 0x20u;
        v90 = 0;
      }
    }
    else
    {
      if ( v12[28] == 17 )
        sub_85BD30((__int64)v12);
      sub_823780(v19);
    }
  }
  if ( (_BYTE)v8 == 9 )
  {
    v77 = **(__int64 ****)(v9 + 408);
    v78 = dword_4F04C64 - 1;
    if ( dword_4F04C64 - 1 < 0 )
    {
LABEL_224:
      sub_85FC20(v77, 0);
    }
    else
    {
      v79 = qword_4F04C68[0] + 776LL * v78;
      while ( *(_BYTE *)(v79 + 4) != 9 || **(__int64 ****)(v79 + 408) != v77 )
      {
        --v78;
        v79 -= 776;
        if ( v78 == -1 )
          goto LABEL_224;
      }
      sub_85BC50(v77, *(_QWORD *)(v79 + 376));
    }
    v27 = *(__int64 **)(v9 + 664);
    if ( v27 )
    {
LABEL_53:
      sub_85B940(v27);
      v28 = *(_QWORD *)(v9 + 664);
      if ( v28 )
      {
        v29 = *(_QWORD **)(v9 + 664);
        do
        {
          v30 = v29;
          v29 = (_QWORD *)*v29;
        }
        while ( v29 );
        *v30 = qword_4F5FD48;
        qword_4F5FD48 = v28;
      }
      *(_QWORD *)(v9 + 664) = 0;
      if ( (_BYTE)v8 == 8 )
        goto LABEL_121;
      if ( (_BYTE)v8 != 9 )
        goto LABEL_59;
    }
LABEL_122:
    v61 = *(_QWORD **)(v9 + 672);
    qword_4F04C18 = v61;
    if ( v61 && v61[2] )
      sub_85BF70((__int64)v61);
    goto LABEL_59;
  }
  v27 = *(__int64 **)(v9 + 664);
  if ( v27 )
    goto LABEL_53;
  if ( (_BYTE)v8 == 8 )
  {
LABEL_121:
    if ( (a1 & 0x800000) == 0 )
      goto LABEL_122;
  }
LABEL_59:
  v31 = *(_QWORD *)(v9 + 408);
  if ( v31 )
  {
    v32 = *(_QWORD *)(v31 + 72);
    if ( v32 )
    {
      for ( i = *(__int64 **)(v32 + 24); i; i = (__int64 *)*i )
      {
        while ( 1 )
        {
          if ( *((_DWORD *)i + 8) == 1 )
          {
            v34 = i[1];
            if ( *(_DWORD *)(v34 + 40) == *(_DWORD *)v9 )
              break;
          }
          i = (__int64 *)*i;
          if ( !i )
            goto LABEL_67;
        }
        *(_QWORD *)(v34 + 88) = 0;
      }
    }
  }
LABEL_67:
  v35 = *(_BYTE *)(v9 + 4) == 17;
  dword_4F04C60 = *(_DWORD *)(v9 + 564);
  if ( v35 && (v73 = *(_QWORD *)(v9 + 688)) != 0 )
  {
    v74 = *(_QWORD **)(v9 + 688);
    v75 = (_QWORD *)(v73 + 128);
    do
    {
      if ( *v74 )
        sub_878490(*v74);
      ++v74;
    }
    while ( v74 != v75 );
    **(_QWORD **)(v9 + 688) = qword_4F5FD20;
    v76 = *(_QWORD *)(v9 + 688);
    *(_QWORD *)(v9 + 688) = 0;
    qword_4F5FD20 = v76;
    if ( v90 )
    {
LABEL_170:
      sub_72A130((__int64)v12);
      sub_734690(v12);
      v87[193] |= 0x20u;
    }
  }
  else if ( v90 )
  {
    goto LABEL_170;
  }
  v36 = &dword_4F04C64;
  v37 = dword_4F04C64 - 1 < 0;
  v38 = --dword_4F04C64;
  if ( !v37 )
  {
    v39 = qword_4F04C68[0] + 776LL * v38;
    v40 = *(unsigned int *)(v39 + 972);
    if ( (_DWORD)v40 != *(_DWORD *)(v39 + 968) )
      sub_7296B0(v40);
    v41 = *(unsigned int *)(v39 + 520);
    dword_4F04C38 = (*(_BYTE *)(v39 + 5) & 8) != 0;
    dword_4F04C58 = *(_DWORD *)(v39 + 400);
    dword_4F04C34 = v41;
    v42 = 0;
    if ( dword_4F04C58 != -1 )
    {
      v36 = (int *)qword_4F04C68;
      v42 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 184);
    }
    qword_4F04C50 = v42;
    dword_4F04C5C = *(_DWORD *)(v39 + 1120);
    dword_4F04C44 = *(_DWORD *)(v39 + 348);
    v43 = &dword_4F04C3C;
    dword_4F04C3C = (*(_BYTE *)(v39 + 7) & 0x10) != 0;
    v44 = *(_QWORD *)(v39 + 1104);
    if ( v44 )
    {
      v43 = *(unsigned int **)(v39 + 336);
      if ( v43 )
      {
        *(_QWORD *)v43 = v44;
        v43 = *(unsigned int **)(v39 + 336);
        *(_QWORD *)(*(_QWORD *)(v39 + 1104) + 8LL) = v43;
      }
      else
      {
        *(_QWORD *)(v39 + 328) = v44;
      }
      *(_QWORD *)(v39 + 336) = *(_QWORD *)(v39 + 1112);
    }
    if ( *(_BYTE *)(v39 + 780) == 10 )
      sub_82C0E0(v40, v41, (__int64)v43, (__int64)v36, a5, a6);
  }
  if ( dword_4F077C4 == 2 )
  {
    if ( (unsigned __int8)(v8 - 6) <= 1u )
      --unk_4F04C28;
    unk_4F04C48 = *(_DWORD *)(v9 + 352);
    unk_4F04C2C = *(_DWORD *)(v9 + 448);
    dword_4F04C40 = *(_DWORD *)(v9 + 472);
    qword_4D03C50 = *(_QWORD *)(v9 + 480);
  }
  if ( (*(_BYTE *)(v9 + 8) & 8) != 0 )
  {
    v45 = *(__int64 **)(v9 + 24);
    if ( !v45 )
      v45 = (__int64 *)(v9 + 32);
    v46 = *v45;
    for ( j = qword_4F04C68[0] + 776LL * dword_4F04C64; v46; v46 = *(_QWORD *)(v46 + 16) )
    {
      while ( 1 )
      {
        sub_878710(v46, &v91);
        if ( (v93 & 0x40) == 0 )
          v92 &= ~0x80u;
        v94 = 0;
        v51 = sub_7D5DD0(&v91, 0, v48, v49, v50);
        if ( v51 )
        {
          if ( *(_DWORD *)(v51 + 40) != *(_DWORD *)j )
            break;
        }
LABEL_86:
        v46 = *(_QWORD *)(v46 + 16);
        if ( !v46 )
          goto LABEL_97;
      }
      v52 = *(_QWORD **)(j + 608);
      if ( v52 )
      {
        while ( v51 != v52[1] )
        {
          v52 = (_QWORD *)*v52;
          if ( !v52 )
            goto LABEL_94;
        }
        goto LABEL_86;
      }
LABEL_94:
      v53 = qword_4F5FCF0;
      if ( qword_4F5FCF0 )
        qword_4F5FCF0 = *(_QWORD *)qword_4F5FCF0;
      else
        v53 = sub_823970(32);
      *(_QWORD *)v53 = 0;
      *(_BYTE *)(v53 + 24) = 0;
      *(_QWORD *)(v53 + 8) = v51;
      *(_QWORD *)(v53 + 16) = v46;
      *(_BYTE *)(v53 + 24) = (*(_BYTE *)(v51 + 83) & 0x10) != 0;
      *(_BYTE *)(v51 + 83) |= 0x10u;
      *(_QWORD *)v53 = *(_QWORD *)(j + 608);
      *(_QWORD *)(j + 608) = v53;
    }
  }
LABEL_97:
  result = &qword_4F04C50;
  if ( !qword_4F04C50 )
  {
    if ( qword_4F5FCE8 )
    {
      if ( *(_BYTE *)(v9 + 4) )
      {
        result = &dword_4F04C44;
        if ( dword_4F04C44 == -1 )
        {
          result = (_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
          if ( (*((_BYTE *)result + 6) & 6) == 0 && *((_BYTE *)result + 4) != 12 )
          {
            result = (_QWORD *)sub_7217A0();
            if ( result )
              return (_QWORD *)sub_8628A0();
          }
        }
      }
    }
  }
  return result;
}
