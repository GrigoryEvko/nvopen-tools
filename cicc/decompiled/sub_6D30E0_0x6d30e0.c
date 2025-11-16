// Function: sub_6D30E0
// Address: 0x6d30e0
//
__int64 __fastcall sub_6D30E0(__int64 a1, __int64 *a2, _QWORD *a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v5; // r12
  _QWORD *v6; // rbx
  _QWORD *v7; // rdx
  unsigned int *v8; // r8
  _QWORD *v9; // r14
  __int64 v10; // rax
  char v11; // al
  int v12; // eax
  __int64 v13; // rsi
  _DWORD *v14; // r15
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  _QWORD *v18; // rdi
  __int64 v19; // r8
  _BYTE *v20; // rax
  __int64 v21; // rax
  int v22; // edx
  unsigned __int16 v24; // ax
  __int64 v25; // rax
  __int64 v26; // r15
  _QWORD *v27; // rdi
  __int64 v28; // rdx
  int v29; // ecx
  char v30; // cl
  __int64 v31; // rax
  char k; // dl
  __int64 v33; // rax
  int v34; // eax
  __int64 v35; // rax
  int v36; // eax
  __int64 v37; // rdx
  char v38; // al
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // rax
  __int64 i; // rdx
  __int64 j; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  int v46; // eax
  unsigned int v47; // eax
  _QWORD *v48; // [rsp-8h] [rbp-358h]
  int v49; // [rsp+Ch] [rbp-344h]
  __int64 v50; // [rsp+10h] [rbp-340h]
  int v51; // [rsp+10h] [rbp-340h]
  int v52; // [rsp+10h] [rbp-340h]
  __int64 v53; // [rsp+10h] [rbp-340h]
  char v54; // [rsp+18h] [rbp-338h]
  _OWORD *v55; // [rsp+18h] [rbp-338h]
  int v56; // [rsp+28h] [rbp-328h]
  __int16 v57; // [rsp+2Eh] [rbp-322h]
  int v58; // [rsp+30h] [rbp-320h]
  _QWORD *v59; // [rsp+30h] [rbp-320h]
  __int64 v60; // [rsp+30h] [rbp-320h]
  unsigned int v61; // [rsp+48h] [rbp-308h] BYREF
  int v62; // [rsp+4Ch] [rbp-304h] BYREF
  __int64 v63; // [rsp+50h] [rbp-300h] BYREF
  __int64 v64; // [rsp+58h] [rbp-2F8h] BYREF
  _BYTE v65[352]; // [rsp+60h] [rbp-2F0h] BYREF
  _OWORD v66[9]; // [rsp+1C0h] [rbp-190h] BYREF
  __m128i v67; // [rsp+250h] [rbp-100h]
  __m128i v68; // [rsp+260h] [rbp-F0h]
  __m128i v69; // [rsp+270h] [rbp-E0h]
  __m128i v70; // [rsp+280h] [rbp-D0h]
  __m128i v71; // [rsp+290h] [rbp-C0h]
  __m128i v72; // [rsp+2A0h] [rbp-B0h]
  __m128i v73; // [rsp+2B0h] [rbp-A0h]
  __m128i v74; // [rsp+2C0h] [rbp-90h]
  __m128i v75; // [rsp+2D0h] [rbp-80h]
  __m128i v76; // [rsp+2E0h] [rbp-70h]
  __m128i v77; // [rsp+2F0h] [rbp-60h]
  __m128i v78; // [rsp+300h] [rbp-50h]
  __m128i v79; // [rsp+310h] [rbp-40h]

  v4 = (__int64)a3;
  v5 = a4;
  v54 = (char)a2;
  v62 = 0;
  if ( a3 )
  {
    if ( *(_BYTE *)(*a3 + 56LL) == 92 )
    {
      a1 = (__int64)a3;
      v6 = v65;
      v9 = 0;
      a2 = (__int64 *)v65;
      sub_6F8AB0(
        (_DWORD)a3,
        (unsigned int)v65,
        (unsigned int)v66,
        0,
        (unsigned int)&v63,
        (unsigned int)&v61,
        (__int64)&v64);
      a3 = v48;
      v58 = 0;
    }
    else
    {
      v6 = v65;
      sub_6F8AB0((_DWORD)a3, (unsigned int)v65, 0, 0, (unsigned int)&v63, (unsigned int)&v61, (__int64)&v64);
      a2 = (__int64 *)v4;
      a1 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v4 + 72LL) + 16LL);
      v58 = 1;
      v9 = (_QWORD *)sub_690A60(a1, v4, v7);
    }
    a4 = *(unsigned int *)(*(_QWORD *)v4 + 44LL);
    v57 = *(_WORD *)(*(_QWORD *)v4 + 48LL);
    v56 = *(_DWORD *)(*(_QWORD *)v4 + 44LL);
    v10 = qword_4D03C50;
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) == 0 )
      goto LABEL_9;
  }
  else
  {
    v8 = &dword_4F063F8;
    v9 = 0;
    v6 = (_QWORD *)a1;
    v58 = 0;
    v63 = *(_QWORD *)&dword_4F063F8;
    v61 = dword_4F06650[0];
    v10 = qword_4D03C50;
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) == 0 )
    {
      v51 = 0;
      goto LABEL_41;
    }
  }
  v11 = *(_BYTE *)(v10 + 16);
  switch ( v11 )
  {
    case 0:
      if ( (unsigned int)sub_6E5430(a1, a2, a3, a4, v8) )
      {
        a2 = &v63;
        a1 = 58;
        sub_6851C0(0x3Au, &v63);
        if ( v4 )
          goto LABEL_47;
      }
      else if ( v4 )
      {
        goto LABEL_47;
      }
LABEL_120:
      v51 = 1;
      goto LABEL_41;
    case 1:
      if ( (unsigned int)sub_6E5430(a1, a2, a3, a4, v8) )
      {
        a1 = 60;
        a2 = &v63;
        sub_6851C0(0x3Cu, &v63);
        if ( v4 )
          goto LABEL_47;
        goto LABEL_120;
      }
      goto LABEL_95;
    case 2:
      if ( (unsigned int)sub_6E5430(a1, a2, a3, a4, v8) )
      {
        a1 = 529;
        a2 = &v63;
        sub_6851C0(0x211u, &v63);
        if ( !v4 )
          goto LABEL_120;
LABEL_47:
        sub_6E6260(v5);
        sub_6E6450(v6);
        if ( !v58 )
        {
          sub_6E6450(v66);
          sub_6E1990(v9);
          if ( v4 )
            goto LABEL_37;
LABEL_49:
          v56 = qword_4F063F0;
          v57 = WORD2(qword_4F063F0);
          sub_7BE280(26, 17, 0, 0);
          --*(_BYTE *)(qword_4F061C8 + 34LL);
          --*(_QWORD *)(qword_4D03C50 + 40LL);
          goto LABEL_37;
        }
        sub_6E6470(v9);
        goto LABEL_36;
      }
LABEL_95:
      if ( v4 )
        goto LABEL_47;
      goto LABEL_120;
  }
  if ( v4 )
    goto LABEL_9;
  v51 = 0;
LABEL_41:
  sub_7B8B50(a1, a2, a3, a4);
  ++*(_BYTE *)(qword_4F061C8 + 34LL);
  ++*(_QWORD *)(qword_4D03C50 + 40LL);
  if ( dword_4D043F8 )
  {
    v24 = word_4F06418[0];
    if ( word_4F06418[0] == 25 )
    {
      sub_684AA0(7u, 0xAE7u, &dword_4F063F8);
      v24 = word_4F06418[0];
    }
  }
  else
  {
    v24 = word_4F06418[0];
  }
  if ( v24 == 73 && dword_4D04428 )
  {
    v45 = sub_6BA760(0, 0);
    a2 = (__int64 *)v66;
    sub_6E9FE0(v45, v66);
  }
  else
  {
    a2 = 0;
    sub_69ED20((__int64)v66, 0, 0, 0x4000);
  }
  v64 = *(_QWORD *)&dword_4F063F8;
  if ( v51 )
    goto LABEL_47;
LABEL_9:
  if ( dword_4F077C4 != 2 )
    goto LABEL_10;
  if ( dword_4F04C44 == -1 )
  {
    a3 = qword_4F04C68;
    v25 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( (*(_BYTE *)(v25 + 6) & 6) == 0 && *(_BYTE *)(v25 + 4) != 12 )
      goto LABEL_53;
  }
  v27 = v6;
  if ( (unsigned int)sub_82ED00(v6, a2, a3) )
  {
    if ( v58 )
LABEL_67:
      sub_721090(v27);
    goto LABEL_91;
  }
  if ( !v58 )
  {
    if ( !(unsigned int)sub_82ED00(v66, 0, v28) )
      goto LABEL_107;
LABEL_91:
    sub_831920(43, 0, (_DWORD)v6, (unsigned int)v66, v5, (unsigned int)&v63, v61, (__int64)&v64);
    v62 = 1;
    goto LABEL_36;
  }
  v27 = v9;
  if ( (unsigned int)sub_82EC50(v9) )
    goto LABEL_67;
LABEL_107:
  if ( dword_4F077C4 != 2 )
    goto LABEL_10;
LABEL_53:
  if ( !(unsigned int)sub_68FE10(v6, 1, 1) )
  {
    if ( v58 )
    {
      v26 = (__int64)v9;
      while ( v26 )
      {
        if ( !*(_BYTE *)(v26 + 8) && (unsigned int)sub_68FE10((_BYTE *)(*(_QWORD *)(v26 + 24) + 8LL), 0, 1) )
          goto LABEL_110;
        if ( !*(_QWORD *)v26 )
          goto LABEL_10;
        if ( *(_BYTE *)(*(_QWORD *)v26 + 8LL) == 3 )
          v26 = sub_6BBB10((_QWORD *)v26);
        else
          v26 = *(_QWORD *)v26;
      }
      goto LABEL_10;
    }
    if ( !(unsigned int)sub_68FE10(v66, 0, 1) )
    {
LABEL_10:
      v12 = v62;
LABEL_11:
      if ( v12 )
        goto LABEL_36;
      goto LABEL_12;
    }
  }
LABEL_110:
  sub_84EC30(43, 0, 1, 1, 0, (_DWORD)v6, (__int64)v66, (__int64)&v63, v61, 0, (__int64)&v64, v5, 0, 0, (__int64)&v62);
  v12 = v62;
  if ( v62 )
  {
    if ( (v54 & 1) != 0 )
    {
      if ( !*(_BYTE *)(v5 + 16) )
        goto LABEL_36;
      v41 = *(_QWORD *)v5;
      for ( i = *(unsigned __int8 *)(*(_QWORD *)v5 + 140LL); (_BYTE)i == 12; i = *(unsigned __int8 *)(v41 + 140) )
        v41 = *(_QWORD *)(v41 + 160);
      if ( !(_BYTE)i )
        goto LABEL_36;
      if ( (unsigned int)sub_6E5430(43, 0, i, v39, v40) )
        sub_6851C0(0x888u, &v63);
      sub_6E6840(v5);
      v12 = v62;
    }
    goto LABEL_11;
  }
LABEL_12:
  if ( HIDWORD(qword_4F077B4) && qword_4F077A8 > 0x9E97u )
  {
    if ( (unsigned int)sub_8D2B80(*v6) )
    {
      sub_6F69D0(v6, 4);
      sub_6F69D0(v66, 0);
      if ( !(unsigned int)sub_8D2930(*(_QWORD *)&v66[0]) )
      {
        v47 = sub_6E92D0();
        sub_6E68E0(v47, v66);
      }
      v60 = sub_6F6F40(v6, 0);
      *(_QWORD *)(v60 + 16) = sub_6F6F40(v66, 0);
      for ( j = *(_QWORD *)v60; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      v44 = sub_73DBF0(93, *(_QWORD *)(j + 160), v60);
      if ( (*(_BYTE *)(v60 + 25) & 1) != 0 )
        *(_BYTE *)(v44 + 25) |= 1u;
      sub_6E7170(v44, v5);
      if ( (*(_BYTE *)(v60 + 25) & 1) != 0 )
        *(_QWORD *)(v5 + 88) = v6[11];
      v62 = 1;
      goto LABEL_36;
    }
    if ( v62 )
      goto LABEL_36;
  }
  v13 = (unsigned int)qword_4F077B4;
  if ( (_DWORD)qword_4F077B4 )
    v13 = 256;
  sub_6F69D0(v6, v13);
  if ( !v62 )
  {
    if ( v58 )
    {
      if ( *v9 )
      {
        v14 = (_DWORD *)sub_6E1A20(v9);
        if ( (unsigned int)sub_6E5430(v9, v13, v15, v16, v17) )
          sub_6851C0(0x8C1u, v14);
        sub_6E6260(v66);
      }
      else
      {
        v37 = v9[3];
        v66[0] = _mm_loadu_si128((const __m128i *)(v37 + 8));
        v66[1] = _mm_loadu_si128((const __m128i *)(v37 + 24));
        v66[2] = _mm_loadu_si128((const __m128i *)(v37 + 40));
        v66[3] = _mm_loadu_si128((const __m128i *)(v37 + 56));
        v66[4] = _mm_loadu_si128((const __m128i *)(v37 + 72));
        v66[5] = _mm_loadu_si128((const __m128i *)(v37 + 88));
        v66[6] = _mm_loadu_si128((const __m128i *)(v37 + 104));
        v66[7] = _mm_loadu_si128((const __m128i *)(v37 + 120));
        v66[8] = _mm_loadu_si128((const __m128i *)(v37 + 136));
        v38 = *(_BYTE *)(v37 + 24);
        if ( v38 == 2 )
        {
          v67 = _mm_loadu_si128((const __m128i *)(v37 + 152));
          v68 = _mm_loadu_si128((const __m128i *)(v37 + 168));
          v69 = _mm_loadu_si128((const __m128i *)(v37 + 184));
          v70 = _mm_loadu_si128((const __m128i *)(v37 + 200));
          v71 = _mm_loadu_si128((const __m128i *)(v37 + 216));
          v72 = _mm_loadu_si128((const __m128i *)(v37 + 232));
          v73 = _mm_loadu_si128((const __m128i *)(v37 + 248));
          v74 = _mm_loadu_si128((const __m128i *)(v37 + 264));
          v75 = _mm_loadu_si128((const __m128i *)(v37 + 280));
          v76 = _mm_loadu_si128((const __m128i *)(v37 + 296));
          v77 = _mm_loadu_si128((const __m128i *)(v37 + 312));
          v78 = _mm_loadu_si128((const __m128i *)(v37 + 328));
          v79 = _mm_loadu_si128((const __m128i *)(v37 + 344));
        }
        else if ( v38 == 5 || v38 == 1 )
        {
          v67.m128i_i64[0] = *(_QWORD *)(v37 + 152);
        }
      }
      v18 = v9;
      v9 = 0;
      sub_6E1990(v18);
    }
    sub_6F69D0(v66, 0);
    v49 = sub_8D2930(*v6);
    if ( v49 )
    {
      v55 = v6;
      LOBYTE(v49) = 1;
      v59 = v66;
    }
    else
    {
      v59 = v6;
      v55 = v66;
    }
    v19 = *v59;
    if ( dword_4F077C0
      && (v53 = *v59, v34 = sub_8D2E30(*v59), v19 = v53, v34)
      && (v35 = sub_8D46C0(v53), v36 = sub_8D2600(v35), v19 = v53, v36) )
    {
      if ( *(char *)(qword_4D03C50 + 20LL) >= 0 )
      {
        v46 = sub_6E53E0(5, 1143, &v63);
        v19 = v53;
        if ( v46 )
        {
          sub_684B30(0x477u, &v63);
          v19 = v53;
        }
      }
      v29 = sub_8D46C0(v19);
    }
    else
    {
      if ( dword_4F077BC )
      {
        if ( !(_DWORD)qword_4F077B4 )
        {
          if ( qword_4F077A8 )
          {
            v20 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
            if ( (dword_4F04C44 != -1 || (v20[6] & 6) != 0 || v20[4] == 12) && (v20[12] & 0x10) == 0 )
            {
              v50 = v19;
              if ( (unsigned int)sub_8D2E30(v19) )
              {
                v21 = sub_8D46C0(v50);
                if ( (unsigned int)sub_8D23B0(v21) )
                {
                  sub_831920(43, 0, (_DWORD)v6, (unsigned int)v66, v5, (unsigned int)&v63, v61, (__int64)&v64);
                  goto LABEL_36;
                }
              }
            }
          }
        }
      }
      if ( (unsigned int)sub_702E30(v59, 142) )
        v29 = sub_8D46C0(*v59);
      else
        v29 = sub_72C930(v59);
    }
    v52 = v29;
    sub_6E9350(v55);
    sub_700E50(92, (_DWORD)v6, (unsigned int)v66, v52, 1, v5, (__int64)&v63, v61, (__int64)&v64);
    v30 = *(_BYTE *)(v5 + 16);
    if ( v30 )
    {
      v31 = *(_QWORD *)v5;
      for ( k = *(_BYTE *)(*(_QWORD *)v5 + 140LL); k == 12; k = *(_BYTE *)(v31 + 140) )
        v31 = *(_QWORD *)(v31 + 160);
      if ( k )
      {
        if ( v30 == 1 && (v49 & 1) != 0 )
        {
          v33 = *(_QWORD *)(v5 + 144);
          if ( *(_BYTE *)(v33 + 24) )
            *(_BYTE *)(v33 + 59) |= 0x20u;
        }
        *(_QWORD *)(v5 + 88) = v59[11];
      }
    }
  }
LABEL_36:
  sub_6E1990(v9);
  if ( !v4 )
    goto LABEL_49;
LABEL_37:
  v22 = *((_DWORD *)v6 + 17);
  *(_WORD *)(v5 + 72) = *((_WORD *)v6 + 36);
  *(_DWORD *)(v5 + 68) = v22;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(v5 + 68);
  *(_DWORD *)(v5 + 76) = v56;
  *(_WORD *)(v5 + 80) = v57;
  *(_QWORD *)&dword_4F061D8 = *(_QWORD *)(v5 + 76);
  sub_6E3280(v5, &v63);
  return sub_6E26D0(1, v5);
}
