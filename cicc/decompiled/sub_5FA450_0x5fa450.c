// Function: sub_5FA450
// Address: 0x5fa450
//
__int64 __fastcall sub_5FA450(__int64 a1, _QWORD *a2, __int64 a3, const __m128i *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  char v8; // dl
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned int v13; // r13d
  __int64 v14; // rbx
  __int64 v15; // r14
  __int64 v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // r13
  __int64 v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // r13
  __int64 v23; // rax
  char v24; // al
  char v25; // al
  char v26; // al
  __int64 v27; // rax
  char v28; // al
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 result; // rax
  char v34; // al
  __int64 v35; // rsi
  __int64 v36; // rdx
  char v37; // al
  __int64 v38; // rdi
  char v39; // al
  __int64 v40; // rax
  __int64 v41; // rax
  char v42; // al
  __int64 v43; // rax
  _QWORD *v44; // r12
  _QWORD *v45; // r14
  int v46; // r15d
  _QWORD *v47; // r12
  __int64 v48; // rax
  int v49; // ebx
  __int64 v50; // r13
  __int64 v51; // rax
  int v52; // r10d
  __int64 v53; // rax
  int v54; // r14d
  void *v55; // r15
  __int64 v56; // rcx
  int v57; // eax
  int v58; // r10d
  _QWORD *v59; // rbx
  int v60; // r12d
  __int64 v61; // r8
  __int64 v62; // rax
  __int64 v63; // rax
  int v64; // eax
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // rax
  char v72; // dl
  __int64 v73; // rcx
  bool v74; // zf
  __int64 v75; // rax
  __int64 v76; // rsi
  __int64 v77; // rdi
  int v78; // eax
  __int64 v79; // rdi
  __int64 v80; // rsi
  __int64 v81; // [rsp+0h] [rbp-D0h]
  __int64 v82; // [rsp+8h] [rbp-C8h]
  __int64 v83; // [rsp+10h] [rbp-C0h]
  __int64 v84; // [rsp+18h] [rbp-B8h]
  _QWORD *v85; // [rsp+20h] [rbp-B0h]
  int v86; // [rsp+20h] [rbp-B0h]
  const __m128i *v88; // [rsp+30h] [rbp-A0h]
  __int64 v89; // [rsp+38h] [rbp-98h]
  __int64 v91; // [rsp+48h] [rbp-88h]
  __int64 v92; // [rsp+50h] [rbp-80h]
  bool v94; // [rsp+62h] [rbp-6Eh]
  char v95; // [rsp+63h] [rbp-6Dh]
  int v96; // [rsp+64h] [rbp-6Ch]
  __int64 v97; // [rsp+68h] [rbp-68h]
  __m128i v98; // [rsp+70h] [rbp-60h]
  __int64 v99; // [rsp+70h] [rbp-60h]
  __int64 v100; // [rsp+70h] [rbp-60h]
  __int64 v101; // [rsp+70h] [rbp-60h]
  __int64 v102; // [rsp+80h] [rbp-50h]
  __int64 v104[7]; // [rsp+98h] [rbp-38h] BYREF

  v6 = a6;
  v8 = *(_BYTE *)(a6 + 269);
  v9 = *(_QWORD *)(a6 + 288);
  v102 = *(_QWORD *)a5;
  v94 = v8 == 2;
  v74 = *(_BYTE *)(v9 + 140) == 12;
  v10 = v9;
  v104[0] = 0;
  if ( v74 )
  {
    do
      v10 = *(_QWORD *)(v10 + 160);
    while ( *(_BYTE *)(v10 + 140) == 12 );
  }
  v11 = **(_QWORD **)(v10 + 168);
  if ( v11 && (*(_BYTE *)(v11 + 35) & 1) != 0 )
  {
    if ( v8 == 2 )
      sub_684AA0(7, 3210, a6 + 32);
    v94 = 1;
  }
  *(_BYTE *)(v6 + 122) = ((a4[4].m128i_i8[0] & 4) != 0) | *(_BYTE *)(v6 + 122) & 0xFE;
  if ( (*(_BYTE *)(a1 + 17) & 0x20) == 0 )
  {
    if ( (*(_BYTE *)(a1 + 16) & 0x40) != 0 )
    {
      sub_6851C0(2488, a1 + 8);
      *(__m128i *)a1 = _mm_loadu_si128(xmmword_4F06660);
      *(__m128i *)(a1 + 16) = _mm_loadu_si128(&xmmword_4F06660[1]);
      *(__m128i *)(a1 + 32) = _mm_loadu_si128(&xmmword_4F06660[2]);
      v12 = *(_QWORD *)dword_4F07508;
      v98 = _mm_loadu_si128(&xmmword_4F06660[3]);
      *(_BYTE *)(a1 + 17) |= 0x20u;
      *(_QWORD *)(a1 + 8) = v12;
      *(__m128i *)(a1 + 48) = v98;
    }
    else if ( (unsigned int)sub_6461D0(a1, v9, 0) )
    {
      v16 = 556;
      if ( (*(_BYTE *)(a1 + 56) & 0xFD) != 1 )
        v16 = 507;
      sub_6851C0(v16, a1 + 8);
      *(_BYTE *)(a1 + 17) |= 0x20u;
      *(_QWORD *)(a1 + 24) = 0;
    }
  }
  sub_646070(v9, v102, a1);
  if ( (*(_BYTE *)(a1 + 17) & 0x20) != 0 || (v40 = sub_5E69C0(a1, v102), (v92 = v40) == 0) )
  {
LABEL_12:
    sub_8DCB20(v9);
    v13 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v102 + 168) + 152LL) + 240LL);
    v14 = sub_647630(20, a1, v13, 0);
    goto LABEL_13;
  }
  if ( *(char *)(v6 + 130) < 0 )
    goto LABEL_75;
  v95 = *(_BYTE *)(v40 + 80);
  v41 = *(_QWORD *)(a3 + 176);
  if ( v95 != 17 )
  {
    v91 = v9;
    v88 = a4;
    v101 = v92;
    v89 = v6;
    v81 = *(_QWORD *)(v41 + 16);
    v42 = v95;
    goto LABEL_71;
  }
  v70 = *(_QWORD *)(v92 + 88);
  v81 = *(_QWORD *)(v41 + 16);
  v101 = v70;
  if ( !v70 )
    goto LABEL_75;
  v91 = v9;
  v42 = *(_BYTE *)(v70 + 80);
  v88 = a4;
  v89 = v6;
  while ( 1 )
  {
LABEL_71:
    v97 = v101;
    if ( v42 != 16 )
      goto LABEL_195;
    if ( (*(_BYTE *)(v101 + 96) & 4) == 0 )
      goto LABEL_73;
    v97 = **(_QWORD **)(v101 + 88);
    v42 = *(_BYTE *)(v97 + 80);
    if ( v42 == 24 )
    {
      v97 = *(_QWORD *)(v97 + 88);
      if ( *(_BYTE *)(v97 + 80) != 20 )
        goto LABEL_73;
    }
    else
    {
LABEL_195:
      if ( v42 != 20 )
        goto LABEL_73;
    }
    v43 = *(_QWORD *)(v97 + 88);
    v82 = *(_QWORD *)(v43 + 176);
    v44 = **(_QWORD ***)(v43 + 328);
    v83 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v43 + 104) + 176LL) + 16LL);
    v84 = *(_QWORD *)(v82 + 152);
    if ( v44 && a2 )
    {
      v85 = **(_QWORD ***)(v43 + 328);
      v45 = v85;
      v46 = 0;
      v47 = a2;
      v96 = 0;
      do
      {
        v48 = sub_892BC0(v45);
        v49 = *(_DWORD *)(v48 + 4);
        v50 = v48;
        v51 = sub_892BC0(v47);
        v52 = *(_DWORD *)(v51 + 4);
        if ( v49 == v52 )
          break;
        if ( v49 < v52 )
        {
          *(_DWORD *)(v50 + 4) = v52;
          v96 = 1;
        }
        else
        {
          *(_DWORD *)(v51 + 4) = v49;
          v46 = 1;
        }
        v45 = (_QWORD *)*v45;
        v47 = (_QWORD *)*v47;
        if ( !v45 )
          break;
      }
      while ( v47 );
      v44 = v85;
      v54 = v46;
    }
    else
    {
      v96 = 0;
      v54 = 0;
      v52 = 0;
      v49 = 0;
    }
    LODWORD(v55) = 0;
    v56 = dword_4F077B8;
    if ( dword_4F077B8 )
    {
      v56 = 0;
      if ( unk_4F04C48 != -1 )
      {
        if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 6) != 0
          || (v55 = &loc_1000000, v56 = 16, v97 != v101) )
        {
          LODWORD(v55) = 0;
          v56 = 0;
        }
      }
    }
    v86 = v52;
    v57 = sub_89B3C0(v44, a2, 0, v56, 0, 8);
    v58 = v86;
    if ( v57 )
    {
      v64 = sub_8DE890(v84, v91, (unsigned int)v55, 0);
      v58 = v86;
      if ( v64 )
      {
        v65 = v84;
        if ( *(_BYTE *)(v84 + 140) == 12 )
        {
          do
            v65 = *(_QWORD *)(v65 + 160);
          while ( *(_BYTE *)(v65 + 140) == 12 );
        }
        else
        {
          v65 = v84;
        }
        v66 = *(_QWORD *)(v65 + 168);
        v67 = v91;
        if ( *(_BYTE *)(v91 + 140) == 12 )
        {
          do
            v67 = *(_QWORD *)(v67 + 160);
          while ( *(_BYTE *)(v67 + 140) == 12 );
        }
        else
        {
          v67 = v91;
        }
        if ( !*(_QWORD *)(v66 + 40) )
          break;
        v68 = *(_QWORD *)(v67 + 168);
        if ( !*(_QWORD *)(v68 + 40)
          || ((*(_BYTE *)(v68 + 18) ^ *(_BYTE *)(v66 + 18)) & 0x7F) == 0
          && ((*(_BYTE *)(v68 + 19) ^ *(_BYTE *)(v66 + 19)) & 0xC0) == 0 )
        {
          break;
        }
      }
    }
    if ( v96 )
    {
      if ( !v44 )
        goto LABEL_73;
      v96 = 0;
      goto LABEL_119;
    }
    if ( v54 && a2 )
    {
      v96 = 0;
LABEL_105:
      v59 = a2;
      v60 = v58;
      do
      {
        *(_DWORD *)(sub_892BC0(v59) + 4) = v60;
        v59 = (_QWORD *)*v59;
      }
      while ( v59 );
      goto LABEL_107;
    }
LABEL_73:
    if ( v95 != 17 || (v53 = *(_QWORD *)(v101 + 8), (v101 = v53) == 0) )
    {
      v9 = v91;
      a4 = v88;
      v6 = v89;
      goto LABEL_75;
    }
    v42 = *(_BYTE *)(v53 + 80);
  }
  if ( !v96 )
  {
    if ( !v54 || !a2 )
      goto LABEL_108;
    v96 = 1;
    goto LABEL_105;
  }
  if ( !v44 )
    goto LABEL_108;
  do
  {
LABEL_119:
    *(_DWORD *)(sub_892BC0(v44) + 4) = v49;
    v44 = (_QWORD *)*v44;
  }
  while ( v44 );
LABEL_107:
  if ( !v96 )
    goto LABEL_73;
LABEL_108:
  if ( !(unsigned int)sub_739400(v83, v81) || !(unsigned int)sub_739400(*(_QWORD *)(v82 + 216), *(_QWORD *)(v89 + 400)) )
    goto LABEL_73;
  v9 = v91;
  a4 = v88;
  v6 = v89;
  if ( v97 == v101 )
  {
    v71 = v84;
    v72 = *(_BYTE *)(v84 + 140);
    if ( v72 == 12 )
    {
      do
        v71 = *(_QWORD *)(v71 + 160);
      while ( *(_BYTE *)(v71 + 140) == 12 );
    }
    else
    {
      v71 = v84;
    }
    v73 = *(unsigned __int8 *)(v91 + 140);
    v74 = *(_QWORD *)(*(_QWORD *)(v71 + 168) + 40LL) == 0;
    v75 = v91;
    if ( (_BYTE)v73 == 12 )
    {
      do
        v75 = *(_QWORD *)(v75 + 160);
      while ( *(_BYTE *)(v75 + 140) == 12 );
    }
    if ( !v74 != (*(_QWORD *)(*(_QWORD *)(v75 + 168) + 40LL) != 0)
      && (!(dword_4F077B4 | dword_4F077BC)
       || (_BYTE)v73 == 7
       && v72 == 7
       && ((v76 = *(_QWORD *)(v91 + 160), v77 = *(_QWORD *)(v84 + 160), v77 == v76)
        || (unsigned int)sub_8D97D0(v77, v76, 0, v73, v61))) )
    {
      sub_6851C0(751, a1 + 8);
    }
    else
    {
      v78 = dword_4F077B8;
      if ( dword_4F077B8 )
      {
        if ( unk_4F04C48 == -1 )
          v78 = 0;
        else
          v78 = (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 6) == 0 ? 0x1000000 : 0;
      }
      if ( v91 != v84 && !(unsigned int)sub_8DED30(v84, v91, v78 | 0x141008u) )
        goto LABEL_75;
      if ( (*(_BYTE *)(v101 + 84) & 0x10) != 0 )
        ++unk_4F07488;
      else
        sub_6854C0(403, a1 + 8, v101);
    }
    *(_BYTE *)(a1 + 17) |= 0x20u;
    *(_QWORD *)(a1 + 24) = 0;
    goto LABEL_12;
  }
  if ( !dword_4F077BC
    || dword_4F077B4
    || *(_BYTE *)(v91 + 140) != 7
    || *(_BYTE *)(v84 + 140) != 7
    || (v79 = *(_QWORD *)(v84 + 160), v80 = *(_QWORD *)(v91 + 160), v79 == v80)
    || (unsigned int)sub_8D97D0(v79, v80, 0, v101, v61) )
  {
    if ( v92 == v101 || (sub_879190(v101, v92), !*(_QWORD *)(v92 + 88)) )
    {
      sub_881DB0(v92);
      goto LABEL_12;
    }
  }
LABEL_75:
  sub_8DCB20(v9);
  v13 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v102 + 168) + 152LL) + 240LL);
  v14 = sub_887500(20, a1, (*(_BYTE *)(v6 + 560) & 2) != 0, v92, v104);
LABEL_13:
  sub_877E20(v14, 0, v102);
  *(_QWORD *)v6 = v14;
  if ( dword_4F07590 )
  {
    if ( (*(_BYTE *)(v14 + 81) & 0x20) != 0 )
      v13 = -1;
  }
  else
  {
    v13 = -1;
  }
  v15 = sub_646F50(v9, 0, v13);
  *(_BYTE *)(v15 + 207) = (2 * *(_BYTE *)(v6 + 125)) & 0x10 | *(_BYTE *)(v15 + 207) & 0xEF;
  switch ( *(_BYTE *)(v14 + 80) )
  {
    case 4:
    case 5:
      v17 = *(_QWORD *)(*(_QWORD *)(v14 + 96) + 80LL);
      break;
    case 6:
      v17 = *(_QWORD *)(*(_QWORD *)(v14 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v17 = *(_QWORD *)(*(_QWORD *)(v14 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v17 = *(_QWORD *)(v14 + 88);
      break;
    default:
      MEMORY[0xB0] = v15;
      BUG();
  }
  v99 = v17;
  *(_QWORD *)(v17 + 176) = v15;
  *(__m128i *)(v17 + 184) = _mm_loadu_si128(a4);
  *(__m128i *)(v17 + 200) = _mm_loadu_si128(a4 + 1);
  *(__m128i *)(v17 + 216) = _mm_loadu_si128(a4 + 2);
  *(__m128i *)(v17 + 232) = _mm_loadu_si128(a4 + 3);
  *(__m128i *)(v17 + 248) = _mm_loadu_si128(a4 + 4);
  *(__m128i *)(v17 + 264) = _mm_loadu_si128(a4 + 5);
  *(_QWORD *)(v17 + 280) = a4[6].m128i_i64[0];
  a4->m128i_i64[1] = 0;
  v18 = sub_880CF0(v14, v15, a2);
  sub_877D80(v15, v18);
  sub_877E20(v18, v15, v102);
  v20 = v99;
  *(_BYTE *)(v15 + 88) = *(_BYTE *)(a5 + 12) & 3 | *(_BYTE *)(v15 + 88) & 0xFC;
  *(_QWORD *)(*(_QWORD *)(v18 + 96) + 64LL) = a4->m128i_i64[0];
  *(_QWORD *)(*(_QWORD *)(v18 + 96) + 104LL) = *(_QWORD *)(v99 + 192);
  a4[4].m128i_i8[1] |= 8u;
  v21 = v102;
  if ( *(_BYTE *)(v102 + 140) == 12 )
  {
    do
      v21 = *(_QWORD *)(v21 + 160);
    while ( *(_BYTE *)(v21 + 140) == 12 );
  }
  else
  {
    v21 = v102;
  }
  v22 = *(_QWORD *)(*(_QWORD *)v21 + 96LL);
  v23 = *(_QWORD *)(v6 + 8);
  if ( (v23 & 0x180000) != 0 )
  {
    v74 = (v23 & 0x100000) == 0;
    v24 = *(_BYTE *)(v15 + 193);
    if ( v74 )
      v25 = v24 | 1;
    else
      v25 = v24 | 4;
    *(_BYTE *)(v15 + 193) = v25;
    *(_BYTE *)(v15 + 193) = v25 | 2;
    if ( !v94 )
      *(_BYTE *)(v22 + 183) |= 8u;
    sub_736C90(v15, 1);
    v20 = v99;
  }
  if ( *(char *)(v15 + 192) >= 0 )
  {
    if ( (a4[4].m128i_i8[0] & 2) == 0 )
      goto LABEL_33;
    v100 = v20;
    sub_736C90(v15, 1);
    v20 = v100;
  }
  if ( (a4[4].m128i_i8[0] & 2) == 0 || dword_4D04824 )
  {
LABEL_33:
    v26 = *(_BYTE *)(v102 + 88);
    *(_BYTE *)(v15 + 172) = 1;
    *(_BYTE *)(v15 + 88) = v26 & 0x70 | *(_BYTE *)(v15 + 88) & 0x8F;
  }
  else
  {
    v39 = *(_BYTE *)(v15 + 88);
    *(_BYTE *)(v15 + 172) = 2;
    *(_BYTE *)(v15 + 88) = v39 & 0x8F | 0x10;
  }
  if ( (dword_4F07590 || *(char *)(v20 + 160) < 0) && (*(_BYTE *)(v14 + 81) & 0x20) == 0 )
    sub_7362F0(v15, 0xFFFFFFFFLL);
  v27 = *(_QWORD *)(v6 + 400);
  if ( v27 )
  {
    *(_QWORD *)(v15 + 216) = v27;
    *(_QWORD *)(v6 + 400) = 0;
  }
  if ( (*(_BYTE *)(a1 + 17) & 0x20) == 0 )
  {
    sub_729470(v15, v6 + 472);
    v28 = *(_BYTE *)(a1 + 16);
    if ( (v28 & 8) == 0 )
      goto LABEL_40;
    goto LABEL_66;
  }
  v28 = *(_BYTE *)(a1 + 16);
  if ( (v28 & 8) != 0 )
  {
LABEL_66:
    sub_725ED0(v15, 5);
    *(_BYTE *)(v15 + 176) = *(_BYTE *)(a1 + 56);
    goto LABEL_42;
  }
LABEL_40:
  if ( (v28 & 0x10) != 0 )
    sub_725ED0(v15, 3);
LABEL_42:
  sub_5F06F0((_BYTE *)v6, (__int64)a4, &dword_4F063F8, v20, v19);
  if ( v104[0] )
    sub_5EE4B0(v104[0]);
  result = a1;
  if ( (*(_BYTE *)(a1 + 17) & 0x20) == 0 )
  {
    v34 = *(_BYTE *)(a1 + 16);
    if ( (v34 & 8) != 0 )
    {
      switch ( *(_BYTE *)(v15 + 176) )
      {
        case 1:
          *(_BYTE *)(v22 + 179) |= 4u;
          break;
        case 2:
          *(_BYTE *)(v22 + 179) |= 0x10u;
          sub_5F93D0(v15, (__int64 *)(v6 + 288));
          break;
        case 3:
          *(_BYTE *)(v22 + 179) |= 8u;
          break;
        case 4:
          *(_BYTE *)(v22 + 179) |= 0x20u;
          sub_5F93D0(v15, (__int64 *)(v6 + 288));
          break;
        case 0xF:
          v69 = *(_QWORD *)(v22 + 32);
          v29 = v104[0];
          if ( v69 )
          {
            if ( *(_BYTE *)(v69 + 80) != 17 )
              *(_QWORD *)(v22 + 32) = v104[0];
          }
          else
          {
            if ( !v104[0] )
              v29 = v14;
            *(_QWORD *)(v22 + 32) = v29;
          }
          break;
        default:
          break;
      }
    }
    else if ( (v34 & 0x10) != 0 )
    {
      sub_5E6C40(v14, v22);
    }
    if ( (*(_BYTE *)(v6 + 560) & 2) != 0 )
    {
      sub_725ED0(v15, 1);
      v62 = *(_QWORD *)(v22 + 8);
      if ( v62 )
      {
        if ( *(_BYTE *)(v62 + 80) != 17 )
          *(_QWORD *)(v22 + 8) = v104[0];
      }
      else
      {
        *(_QWORD *)(v22 + 8) = v14;
      }
      if ( dword_4D04428 )
        sub_5E4BC0(v15, v102);
      v29 = *(unsigned __int8 *)(v6 + 131) << 7;
      *(_BYTE *)(v15 + 194) = (*(_BYTE *)(v6 + 131) << 7) | *(_BYTE *)(v15 + 194) & 0x7F;
      if ( *(char *)(v6 + 130) < 0 )
      {
        v63 = *(_QWORD *)(v6 + 320);
        *(_QWORD *)(v15 + 232) = v63;
        if ( v63 )
        {
          if ( (*(_BYTE *)(v63 + 198) & 0x10) != 0 )
            *(_BYTE *)(v15 + 198) |= 0x10u;
          if ( (*(_BYTE *)(*(_QWORD *)(v6 + 320) + 198LL) & 8) != 0 )
            *(_BYTE *)(v15 + 198) |= 8u;
        }
      }
    }
    sub_644920(v6, (a4[4].m128i_i8[0] & 4) != 0, v29, v30, v31, v32);
    v35 = v6 + 224;
    sub_648B00(v15, v6 + 224, a1 + 8, 0, (a4[4].m128i_i8[0] & 4) != 0, (a4[4].m128i_i8[0] & 2) != 0);
    v37 = *(_BYTE *)(v15 + 200);
    if ( (v37 & 7) == 0 )
    {
      v35 = v102;
      v36 = *(_BYTE *)(*(_QWORD *)(v102 + 168) + 109LL) & 7;
      *(_BYTE *)(v15 + 200) = v36 | v37 & 0xF8;
    }
    result = dword_4F04C3C;
    if ( !dword_4F04C3C )
    {
      v38 = *(_QWORD *)(v6 + 352);
      if ( v38 )
      {
        v35 = (unsigned int)dword_4F04C64;
        sub_869FD0(v38, (unsigned int)dword_4F04C64);
        *(_QWORD *)(v6 + 352) = 0;
      }
      if ( (a4[4].m128i_i8[0] & 4) != 0 )
      {
        *(_QWORD *)(v15 + 264) = a4[5].m128i_i64[0];
        *(_BYTE *)(v15 + 173) = *(_BYTE *)(v6 + 268);
      }
      else
      {
        *(_QWORD *)(sub_8921F0(a3, v35, v36) + 32) = a4[5].m128i_i64[0];
      }
      *(_QWORD *)(v6 + 352) = *(_QWORD *)(a3 + 96);
      return sub_65C210(v6);
    }
  }
  return result;
}
