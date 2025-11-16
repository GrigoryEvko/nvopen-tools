// Function: sub_5F1000
// Address: 0x5f1000
//
__int64 __fastcall sub_5F1000(__m128i *a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  __int64 v8; // rbx
  __int64 v9; // rax
  char v10; // dl
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rcx
  __int64 v14; // r8
  _BYTE *v15; // rax
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  __int64 *v22; // rsi
  __m128i v23; // xmm7
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  char v27; // dl
  int v28; // eax
  __int8 v29; // al
  __int64 v30; // rdx
  _QWORD *v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rcx
  char v35; // al
  __int64 *v36; // r8
  __int64 v37; // r9
  char v38; // al
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rdx
  __int64 *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // [rsp-10h] [rbp-90h]
  __int64 v54; // [rsp-8h] [rbp-88h]
  __int64 v55; // [rsp+8h] [rbp-78h]
  __int64 v56; // [rsp+10h] [rbp-70h]
  __int64 v57; // [rsp+10h] [rbp-70h]
  __int64 v58; // [rsp+18h] [rbp-68h]
  char v59; // [rsp+18h] [rbp-68h]
  unsigned __int64 v60; // [rsp+20h] [rbp-60h]
  __int64 v61; // [rsp+20h] [rbp-60h]
  __int64 v63; // [rsp+28h] [rbp-58h]
  _BYTE v64[4]; // [rsp+3Ch] [rbp-44h] BYREF
  char v65[8]; // [rsp+40h] [rbp-40h] BYREF
  _DWORD v66[14]; // [rsp+48h] [rbp-38h] BYREF

  v8 = *((_QWORD *)a4 + 36);
  v9 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v10 = *(_BYTE *)(v9 + 6) & 6;
  if ( unk_4F04C44 != -1 )
  {
    if ( v10 )
      goto LABEL_3;
    goto LABEL_36;
  }
  if ( !v10 && *(_BYTE *)(v9 + 4) == 12 )
  {
LABEL_36:
    a1[1].m128i_i8[1] |= 0x20u;
    a1[1].m128i_i64[1] = 0;
    sub_854B40();
  }
LABEL_3:
  if ( (a1[1].m128i_i8[1] & 0x20) != 0 )
  {
LABEL_4:
    v63 = sub_646F50(v8, 2, 0xFFFFFFFFLL);
    v11 = sub_885AD0(11, a1, 0, 0);
    v12 = v11;
    *(_QWORD *)(v11 + 88) = v63;
    *(_QWORD *)a4 = v11;
    a4[127] |= 0x10u;
    *((_QWORD *)a4 + 37) = 0;
    if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0 )
      *(_BYTE *)(v63 + 195) |= 9u;
    sub_877D80(*(_QWORD *)(v11 + 88), v11);
    goto LABEL_7;
  }
  v17 = a1[1].m128i_i64[1];
  if ( !v17 && (a1[1].m128i_i8[2] & 1) != 0 )
    v17 = sub_7D5DD0(a1, 0x4000);
  if ( unk_4F04C44 != -1
    || (v20 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v20 + 6) & 6) != 0)
    || *(_BYTE *)(v20 + 4) == 12 )
  {
    if ( (*(_BYTE *)(a3 + 64) & 4) != 0 )
    {
      if ( (a1[1].m128i_i8[0] & 1) != 0 )
      {
        sub_6854C0(551, &a1->m128i_u64[1], v17);
        *a1 = _mm_loadu_si128(xmmword_4F06660);
        a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
        a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
        v25 = *(_QWORD *)dword_4F07508;
        a1[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
        a1[1].m128i_i8[1] |= 0x20u;
        a1->m128i_i64[1] = v25;
        goto LABEL_4;
      }
      a4[122] |= 1u;
    }
    else
    {
      v26 = *((_QWORD *)a4 + 50);
      if ( v26 )
        sub_6851C0(3159, v26 + 8);
    }
    goto LABEL_27;
  }
  if ( (a4[133] & 4) != 0 )
  {
    a4[122] = ((*(_BYTE *)(a3 + 64) & 4) != 0) | a4[122] & 0xFE;
LABEL_27:
    v18 = sub_5EDE90(1u, 0, a1, (__int64)a4, a3, (__int64)(a4 + 472));
    a4[127] |= 0x10u;
    *(_QWORD *)a4 = v18;
    v12 = v18;
    goto LABEL_28;
  }
  if ( (a1[1].m128i_i8[1] & 0x20) != 0 )
    goto LABEL_4;
  if ( !unk_4D04764 )
    goto LABEL_111;
  if ( !v17 )
  {
    v34 = a1->m128i_i64[0];
    v27 = *(_BYTE *)(a3 + 64);
    v35 = *(_BYTE *)(a1->m128i_i64[0] + 73) & 2;
    if ( (v27 & 4) != 0 )
    {
      if ( !v35 || strcmp(*(const char **)(v34 + 8), "main") )
      {
        LODWORD(v60) = 259;
        goto LABEL_114;
      }
      LODWORD(v60) = 259;
    }
    else
    {
      if ( !v35 || strcmp(*(const char **)(v34 + 8), "main") )
      {
        sub_6523A0(
          (_DWORD)a1,
          (_DWORD)a4,
          a3,
          257,
          (unsigned int)v64,
          (unsigned int)v66,
          (__int64)v65,
          (__int64)(a4 + 472));
LABEL_75:
        v32 = v53;
        v12 = *(_QWORD *)a4;
        v33 = v54;
        if ( (*(_BYTE *)(a3 + 64) & 4) != 0 && (a1[1].m128i_i8[1] & 0x20) == 0 && (*(_BYTE *)(a2 + 89) & 1) != 0 )
          sub_6854C0(551, &dword_4F063F8, *(_QWORD *)a4);
        if ( unk_4D047A0 )
        {
          if ( unk_4F04C44 == -1 )
          {
            v31 = qword_4F04C68;
            v50 = qword_4F04C68[0] + 776LL * dword_4F04C64;
            if ( (*(_BYTE *)(v50 + 6) & 6) == 0
              && *(_BYTE *)(v50 + 4) != 12
              && (*(_DWORD *)(a2 + 176) & 0x11000) == 0x1000 )
            {
              v30 = *(_QWORD *)(v12 + 88);
              if ( *(char *)(v30 + 202) < 0 && (*(_BYTE *)(a3 + 64) & 0x1C) == 4 )
              {
                *(_QWORD *)(v30 + 344) = qword_4CF8008;
                v30 = (a4[133] & 2) != 0;
                *(_BYTE *)(v12 + 104) = v30 | *(_BYTE *)(v12 + 104) & 0xFE;
              }
            }
          }
        }
        if ( unk_4D047D8 && (*(_BYTE *)(v12 + 83) & 0x40) != 0 )
          sub_87FDB0(v12, a2, v30, v31, v32, v33);
        goto LABEL_49;
      }
      LODWORD(v60) = 257;
    }
LABEL_69:
    v29 = a1[1].m128i_i8[0];
    if ( (v29 & 1) != 0 )
    {
      if ( (v29 & 4) == 0 )
        goto LABEL_71;
LABEL_142:
      *(_BYTE *)(a3 + 64) = v27 | 0x20;
      v66[0] = (v27 & 2) != 0;
      sub_650B30(a3, v8, a4, v66, &a1->m128i_u64[1]);
      *(_BYTE *)(a3 + 64) = (2 * (v66[0] & 1)) | *(_BYTE *)(a3 + 64) & 0xFD;
      goto LABEL_74;
    }
    if ( !unk_4F04C34 )
      goto LABEL_142;
LABEL_71:
    if ( (v27 & 4) == 0 )
    {
      if ( !v17 || *(_BYTE *)(v17 + 80) != 24 || (*(_BYTE *)(v17 + 82) & 4) == 0 )
        goto LABEL_74;
      v51 = a1[1].m128i_i64[1];
      if ( v51 && (*(_BYTE *)(v51 + 82) & 4) != 0 )
        sub_87DC80(a1, 0, 0, 1, a5, v66);
      goto LABEL_131;
    }
LABEL_114:
    if ( (*(_BYTE *)(a2 + 89) & 1) != 0 )
    {
      *(_BYTE *)(a3 + 64) &= ~2u;
LABEL_74:
      sub_6523A0(
        (_DWORD)a1,
        (_DWORD)a4,
        a3,
        v60,
        (unsigned int)v64,
        (unsigned int)v66,
        (__int64)v65,
        (__int64)(a4 + 472));
      goto LABEL_75;
    }
    if ( ((a1[1].m128i_i8[2] & 2) != 0 || !a1[2].m128i_i64[0]) && (a1[1].m128i_i8[0] & 4) == 0 )
      goto LABEL_74;
    sub_6851C0(998, &a1->m128i_u64[1]);
LABEL_131:
    sub_878790(a1);
    a1[1].m128i_i8[1] |= 0x20u;
    a1[1].m128i_i64[1] = 0;
    goto LABEL_74;
  }
  if ( (*(_BYTE *)(v17 + 82) & 4) != 0 )
  {
LABEL_111:
    if ( dword_4F077C4 == 2 )
    {
      v46 = a1[1].m128i_i64[1];
      if ( v46 )
      {
        if ( (*(_DWORD *)(v46 + 80) & 0x41000) != 0 )
          sub_8841F0(a1, 0, 0, 0);
      }
    }
    v27 = *(_BYTE *)(a3 + 64);
    v60 = (-(__int64)((v27 & 4) == 0) & 0xFFFFFFFFFFFFFFFELL) + 259;
    v58 = *(_QWORD *)(a3 + 72);
    if ( !v17 )
      goto LABEL_67;
  }
  else
  {
    v58 = *(_QWORD *)(a3 + 72);
    v60 = (-(__int64)((*(_BYTE *)(a3 + 64) & 4) == 0) & 0xFFFFFFFFFFFFFFFELL) + 259;
  }
  if ( (*(_BYTE *)(v17 + 81) & 0x10) == 0 )
  {
    if ( (unsigned int)sub_8782B0(v17) )
      goto LABEL_90;
    v27 = *(_BYTE *)(a3 + 64);
LABEL_67:
    if ( (*(_BYTE *)(a1->m128i_i64[0] + 73) & 2) == 0 )
      goto LABEL_71;
    v59 = v27;
    v28 = strcmp(*(const char **)(a1->m128i_i64[0] + 8), "main");
    v27 = v59;
    if ( v28 )
      goto LABEL_71;
    goto LABEL_69;
  }
  v21 = *(unsigned __int8 *)(v17 + 80);
  v22 = &a1->m128i_i64[1];
  if ( (unsigned __int8)v21 > 0x14u )
    goto LABEL_47;
  v48 = 1180672;
  if ( !_bittest64(&v48, v21) )
  {
    v22 = &a1->m128i_i64[1];
    if ( (_BYTE)v21 == 16 )
    {
      sub_6851C0(298, v22);
      goto LABEL_48;
    }
LABEL_47:
    sub_6854C0(147, v22, v17);
LABEL_48:
    v12 = 0;
    *a1 = _mm_loadu_si128(xmmword_4F06660);
    a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
    a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
    v23 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v24 = *(_QWORD *)dword_4F07508;
    a1[1].m128i_i8[1] |= 0x20u;
    a1[3] = v23;
    a1->m128i_i64[1] = v24;
    goto LABEL_49;
  }
LABEL_90:
  if ( *(_QWORD *)(v17 + 64) == a2 )
  {
    v49 = 5;
    if ( unk_4D04964 )
      v49 = unk_4F07472;
    sub_684AC0(v49, 522);
  }
  v12 = sub_8B8140(v17, (_DWORD)a4, (_DWORD)a1, 0, a1[1].m128i_i8[2] & 1, 0, 0, 8, (__int64)v66);
  if ( v12 )
  {
    if ( (*(_BYTE *)(a3 + 64) & 6) == 2 && *(char *)(*(_QWORD *)(v12 + 88) + 192LL) >= 0 )
      sub_6851C0(326, dword_4F07508);
    v36 = &a1->m128i_i64[1];
    if ( (*(_BYTE *)(v12 + 81) & 2) == 0 || (*(_BYTE *)(a3 + 64) & 4) == 0 )
    {
      v37 = *(_QWORD *)(v12 + 88);
      *(_QWORD *)a4 = v12;
      *((_QWORD *)a4 + 37) = *(_QWORD *)(v37 + 152);
      if ( (*(_BYTE *)(a3 + 64) & 4) != 0 )
      {
        v55 = v37;
        sub_6854C0(551, &a1->m128i_u64[1], v12);
        v37 = v55;
        v36 = &a1->m128i_i64[1];
      }
      v38 = *(_BYTE *)(v37 + 193);
      if ( (v38 & 0x10) != 0 )
      {
        *(_BYTE *)(v37 + 193) = v38 & 0xEF;
        v56 = v37;
        sub_8756F0(v60, v12, v36, v58);
        v41 = v56;
        *(_BYTE *)(v56 + 193) |= 0x10u;
      }
      else
      {
        v57 = v37;
        sub_8756F0(v60, v12, v36, v58);
        v41 = v57;
      }
      if ( (*(_BYTE *)(a3 + 64) & 4) == 0 )
      {
        v42 = 0;
        if ( dword_4F04C64 != -1
          && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 2) != 0
          && dword_4F077C4 == 2
          && (*(_BYTE *)(v41 - 8) & 1) != 0
          && (a1[1].m128i_i8[2] & 0x40) == 0 )
        {
          v61 = v41;
          v52 = sub_7CAFF0(a1, v41, 0);
          v41 = v61;
          v42 = v52;
        }
        sub_86A3D0(v41, *(_QWORD *)(a3 + 80), v42, 2, a4 + 472);
      }
      v43 = (__int64 *)qword_4CF7FD0;
      if ( qword_4CF7FD0 )
        qword_4CF7FD0 = *(_QWORD *)qword_4CF7FD0;
      else
        v43 = (__int64 *)sub_823970(40);
      v44 = qword_4CF7FC8;
      v43[1] = v12;
      v43[2] = 0;
      *v43 = v44;
      v43[3] = v8;
      v45 = *(_QWORD *)(a3 + 24);
      qword_4CF7FC8 = (__int64)v43;
      v43[4] = v45;
      sub_644920(a4, 0, v45, v39, v40, v41);
      goto LABEL_49;
    }
    sub_685920(&a1->m128i_u64[1], v12, 8);
  }
  *a1 = _mm_loadu_si128(xmmword_4F06660);
  a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
  a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
  v47 = *(_QWORD *)dword_4F07508;
  a1[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
  a1[1].m128i_i8[1] |= 0x20u;
  a1->m128i_i64[1] = v47;
LABEL_49:
  sub_65C210(a4);
  if ( (a1[1].m128i_i8[1] & 0x20) != 0 )
    goto LABEL_4;
  if ( (*(_BYTE *)(a2 + 177) & 0x20) == 0 || dword_4F07590 )
    sub_5EDDD0(*(_QWORD *)(v12 + 88), a2);
  if ( unk_4D04964 && (*(_BYTE *)(a3 + 65) & 1) != 0 )
  {
    if ( (*(_BYTE *)(v12 + 81) & 0x10) != 0 )
    {
      sub_684AA0(unk_4F07472, 1428, a4 + 24);
    }
    else
    {
      if ( (*(_BYTE *)(a3 + 64) & 4) != 0 )
        goto LABEL_8;
      sub_684AA0(unk_4F07472, 1429, a4 + 24);
    }
LABEL_7:
    if ( (*(_BYTE *)(a3 + 64) & 4) == 0 )
      goto LABEL_15;
LABEL_8:
    if ( *(_BYTE *)(v12 + 80) != 11 )
      goto LABEL_9;
    goto LABEL_31;
  }
LABEL_28:
  if ( (*(_BYTE *)(a3 + 64) & 4) == 0 )
    goto LABEL_15;
  if ( v12 && *(_BYTE *)(v12 + 80) == 11 )
  {
LABEL_31:
    v19 = *(_QWORD *)(v12 + 88);
    if ( v19 && (*(_BYTE *)(v19 + 198) & 0x20) != 0 )
      sub_684AA0(7, 3643, a4 + 24);
  }
LABEL_9:
  sub_64F530(v12);
  if ( *(_QWORD *)a4 )
    sub_5F06F0(a4, a3, &dword_4F063F8, v13, v14);
  if ( unk_4F04C48 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 6) != 0 )
    goto LABEL_15;
  *(_BYTE *)(*(_QWORD *)(v12 + 88) + 208LL) |= 2u;
  while ( *(_BYTE *)(v8 + 140) == 12 )
  {
    v8 = *(_QWORD *)(v8 + 160);
LABEL_15:
    ;
  }
  v15 = *(_BYTE **)(*(_QWORD *)(v8 + 168) + 56LL);
  if ( v15 && (*v15 & 0x20) != 0 )
  {
    if ( unk_4F04C48 == -1 || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 6) != 0 )
      *(_BYTE *)(qword_4CF8008 + 184) |= 8u;
    else
      *(_BYTE *)(a3 + 65) |= 8u;
  }
  sub_854980(v12, 0);
  return v12;
}
