// Function: sub_8BE350
// Address: 0x8be350
//
__int64 __fastcall sub_8BE350(__int64 a1, __int64 a2, unsigned int a3, __int64 *a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rdi
  unsigned int v9; // ecx
  unsigned __int64 v10; // rdi
  unsigned int v11; // edx
  __int16 v12; // r15
  unsigned __int64 v13; // r8
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  char j; // dl
  unsigned __int16 v19; // ax
  __int64 *v20; // rdx
  __int64 v21; // r15
  unsigned __int64 v22; // rax
  char v23; // al
  unsigned int v24; // edi
  __int64 result; // rax
  int v26; // eax
  unsigned __int64 v27; // rdi
  __int64 *v28; // rdi
  __int64 v29; // rax
  _QWORD *v30; // rdi
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // r14
  char v34; // al
  unsigned __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 *v40; // rdx
  __int64 v41; // r14
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  char v46; // al
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rdi
  __int64 v50; // rsi
  int v51; // edx
  unsigned int v52; // ecx
  _BOOL4 v53; // eax
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // r14
  char v57; // al
  __int64 v58; // rsi
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 k; // rax
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 **m; // rsi
  __int64 i; // rax
  __int64 v69; // rax
  __int64 v70; // [rsp+8h] [rbp-168h]
  _BYTE *v71; // [rsp+10h] [rbp-160h]
  unsigned __int8 v72; // [rsp+18h] [rbp-158h]
  int v74; // [rsp+2Ch] [rbp-144h] BYREF
  __m128i v75; // [rsp+30h] [rbp-140h] BYREF
  __m128i v76; // [rsp+40h] [rbp-130h]
  __m128i v77; // [rsp+50h] [rbp-120h]
  __m128i v78; // [rsp+60h] [rbp-110h]
  __m128i v79[6]; // [rsp+70h] [rbp-100h] BYREF
  int v80; // [rsp+D0h] [rbp-A0h] BYREF
  _QWORD *v81; // [rsp+D8h] [rbp-98h] BYREF
  unsigned int v82[10]; // [rsp+E8h] [rbp-88h] BYREF
  char v83; // [rsp+111h] [rbp-5Fh]

  v7 = a1 + 8;
  v72 = a2;
  *(_QWORD *)(v7 - 8) = 0;
  *(_QWORD *)(v7 + 456) = 0;
  v7 &= 0xFFFFFFFFFFFFFFF8LL;
  v9 = (unsigned int)(a1 - v7 + 472) >> 3;
  memset((void *)v7, 0, 8LL * v9);
  v10 = v7 + 8LL * v9;
  *(_QWORD *)(a1 + 152) = a1;
  *(_QWORD *)(a1 + 24) = *(_QWORD *)&dword_4F063F8;
  v11 = dword_4F077BC;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    *(_BYTE *)(a1 + 178) |= 1u;
  *(_QWORD *)(a1 + 123) = *(_QWORD *)(a1 + 123) & 0xF7FFFFFFFFFFFFF7LL
                        | (8LL * (word_4D04430 & 1))
                        | 0x800000000000000LL;
  *(_QWORD *)(a1 + 24) = *a4;
  if ( a3 )
  {
    v71 = 0;
    v12 = 9;
    memset(v79, 0, 0x58u);
  }
  else
  {
    v12 = 75;
    v71 = sub_869D30();
    sub_7B8B50(v10, (unsigned int *)a2, v42, v43, v44, v45);
    *a4 = *(_QWORD *)&dword_4F063F8;
    memset(v79, 0, 0x58u);
    v11 = dword_4F077BC;
  }
  if ( v11 && qword_4F077A8 <= 0x76BFu )
  {
    if ( dword_4F077C4 == 2 )
    {
      if ( (word_4F06418[0] != 1 || (word_4D04A10 & 0x200) == 0)
        && !(unsigned int)sub_7C0F00(0, 0, word_4F06418[0], 0, a5, a6) )
      {
        goto LABEL_8;
      }
    }
    else if ( word_4F06418[0] != 1 )
    {
      goto LABEL_8;
    }
    if ( (unsigned __int16)sub_7BE840(0, 0) != v12 )
      goto LABEL_8;
    v80 = 0;
    v32 = sub_7BF130(0, 0, &v80);
    v33 = v32;
    if ( !v32 )
      goto LABEL_65;
    v34 = *(_BYTE *)(v32 + 80);
    if ( v34 == 3 )
    {
      for ( i = *(_QWORD *)(v33 + 88); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v33 = *(_QWORD *)i;
      if ( v80 )
      {
        if ( !v33 )
          goto LABEL_65;
        goto LABEL_64;
      }
      if ( !v33 )
        goto LABEL_65;
      v34 = *(_BYTE *)(v33 + 80);
    }
    else if ( v80 )
    {
      goto LABEL_64;
    }
    if ( (unsigned __int8)(v34 - 4) <= 1u )
    {
      v69 = *(_QWORD *)(v33 + 88);
      if ( v69 )
      {
        if ( (*(_BYTE *)(v69 + 177) & 0x30) == 0x10 && (*(_BYTE *)(v69 + 178) & 1) == 0 )
        {
          v35 = v33;
          a2 = (unsigned __int8)a2;
          sub_8B1370(v33, a2, a4, a3, 1, 0);
          if ( !a3 )
          {
            v79[1].m128i_i64[0] = *a4;
            v79[1].m128i_i64[1] = qword_4F063F0;
            if ( !dword_4F04C3C )
            {
              a2 = a1;
              v35 = v72;
              sub_88F760(v72, (_QWORD *)a1, v33, v71, v79);
            }
          }
          goto LABEL_66;
        }
      }
    }
LABEL_64:
    if ( (*(_BYTE *)(v33 + 81) & 0x20) == 0 )
    {
      a2 = v33;
      v35 = 485;
      sub_6854E0(0x1E5u, v33);
      goto LABEL_66;
    }
LABEL_65:
    a2 = (__int64)dword_4F07508;
    v35 = 484;
    sub_6851C0(0x1E4u, dword_4F07508);
LABEL_66:
    sub_7B8B50(v35, (unsigned int *)a2, v36, v37, v38, v39);
    goto LABEL_23;
  }
LABEL_8:
  *(_QWORD *)(a1 + 184) = sub_5CC970(1);
  memset(v79, 0, 0x58u);
  sub_672A20((-(__int64)(dword_4F077BC == 0) & 0xFFFFFFFFFFBFFFFFLL) + 4259859, a1, (__int64)v79, 0, v13);
  v17 = *(_QWORD *)(a1 + 288);
  for ( j = *(_BYTE *)(v17 + 140); j == 12; j = *(_BYTE *)(v17 + 140) )
    v17 = *(_QWORD *)(v17 + 160);
  v19 = word_4F06418[0];
  if ( !j )
  {
    if ( word_4F06418[0] == 1 )
    {
      if ( dword_4F077C4 != 2 )
        goto LABEL_37;
      v20 = &qword_4D04A00;
      if ( (word_4D04A10 & 0x200) != 0 )
      {
        if ( (unk_4D04A12 & 1) == 0 )
          goto LABEL_37;
        goto LABEL_15;
      }
      v26 = sub_7C0F00(0, 0, (__int64)&qword_4D04A00, v14, v15, v16);
      v20 = &qword_4D04A00;
      if ( v26 && (unk_4D04A12 & 1) != 0 )
        goto LABEL_15;
      v19 = word_4F06418[0];
    }
    else if ( word_4F06418[0] != 34 && word_4F06418[0] != 27 )
    {
      v20 = (__int64 *)&dword_4F077C4;
      if ( dword_4F077C4 == 2 )
      {
        if ( word_4F06418[0] == 33 || dword_4D04474 && word_4F06418[0] == 52 )
          goto LABEL_37;
        v20 = (__int64 *)&dword_4D0485C;
        if ( dword_4D0485C )
        {
          if ( word_4F06418[0] == 25 )
            goto LABEL_37;
        }
        if ( word_4F06418[0] == 156 )
          goto LABEL_37;
      }
LABEL_15:
      v76 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v75.m128i_i64[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      v76.m128i_i8[1] |= 0x20u;
      v77 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v75.m128i_i64[1] = *(_QWORD *)dword_4F07508;
      v78 = _mm_loadu_si128(&xmmword_4F06660[3]);
      goto LABEL_16;
    }
  }
  if ( v12 == v19 && (*(_BYTE *)(a1 + 9) & 2) != 0 )
  {
    if ( (*(_BYTE *)(a1 + 124) & 0x20) != 0 )
      sub_6451E0(a1);
    v40 = *(__int64 **)(a1 + 288);
    v41 = *v40;
    if ( (unsigned __int8)(*(_BYTE *)(*v40 + 80) - 4) <= 1u )
    {
      v61 = *(_QWORD *)(v41 + 88);
      if ( v61 && (*(_DWORD *)(v61 + 176) & 0x13000) == 0x1000 )
      {
        *((_BYTE *)v40 + 178) &= ~1u;
        sub_8B1370(v41, a2, a4, a3, 1, 0);
        if ( !(dword_4F04C3C | a3) )
          sub_88F760(a2, (_QWORD *)a1, v41, v71, v79);
        goto LABEL_23;
      }
      if ( *(_QWORD *)(*(_QWORD *)(v41 + 96) + 72LL) )
        goto LABEL_23;
    }
    sub_6854E0(0x1E5u, *v40);
    goto LABEL_23;
  }
LABEL_37:
  if ( (_BYTE)a2 == 18 )
    sub_6851C0(0x146u, (_DWORD *)(a1 + 24));
  sub_87E3B0((__int64)&v80);
  v27 = 8709;
  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    v27 = (*(_BYTE *)(a1 + 120) & 0x7F) == 0 ? 74245LL : 8709LL;
  sub_626F50(v27, a1, 0, (__int64)&v75, (__int64)&v80, v79);
  v28 = *(__int64 **)(a1 + 200);
  if ( v28 )
    sub_66ABD0(v28);
  sub_876830((__int64)&v80);
  sub_87E350((__int64)&v80);
  if ( dword_4F04C64 != -1
    && (v20 = qword_4F04C68, v29 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v29 + 7) & 1) != 0)
    && ((v20 = (__int64 *)&dword_4F04C44, dword_4F04C44 != -1) || (*(_BYTE *)(v29 + 6) & 2) != 0)
    || (v83 & 8) != 0 )
  {
    v30 = *(_QWORD **)(a1 + 352);
    if ( !v30 )
      goto LABEL_16;
LABEL_47:
    sub_869FD0(v30, dword_4F04C64);
    v21 = v76.m128i_i64[1];
    *(_QWORD *)(a1 + 352) = 0;
    if ( v21 )
      goto LABEL_17;
    goto LABEL_48;
  }
  sub_87E280(&v81);
  v30 = *(_QWORD **)(a1 + 352);
  if ( v30 )
    goto LABEL_47;
LABEL_16:
  v21 = v76.m128i_i64[1];
  if ( v76.m128i_i64[1] )
  {
LABEL_17:
    sub_88F5B0(a1, (__int64)&v75);
    goto LABEL_18;
  }
LABEL_48:
  sub_7D5DD0(&v75, 0x20u, (__int64)v20, v14, v15);
  v21 = v76.m128i_i64[1];
  sub_88F5B0(a1, (__int64)&v75);
  if ( !v21 )
  {
    if ( (v76.m128i_i8[1] & 0x20) == 0 )
    {
      v31 = *(_QWORD *)(a1 + 288);
      if ( !v31 || (unsigned int)sub_8D2310(v31) )
      {
        sub_6849F0(8u, 0x321u, &v75.m128i_i32[2], *(_QWORD *)(v75.m128i_i64[0] + 8));
        goto LABEL_23;
      }
    }
LABEL_22:
    sub_6851C0(0x1E4u, a4);
    goto LABEL_23;
  }
LABEL_18:
  if ( (*(_WORD *)(v21 + 80) & 0x10FF) == 0x1010 )
  {
    sub_6851C0(0x12Au, &v75.m128i_i32[2]);
    v22 = *(unsigned __int8 *)(v21 + 80);
    if ( (_BYTE)v22 == 16 )
    {
      v21 = **(_QWORD **)(v21 + 88);
      v22 = *(unsigned __int8 *)(v21 + 80);
    }
    if ( (_BYTE)v22 == 24 )
    {
      v21 = *(_QWORD *)(v21 + 88);
      v22 = *(unsigned __int8 *)(v21 + 80);
    }
  }
  else
  {
    v22 = *(unsigned __int8 *)(v21 + 80);
    if ( (_BYTE)v22 == 24 )
    {
      v70 = *(_QWORD *)(v21 + 88);
      v53 = sub_8808B0(v21, v70);
      v54 = v70;
      if ( !v53 )
      {
        sub_6854E0(0x2F7u, v70);
        v54 = v70;
      }
      v22 = *(unsigned __int8 *)(v54 + 80);
      v21 = v54;
    }
  }
  if ( (((_BYTE)v22 - 7) & 0xFD) == 0 )
  {
    *(_QWORD *)a1 = v21;
    if ( sub_892240(v21) )
    {
      v46 = *(_BYTE *)(v21 + 80);
      if ( v46 == 9 || v46 == 7 )
      {
        v47 = *(_QWORD *)(v21 + 88);
      }
      else
      {
        if ( v46 != 21 )
          BUG();
        v47 = *(_QWORD *)(*(_QWORD *)(v21 + 88) + 192LL);
      }
      v49 = *(_QWORD *)(a1 + 288);
      v50 = *(_QWORD *)(v47 + 120);
      if ( v49 == v50 || (unsigned int)sub_8DED30(v49, v50, 5) )
      {
        if ( !unk_4D03FD8 && sub_891C80(v21, 1, a3, v72) )
          sub_8ACB90(v21, v72, a4, 0, a3, 0, 1);
        if ( !(dword_4F04C3C | a3) )
          sub_88F760(v72, (_QWORD *)a1, v21, v71, v79);
      }
      else
      {
        sub_6853B0(8u, 0x93u, (FILE *)&v75.m128i_u64[1], v21);
      }
      v51 = 0;
      v52 = dword_4F077BC;
      goto LABEL_92;
    }
    sub_6854E0(0x1E5u, v21);
LABEL_23:
    v23 = *(_BYTE *)(a1 + 268);
    if ( !v23 )
      goto LABEL_27;
    goto LABEL_24;
  }
  if ( (unsigned __int8)v22 > 0x14u )
    goto LABEL_22;
  v48 = 1182720;
  if ( !_bittest64(&v48, v22) )
    goto LABEL_22;
  if ( !(unsigned int)sub_8D2310(*(_QWORD *)(a1 + 288)) )
  {
    sub_6854C0(0x93u, (FILE *)&v75.m128i_u64[1], v21);
    goto LABEL_23;
  }
  v55 = sub_8B8140((_BYTE *)v21, a1, (__int64)&v75, 0, 1, 0, 0, 8u, &v74);
  v56 = v55;
  if ( v55 )
  {
    *(_QWORD *)a1 = v55;
    if ( *(_BYTE *)(v55 + 80) == 10 )
    {
      for ( k = *(_QWORD *)(*(_QWORD *)(v55 + 88) + 152LL); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
        ;
      if ( *(_QWORD *)(*(_QWORD *)(k + 168) + 40LL) )
      {
        v63 = *(_QWORD *)(v56 + 64);
        v64 = *(_QWORD *)(*(_QWORD *)(a1 + 288) + 168LL);
        *(_BYTE *)(v64 + 21) |= 1u;
        *(_QWORD *)(v64 + 40) = v63;
      }
    }
    if ( !unk_4D03FD8 && sub_891C80(v56, 1, a3, a2) )
      sub_8ACB90(v56, a2, a4, 0, a3, 0, 1);
    if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 288) + 168LL) + 56LL) )
    {
      sub_894C00(v56);
      sub_6464A0(*(_QWORD *)(a1 + 288), v56, v82, 1u);
    }
    sub_644920((_QWORD *)a1, 0);
    sub_648B00(*(_QWORD *)(v56 + 88), (_BYTE *)(a1 + 224), (__int64)&v75.m128i_i64[1]);
    v57 = *(_BYTE *)(v21 + 80);
    v58 = *(_QWORD *)(a1 + 288);
    if ( v57 != 20 )
    {
      if ( !*(_BYTE *)(*(_QWORD *)(v58 + 168) + 25LL) )
        goto LABEL_115;
      goto LABEL_109;
    }
    if ( (*(_BYTE *)(*(_QWORD *)(v56 + 88) + 198LL) & 0x20) != 0 && (v65 = *(_QWORD *)(v21 + 88)) != 0 )
    {
      while ( *(_BYTE *)(v58 + 140) == 12 )
        v58 = *(_QWORD *)(v58 + 160);
      v66 = *(_QWORD *)(*(_QWORD *)(v65 + 176) + 152LL);
      for ( m = **(__int64 ****)(v58 + 168); *(_BYTE *)(v66 + 140) == 12; v66 = *(_QWORD *)(v66 + 160) )
        ;
      sub_826B90(**(__int64 ****)(v66 + 168), m, 0xE75u, (_DWORD *)(a1 + 32), (_QWORD *)(v21 + 48));
      if ( !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 288) + 168LL) + 25LL) )
        goto LABEL_115;
      v57 = *(_BYTE *)(v21 + 80);
      if ( v57 != 20 )
      {
LABEL_109:
        if ( v57 == 10 )
        {
          v59 = *(_QWORD *)(v21 + 96);
          if ( v59 )
          {
            v60 = *(_QWORD *)(v59 + 56);
LABEL_112:
            if ( v60 && !(unsigned int)sub_8D7310(*(_QWORD *)(*(_QWORD *)(v60 + 176) + 152LL)) )
              sub_6851C0(0x287u, (_DWORD *)(a1 + 32));
          }
        }
LABEL_115:
        if ( !(dword_4F04C3C | a3) )
          sub_88F760(v72, (_QWORD *)a1, v56, v71, v79);
        v51 = 0;
        v52 = dword_4F077BC;
        if ( dword_4F077BC )
          v51 = ((*(_BYTE *)(v56 + 81) >> 4) ^ 1) & 1;
        goto LABEL_92;
      }
    }
    else if ( !*(_BYTE *)(*(_QWORD *)(v58 + 168) + 25LL) )
    {
      goto LABEL_115;
    }
    v60 = *(_QWORD *)(v21 + 88);
    goto LABEL_112;
  }
  v52 = 0;
  v51 = 0;
LABEL_92:
  v23 = *(_BYTE *)(a1 + 268);
  if ( !v23 )
    goto LABEL_27;
  if ( v52 && v23 == 1 || v23 == 2 && v51 )
  {
    sub_684B30(0x56Au, (_DWORD *)(a1 + 260));
    goto LABEL_27;
  }
LABEL_24:
  v24 = 935;
  if ( v23 != 4 )
    v24 = 80;
  sub_6851C0(v24, (_DWORD *)(a1 + 260));
LABEL_27:
  result = sub_643EB0(a1, 0);
  if ( (a3 & 1) == 0 )
  {
    result = (__int64)v71;
    if ( v71 )
    {
      if ( !v71[16] )
        return sub_869FD0(v71, dword_4F04C64);
    }
  }
  return result;
}
